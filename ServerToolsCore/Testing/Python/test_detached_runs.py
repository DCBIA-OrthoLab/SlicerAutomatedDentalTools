"""A run that outlives the request that asked for it.

The POST used to stay open for the whole run, and two things followed from
that. The read timeout is 600 s by default and an hour at the very most a user
can dial in, while a cohort legitimately runs for longer. And a dropped
connection never stopped the run -- nothing on the server cancels a worker
thread -- it only threw the answer away after the GPU had been spent on it.

Detached, the POST answers 202 as soon as the inputs are staged and the result
arrives on the event stream this client already watches. The stream reconnects
and dedupes on `seq`, so the wait survives what a blocking POST could not.

Against a REAL socket, like test_run_progress.py and for the same reason: what
is under test is a response arriving before the work starts, and a mocked
Session cannot show that ordering.

Usage:
    python3 -m unittest ServerToolsCore/Testing/Python/test_detached_runs.py
"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.parse import urlsplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tls_fixture import serve_tls, trust_test_ca  # noqa: E402
from ServerToolsCoreLib.client import (
    ToolServerClient,
    new_run_id,
    normalise_run_event,
)
from ServerToolsCoreLib.errors import RunCancelled, ServerToolError

_TOOL = "Probe"
# One optional path argument, so the same fake tool serves both the runs that
# send no file and the ones that send a small one.
_SCHEMA = [{"name": _TOOL, "output_kind": "text",
            "arguments": {"input": {"type": "path", "required": False}}}]


def _terminal(seq, state="done", result=None, message=""):
    payload = {"seq": seq, "at": 1757400000.0, "state": state, "phase": state,
               "fraction": 1.0, "message": message, "depth": 0}
    if result is not None:
        payload["result"] = result
    return payload


class _State:
    def __init__(self):
        self.events = []
        self.delivery = []       # every X-Run-Delivery the POST saw
        self.run_ids = []
        self.detached_status = 202
        self.streams = 0
        self.uploads = []        # every file opened through POST /uploads
        self.parts = 0
        self.run_body_bytes = 0  # how much travelled inside the /run request
        self.has_uploads_field = None
        self.lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    state = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path == "/uploads":
            opened = json.loads(body or b"{}")
            with self.state.lock:
                self.state.uploads.append(opened.get("filename"))
            return self._json({"upload_id": "u-%d" % len(self.state.uploads),
                               "chunk_size": 8 * 1024 * 1024, "part_count": 1})
        with self.state.lock:
            self.state.run_body_bytes = len(body)
            self.state.has_uploads_field = b"__uploads__" in body
        delivery = self.headers.get("X-Run-Delivery")
        run_id = self.headers.get("X-Run-Id")
        with self.state.lock:
            self.state.delivery.append(delivery)
            self.state.run_ids.append(run_id)
        if delivery == "detached" and self.state.detached_status == 202:
            return self._json({"run_id": run_id, "status": "accepted"}, status=202)
        # Either the client did not ask, or this server does not know the
        # header: the ordinary answer, in the body, as it always was.
        return self._json({"result": "finished"})

    def do_GET(self):
        if self.path == "/tools":
            return self._json(_SCHEMA)
        # The path alone: the watcher asks for `?logs=user`, and a server
        # that has no log lines ignores the query rather than 404ing on it.
        if not urlsplit(self.path).path.endswith("/events"):
            return self._json({"detail": "not found"}, status=404)
        with self.state.lock:
            self.state.streams += 1
            replayed = self.state.streams > 1
        if replayed:
            # The run is gone. `watch_run` reconnects a stream that ended
            # without a terminal event -- correctly, because a real run lasts
            # hours and proxies drop idle connections -- and a 404 after events
            # have arrived is how it learns the run was reaped rather than
            # merely quiet. Without this the fake server would answer 200 for
            # ever and no client could tell the difference.
            return self._json({"detail": "unknown run"}, status=404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for event in self.state.events:
            self._chunk(f"data: {json.dumps(event)}\n\n".encode())
        self._chunk(b"")

    def do_PUT(self):
        """A part of a chunked upload. Accepted without inspection: what is
        under test is WHICH path the client chose, not the transfer itself."""
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        with self.state.lock:
            self.state.parts += 1
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _chunk(self, body):
        self.wfile.write(b"%x\r\n" % len(body) + body + b"\r\n")
        self.wfile.flush()


class _Live(unittest.TestCase):
    def setUp(self):
        self.state = _State()
        handler = type("_Bound", (_Handler,), {"state": self.state})
        trust_test_ca(self)
        self.server = serve_tls(ThreadingHTTPServer(("127.0.0.1", 0), handler))
        self.server.handle_error = lambda *_args: None
        threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"https://localhost:{self.server.server_address[1]}"

    def _client(self, detached=True):
        return ToolServerClient(self.url, token="t", 
                                timeout=10, detached_runs=detached)


# ----------------------------------------------------------------------
# The event carries the answer
# ----------------------------------------------------------------------

class EventShapeTest(unittest.TestCase):
    def test_a_terminal_event_can_carry_where_the_result_is(self):
        """Because the response that used to carry it was a 202, sent before
        the tool had started."""
        reference = {"result_ref": {"result_id": "abc", "size": 12}}
        event = normalise_run_event(_terminal(4, result=reference))
        self.assertEqual(event["result"], reference)

    def test_an_ordinary_event_carries_none(self):
        self.assertNotIn("result", normalise_run_event(_terminal(1, result=None)))

    def test_a_result_that_is_not_an_object_is_dropped(self):
        for junk in ("/etc/passwd", 7, ["a"], None):
            self.assertNotIn("result", normalise_run_event(_terminal(1, result=junk)))


# ----------------------------------------------------------------------
# The bargain: a client that says nothing changes nothing
# ----------------------------------------------------------------------

class OptInTest(_Live):
    def test_off_by_default_no_header_is_sent(self):
        result = self._client(detached=False).run(_TOOL, run_id=new_run_id())
        self.assertEqual(result.text, "finished")
        self.assertEqual(self.state.delivery, [None])

    def test_on_the_header_is_sent(self):
        self.state.events = [_terminal(1, result={"result": "finished"})]
        self._client().run(_TOOL, run_id=new_run_id())
        self.assertEqual(self.state.delivery, ["detached"])

    def test_without_a_run_id_it_stays_blocking(self):
        """Detaching needs an id to report through. A caller that minted none
        keeps the blocking contract whatever the setting says."""
        result = self._client().run(_TOOL)
        self.assertEqual(result.text, "finished")
        self.assertEqual(self.state.delivery, [None])

    def test_a_server_that_answers_200_is_still_understood(self):
        """An older server ignores the header and sends the ordinary body.
        One client, both servers."""
        self.state.detached_status = 200
        result = self._client().run(_TOOL, run_id=new_run_id())
        self.assertEqual(result.text, "finished")


# ----------------------------------------------------------------------
# Collecting the answer
# ----------------------------------------------------------------------

class CollectTest(_Live):
    def test_a_text_answer_comes_back_on_the_terminal_event(self):
        self.state.events = [
            _terminal(1, state="running"),
            _terminal(2, result={"result": "finished"}),
        ]
        self.assertEqual(self._client().run(_TOOL, run_id=new_run_id()).text, "finished")

    def test_a_reference_is_fetched_the_way_a_blocking_run_fetches_one(self):
        reference = {"result_id": "abc", "filename": "out.zip", "size": 12}
        self.state.events = [_terminal(1, result={"result_ref": reference})]
        client = self._client()
        with mock.patch.object(client, "_download_reference") as download:
            client.run(_TOOL, run_id=new_run_id(), output_dir="/tmp")
        download.assert_called_once()
        self.assertEqual(download.call_args[0][1], reference)

    def test_progress_still_reaches_the_caller(self):
        """The same stream carries both. Detaching must not cost the panel its
        progress bar."""
        self.state.events = [
            _terminal(1, state="running", message="one of four"),
            _terminal(2, result={"result": "finished"}),
        ]
        seen = []
        self._client().run(_TOOL, run_id=new_run_id(), event_cb=seen.append)
        self.assertEqual([e["message"] for e in seen], ["one of four", ""])

    def test_a_log_line_reaches_the_caller_and_is_never_taken_for_the_end(self):
        """A detached run is collected from the terminal event it waits for. A
        log line claiming `done` must neither be that event nor stop the wait
        before the real one arrives with the answer on it."""
        self.state.events = [
            {"seq": 1, "at": 1757400000.0, "kind": "log", "state": "done",
             "phase": "done", "level": "warning", "message": "scan 4 skipped",
             "depth": 0},
            _terminal(2, result={"result": "finished"}),
        ]
        seen = []
        result = self._client().run(_TOOL, run_id=new_run_id(), event_cb=seen.append)
        self.assertEqual(result.text, "finished")
        self.assertEqual([e.get("kind") for e in seen], ["log", None])


# ----------------------------------------------------------------------
# How it ends badly
# ----------------------------------------------------------------------

class SmallFileTest(_Live):
    """A detached run has to send EVERY file through POST /uploads.

    The client only chunked files of at least 2 x chunk_size -- 16 MB by
    default -- and sent anything smaller as a multipart part of the /run
    request. The server refuses exactly that on a detached run, and correctly:
    it answers 202 before the tool starts, so there is no moment at which it
    could stage a body it has already replied to.

    Nothing tested either feature against the other, so every detached run
    taking a small input -- a landmark file, an ROI box, a spreadsheet -- was a
    400 the moment DETACHED_RUNS was turned on. Both halves worked; the pair
    did not.
    """

    def _tiny(self):
        import tempfile
        handle = tempfile.NamedTemporaryFile(suffix=".vtk", delete=False)
        handle.write(b"# vtk DataFile Version 3.0\n")
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_a_small_file_goes_through_uploads_when_detached(self):
        self.state.events = [_terminal(1, result={"result": "finished"})]
        path = self._tiny()
        self._client().run(_TOOL, run_id=new_run_id(), files={"input": path})
        self.assertEqual(self.state.uploads, [os.path.basename(path)])
        self.assertGreaterEqual(self.state.parts, 1)
        self.assertTrue(self.state.has_uploads_field,
                        "the run request must reference the upload, not carry the file")

    def test_a_small_file_still_rides_the_request_when_blocking(self):
        """The threshold is right for a blocking run: two extra round trips
        cost more than a 26-byte multipart part. Only detaching changes it."""
        path = self._tiny()
        self._client(detached=False).run(_TOOL, run_id=new_run_id(),
                                         files={"input": path})
        self.assertEqual(self.state.uploads, [])
        self.assertFalse(self.state.has_uploads_field)
        self.assertGreater(self.state.run_body_bytes, 0)


class FailureTest(_Live):
    def test_a_failed_run_raises_with_what_the_server_said(self):
        self.state.events = [_terminal(1, state="failed", message="landmark 'Zz' is unknown")]
        with self.assertRaises(ServerToolError) as caught:
            self._client().run(_TOOL, run_id=new_run_id())
        self.assertIn("Zz", str(caught.exception))

    def test_a_failed_run_s_message_is_capped(self):
        from ServerToolsCoreLib.client import _SERVER_MESSAGE_MAX_LEN
        self.state.events = [_terminal(1, state="failed", message="x" * 10_000)]
        with self.assertRaises(ServerToolError) as caught:
            self._client().run(_TOOL, run_id=new_run_id())
        self.assertLessEqual(len(str(caught.exception)), _SERVER_MESSAGE_MAX_LEN)

    def test_a_cancelled_run_is_not_a_failure(self):
        """One closes a panel quietly, the other opens an error dialog."""
        self.state.events = [_terminal(1, state="cancelled")]
        with self.assertRaises(RunCancelled):
            self._client().run(_TOOL, run_id=new_run_id())

    def test_a_stream_that_never_says_how_it_ended_names_the_run(self):
        """The run may well have finished. Saying so, with the id, is what lets
        someone go and look rather than guess."""
        run_id = new_run_id()
        self.state.events = [_terminal(1, state="running")]
        with self.assertRaises(ServerToolError) as caught:
            self._client().run(_TOOL, run_id=run_id)
        self.assertIn(run_id, str(caught.exception))

    def test_a_run_that_finished_saying_nothing_useful_is_reported(self):
        self.state.events = [_terminal(1, result={"unexpected": True})]
        with self.assertRaises(ServerToolError) as caught:
            self._client().run(_TOOL, run_id=new_run_id())
        self.assertIn("where its result is", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
