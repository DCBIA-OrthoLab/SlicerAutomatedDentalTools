"""A run that says what it is doing, and a run that can be stopped.

Most of these drive the real client against a REAL HTTP server (stdlib
ThreadingHTTPServer, speaking the run contract's `/runs/{id}/events` and
`DELETE /runs/{id}`), the same choice `test_transfer.py` makes and for the same
reason: what is under test here is a SECOND connection held open while the
first one is blocked, and a mocked `requests.Session` cannot show two requests
overlapping. `test_the_event_stream_really_is_open_while_the_run_is` proves it
with a barrier neither request can pass alone.

The grace and retry windows are patched down to fractions of a second in the
cases that exercise them: their shipped values are measured in seconds because
a real upload takes seconds, and a test suite that waited them out would take
minutes to say something it can say in a tenth of one.

Usage:
    python3 -m unittest ServerToolsCore/Testing/Python/test_run_progress.py
"""

import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.parse import urlsplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tls_fixture import serve_tls, trust_test_ca  # noqa: E402
from ServerToolsCoreLib import client as client_module
from ServerToolsCoreLib.client import (ToolServerClient, is_log_event, new_run_id,
                                      normalise_run_event)
from ServerToolsCoreLib.errors import RunCancelled, ServerToolError

_TOOL = "Probe"
_SCHEMA = [{"name": _TOOL, "arguments": {}, "output_kind": "text"}]


def _event(seq, **fields):
    payload = {"seq": seq, "at": 1757400000.0, "state": "running",
               "phase": "running", "fraction": None, "message": "", "depth": 0}
    payload.update(fields)
    return payload


def _log(seq, **fields):
    """A log line as the server sends it: a running tool's state and phase
    riding along, and no fraction at all."""
    payload = {"seq": seq, "at": 1757400000.0, "kind": "log", "state": "running",
               "phase": "running", "level": "info", "audience": "user",
               "message": "", "depth": 0}
    payload.update(fields)
    return payload


class _State:
    """What the fake server holds, plus the knobs a test uses to make it
    behave like a server that is older, slower or busier than this one."""

    def __init__(self):
        self.events = []                # what GET /runs/{id}/events replays
        self.run_ids = []               # every X-Run-Id header the POST saw
        self.posts = 0
        self.deleted = []               # every id DELETE /runs/{id} was given
        self.events_404_left = 0        # answer 404 this many times first
        self.events_requests = 0
        self.events_paths = []          # every path the events stream was asked on
        self.close_after = None         # cut the FIRST stream after N events
        self.post_status = 200
        # Set by a test to hold the run's answer back until something else has
        # happened -- which is what a real run looks like: the progress arrives
        # WHILE the POST is still in flight, not after it.
        self.finish_run_when = None
        self.lock = threading.Lock()
        # A barrier both the run and its event stream must reach before either
        # may answer. Deterministic where counting concurrent requests is not:
        # if the client ever stops opening a second connection during the POST,
        # this times out and the test fails rather than passing by luck.
        self.barrier = None

    def rendezvous(self):
        if self.barrier is None:
            return
        try:
            self.barrier.wait(timeout=10)
        except threading.BrokenBarrierError:
            pass


class _Handler(BaseHTTPRequestHandler):
    state = None  # set per server instance

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass  # the test output is not a web server log

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        # Read the body before answering, exactly as the real server does: on
        # the multipart path parsing the form IS receiving the upload, which is
        # the whole reason the event stream has to be open by now.
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        with self.state.lock:
            self.state.posts += 1
            self.state.run_ids.append(self.headers.get("X-Run-Id"))
        self.state.rendezvous()
        if self.state.finish_run_when is not None:
            self.state.finish_run_when.wait(timeout=10)
        if self.state.post_status != 200:
            return self._json({"detail": "Run cancelled by the client."},
                              status=self.state.post_status)
        return self._json({"result": "finished"})

    def do_DELETE(self):
        self.state.deleted.append(self.path[len("/runs/"):])
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/tools":
            return self._json(_SCHEMA)
        # The path alone: the watcher asks for `?logs=user`, and a server
        # that has no log lines ignores the query rather than 404ing on it.
        if not urlsplit(self.path).path.endswith("/events"):
            return self._json({"detail": "not found"}, status=404)

        with self.state.lock:
            self.state.events_requests += 1
            self.state.events_paths.append(self.path)
            if self.state.events_404_left > 0:
                self.state.events_404_left -= 1
                return self._json({"detail": "unknown run"}, status=404)
        self.state.rendezvous()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        # Chunked rather than a length: a stream whose end is not known in
        # advance is the shape the contract describes.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        # One-shot: the connection that gets cut is the first one, and the
        # client's next attempt is served in full. A permanent cut would only
        # prove that a loop can spin.
        with self.state.lock:
            cut_after, self.state.close_after = self.state.close_after, None

        sent = 0
        for event in self.state.events:
            self._frame(json.dumps(event))
            sent += 1
            if cut_after is not None and sent >= cut_after:
                # A dropped connection / proxy idle timeout: the run is still
                # going and the client is expected to come back for the rest.
                break
        self._chunk(b"")

    def _frame(self, payload):
        self._chunk(f"data: {payload}\n\n".encode())

    def _chunk(self, body):
        self.wfile.write(b"%x\r\n" % len(body) + body + b"\r\n")
        self.wfile.flush()


class _LiveServerTest(unittest.TestCase):
    """Base for the tests that talk to a real socket."""

    def setUp(self):
        self.state = _State()
        handler = type("_BoundHandler", (_Handler,), {"state": self.state})
        trust_test_ca(self)
        self.server = serve_tls(ThreadingHTTPServer(("127.0.0.1", 0), handler))
        # A watcher that has read its terminal event closes the socket without
        # draining it, which is correct and which the stdlib server reports as
        # an unhandled ConnectionResetError all over the test output.
        self.server.handle_error = lambda *_args: None
        # A short poll interval only so the suite is quick: shutdown() blocks
        # until serve_forever next looks, and the default half second, paid by
        # every case here, is most of this file's runtime.
        threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        url = f"https://localhost:{self.server.server_address[1]}"
        self.client = ToolServerClient(url, token="t", timeout=10)

    def _watch(self, run_id="run-id-0123456789abcdef", **kwargs):
        received = []
        delivered = self.client.watch_run(run_id, received.append, **kwargs)
        return delivered, received


class RunIdTest(unittest.TestCase):
    def test_it_matches_the_shape_the_server_enforces(self):
        for _ in range(20):
            run_id = new_run_id()
            self.assertRegex(run_id, r"^[A-Za-z0-9_-]{16,64}$")

    def test_two_runs_never_share_one(self):
        """It is a capability, not a label: knowing it plus the token is what
        authorises reading a run's progress and cancelling it."""
        self.assertEqual(len({new_run_id() for _ in range(500)}), 500)


class EventShapeTest(unittest.TestCase):
    """Whatever arrives, what reaches a panel is renderable."""

    def test_a_full_event_survives_intact(self):
        event = normalise_run_event(_event(12, fraction=0.35, message="patient 14 of 40", depth=1))
        self.assertEqual(event["seq"], 12)
        self.assertEqual(event["fraction"], 0.35)
        self.assertEqual(event["message"], "patient 14 of 40")
        self.assertEqual(event["depth"], 1)

    def test_an_event_without_a_usable_seq_is_not_an_event(self):
        """`seq` is what the client dedupes and orders on, so an event that
        cannot be placed cannot be shown either."""
        self.assertIsNone(normalise_run_event(_event(None)))
        self.assertIsNone(normalise_run_event("running"))

    def test_an_unknown_fraction_stays_unknown(self):
        """Never fabricated on either side: a determinate bar is only ever
        shown for a number a tool really sent."""
        self.assertIsNone(normalise_run_event(_event(1))["fraction"])
        self.assertIsNone(normalise_run_event(_event(1, fraction="0.4"))["fraction"])

    def test_a_fraction_outside_the_range_is_clamped(self):
        self.assertEqual(normalise_run_event(_event(1, fraction=1.4))["fraction"], 1.0)
        self.assertEqual(normalise_run_event(_event(1, fraction=-2))["fraction"], 0.0)

    def test_a_message_is_truncated_here_too(self):
        """The server truncates it. Trusting that would leave one forgetful
        server able to push a megabyte of text into a QLabel."""
        event = normalise_run_event(_event(1, message="x" * 5000))
        self.assertEqual(len(event["message"]), 200)

    def test_a_negative_depth_is_no_depth(self):
        self.assertEqual(normalise_run_event(_event(1, depth=-3))["depth"], 0)


class LogEventShapeTest(unittest.TestCase):
    """A tool's log line: renderable, and impossible to mistake for progress."""

    def test_a_log_line_is_its_own_kind_and_carries_nothing_of_progress(self):
        event = normalise_run_event(_log(7, level="warning", message="scan 4 skipped",
                                         depth=1, source="ALI_CBCT"))
        self.assertTrue(is_log_event(event))
        self.assertEqual(event["seq"], 7)
        self.assertEqual(event["level"], "warning")
        self.assertEqual(event["message"], "scan 4 skipped")
        self.assertEqual(event["depth"], 1)
        self.assertEqual(event["source"], "ALI_CBCT")
        # Left out on purpose, so no reader of progress can read one here.
        for key in ("state", "phase", "fraction"):
            self.assertNotIn(key, event)

    def test_a_progress_event_is_not_a_log_line(self):
        self.assertFalse(is_log_event(normalise_run_event(_event(1))))

    def test_the_known_levels_survive_and_anything_else_reads_as_info(self):
        for level in ("debug", "info", "warning", "error", "metric"):
            self.assertEqual(normalise_run_event(_log(1, level=level))["level"], level)
        self.assertEqual(normalise_run_event(_log(1, level="WARNING"))["level"], "warning")
        for level in ("critical", "", None, 3, "<b>error</b>"):
            self.assertEqual(normalise_run_event(_log(1, level=level))["level"], "info")

    def test_a_log_message_is_truncated_like_a_progress_one(self):
        event = normalise_run_event(_log(1, message="x" * 5000))
        self.assertEqual(len(event["message"]), 200)

    def test_a_source_that_is_not_a_tool_name_is_dropped(self):
        """It is drawn next to a clinician's data; an identifier is the one
        shape that cannot carry a file name or a sentence."""
        for source in ("../etc/passwd", "scan_01.nii.gz", "two words", "x" * 65,
                       "", 12, None, ["ALI"]):
            self.assertNotIn("source", normalise_run_event(_log(1, source=source)), source)
        self.assertEqual(normalise_run_event(_log(1, source="AREG-v2"))["source"], "AREG-v2")

    def test_an_unusable_time_is_unknown_rather_than_wrong(self):
        for at in ("yesterday", -1, float("nan"), float("inf"), True, None):
            self.assertIsNone(normalise_run_event(_log(1, at=at))["at"], at)

    def test_a_log_line_without_a_usable_seq_is_not_an_event(self):
        self.assertIsNone(normalise_run_event(_log(None)))


class NestedProgressShapeTest(unittest.TestCase):
    """The optional fields a weighted chain adds to a progress event."""

    def test_own_fraction_tool_edge_and_call_are_kept_when_well_formed(self):
        event = normalise_run_event(_event(3, fraction=0.4, own_fraction=0.8, tool="ALI_CBCT",
                                           edge="open", call="1.2", depth=2))
        self.assertEqual(event["fraction"], 0.4)
        self.assertEqual(event["own_fraction"], 0.8)
        self.assertEqual(event["tool"], "ALI_CBCT")
        self.assertEqual(event["edge"], "open")
        self.assertEqual(event["call"], "1.2")

    def test_malformed_ones_are_dropped_and_the_event_still_arrives(self):
        event = normalise_run_event(_event(3, own_fraction=1.5, tool="a/b", edge="sideways",
                                           call="1;rm"))
        self.assertIsNotNone(event)
        for key in ("own_fraction", "tool", "edge", "call"):
            self.assertNotIn(key, event)

    def test_an_older_server_s_event_gains_nothing(self):
        event = normalise_run_event(_event(3, fraction=0.4))
        for key in ("own_fraction", "tool", "edge", "call", "kind"):
            self.assertNotIn(key, event)


class WatchRunTest(_LiveServerTest):
    def test_the_stream_is_asked_for_the_user_s_log_lines(self):
        """`?logs=user`, on every connection: an older server ignores it, and
        a newer one sends no log line without it."""
        self.state.events = [_event(0, state="done", phase="done")]

        self._watch()

        self.assertTrue(self.state.events_paths)
        for path in self.state.events_paths:
            self.assertTrue(path.endswith("/events?logs=user"), path)

    def test_a_log_line_never_ends_the_watch_whatever_state_it_claims(self):
        """Only progress can be terminal. A log line claiming `done` is still a
        line, and the run it belongs to is still going."""
        self.state.events = [_event(0), _log(1, state="done", phase="done", message="noise"),
                             _event(2, fraction=0.5), _event(3, state="done", phase="done")]

        delivered, received = self._watch()

        self.assertTrue(delivered)
        self.assertEqual([event["seq"] for event in received], [0, 1, 2, 3])
        self.assertTrue(is_log_event(received[1]))
        self.assertEqual(received[-1]["state"], "done")

    def test_one_seq_dedupes_log_lines_and_progress_alike(self):
        """They share the run's numbering, so a reconnect's replay re-announces
        neither a progress event nor a warning the reader has already seen."""
        self.state.events = [_event(0), _log(1, level="warning", message="scan 4 skipped"),
                             _event(2), _log(3), _event(4, state="done", phase="done")]
        self.state.close_after = 2  # the first connection dies after the warning

        with mock.patch.object(client_module, "_RUN_EVENTS_RETRY_SECONDS", 0.01):
            delivered, received = self._watch()

        self.assertTrue(delivered)
        self.assertEqual([event["seq"] for event in received], [0, 1, 2, 3, 4])
        self.assertGreaterEqual(self.state.events_requests, 2)


    def test_events_arrive_in_order_and_the_stream_ends_on_a_terminal_one(self):
        self.state.events = [_event(0, phase="received"), _event(1, phase="running"),
                             _event(2, state="done", phase="done")]

        delivered, received = self._watch()

        self.assertTrue(delivered)
        self.assertEqual([event["seq"] for event in received], [0, 1, 2])
        self.assertEqual(received[-1]["state"], "done")

    def test_a_resumed_run_is_watched_from_where_its_first_watch_stopped(self):
        """The resume of a stopped run keeps its id and its event file; a new
        watcher replaying it from the start would print every line before the
        stop a second time."""
        self.state.events = [_event(0), _event(1, state="paused", phase="paused")]
        # A pause ends nothing on the stream: `run` stops its watcher once the
        # POST has answered with the checkpoint, which is what this stands for.
        stop = threading.Event()
        self.client.watch_run("run-id-0123456789abcdef",
                              lambda event: event["seq"] == 1 and stop.set(),
                              stop_event=stop)
        self.state.events += [_event(2), _event(3, state="done", phase="done")]

        seen = self.client.__dict__["_delivered_seq"]["run-id-0123456789abcdef"]
        _delivered, received = self._watch(after_seq=seen)

        self.assertEqual([event["seq"] for event in received], [2, 3])

    def test_a_reconnect_does_not_re_announce_an_hour_of_progress(self):
        """Every connection replays the run from the beginning by design (a
        watcher that attaches late is never behind), so deduplication on `seq`
        is what makes reconnecting safe rather than noisy."""
        self.state.events = [_event(0), _event(1), _event(2, state="done", phase="done")]
        self.state.close_after = 2  # the first connection dies mid-run

        with mock.patch.object(client_module, "_RUN_EVENTS_RETRY_SECONDS", 0.01):
            delivered, received = self._watch()

        self.assertTrue(delivered)
        self.assertEqual([event["seq"] for event in received], [0, 1, 2])
        self.assertGreaterEqual(self.state.events_requests, 2)

    def test_a_404_is_tolerated_while_the_run_is_still_being_registered(self):
        """The run only exists server-side once the POST gets there, and on the
        multipart path the POST IS the upload. A watcher that gave up on the
        first 404 would go quiet on exactly the long uploads this exists for."""
        self.state.events_404_left = 3
        self.state.events = [_event(0, state="done", phase="done")]

        with mock.patch.object(client_module, "_RUN_EVENTS_RETRY_SECONDS", 0.01):
            delivered, received = self._watch()

        self.assertTrue(delivered)
        self.assertEqual(len(received), 1)

    def test_a_server_that_never_has_the_endpoint_is_given_up_on_quietly(self):
        """An older deployment. Nothing raised, nothing alarming logged: the
        panel keeps the elapsed timer it has always shown."""
        self.state.events_404_left = 10_000

        with mock.patch.object(client_module, "_RUN_EVENTS_STARTUP_GRACE_SECONDS", 0.2), \
                mock.patch.object(client_module, "_RUN_EVENTS_RETRY_SECONDS", 0.01):
            delivered, received = self._watch()

        self.assertFalse(delivered)
        self.assertEqual(received, [])

    def test_a_404_after_the_first_event_means_reaped_and_stops_at_once(self):
        """The startup window is over the moment anything arrives: a 404 then
        is a run that has been swept, and there is nothing left to wait for."""
        self.state.events = [_event(0)]  # no terminal event, so it reconnects

        with mock.patch.object(client_module, "_RUN_EVENTS_RETRY_SECONDS", 0.01):
            def stop_serving_after_the_first_attempt():
                while self.state.events_requests < 1:
                    time.sleep(0.005)
                self.state.events_404_left = 10_000

            threading.Thread(target=stop_serving_after_the_first_attempt, daemon=True).start()
            delivered, received = self._watch()

        self.assertTrue(delivered)
        self.assertEqual([event["seq"] for event in received], [0])

    def test_a_stop_event_ends_the_watch(self):
        self.state.events = [_event(0)]  # never terminal: only the stop can end it
        stop = threading.Event()

        received = []

        def deliver(event):
            received.append(event)
            # Stopped once the client HAS the event, not once the server was
            # asked for it: the response is still in flight at that point
            # (a TLS handshake later), and a stop that wins that race
            # rightly delivers nothing.
            stop.set()

        with mock.patch.object(client_module, "_RUN_EVENTS_RETRY_SECONDS", 0.01):
            delivered = self.client.watch_run("run-id-0123456789abcdef", deliver, stop_event=stop)

        self.assertTrue(delivered)
        self.assertEqual([event["seq"] for event in received], [0])

    def test_an_unreadable_frame_costs_one_frame_and_not_the_stream(self):
        self.state.events = [_event(0), "not json at all", _event(2, state="done", phase="done")]

        delivered, received = self._watch()

        self.assertTrue(delivered)
        self.assertEqual([event["seq"] for event in received], [0, 2])


class RunWithProgressTest(_LiveServerTest):
    def test_the_run_id_travels_as_a_header(self):
        run_id = new_run_id()
        self.client.run(_TOOL, run_id=run_id)
        self.assertEqual(self.state.run_ids, [run_id])

    def test_without_one_the_request_is_exactly_what_it_always_was(self):
        """The whole feature is optional on both sides: a run sent without an
        id must reach an old server byte for byte as before."""
        self.client.run(_TOOL)
        self.assertEqual(self.state.run_ids, [None])
        self.assertEqual(self.state.events_requests, 0, "nothing to watch, nothing opened")

    def test_the_event_stream_really_is_open_while_the_run_is(self):
        """The point of the whole design, and the one thing a mock cannot show.

        The server answers NEITHER request until both have arrived. The POST
        blocks for the length of an inference, so the events request can only
        be there if the client genuinely opened a second connection alongside
        it; if it ever goes back to one connection, the barrier times out and
        this fails rather than passing quietly.
        """
        self.state.barrier = threading.Barrier(2)
        self.state.events = [_event(0, phase="running", fraction=0.5),
                             _event(1, state="done", phase="done")]
        received = []
        # The run does not answer until its progress has been DELIVERED, which
        # is the ordering a real run has: the panel learns it is at 50% while
        # the request it is waiting on has not returned. Without this the POST
        # could answer first on loopback and the test would prove nothing.
        delivered = threading.Event()
        self.state.finish_run_when = delivered

        def collect(event):
            received.append(event)
            if event["state"] == "done":
                delivered.set()

        result = self.client.run(_TOOL, run_id=new_run_id(), event_cb=collect)

        self.assertFalse(self.state.barrier.broken, "the two requests never overlapped")
        self.assertTrue(delivered.is_set(), "the run answered before any progress arrived")
        self.assertEqual(result.text, "finished")
        self.assertEqual([event["seq"] for event in received], [0, 1])
        self.assertEqual(received[0]["fraction"], 0.5)

    def test_no_watcher_without_somewhere_to_deliver(self):
        """An id but no callback is a run that wants to be cancellable and
        nothing more; it must not pay for a thread and a connection."""
        self.client.run(_TOOL, run_id=new_run_id())
        self.assertEqual(self.state.events_requests, 0)

    def test_a_499_is_a_cancellation_and_not_a_failure(self):
        """A dedicated class, because one closes the panel quietly and the
        other opens an error dialog, and no message should have to be parsed to
        tell them apart."""
        self.state.post_status = 499

        with self.assertRaises(RunCancelled) as raised:
            self.client.run(_TOOL, run_id=new_run_id())

        self.assertEqual(raised.exception.status_code, 499)
        self.assertIsInstance(raised.exception, ServerToolError)

    def test_a_run_cancelled_before_it_is_sent_never_reaches_the_server(self):
        cancel = threading.Event()
        cancel.set()

        with self.assertRaises(RunCancelled):
            self.client.run(_TOOL, cancel_event=cancel)

        self.assertEqual(self.state.posts, 0)

    def test_cancelling_asks_the_server_to_stop_the_run(self):
        run_id = new_run_id()
        self.assertTrue(self.client.cancel_run(run_id))
        self.assertEqual(self.state.deleted, [run_id])

    def test_an_unreachable_server_is_a_false_and_never_an_exception(self):
        """The panel has already released the run by the time this is called;
        there is nothing a user could do with the news, and a raise here would
        turn letting go of a run into a second error dialog."""
        offline = ToolServerClient("https://127.0.0.1:1", token="t", timeout=1)
        self.assertFalse(offline.cancel_run(new_run_id()))


class SseFramingTest(unittest.TestCase):
    """The framing itself, apart from any socket."""

    @staticmethod
    def _frames(*lines):
        return list(client_module._sse_data_frames(lines))

    def test_one_frame_per_blank_line(self):
        self.assertEqual(self._frames("data: a", "", "data: b", ""), ["a", "b"])

    def test_a_comment_is_a_keep_alive_and_never_an_event(self):
        self.assertEqual(self._frames(": ping", "data: a", ""), ["a"])

    def test_a_frame_split_over_several_data_lines_is_rejoined(self):
        self.assertEqual(self._frames("data: {", 'data: "seq": 1}', ""), ['{\n"seq": 1}'])

    def test_a_last_frame_with_no_trailing_blank_line_is_not_lost(self):
        self.assertEqual(self._frames("data: a"), ["a"])

    def test_bytes_are_decoded(self):
        self.assertEqual(self._frames(b"data: a", b""), ["a"])


if __name__ == "__main__":
    unittest.main()
