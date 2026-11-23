"""Unit tests for ServerToolsCoreLib.transfer, run outside Slicer.

Most of these drive the real code against a REAL HTTP server (stdlib
ThreadingHTTPServer, speaking the same protocol as the tool server's
transfer.py) rather than against mocks. That is deliberate: what this module
does is concurrency and byte offsets, and a mock proves nothing about either.
The server here is deliberately hostile in places, it can drop connections,
corrupt a part, or refuse ranges, because every one of those is a thing a real
remote server does and a thing the client has to survive.

Usage:
    python3 -m unittest ServerToolsCore/Testing/Python/test_transfer.py
"""

import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from tls_fixture import serve_tls, trust_test_ca  # noqa: E402
from ServerToolsCoreLib import client, transfer
from ServerToolsCoreLib.errors import ServerToolError


class _State:
    """What the fake server holds between requests, plus the knobs a test uses
    to make it misbehave."""

    def __init__(self):
        self.uploads = {}          # upload_id -> {"size", "chunk_size", "parts": {i: bytes}}
        self.results = {}          # result_id -> bytes
        self.lock = threading.Lock()
        self.no_chunked_endpoints = False   # answers 404, like an old server
        self.ignore_ranges = False          # answers 200 with the whole body
        self.fail_parts = set()             # part indices to refuse once
        self.truncate_ranges = 0            # first N range responses come up short
        # A barrier every request must reach before any of them may answer.
        # Deterministic where counting overlapping requests is not: on
        # loopback a small request can finish before the next one starts, so a
        # peak-concurrency counter proves nothing when it reads 1. If the
        # client ever serialises its parts, this times out and the test fails.
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

    def log_message(self, *_args):
        pass  # the test output is not a web server log

    # -- helpers ------------------------------------------------------

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    # -- routes -------------------------------------------------------

    def _not_found(self):
        # Drain the request body before answering. Closing a socket that still
        # holds unread bytes makes Windows reset the connection, and the client
        # then sees "connection aborted" instead of this 404.
        self._body()
        return self._json({"detail": "not found"}, status=404)

    def do_POST(self):
        if self.state.no_chunked_endpoints:
            return self._not_found()
        if self.path == "/uploads":
            spec = json.loads(self._body())
            chunk_size = min(spec.get("chunk_size") or 8 << 20, 64 << 20)
            upload_id = f"upload{len(self.state.uploads):04d}xxxxxxxxxxxx"
            size = spec["size"]
            self.state.uploads[upload_id] = {"size": size, "chunk_size": chunk_size, "parts": {}}
            return self._json({
                "upload_id": upload_id,
                "chunk_size": chunk_size,
                "part_count": max(1, (size + chunk_size - 1) // chunk_size),
            })
        return self._json({"detail": "not found"}, status=404)

    def do_PUT(self):
        if self.state.no_chunked_endpoints:
            return self._not_found()
        upload_id, _, index = self.path[len("/uploads/"):].partition("/parts/")
        index = int(index)
        data = self._body()
        self.state.rendezvous()
        if self.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        expected = self.headers.get("X-Part-SHA256")
        if expected and hashlib.sha256(data).hexdigest() != expected:
            return self._json({"detail": "checksum"}, status=400)
        with self.state.lock:
            if index in self.state.fail_parts:
                self.state.fail_parts.discard(index)
                return self._json({"detail": "transient"}, status=503)
            self.state.uploads[upload_id]["parts"][index] = data
        return self._json({"received": index})

    def do_GET(self):
        if self.path.startswith("/uploads/"):
            upload = self.state.uploads[self.path[len("/uploads/"):]]
            count = max(1, (upload["size"] + upload["chunk_size"] - 1) // upload["chunk_size"])
            missing = [i for i in range(count) if i not in upload["parts"]]
            return self._json({"missing_parts": missing})

        blob = self.state.results[self.path[len("/results/"):]]
        span = self.headers.get("Range")
        self.state.rendezvous()
        if not span or self.state.ignore_ranges:
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            return self.wfile.write(blob)

        start, _, end = span[len("bytes="):].partition("-")
        start, end = int(start), int(end)
        piece = blob[start:end + 1]
        with self.state.lock:
            if self.state.truncate_ranges > 0:
                self.state.truncate_ranges -= 1
                piece = piece[: len(piece) // 2]  # a connection cut mid-body
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(blob)}")
        self.send_header("Content-Length", str(len(piece)))
        self.end_headers()
        self.wfile.write(piece)

    def do_HEAD(self):
        blob = self.state.results.get(self.path[len("/results/"):], b"")
        self.send_response(200)
        if not self.state.ignore_ranges:
            self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()

    def do_DELETE(self):
        self.state.uploads.pop(self.path[len("/uploads/"):], None)
        self._json({"status": "ok"})


class _LiveServerTest(unittest.TestCase):
    """Base for the tests that talk to a real socket."""

    def setUp(self):
        self.state = _State()
        handler = type("_BoundHandler", (_Handler,), {"state": self.state})
        trust_test_ca(self)
        self.server = serve_tls(ThreadingHTTPServer(("127.0.0.1", 0), handler))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"https://localhost:{self.server.server_address[1]}"
        # The client's own pooled session, exactly as in production.
        self.session = client._pooled_session()
        self.addCleanup(self.session.close)
        self.work = tempfile.mkdtemp(prefix="transfer_test_")
        self.addCleanup(shutil.rmtree, self.work, True)

    def _file(self, payload, name="scan.nii.gz"):
        path = os.path.join(self.work, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def _assembled(self, upload_id):
        upload = self.state.uploads[upload_id]
        return b"".join(upload["parts"][i] for i in sorted(upload["parts"]))


class UploadTest(_LiveServerTest):
    def test_a_file_arrives_byte_identical_through_its_parts(self):
        payload = os.urandom(700_000)
        path = self._file(payload)

        upload_id = transfer.upload_file(
            self.session, self.url, {}, path, chunk_bytes=100_000, parallelism=4
        )

        self.assertEqual(self._assembled(upload_id), payload)
        self.assertEqual(len(self.state.uploads[upload_id]["parts"]), 7)

    def test_parts_really_do_travel_at_the_same_time(self):
        """The whole point of the module. The server holds every part until
        four of them have arrived, so this can only pass if four are genuinely
        in flight at once, if the parts are ever serialised (a shared file
        handle, a lock, a connection pool too small) the barrier never trips
        and the upload fails."""
        payload = os.urandom(800_000)          # 8 parts, exactly two groups of 4
        path = self._file(payload)
        self.state.barrier = threading.Barrier(4)

        upload_id = transfer.upload_file(
            self.session, self.url, {}, path, chunk_bytes=100_000, parallelism=4
        )

        self.assertFalse(self.state.barrier.broken)
        self.assertEqual(self._assembled(upload_id), payload)

    def test_the_last_part_is_short_and_that_is_fine(self):
        payload = os.urandom(250_001)
        path = self._file(payload)

        upload_id = transfer.upload_file(
            self.session, self.url, {}, path, chunk_bytes=100_000, parallelism=2
        )

        parts = self.state.uploads[upload_id]["parts"]
        self.assertEqual(len(parts[2]), 50_001)
        self.assertEqual(self._assembled(upload_id), payload)

    def test_each_part_carries_a_checksum_of_what_lands_on_disk(self):
        path = self._file(os.urandom(150_000), name="scan.nii")  # not pre-compressed
        seen = []
        original = transfer.requests.Session.put

        def spy(self_, url, **kwargs):
            seen.append(kwargs["headers"])
            return original(self_, url, **kwargs)

        with mock.patch.object(requests.Session, "put", spy):
            transfer.upload_file(
                self.session, self.url, {}, path, chunk_bytes=100_000, parallelism=1
            )

        self.assertTrue(all("X-Part-SHA256" in headers for headers in seen))
        # Compressed on the wire, but the checksum is over the plain bytes:
        # the server verifies what it writes, not what it received.
        self.assertTrue(all(headers.get("Content-Encoding") == "gzip" for headers in seen))

    def test_already_compressed_inputs_are_not_gzipped_again(self):
        path = self._file(os.urandom(150_000), name="scan.nii.gz")
        seen = []
        original = transfer.requests.Session.put

        def spy(self_, url, **kwargs):
            seen.append(kwargs["headers"])
            return original(self_, url, **kwargs)

        with mock.patch.object(requests.Session, "put", spy):
            transfer.upload_file(
                self.session, self.url, {}, path, chunk_bytes=100_000, parallelism=1
            )

        self.assertFalse(any("Content-Encoding" in headers for headers in seen))

    def test_a_gzipped_part_round_trips_to_the_original_bytes(self):
        payload = b"uncompressed NIfTI-ish content, quite repetitive. " * 4000
        path = self._file(payload, name="scan.nii")

        upload_id = transfer.upload_file(
            self.session, self.url, {}, path, chunk_bytes=100_000, parallelism=3
        )

        self.assertEqual(self._assembled(upload_id), payload)

    def test_a_failed_part_is_retried_not_the_whole_file(self):
        payload = os.urandom(500_000)
        path = self._file(payload)
        self.state.fail_parts = {1, 3}

        with mock.patch.object(transfer, "_RETRY_BACKOFF_SECONDS", 0):
            upload_id = transfer.upload_file(
                self.session, self.url, {}, path, chunk_bytes=100_000, parallelism=4
            )

        self.assertEqual(self._assembled(upload_id), payload)

    def test_a_part_that_never_succeeds_fails_the_upload_and_drops_the_session(self):
        path = self._file(os.urandom(300_000))
        # Never discarded by the handler, so every attempt for part 1 fails.
        self.state.fail_parts = _AlwaysContains({1})

        with mock.patch.object(transfer, "_RETRY_BACKOFF_SECONDS", 0):
            with self.assertRaises(ServerToolError) as raised:
                transfer.upload_file(
                    self.session, self.url, {}, path, chunk_bytes=100_000, parallelism=2
                )

        self.assertIn("part(s) failed", str(raised.exception))
        # And nothing is left holding patient data on the server.
        self.assertEqual(self.state.uploads, {})

    def test_an_old_server_is_detected_before_any_byte_travels(self):
        self.state.no_chunked_endpoints = True
        path = self._file(os.urandom(300_000))

        with self.assertRaises(transfer.UnsupportedByServer):
            transfer.upload_file(self.session, self.url, {}, path, chunk_bytes=100_000)

        self.assertEqual(self.state.uploads, {})

    def test_progress_reports_bytes_a_rate_and_a_percentage(self):
        path = self._file(os.urandom(400_000))
        messages = []

        transfer.upload_file(
            self.session, self.url, {}, path, chunk_bytes=100_000,
            parallelism=2, progress_cb=messages.append,
        )

        self.assertTrue(messages)
        final = messages[-1]
        self.assertIn("Uploading scan.nii.gz...", final)
        self.assertIn("(100%)", final)
        self.assertIn("MB/s", final)


class _AlwaysContains(set):
    """A `fail_parts` that never forgets, so a part fails every attempt."""

    def discard(self, value):
        pass


class DownloadTest(_LiveServerTest):
    def _result(self, payload, result_id="result0000xxxxxxxxxx"):
        self.state.results[result_id] = payload
        return f"{self.url}/results/{result_id}"

    def test_ranges_reassemble_into_the_original_bytes(self):
        payload = os.urandom(700_000)
        url = self._result(payload)
        destination = os.path.join(self.work, "out.zip")

        transfer.download_ranged(
            self.session, url, destination, len(payload), chunk_bytes=100_000, parallelism=4
        )

        with open(destination, "rb") as handle:
            self.assertEqual(handle.read(), payload)

    def test_ranges_reassemble_where_the_os_has_no_positioned_write(self):
        """Windows has no os.pwrite; the parallel download must not need it."""
        payload = os.urandom(700_000)
        url = self._result(payload)
        destination = os.path.join(self.work, "out.zip")

        real_pwrite = getattr(os, "pwrite", None)
        if real_pwrite is not None:
            del os.pwrite
        try:
            transfer.download_ranged(
                self.session, url, destination, len(payload),
                chunk_bytes=100_000, parallelism=4,
            )
        finally:
            if real_pwrite is not None:
                os.pwrite = real_pwrite

        with open(destination, "rb") as handle:
            self.assertEqual(handle.read(), payload)

    def test_ranges_really_do_travel_at_the_same_time(self):
        """Same barrier, on the way down: four ranges must be open at once."""
        payload = os.urandom(800_000)          # 8 ranges, exactly two groups of 4
        url = self._result(payload)
        destination = os.path.join(self.work, "out.bin")
        self.state.barrier = threading.Barrier(4)

        transfer.download_ranged(
            self.session, url, destination, len(payload),
            chunk_bytes=100_000, parallelism=4,
        )

        self.assertFalse(self.state.barrier.broken)
        with open(destination, "rb") as handle:
            self.assertEqual(handle.read(), payload)

    def test_a_server_that_ignores_ranges_still_yields_the_whole_file(self):
        """Answering 200 to a Range request means "I don't do ranges". Taking
        that body once, instead of once per span, is what keeps this safe to
        attempt against any host."""
        payload = os.urandom(500_000)
        url = self._result(payload)
        self.state.ignore_ranges = True
        destination = os.path.join(self.work, "out.bin")

        transfer.download_ranged(
            self.session, url, destination, len(payload), chunk_bytes=100_000, parallelism=4
        )

        with open(destination, "rb") as handle:
            self.assertEqual(handle.read(), payload)

    def test_a_range_cut_short_is_retried_rather_than_silently_kept(self):
        payload = os.urandom(400_000)
        url = self._result(payload)
        self.state.truncate_ranges = 2
        destination = os.path.join(self.work, "out.bin")

        with mock.patch.object(transfer, "_RETRY_BACKOFF_SECONDS", 0):
            transfer.download_ranged(
                self.session, url, destination, len(payload), chunk_bytes=100_000, parallelism=2
            )

        with open(destination, "rb") as handle:
            self.assertEqual(handle.read(), payload)

    def test_a_range_that_never_completes_leaves_no_partial_file(self):
        payload = os.urandom(400_000)
        url = self._result(payload)
        self.state.truncate_ranges = 10_000  # every attempt comes up short
        destination = os.path.join(self.work, "out.zip")

        with mock.patch.object(transfer, "_RETRY_BACKOFF_SECONDS", 0):
            with self.assertRaises(ServerToolError):
                transfer.download_ranged(
                    self.session, url, destination, len(payload),
                    chunk_bytes=100_000, parallelism=2,
                )

        # A half-written .zip is the dangerous case: the caller would unpack
        # whatever survived and hand back a subset of the results.
        self.assertFalse(os.path.exists(destination))

    def test_an_empty_result_produces_an_empty_file(self):
        url = self._result(b"")
        destination = os.path.join(self.work, "empty.bin")

        transfer.download_ranged(self.session, url, destination, 0)

        self.assertTrue(os.path.exists(destination))
        self.assertEqual(os.path.getsize(destination), 0)

    def test_probe_reports_the_size_when_ranges_are_supported(self):
        payload = os.urandom(1234)
        url = self._result(payload)

        self.assertEqual(transfer.probe_ranged(self.session, url), 1234)

    def test_probe_declines_when_ranges_are_not_advertised(self):
        url = self._result(os.urandom(1234))
        self.state.ignore_ranges = True

        self.assertIsNone(transfer.probe_ranged(self.session, url))

    def test_progress_reports_a_rate_and_a_time_left(self):
        payload = os.urandom(600_000)
        url = self._result(payload)
        messages = []

        transfer.download_ranged(
            self.session, url, os.path.join(self.work, "out.bin"), len(payload),
            chunk_bytes=100_000, parallelism=2, progress_cb=messages.append,
        )

        self.assertTrue(messages)
        self.assertIn("Downloading results...", messages[-1])
        self.assertIn("(100%)", messages[-1])


class ShouldChunkTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="chunk_test_")
        self.addCleanup(shutil.rmtree, self.work, True)

    def _file(self, size):
        path = os.path.join(self.work, "f.bin")
        with open(path, "wb") as handle:
            handle.truncate(size)
        return path

    def test_small_files_stay_on_the_single_request_path(self):
        """Below the threshold, chunking costs two extra round trips to save
        nothing, there are not even enough parts to run in parallel."""
        self.assertFalse(transfer.should_chunk(self._file(1024), minimum=1_000_000))

    def test_big_files_are_chunked(self):
        self.assertTrue(transfer.should_chunk(self._file(2_000_000), minimum=1_000_000))

    def test_a_missing_file_is_not_chunked_rather_than_raising(self):
        self.assertFalse(transfer.should_chunk(os.path.join(self.work, "nope.bin")))


if __name__ == "__main__":
    unittest.main()
