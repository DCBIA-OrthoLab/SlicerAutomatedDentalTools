"""Moving big files over one connection is what made a 100 MB scan take minutes.

A single HTTP request rides a single TCP connection, and a single TCP
connection to a remote server is bound by its congestion window long before it
is bound by anyone's bandwidth: on a link with a 100 ms round trip, one stream
plateaus around a tenth of what the line can actually carry. Adding streams is
what fixes that, and nothing else does -- the wire is not full, the window is.

So a file goes up as independent parts sent over several connections at once
(`upload_file`) and a result comes down as several byte ranges pulled at once
(`download_ranged`). Both talk to the endpoints in the server's transfer.py.

Three things fall out of the same design, and they matter as much as the speed:

- **A failure costs one part, not the file.** A dropped connection at 95% used
  to mean starting again from zero. Now the part is retried, and a whole pass
  of failures re-reads what the server is still missing rather than guessing.
- **Progress is real.** The old upload was one opaque `requests.post` that
  reported nothing until the server answered; here every part that lands moves
  a counter, so the panel can show bytes, rate and a time left.
- **Integrity is checked per part.** Each part carries its SHA-256 and the
  server refuses to write one that does not match. The parts tile the file
  exactly, so the whole thing is verified without either side making a second
  pass over it -- which for medical imaging is the point: a silently truncated
  CBCT is a wrong result, not an error.

Imports neither `slicer` nor `qt` (see ARCHITECTURE.md dependency rule), so it
is testable in plain CI with `requests` mocked out.
"""

import gzip
import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import requests

from . import tls
from .errors import ServerToolError

logger = logging.getLogger("ServerToolsCore.transfer")

# How many parts/ranges are in flight at once. Four is the knee of the curve on
# a long-haul link: it multiplies the effective window by four, while staying
# well under what a shared uplink or a server's connection pool would notice.
# Raising it past ~8 buys little and starts to compete with itself.
DEFAULT_PARALLELISM = 4

# Requested part size. The server clamps it and answers with what it actually
# used, which is the value both sides then compute offsets from -- this is only
# a preference. 8 MB keeps a 100 MB scan at 13 parts: enough to keep four
# connections busy, few enough that per-request overhead stays invisible.
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024

# Below this, chunking is a pessimisation: the session handshake plus the run
# request is three round trips where a plain multipart upload is one, and there
# are not enough parts to parallelise anyway.
MIN_CHUNKED_BYTES = 2 * DEFAULT_CHUNK_BYTES

# Attempts per part before the transfer is declared failed. Parts fail for
# exactly the reason this module exists (a connection that did not survive), so
# retrying one is nearly always enough.
_MAX_ATTEMPTS = 4
_RETRY_BACKOFF_SECONDS = 1.5

# A part is pure I/O, so it must not inherit the run timeout (up to 600s, sized
# for inference). A stalled connection has to be noticed and retried in
# seconds, not sat on for ten minutes.
_PART_TIMEOUT = (15, 180)

# Extensions whose bytes are already compressed: gzipping a part of one of
# these spends CPU on both ends to save ~0%. Everything NOT listed here is
# worth compressing on the wire -- an uncompressed .nii or a .vtk mesh is
# roughly a third of its size deflated, which on a remote link is directly a
# third of the time. Mirrors the server's own table in file_utils.py.
_PRECOMPRESSED_EXTENSIONS = (
    ".gz", ".bz2", ".xz", ".zip", ".7z",
    ".xlsx", ".ods", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg",
)

# Level 1, not 6: measured on the server side of the same trade-off, level 1
# compresses at roughly twice the rate for ~3% more bytes. The wire is the
# scarce resource here, but only up to the point where compressing becomes
# slower than sending -- level 1 on four threads stays ahead of any link this
# is used over.
_GZIP_LEVEL = 1

_READ_BUFFER_BYTES = 1024 * 1024


class _Meter:
    """Byte counter shared by the worker threads, throttled on the way out.

    Progress lands on a queue drained by a Qt timer (see worker.py). Emitting
    per part would be too coarse to look alive on a slow link and per socket
    read would flood the queue, so this reports on a clock instead: at most
    every _INTERVAL, plus one final call so the last message is never a stale
    99%.
    """

    _INTERVAL = 0.25

    def __init__(self, label: str, total: int, progress_cb: Optional[Callable[[str], None]]):
        self._label = label
        self._total = total
        self._progress_cb = progress_cb
        self._done = 0
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._last_emit = 0.0

    def add(self, count: int) -> None:
        if self._progress_cb is None:
            return
        with self._lock:
            self._done += count
            now = time.monotonic()
            if now - self._last_emit < self._INTERVAL:
                return
            self._last_emit = now
            message = self._render(self._done, now)
        # Outside the lock: the callback puts on a queue, and no worker thread
        # should ever wait on another one's turn to do that.
        self._progress_cb(message)

    def finish(self) -> None:
        if self._progress_cb is None:
            return
        with self._lock:
            done, now = self._done, time.monotonic()
        self._progress_cb(self._render(done, now))

    def _render(self, done: int, now: float) -> str:
        elapsed = max(1e-6, now - self._started)
        rate = done / elapsed
        text = f"{self._label} {done / (1024 * 1024):.1f}"
        if self._total:
            percent = min(100, round(100 * done / self._total))
            text += f" / {self._total / (1024 * 1024):.1f} MB ({percent}%)"
        else:
            text += " MB"
        if rate > 0:
            text += f" at {rate / (1024 * 1024):.1f} MB/s"
            if self._total and done < self._total:
                text += f", {_human_seconds((self._total - done) / rate)} left"
        return text


def _human_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{max(1, int(seconds))}s"
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s"


def should_chunk(path: str, minimum: int = MIN_CHUNKED_BYTES) -> bool:
    try:
        return os.path.getsize(path) >= minimum
    except OSError:
        return False


def _worth_compressing(path: str) -> bool:
    return not path.lower().endswith(_PRECOMPRESSED_EXTENSIONS)


def _retry_delay(attempt: int) -> float:
    return _RETRY_BACKOFF_SECONDS * (2 ** attempt)


class UnsupportedByServer(Exception):
    """The server has no chunked-transfer endpoints (it predates them).

    Not a ServerToolError: it is never shown to a user and never means the run
    failed. It is the signal to fall back to the single-request path, which is
    what keeps this extension working against a server that has not been
    redeployed yet.
    """


# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------

def upload_file(
    session: requests.Session,
    server_url: str,
    headers: dict,
    path: str,
    parallelism: int = DEFAULT_PARALLELISM,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    compress: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
    label: Optional[str] = None,
) -> str:
    """Send `path` in parallel parts; return the upload id /run then refers to.

    Raises UnsupportedByServer (and sends nothing) when the server does not
    have the endpoints, so the caller can fall back.
    """
    size = os.path.getsize(path)
    name = os.path.basename(path)
    opened = _open_session(session, server_url, headers, name, size, chunk_bytes)
    upload_id = opened["upload_id"]
    chunk_size = opened["chunk_size"]
    part_count = opened["part_count"]

    meter = _Meter(label or f"Uploading {name}...", size, progress_cb)
    gzip_parts = compress and _worth_compressing(path)
    part_url = f"{server_url}/uploads/{upload_id}/parts"

    try:
        pending = list(range(part_count))
        for attempt in range(_MAX_ATTEMPTS):
            failures = _send_parts(
                session, part_url, headers, path, pending, chunk_size, size,
                gzip_parts, parallelism, meter,
            )
            if not failures:
                break
            if attempt == _MAX_ATTEMPTS - 1:
                raise ServerToolError(
                    f"Could not upload '{name}': {len(failures)} part(s) failed after "
                    f"{_MAX_ATTEMPTS} attempts. Last error: {failures[0]}"
                )
            time.sleep(_retry_delay(attempt))
            # Ask the server what is actually missing rather than retrying the
            # list we think failed: a part whose response was lost on the way
            # back did land, and re-sending it would be wasted bandwidth on a
            # link that just proved it has none to spare.
            pending = _missing_parts(session, server_url, headers, upload_id, pending)
            if not pending:
                break
        meter.finish()
    except Exception:
        # Never leave a half-filled session holding patient data on the server;
        # it has a reaper, but only as a safety net.
        _discard(session, server_url, headers, upload_id)
        raise

    logger.info("uploaded %s (%d byte(s)) as %d part(s)", name, size, part_count)
    return upload_id


def _open_session(session, server_url, headers, name, size, chunk_bytes) -> dict:
    try:
        response = session.post(
            f"{server_url}/uploads",
            headers=headers,
            json={"filename": name, "size": size, "chunk_size": chunk_bytes},
            timeout=_PART_TIMEOUT,
            verify=tls.ca_bundle(),
        )
    except requests.RequestException as exc:
        raise ServerToolError(f"Could not start the upload of '{name}': {exc}") from exc

    if response.status_code in (404, 405):
        raise UnsupportedByServer()
    if not response.ok:
        raise ServerToolError(f"Could not start the upload of '{name}': {_message(response)}")
    try:
        opened = response.json()
        # The layout both sides compute every offset from. Missing or
        # unparseable means we are not talking to the server we think we are,
        # and guessing a default here would write parts at the wrong offsets.
        return {
            "upload_id": opened["upload_id"],
            "chunk_size": int(opened["chunk_size"]),
            "part_count": int(opened["part_count"]),
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise ServerToolError(
            f"Could not start the upload of '{name}': malformed session response ({exc})."
        ) from exc


def _send_parts(
    session, part_url, headers, path, indices, chunk_size, size,
    gzip_parts, parallelism, meter,
):
    """Push `indices` concurrently; return the errors, one per part that failed.

    Each worker opens its own file handle and seeks: one shared handle would
    need a lock around every read, which is exactly the serialisation the
    parallelism is here to avoid.
    """
    failures = []
    failures_lock = threading.Lock()

    def send(index):
        offset = index * chunk_size
        length = min(chunk_size, size - offset)
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise ServerToolError(
                f"'{os.path.basename(path)}' changed while it was being uploaded."
            )

        part_headers = dict(headers)
        # Over the DECOMPRESSED bytes: it is what the server writes to disk,
        # so it is what has to be verified.
        part_headers["X-Part-SHA256"] = hashlib.sha256(data).hexdigest()
        part_headers["Content-Type"] = "application/octet-stream"
        if gzip_parts:
            data = gzip.compress(data, _GZIP_LEVEL)
            part_headers["Content-Encoding"] = "gzip"

        response = session.put(
            f"{part_url}/{index}",
            headers=part_headers,
            data=data,
            timeout=_PART_TIMEOUT,
            verify=tls.ca_bundle(),
        )
        if not response.ok:
            raise ServerToolError(f"Part {index} refused: {_message(response)}")
        meter.add(length)

    def guarded(index):
        try:
            send(index)
        except Exception as exc:  # noqa: BLE001 - collected, reported, retried
            logger.debug("part %d failed: %s", index, exc)
            with failures_lock:
                failures.append(exc)

    workers = max(1, min(parallelism, len(indices)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(guarded, indices))
    return failures


def _missing_parts(session, server_url, headers, upload_id, fallback):
    try:
        response = session.get(
            f"{server_url}/uploads/{upload_id}",
            headers=headers,
            timeout=_PART_TIMEOUT,
            verify=tls.ca_bundle(),
        )
        if response.ok:
            return response.json().get("missing_parts", fallback)
    except (requests.RequestException, ValueError) as exc:
        logger.debug("could not read upload status, retrying what failed: %s", exc)
    return fallback


def _discard(session, server_url, headers, upload_id) -> None:
    try:
        session.delete(
            f"{server_url}/uploads/{upload_id}",
            headers=headers,
            timeout=_PART_TIMEOUT,
            verify=tls.ca_bundle(),
        )
    except requests.RequestException as exc:
        logger.debug("could not discard upload %s: %s", upload_id, exc)


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------

# Per-part durations of the most recent ranged download, for whoever wants to
# report them. Module-level and cleared on entry: one transfer runs at a time
# per client, and a caller that cares reads it straight after.
last_part_times = []


def download_ranged(
    session: requests.Session,
    url: str,
    destination: str,
    size: int,
    headers: Optional[dict] = None,
    parallelism: int = DEFAULT_PARALLELISM,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    progress_cb: Optional[Callable[[str], None]] = None,
    label: str = "Downloading results...",
) -> str:
    """Pull `url` into `destination` over `parallelism` concurrent ranges.

    Ranges are a fixed size rather than one per worker: equal shares would make
    the whole download wait for whichever connection happened to be slowest,
    while a queue of small ranges lets the fast ones pick up the slack.

    Falls back to a plain sequential read of the whole body when the server
    answers a Range request with 200 (i.e. it does not support ranges), so this
    is safe to attempt anywhere.
    """
    headers = dict(headers or {})
    meter = _Meter(label, size, progress_cb)

    # Filled in as the parts land; the caller prints it. A list rather than a
    # return value so the existing signature is untouched.
    last_part_times.clear()

    # O_BINARY exists only on Windows, where its absence would let the C
    # runtime turn every \n written into \r\n.
    file_descriptor = os.open(
        destination, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    )
    try:
        if not size:
            # Nothing to range over. The empty file has just been created,
            # which is the whole of what this call had to produce.
            meter.finish()
            return destination
        # Reserve the full length up front so every worker can write at its own
        # offset from the start; on any real filesystem this is a sparse file
        # and costs no blocks until the bytes arrive.
        os.ftruncate(file_descriptor, size)
        spans = [
            (start, min(start + chunk_bytes, size) - 1)
            for start in range(0, size, chunk_bytes)
        ]
        failures = []
        failures_lock = threading.Lock()
        whole_body = threading.Event()

        # One line per part, so "the download was slow" can be read as either
        # "every part carried a fixed cost" (a per-connection problem) or "one
        # part straggled and the rest waited" (contention). The two have
        # nothing in common and no single total distinguishes them.
        part_lock = threading.Lock()

        def fetch(span):
            start, end = span
            for attempt in range(_MAX_ATTEMPTS):
                if whole_body.is_set():
                    return
                try:
                    part_started = time.monotonic()
                    _fetch_span(
                        session, url, headers, file_descriptor,
                        start, end, size, meter, whole_body,
                    )
                    with part_lock:
                        last_part_times.append(time.monotonic() - part_started)
                    return
                except Exception as exc:  # noqa: BLE001 - retried, then reported
                    logger.debug("range %d-%d failed (attempt %d): %s", start, end, attempt, exc)
                    if attempt == _MAX_ATTEMPTS - 1:
                        with failures_lock:
                            failures.append(exc)
                        return
                    time.sleep(_retry_delay(attempt))

        workers = max(1, min(parallelism, len(spans)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, spans))

        if failures:
            os.close(file_descriptor)
            file_descriptor = None
            # A partial file must never survive: for a .zip the caller would
            # unpack whatever central directory happens to be intact and
            # deliver a SUBSET of the results, which is the worst outcome
            # there is for medical data.
            os.remove(destination)
            raise ServerToolError(
                f"Could not download the result: {len(failures)} range(s) failed. "
                f"Last error: {failures[0]}"
            )
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)

    # No byte-count check here on purpose: every span verifies its own length
    # before returning (see _fetch_span), and the spans tile the file exactly,
    # so "no failures" already means "complete".
    meter.finish()
    return destination


def _fetch_span(session, url, headers, file_descriptor, start, end, size, meter, whole_body):
    """One ranged GET, written straight to its offset in the destination.

    Verifies its own length before returning: `iter_content` ending early is
    how a connection cut mid-body looks from here, and it raises nothing on its
    own. Checking per span is what turns that into one retried range instead of
    a silently short file.
    """
    request_headers = dict(headers)
    request_headers["Range"] = f"bytes={start}-{end}"
    # Identity, not gzip: the range is a slice of the file's real bytes, and a
    # transfer-compressed body would not line up with the offsets we write at.
    request_headers["Accept-Encoding"] = "identity"

    with session.get(
        url, headers=request_headers, stream=True, timeout=_PART_TIMEOUT, verify=tls.ca_bundle()
    ) as response:
        if not response.ok:
            raise ServerToolError(f"Range {start}-{end} refused: {_message(response)}")
        if response.status_code != 206:
            # No Range support: this response is the WHOLE file, so take it and
            # tell the other workers to stand down rather than downloading it
            # once per span.
            if whole_body.is_set():
                return
            whole_body.set()
            offset = 0
            for chunk in response.iter_content(_READ_BUFFER_BYTES):
                offset += _pwrite_all(file_descriptor, chunk, offset)
                meter.add(len(chunk))
            if offset != size:
                whole_body.clear()
                raise ServerToolError(f"Truncated body: {offset} of {size} bytes.")
            return

        offset = start
        for chunk in response.iter_content(_READ_BUFFER_BYTES):
            offset += _pwrite_all(file_descriptor, chunk, offset)
            meter.add(len(chunk))
        if offset != end + 1:
            raise ServerToolError(
                f"Range {start}-{end} stopped at {offset - 1}; the connection was cut."
            )


# Windows has no os.pwrite: there, a seek and a write stand in for it, and the
# lock keeps another worker's seek from landing between the two.
_seek_and_write_lock = threading.Lock()


def _pwrite(file_descriptor, data, offset) -> int:
    if hasattr(os, "pwrite"):
        return os.pwrite(file_descriptor, data, offset)
    with _seek_and_write_lock:
        os.lseek(file_descriptor, offset, os.SEEK_SET)
        return os.write(file_descriptor, data)


def _pwrite_all(file_descriptor, data, offset) -> int:
    """A positioned write may write short; the offset is ours to advance either way."""
    written = 0
    while written < len(data):
        written += _pwrite(file_descriptor, data[written:], offset + written)
    return written


def probe_ranged(session, url, headers=None):
    """Total size if `url` can be fetched in ranges, else None.

    A HEAD is one round trip against a transfer that is about to be many, and
    knowing the length up front is what allows the parallel plan at all.
    """
    try:
        response = session.head(
            url, headers=dict(headers or {}), timeout=_PART_TIMEOUT,
            # Never followed: a redirect is how https:// becomes http://.
            verify=tls.ca_bundle(), allow_redirects=False,
        )
    except requests.RequestException as exc:
        logger.debug("range probe failed for %s: %s", url, exc)
        return None
    if not response.ok:
        return None
    if response.headers.get("Accept-Ranges", "").lower() != "bytes":
        return None
    try:
        size = int(response.headers.get("Content-Length") or 0)
    except ValueError:
        return None
    return size or None


def _message(response) -> str:
    """The server's own explanation, when it sent one."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message")
            if detail:
                return str(detail)
    except ValueError:
        pass
    return f"HTTP {response.status_code}"
