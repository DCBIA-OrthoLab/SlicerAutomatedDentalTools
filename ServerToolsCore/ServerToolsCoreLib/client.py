"""The only class in the extension that speaks HTTP to the tool server.

Imports neither `slicer` nor `qt` - see ARCHITECTURE.md dependency rule. This
makes it testable in plain CI with `requests` mocked out (see
ServerToolsCore/Testing/Python/test_client.py).

Bulk transfer is the one thing this file delegates: `transfer.py` moves a big
input up in parallel parts and pulls a big result down in parallel ranges,
because one file over one connection is throughput-bound by that connection's
congestion window rather than by the link. Everything about WHICH bytes travel
and what they mean stays here; that module only moves them.
"""

import json
import logging
import math
import mimetypes
import os
import secrets
import threading
import time
import re
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import quote, urlsplit

import requests

from . import tls, transfer
# Re-exported: the one loopback predicate, which settings_qt and others import
# from here.
from .tls import is_loopback  # noqa: F401
from .errors import RunCancelled, ServerToolError, error_for_status

logger = logging.getLogger("ServerToolsCore.client")

# A health check feeds the status banner on every enter(); it must never hang
# for as long as a real tool run (self._timeout, up to 600s).
_HEALTH_CHECK_TIMEOUT = 10

# get_tool_schema() is called synchronously from a module's setup() (building
# the GUI needs the schema before the first paint) - a slow/unreachable server
# must not be able to freeze Slicer for up to 600s just to open a module.
_TOOLS_FETCH_TIMEOUT = 15

_CONTENT_DISPOSITION_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?')
_SERVER_MESSAGE_MAX_LEN = 500

# A result archive can weigh hundreds of MB (AMASSS: one .nii.gz + .vtk per
# structure and per scan). It is streamed to disk in chunks of this size, so
# the whole body is never held in Slicer's RAM.
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# Header asking the server to hand back a POINTER to the result instead of
# streaming it down the same connection that carried the run, so it can be
# pulled in parallel ranges (see transfer.download_ranged). A server that
# predates it ignores an unknown header and answers exactly as it always did,
# which is what makes this safe to send unconditionally.
_RESULT_DELIVERY_HEADER = {"X-Result-Delivery": "reference"}

# Opts one run out of the blocking contract: the POST answers 202 as soon as the
# inputs are staged, and the result reference arrives on the terminal event of
# the stream this client already watches.
#
# What it fixes is not hypothetical. The POST's read timeout is TIMEOUT seconds
# (600 by default, an hour at the very most a user can dial in), while a cohort
# legitimately runs for longer; and a dropped connection never stopped the run,
# it only threw the answer away after the GPU had been spent on it.
_RUN_DELIVERY_HEADER = {"X-Run-Delivery": "detached"}

# Connections the pool keeps alive per host. Must exceed the transfer
# parallelism, or the parallel parts queue up on each other inside urllib3 and
# the whole point is lost.
_CONNECTION_POOL_SIZE = 16

# Form field naming the inputs that already travelled through the upload
# endpoints, as {argument name: upload id}. Must match the server's own
# _UPLOADS_FIELD; double-underscored so it can never collide with a tool's
# argument name.
_UPLOADS_FIELD = "__uploads__"

# The argument the server injects on a tool whose chain can be interrupted: a
# multichoice of the steps a run may stop after, every option off by default
# (server-side `registry/schema_tool.STOP_AFTER_ARGUMENT`). This side only
# ever READS it -- it is published in the schema like any other argument and
# rendered like any other multichoice.
STOP_AFTER_ARGUMENT = "stop_after"

# ----------------------------------------------------------------------
# Run progress and cancellation (see the wire contract shared with the
# server repository). Everything here is OPTIONAL ON BOTH SIDES: a run sent
# without an id behaves exactly as it always did, and a client using these
# endpoints against a server that has none of them gets a 404 and falls back
# to the elapsed-time tick it already shows. That is a hard requirement, not
# a nicety - the extension ships on its own schedule.
# ----------------------------------------------------------------------

# The id travels as a request header on the run, not in the body: it has to be
# known to BOTH sides before the response exists, since the whole point is to
# say something while the request is still in flight.
RUN_ID_HEADER = "X-Run-Id"
# A run that is one batch of a divided cohort says which, so the server can
# group the batches and decide whether they run side by side. Headers for the
# same reason the run id is one: a server that has never heard of them ignores
# them and answers exactly as before.
BATCH_ID_HEADER = "X-Batch-Id"
BATCH_INDEX_HEADER = "X-Batch-Index"
BATCH_TOTAL_HEADER = "X-Batch-Total"

# The closed set of terminal states. Reaching one ends the stream, on both
# sides: the server stops writing, and the watcher stops reading rather than
# reconnecting to a run that will never say anything again.
TERMINAL_RUN_STATES = ("done", "failed", "cancelled")

# A progress message is written by a tool and may name a file, so the server
# truncates it. Truncated again here rather than trusted: this text goes
# straight onto a panel, and a server that forgot its own cap must not be able
# to push a megabyte of it into a QLabel.
_RUN_MESSAGE_MAX_LEN = 200

# A log line is an event of its own kind (`"kind": "log"`), interleaved in the
# same stream and numbered on the same `seq`, and it is NEVER progress: it
# carries no fraction and says nothing about where the run is. The server sends
# them only to a watcher that asked with `?logs=user`, and an older server
# ignores the parameter and simply sends none -- which is what makes it safe to
# ask every time.
RUN_LOG_KIND = "log"
_RUN_LOGS_QUERY = {"logs": "user"}

# The contract's four levels. Anything else a server or a tool sends reads as
# "info": a level is only ever used to pick a colour, so an unknown one must
# degrade to the plain line rather than to no line at all.
# "metric" is a number the user may want and can skip -- a residual, a volume --
# printed in its own colour; an older server never sends it.
RUN_LOG_LEVELS = ("debug", "info", "warning", "error", "metric")

# A tool name as the server publishes one (`source` on a log line, `tool` on a
# nested call's marker). Held to an identifier on this side too, because it is
# drawn on the panel next to a clinician's data: a name is the one fact the
# drawing needs, and an identifier cannot carry a file name or a sentence.
_RUN_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

# Which supervised call wrote a record ("1", "1.2"). Only ever used as a key
# that ties a nested call's records to the tool its opening marker named.
_RUN_CALL_RE = re.compile(r"[0-9]{1,6}(?:\.[0-9]{1,6}){0,15}")

# Read timeout on the event stream. Long on purpose: the contract has no
# heartbeat, so a server that is simply busy inferring sends NOTHING between
# `running` and `packaging` - minutes of it - and a short timeout would mean
# reconnecting (and replaying the backlog) over and over for no information.
# What it costs is that an abandoned watcher, one whose run finished without a
# terminal event, takes this long to notice. It is a daemon thread doing
# nothing, so that is the cheap side of the trade.
_RUN_EVENTS_READ_TIMEOUT = 30

# How long a watcher tolerates a 404 before concluding the server simply has
# no such endpoint.
#
# This window is not defensiveness, it is the normal case: the watcher opens
# the stream as soon as the id is minted, and the run only exists server-side
# once the POST gets there. On the multipart path the POST *is* the upload, so
# those two are separated by however long the input takes to travel. A watcher
# that gave up on the first 404 would therefore go quiet on exactly the long
# uploads this feature exists for. The server registers the run before it
# parses the form, which narrows the window to microseconds in the normal
# case; this closes what is left of it.
#
# After the first event has been delivered the window is over for good: a 404
# then means the run was reaped, and there is nothing left to wait for.
_RUN_EVENTS_STARTUP_GRACE_SECONDS = 15.0

# Between two attempts, whatever ended the last one. Short enough that a
# reconnect is invisible next to an inference, long enough that a server
# closing the stream instantly cannot become a hot loop.
_RUN_EVENTS_RETRY_SECONDS = 0.5


def new_run_id() -> str:
    """A fresh run id: 32 URL-safe characters from the OS CSPRNG.

    The id is a CAPABILITY - knowing it, plus the bearer token, is what
    authorises reading a run's progress and cancelling it, the same model the
    result ids already use. So it must come from `secrets`, never from a
    counter, a timestamp or anything derived from a patient. The server checks
    the shape (`[A-Za-z0-9_-]{16,64}`) and nothing more, because it cannot
    check entropy.
    """
    return secrets.token_urlsafe(24)


class _AnyEvent:
    """Any one of several `threading.Event`s being set means "stop".

    A watcher has two independent reasons to stop - its run finished, or the
    user cancelled - and the events for those belong to different owners. This
    is the smallest thing that lets `watch_run` keep taking one stop object
    while honouring both, rather than growing a second parameter that every
    caller would have to thread through.
    """

    def __init__(self, events):
        self._events = [event for event in events if event is not None]

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)

    def wait(self, timeout=None) -> bool:
        """Sleep, but wake the moment any of them is set.

        Polled rather than combined with a condition variable: the events are
        other people's, `threading.Event` has no "wait for any", and the
        alternative (a thread per event) costs more than a 50 ms poll of a
        boolean does.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(min(0.05, timeout if timeout is not None else 0.05))
        return True


def _sse_data_frames(lines):
    """Yield the payload of each Server-Sent Events `data:` frame.

    The contract sends one JSON object per frame and each object fits on one
    line, so this could be a single `startswith("data:")`. It is written to the
    actual SSE framing anyway - accumulate `data:` lines, a blank line ends the
    frame - because that costs four lines and means a server that later sends a
    comment heartbeat (`: ping`), or splits a long message, does not break the
    client it has to stay compatible with.
    """
    payload = []
    for raw_line in lines:
        line = raw_line if isinstance(raw_line, str) else (raw_line or b"").decode("utf-8", "replace")
        if not line:
            if payload:
                yield "\n".join(payload)
                payload = []
            continue
        if line.startswith(":"):
            # A comment: SSE's own keep-alive. Never an event.
            continue
        if line.startswith("data:"):
            payload.append(line[len("data:"):].lstrip())
    if payload:
        yield "\n".join(payload)


def normalise_run_event(payload) -> Optional[dict]:
    """One event of the contract, with every field made safe to render.

    Returns None for anything that is not a usable event. A tool's log line
    (`"kind": "log"`) comes back in a shape of its own -- see
    _normalise_log_event and is_log_event -- and never as progress.

    The panel that shows these must not have to ask whether `fraction` came
    back as the string "0.35", whether `depth` is negative, or whether
    `message` is a novel: an event either arrives here in the shape the UI
    expects, or it does not arrive at all.
    """
    if not isinstance(payload, dict):
        return None
    try:
        seq = int(payload.get("seq"))
    except (TypeError, ValueError):
        # `seq` is what the client dedupes and orders on, so an event without
        # a usable one cannot be placed and is not an event.
        return None

    if payload.get("kind") == RUN_LOG_KIND:
        return _normalise_log_event(payload, seq)

    fraction = payload.get("fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        # Never fabricated, on either side: unknown stays unknown, and a
        # determinate bar is only ever shown for a number the tool really sent.
        fraction = None
    else:
        fraction = min(1.0, max(0.0, float(fraction)))

    depth = _event_depth(payload)

    event = {
        "seq": seq,
        "at": payload.get("at"),
        "state": payload.get("state") or "",
        "phase": payload.get("phase") or "",
        "fraction": fraction,
        "message": _event_message(payload),
        "depth": depth,
    }
    # Optional, and absent from every server that predates weighted chains.
    # `fraction` is what the bar draws -- when a parent weighted its nested
    # calls the server has already folded it into the WHOLE run's position --
    # and `own_fraction` is what the record itself said on its own tool's
    # 0..1. Kept only when it is really a number in range, like `fraction`.
    own = payload.get("own_fraction")
    if not isinstance(own, bool) and isinstance(own, (int, float)) and 0.0 <= own <= 1.0:
        event["own_fraction"] = float(own)
    # A nested call's opening and closing markers name the callee. `call` ties
    # the records written inside that call back to the name, which is what
    # lets a panel say WHICH tool of a chain is talking.
    tool = _run_tool_name(payload.get("tool"))
    if tool:
        event["tool"] = tool
        if payload.get("edge") in ("open", "close"):
            event["edge"] = payload["edge"]
    call = _run_call_id(payload.get("call"))
    if call:
        event["call"] = call
    # Only on a detached run's terminal event, and it is how the answer gets
    # back at all: the response that used to carry it was a 202, sent before
    # the tool had started. Kept exactly as the server sent it -- this side
    # does not interpret it, it hands it to _download_reference.
    if isinstance(payload.get("result"), dict):
        event["result"] = payload["result"]
    return event


def is_log_event(event) -> bool:
    """Whether a normalised event is a tool's log line rather than progress.

    The one test every consumer makes before reading an event as progress: a
    log line must never overwrite a run's phase, message or fraction, never end
    a watcher, and never move a cohort's bar.
    """
    return isinstance(event, dict) and event.get("kind") == RUN_LOG_KIND


def _normalise_log_event(payload: dict, seq: int) -> dict:
    """A log line, in a shape of its own that no progress reader can mistake.

    Deliberately WITHOUT `state`, `phase` and `fraction`: the server sends a
    log line with a running tool's state so that an older reader takes it as
    harmless, but on this side the line is not progress at all, and leaving
    those keys out means nothing downstream can read it as such by accident.
    """
    level = str(payload.get("level") or "").lower()
    at = payload.get("at")
    if (isinstance(at, bool) or not isinstance(at, (int, float))
            or not math.isfinite(at) or at <= 0):
        # Only ever used to print a time of day, so an unusable one is simply
        # "unknown" and the panel stamps the line when it arrives.
        at = None
    event = {
        "seq": seq,
        "at": float(at) if at is not None else None,
        "kind": RUN_LOG_KIND,
        "level": level if level in RUN_LOG_LEVELS else "info",
        "message": _event_message(payload),
        "depth": _event_depth(payload),
    }
    source = _run_tool_name(payload.get("source"))
    if source:
        event["source"] = source
    return event


def _event_depth(payload: dict) -> int:
    try:
        return max(0, int(payload.get("depth") or 0))
    except (TypeError, ValueError):
        return 0


def _event_message(payload: dict) -> str:
    message = payload.get("message") or ""
    if not isinstance(message, str):
        message = str(message)
    return message[:_RUN_MESSAGE_MAX_LEN]


def _run_tool_name(value) -> Optional[str]:
    """A tool name, or None for anything that is not strictly one."""
    if isinstance(value, str) and _RUN_TOOL_NAME_RE.fullmatch(value):
        return value
    return None


def _run_call_id(value) -> Optional[str]:
    if isinstance(value, str) and _RUN_CALL_RE.fullmatch(value):
        return value
    return None


def _download_message(received: int, expected: Optional[int], label: str = "results") -> str:
    """"Downloading results... 8.2 / 14.1 MB (58%)", or without the total when
    the server sent no usable Content-Length."""
    received_mb = received / (1024 * 1024)
    if not expected:
        return f"Downloading {label}... {received_mb:.1f} MB"
    expected_mb = expected / (1024 * 1024)
    percent = min(100, round(100 * received / expected))
    return f"Downloading {label}... {received_mb:.1f} / {expected_mb:.1f} MB ({percent}%)"


# What a packaged tool declares for a file-or-folder argument. One name, from
# the tool contract in SADT-VISOR; see is_file_type and accepts_folder.
PATH_TYPE = "path"


def _canonical_tool_name(name: str) -> str:
    """A tool name with case and separators removed, matching the server's rule.

    Only for LOOKUP. Anything sent to the server, or shown to a user, stays the
    name the schema published.
    """
    return "".join(character for character in name.lower() if character.isalnum())


def is_file_type(type_name: str) -> bool:
    """Whether a schema argument `type` denotes a file upload.

    The server is not limited to a generic "file" type - it can (and does,
    e.g. "nifti_file", "zip_file") use more specific type names to hint at
    what kind of file is expected. Treating any "..._file" type (plus the
    literal "file", for tools that don't bother being specific) as a file
    argument means a new file-ish type the server introduces later needs no
    client-side code change - the whole point of a schema-driven client.

    "path" is the exception that rule did not survive: it is what a PACKAGED
    tool (SADT-VISOR) declares for every file or folder it takes, and it ends
    in neither. Left out, such a tool's schema reports NO file arguments at
    all and the panel refuses to build:

        FILE_INPUTS declares ['scans'] but the server's 'AMASSS' schema
        doesn't have them as file arguments (it has: []).

    It is listed rather than pattern-matched because it is one name, fixed by
    the tool contract, and guessing at "anything not obviously scalar" would
    turn every unknown type into a file dialog.
    """
    return type_name in ("file", PATH_TYPE) or type_name.endswith("_file")


# Fallback only. The server publishes each file type's extensions in its
# `types`' company (see file_extensions_for), which is the single source of
# truth; this table is what a *pre-`extensions`* server leaves us guessing
# with, and it is a copy of that server's own FILE_TYPES - the kind of
# duplication that drifts. It once did: "volume_or_zip_file" was missing here
# and derived as ".volume_or_zip", a file dialog matching nothing.
#
# Do not grow it for a new type. Publish the type's extensions server-side
# instead; anything not listed still falls back to the obvious ".<x>"
# ("csv_file" -> ".csv") when the name spells one out.
_FILE_TYPE_EXTENSIONS = {
    "file": (),  # deliberately unrestricted: the generic type accepts anything
    "nifti_file": (".nii", ".nii.gz"),
    "zip_file": (".zip",),
    # A medical volume or a zip of a folder of them (AMASSS's `input`): the
    # type name doesn't spell out an extension, so it needs an entry here.
    "volume_or_zip_file": (".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz", ".zip"),
}

# The one non-file type that may appear alongside file types in `types`: it is
# a *local* selection kind, not something HTTP can carry - a folder is zipped
# client-side and uploaded as the .zip the server then unpacks.
FOLDER_TYPE = "folder"


def _guessed_extension(type_name: str) -> tuple:
    """The extension a "<x>_file" type name spells out, when it spells one.

    `"csv_file"` -> `(".csv",)`. But a compound name like
    `"volume_or_zip_file"` names a *set* of formats, not an extension: guessing
    `".volume_or_zip"` there produces a file dialog that matches nothing, which
    is worse than not filtering at all. Such a name (recognisable by the
    underscores left once "_file" is stripped) that has no entry in
    _FILE_TYPE_EXTENSIONS falls back to no restriction, so a new one the server
    introduces degrades to an unfiltered picker instead of an empty one.
    """
    if not type_name.endswith("_file"):
        return ()
    stem = type_name[: -len("_file")]
    return () if "_" in stem else (f".{stem}",)


def argument_types(spec: dict) -> list:
    """Every type a schema argument accepts.

    The server sends both a single `type` (the primary/first one) and the full
    `types` list; an argument accepting several - e.g. example_tool's `input`:
    `["csv_file", "folder"]` - is only fully described by the latter. Falls
    back to `[type]` so a schema predating the `types` field still works.
    """
    types = spec.get("types")
    if types:
        return list(types)
    type_name = spec.get("type")
    return [type_name] if type_name else []


def accepts_folder(spec: dict) -> bool:
    """Whether the user may pick a whole folder for this argument (which the
    client then zips before uploading - see slicer_io.zip_folder).

    A packaged tool's "path" always does: the tool contract requires every path
    argument to take a directory as readily as one file, because the server
    pays a process start-up cost per call and a cohort of forty scans has to be
    one run rather than forty."""
    types = argument_types(spec)
    return FOLDER_TYPE in types or PATH_TYPE in types


def file_extensions_for(spec: dict) -> tuple:
    """The extensions a file picker should offer for this argument - 
    `["csv_file", "folder"]` gives `(".csv",)`.

    Read from the schema's own `extensions` (`{type name: [extension, ...]}`,
    the server's FILE_TYPES table published alongside `types`), so the client
    holds no copy of it. Only the *file* types count: `"folder"`'s extensions
    say what a zipped folder may be uploaded as, not what a file picker should
    show.

    A server that predates the field leaves it out, and each type then falls
    back to _FILE_TYPE_EXTENSIONS or to what its name spells out.

    An empty tuple means "don't restrict": either the argument declares the
    generic "file" type, or it accepts no file type at all (folder only).
    """
    published = spec.get("extensions") or {}
    extensions = []
    for type_name in argument_types(spec):
        if not is_file_type(type_name):
            continue
        known = published.get(type_name, _FILE_TYPE_EXTENSIONS.get(type_name))
        if known is None:
            known = _guessed_extension(type_name)
        if not known:
            # Either the generic "file", or a type the server declines to
            # restrict: anything goes, so no filter at all.
            return ()
        extensions.extend(extension for extension in known if extension not in extensions)
    return tuple(extensions)


# The two shapes GET /tools/{tool}/data can answer with for its test files.
# `entries` is what a current server publishes -- a name, a `kind`
# ("file"/"folder") and a `size` in bytes; `testfiles` is the flat list of
# names every server has always sent. Both are read, so a panel built against
# the richer one keeps working against a server that only sends the older.
TESTFILE_KINDS = ("file", "folder")


def testfile_entries(data: dict) -> list:
    """`[{"name": str, "kind": str|None, "size": int|None}, ...]` for the test
    files a tool hosts, from a `list_tool_data` payload.

    `kind` and `size` are what let a picker say "folder, 339 MB" before
    fetching 339 MB, and BOTH may legitimately be absent: an older server
    publishes no `entries` at all, and a data-store backend that cannot size a
    tree cheaply sends `null`. Absent is therefore never an error and never a
    zero -- it is "unknown", which a caller renders as nothing rather than as
    "0 B".

    The flat `testfiles` list stays authoritative for WHICH names exist: it is
    the field every server sends, and an entry it does not mention is dropped
    rather than offered as a name the run endpoint would not resolve.
    """
    names = list(data.get("testfiles") or [])
    described = {}
    for entry in (data.get("entries") or {}).get("testfiles") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name:
            described[name] = entry

    entries = []
    for name in names:
        entry = described.get(name, {})
        kind = entry.get("kind")
        size = entry.get("size")
        entries.append({
            "name": name,
            "kind": kind if kind in TESTFILE_KINDS else None,
            "size": size if isinstance(size, int) and size >= 0 else None,
        })
    return entries


def _asks_to_stop(args: dict) -> bool:
    """Whether this request arms a quality-control checkpoint.

    Read off the argument the caller is already sending rather than declared
    anywhere: the panel renders `stop_after` like any other multichoice, so
    the complete `{step: ticked}` dict arrives here and the only question is
    whether anything in it is on. A comma-separated string and a list are
    accepted too, those being the other two spellings the server takes.
    """
    wanted = (args or {}).get(STOP_AFTER_ARGUMENT)
    if isinstance(wanted, dict):
        return any(bool(on) for on in wanted.values())
    if isinstance(wanted, (list, tuple, set)):
        return any(str(step).strip() for step in wanted)
    return bool(str(wanted or "").strip())


class PlainHttpRefused(requests.exceptions.InvalidSchema):
    """A request to an http:// URL reached the session: refused, never sent."""


class RedirectRefused(requests.RequestException):
    """The server answered with a redirect: refused, never followed."""


class _RefusePlainHttp(requests.adapters.BaseAdapter):
    """Mounted on http:// so nothing -- a caller's bug, or a redirect that
    slipped past `_refuse_redirects` -- ever leaves this machine in clear."""

    def send(self, request, **_kwargs):
        raise PlainHttpRefused(tls.HTTPS_REQUIRED_MESSAGE, request=request)

    def close(self):
        pass


def _refuse_redirects(response, *_args, **_kwargs):
    """Response hook: any redirect is an error, whatever the method.

    A redirect is how https:// becomes http:// without the URL ever passing
    tls.require_https: a 307 to an http:// host would have the multipart body
    -- scans and token -- resent in clear and its reply accepted. The server
    never redirects, so there is nothing legitimate to follow. Raised here,
    before requests resolves the redirect, and whether or not the call asked
    for redirects to be followed: a 3xx counts as `ok` in requests and would
    otherwise be parsed as an answer.
    """
    if response.is_redirect or 300 <= response.status_code < 400:
        response.close()
        raise RedirectRefused(
            f"The server answered HTTP {response.status_code} with a redirect; "
            "redirects are never followed.", response=response,
        )
    return response


def _pooled_session() -> requests.Session:
    """One Session for every call this client makes, instead of a fresh
    connection per request.

    Two reasons, and the second is the load-bearing one. A module's setup()
    alone costs /health + /tools + /tools/{name}/data, each of which paid its
    own TCP and TLS handshake, several round trips against a remote server,
    every time a panel is opened. And a chunked transfer needs the pool to hand
    out as many connections as it has parts in flight; urllib3's default
    (10 per host, blocking above that) would quietly serialise them.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=_CONNECTION_POOL_SIZE,
        pool_maxsize=_CONNECTION_POOL_SIZE,
        # Retries stay with the callers: transfer.py resends the one part that
        # failed and knows what the server is still missing, which urllib3's
        # blind per-request retry cannot do.
        max_retries=0,
    )
    # Only https:// gets this adapter: every server URL went through
    # tls.require_https, so a plain-HTTP URL reaching the session is a bug.
    # Every request passes verify=tls.ca_bundle(): the public authorities of requests'
    # own bundle, hostname included, with no knob anywhere (see tls.py).
    session.mount("https://", adapter)
    session.mount("http://", _RefusePlainHttp())
    session.max_redirects = 0
    session.hooks["response"].append(_refuse_redirects)
    return session


@dataclass
class RunCheckpoint:
    """A run that stopped where it was asked to, and can be told to carry on.

    `produced` is the server's own folder names for the steps that ran, in
    order -- `("01_ALI_CBCT",)`. They are not decoration: `resume_run` sends
    one correction per step, named after exactly these, and the server refuses
    a field named anything else.

    `path` is the archive of what those steps produced, already on disk, or
    None when the checkpoint collected nothing. Only outputs of SUPERVISED
    calls are ever shipped from a stopped run, so a tool that stopped inside
    its own work has a checkpoint with nothing to look at.
    """

    run_id: str
    stopped_after: str
    produced: tuple = ()
    path: Optional[str] = None
    # `(folder, digests)` once a panel's worker thread has unpacked `path` for
    # review (base_widget._unpackedOnWorker), so the main thread does not.
    unpacked: Optional[tuple] = None


@dataclass
class ToolResult:
    """Uniform result regardless of output_kind."""

    kind: str  # "text" | "file" | "checkpoint"
    text: Optional[str] = None
    path: Optional[str] = None
    # Set only for kind "checkpoint": the run has NOT finished and is waiting
    # on `resume_run`. A caller that has never heard of this reads None and
    # behaves exactly as it always did.
    checkpoint: Optional[RunCheckpoint] = None
    # Every file written, once a panel's worker thread has unpacked the
    # archive at `path` -- which is then gone -- into the folder holding it
    # (base_widget._unpackedOnWorker). None: nothing was unpacked yet.
    extracted: Optional[list] = None


def checked_server_url(server_url: str) -> str:
    """`server_url` as an https:// URL without its trailing slash, or raise.

    Every path that accepts a server URL goes through here (the constructor,
    `configure()`, the saved settings, the settings panel), and the rule itself
    lives in tls.require_https: the server has no plain-HTTP listener at all,
    and `http://` would put the scans, the DICOM headers and the API token on
    the wire in the clear, so every `http://` URL is refused -- loopback
    included, `http://localhost:8000` (the default of older clients) too.
    Nothing is upgraded behind the user's back. The certificate is always
    verified against the public authorities, for the same reason (tls.py).

    Raises tls.InsecureServerUrl, a ValueError, which the settings panel turns
    into a message.
    """
    return tls.require_https(server_url)


def _never_resolves(server_url: str) -> bool:
    """True for a host under `.invalid`, which RFC 6761 reserves so that it
    never resolves: the placeholder a client has until a server is set."""
    host = (urlsplit(server_url).hostname or "").rstrip(".").lower()
    return host == "invalid" or host.endswith(".invalid")


class ToolServerClient:
    def __init__(
        self,
        server_url,
        token,
        timeout=600,
        parallelism=transfer.DEFAULT_PARALLELISM,
        chunk_bytes=transfer.DEFAULT_CHUNK_BYTES,
        compress_uploads=True,
        detached_runs=False,
    ):
        self._server_url = checked_server_url(server_url)
        self._token = token
        self._timeout = timeout
        self._parallelism = parallelism
        self._chunk_bytes = chunk_bytes
        self._compress_uploads = compress_uploads
        # Opt in to the detached contract. Off by default: it needs a server
        # that knows the header, and a client that says nothing keeps the
        # behaviour it always had, byte for byte.
        self._detached_runs = detached_runs
        self._tools_cache = None
        # None until the first big upload tells us; False pins every later one
        # to the single-request path, so an old server costs one failed probe
        # per session rather than one per file.
        self._chunked_uploads = None
        self._session = _pooled_session()

    # ------------------------------------------------------------------
    # Live (re)configuration - e.g. from a user-facing settings panel
    # ------------------------------------------------------------------

    @property
    def server_url(self) -> str:
        return self._server_url

    @property
    def token(self) -> str:
        return self._token

    @property
    def timeout(self) -> int:
        return self._timeout

    def configure(self, server_url=None, token=None, timeout=None) -> None:
        """Update connection settings on the already-constructed singleton in
        place, so every module sharing get_client() sees the change without a
        Slicer restart. Drops the cached /tools schema unconditionally - it
        may belong to a different server entirely once any of these change.

        A `server_url` that is not https:// raises tls.InsecureServerUrl
        before anything is changed, loopback included.
        """
        if server_url is not None:
            self._server_url = checked_server_url(server_url)
        if token is not None:
            self._token = token
        if timeout is not None:
            self._timeout = timeout
        self._tools_cache = None
        # Dropped for the same reason as the schema cache: "this server has no
        # chunked upload" is a fact about the server that just changed.
        self._chunked_uploads = None

    # ------------------------------------------------------------------
    # Schema discovery
    # ------------------------------------------------------------------

    def health(self) -> bool:
        # The placeholder URL (config.SERVER_URL) names a host that can never
        # resolve, so asking is pointless -- and harmful: the check runs on a
        # daemon thread, and one still waiting on the resolver when Slicer
        # quits logs its failure into Slicer's log handler while the
        # application is being destroyed, which crashes Slicer on exit.
        if _never_resolves(self._server_url):
            return False
        try:
            response = self._session.get(
                f"{self._server_url}/health", timeout=_HEALTH_CHECK_TIMEOUT, verify=tls.ca_bundle()
            )
            return bool(response.ok and response.json().get("status") == "ok")
        except (requests.RequestException, ValueError) as exc:
            logger.debug("Health check failed: %s", exc)
            return False

    def batch_policy(self) -> Optional[dict]:
        """How the server will run this workstation's cohort batches, or None.

        `{"batches": "serial" | "parallel", "max_parallel": N}`, decided per
        workstation by the server's operator. None from a server that predates
        the endpoint, or that cannot be reached: the caller then keeps its own
        configured concurrency, which is what it did before.
        """
        try:
            response = self._session.get(
                f"{self._server_url}/clients/me", headers={"Authorization": f"Bearer {self._token}"},
                timeout=_HEALTH_CHECK_TIMEOUT, verify=tls.ca_bundle(),
            )
            if not response.ok:
                return None
            answer = response.json()
            return answer if isinstance(answer, dict) else None
        except (requests.RequestException, ValueError) as exc:
            logger.debug("Batch policy unavailable: %s", exc)
            return None

    def list_tools(self, force_refresh: bool = False) -> dict:
        """Return {tool_name: schema}, cached after the first call."""
        if self._tools_cache is None or force_refresh:
            self._tools_cache = self._fetch_tools()
        return self._tools_cache

    def get_tool_schema(self, tool_name: str, force_refresh: bool = False) -> dict:
        """`force_refresh` re-fetches /tools instead of trusting the cache - 
        used when retrying after a failure, where the cached list may be the
        very reason the tool wasn't found."""
        tools = self.list_tools(force_refresh=force_refresh)
        if tool_name in tools:
            return tools[tool_name]

        # Case and separators, before calling this unknown. A module names its
        # tool in a constant, and `SurgMovPred` became `Surg_Mov_Pred` when the
        # tool was packaged: the panel then showed "Unknown tool", which is what
        # a typo shows, for a rename that changed nothing else. The server
        # applies the same rule and refuses to serve two names that differ only
        # this way, so at most one can match here.
        canonical = _canonical_tool_name(tool_name)
        for served, schema in tools.items():
            if _canonical_tool_name(served) == canonical:
                return schema

        available = ", ".join(sorted(tools)) or "none"
        raise ServerToolError(f"Unknown tool '{tool_name}'. Available: {available}")

    def list_tool_data(self, tool_name: str) -> dict:
        """Return {"models": [...], "testfiles": [...], "entries": {...}} - what
        the server hosts for this tool (GET /tools/{tool}/data, Bearer-protected).

        This is what lets a server_selectable argument (e.g. SurgMovPred's
        "model") be offered as a dropdown of server-side choices instead of a
        local file picker. Not cached: called once per module setup(), and the
        server-side list can change independently of the /tools schema.

        `entries` is the richer half, added by a later server: the same names
        with a `kind` ("file"/"folder") and a `size` in bytes, so a test-file
        picker can say what it is about to fetch. Passed through verbatim and
        normalised by `testfile_entries`, which also covers a server that sends
        only the flat name lists.
        """
        try:
            response = self._session.get(
                f"{self._server_url}/tools/{tool_name}/data",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=_TOOLS_FETCH_TIMEOUT,
                verify=tls.ca_bundle(),
            )
        except requests.RequestException as exc:
            raise ServerToolError(f"Could not reach the tool server: {exc}") from exc

        if not response.ok:
            raise error_for_status(response.status_code, self._server_message(response))

        try:
            data = response.json()
        except ValueError as exc:
            raise ServerToolError(f"Malformed response from the tool server: {exc}") from exc

        logger.info(
            "GET %s/tools/%s/data -> %d model(s), %d testfile(s)",
            self._server_url, tool_name, len(data.get("models", [])), len(data.get("testfiles", [])),
        )
        entries = data.get("entries")
        # One list per SCOPE, beside the flat one, for a tool whose deployment
        # points an argument at a subfolder of its hosted files -- AREG's CBCT
        # baseline picker must not offer the intraoral cohorts staged beside
        # them. Rebuilt rather than passed through, like the rest of this
        # payload, which is exactly how the section was dropped on the floor
        # once already: the server sent it, the panel never saw it, and the
        # picker quietly went on showing everything.
        scoped = data.get("scoped")
        return {
            "models": data.get("models", []),
            "testfiles": data.get("testfiles", []),
            # {} rather than None for an older server, so every caller can
            # index it without asking which server it is talking to.
            "entries": entries if isinstance(entries, dict) else {},
            "scoped": scoped if isinstance(scoped, dict) else {},
        }

    def download_testfile(
        self,
        tool_name: str,
        filename: str,
        destination: str,
        progress_cb: Optional[Callable[[str], None]] = None,
        scope: str = "",
    ) -> str:
        """Fetch one of the tool's server-hosted test files to `destination`.

        `GET /tools/{tool}/testfiles/{name}`, Bearer-protected. A hosted
        *folder* arrives as a .zip the server builds for us; unpacking it is
        the caller's business, this only moves the bytes.

        Pulled over parallel byte ranges, the same path a large result takes
        (`_download_reference`): these are whole cohorts -- the CBCT
        semi-automated set is 648 MB -- and one connection is bound by its own
        congestion window long before it is bound by the link. Falls back to a
        single streamed read when the server does not advertise ranges, so it
        is safe against any server that serves the endpoint at all.
        """
        # quote with no safe characters: the name comes from the server's own
        # listing, but it lands in a URL path and a stray "/" or "?" in it must
        # address the same file rather than a different route.
        url = f"{self._server_url}/tools/{tool_name}/testfiles/{quote(filename, safe='')}"
        # The subfolder the name was LISTED under, when a deployment scopes an
        # argument's hosted files. Said rather than guessed: a name is bare and
        # two scopes may hold the same one. Without it the server looks in the
        # tool's own folder and answers "No such testfile" for an entry its own
        # picker had just offered.
        if scope:
            url += f"?scope={quote(scope, safe='')}"
        headers = {"Authorization": f"Bearer {self._token}"}
        label = f"Downloading {filename}..."

        # Timed in three pieces, because "the download took sixteen seconds"
        # can mean the HEAD, the transfer, or the disk, and only one of the
        # three is worth optimising. Measured against curl on the same file the
        # server answers in 0.17 s.
        probe_started = time.perf_counter()
        size = transfer.probe_ranged(
            self._session, url, headers=headers
        )
        probe_took = time.perf_counter() - probe_started
        if size and size >= transfer.MIN_CHUNKED_BYTES:
            body_started = time.perf_counter()
            streams = self._parallelism
            transfer.download_ranged(
                self._session,
                url,
                destination,
                size,
                headers=headers,
                parallelism=streams,
                chunk_bytes=self._chunk_bytes,
                progress_cb=progress_cb,
                label=label,
            )
            body_took = time.perf_counter() - body_started
            parts = sorted(transfer.last_part_times)
            spread = ""
            if parts:
                spread = "; {} parts, fastest {:.2f}s, median {:.2f}s, slowest {:.2f}s".format(
                    len(parts), parts[0], parts[len(parts) // 2], parts[-1]
                )
            print(
                "[transfer] {} ranged {} stream(s), {:.1f} MB: probe {:.2f}s, "
                "body {:.2f}s = {:.1f} MB/s{}".format(
                    filename, streams, size / 1048576,
                    probe_took, body_took, size / 1048576 / max(body_took, 1e-9),
                    spread,
                )
            )
            logger.info(
                "GET %s -> %d byte(s) saved to %s (ranged, %d stream(s), "
                "probe %.2fs, body %.2fs)",
                url, size, destination, streams, probe_took, body_took,
            )
            return destination

        # The sequential fallback. Which of the two paths ran is the first
        # thing anyone asks when a transfer is slow, and until this print the
        # log said nothing at all when it was this one.
        body_started = time.perf_counter()
        try:
            response = self._session.get(
                url,
                headers=headers,
                stream=True,
                timeout=self._timeout,
                verify=tls.ca_bundle(),
            )
        except requests.RequestException as exc:
            raise ServerToolError(f"Could not reach the tool server: {exc}") from exc

        with response:
            if not response.ok:
                raise error_for_status(response.status_code, self._server_message(response))
            expected = self._expected_length(response) or 0
            received = 0
            with open(destination, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    out_file.write(chunk)
                    received += len(chunk)
                    if progress_cb:
                        progress_cb(_download_message(received, expected, filename))
        body_took = time.perf_counter() - body_started
        print(
            "[transfer] {} sequential (probe said {}), {:.1f} MB: probe {:.2f}s, "
            "body {:.2f}s = {:.1f} MB/s".format(
                filename, size if size else "no ranges", received / 1048576,
                probe_took, body_took, received / 1048576 / max(body_took, 1e-9),
            )
        )
        logger.info(
            "GET %s -> %d byte(s) saved to %s (sequential, probe %.2fs, body %.2fs)",
            url, received, destination, probe_took, body_took,
        )
        return destination

    def _fetch_tools(self) -> dict:
        try:
            response = self._session.get(
                f"{self._server_url}/tools", timeout=_TOOLS_FETCH_TIMEOUT, verify=tls.ca_bundle()
            )
        except requests.RequestException as exc:
            raise ServerToolError(f"Could not reach the tool server: {exc}") from exc

        if not response.ok:
            raise error_for_status(response.status_code, self._server_message(response))

        try:
            tools = response.json()
        except ValueError as exc:
            raise ServerToolError(f"Malformed response from the tool server: {exc}") from exc

        by_name = {tool["name"]: tool for tool in tools}
        logger.info("GET %s/tools -> %d tool(s): %s", self._server_url, len(by_name), sorted(by_name.keys()))
        return by_name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        tool_name: str,
        args: Optional[dict] = None,
        files: Optional[dict] = None,
        output_dir: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        run_id: Optional[str] = None,
        event_cb: Optional[Callable[[dict], None]] = None,
        cancel_event=None,
        batch: Optional[dict] = None,
    ) -> ToolResult:
        """`files`: {schema_argument_name: local_file_path}, one entry per
        `type: "file"` argument you're providing. Each is uploaded as its own
        multipart field named after its schema argument - a tool can declare
        several independent file arguments (e.g. SurgMovPred's "model" +
        "input"); there is no single reserved "file" key.

        `run_id` (see new_run_id) names this run to the server, which is what
        makes its progress readable and the run cancellable. Omitting it is
        exactly today's request, byte for byte.

        `event_cb` receives the run's progress events (see
        normalise_run_event), from a SECOND thread opened for the duration of
        the POST. It has to be a second connection: the thread calling this
        method is blocked inside the POST for the whole inference, which is
        precisely the window there is nothing to say from. Ignored without a
        `run_id`, there being nothing to subscribe to.

        `cancel_event` is a `threading.Event` the caller sets to withdraw the
        run. It cannot interrupt the POST itself - only the server's DELETE
        does that - but it stops this side doing any more work for a run
        nobody wants: no further parts uploaded, and above all no result
        archive pulled down after the answer arrives.
        """
        args = args or {}
        files = files or {}
        schema = self.get_tool_schema(tool_name)
        self._validate_against_schema(schema, args, files)

        headers = {"Authorization": f"Bearer {self._token}"}
        data = self._stringify(args)

        self._raise_if_cancelled(cancel_event, tool_name)

        # Detaching needs an id to report through, so a caller that minted none
        # keeps the blocking contract whatever the setting says. Decided BEFORE
        # the uploads, because it changes which of them travel in the request.
        #
        # ...and a run that asked to stop at a checkpoint keeps it too. The
        # server delivers the quality-control record -- what it produced, and
        # the reference to fetch it by -- in the RESPONSE BODY and nowhere
        # else: its detached path answers 202, writes a non-terminal `paused`
        # event and drops the payload it had built. So a detached run of this
        # kind would wait on a terminal event that never comes, then fail on
        # the stream's idle timeout with a message about losing track of it.
        # Blocking, the same run works, and the ceiling detaching exists to
        # lift is not the binding one here: the wait is a person reading
        # scans, and the POST is answered as soon as the run stops.
        detached = bool(self._detached_runs and run_id) and not _asks_to_stop(args)

        # Anything big enough to be worth it goes up FIRST, in parallel parts,
        # and this request then only references it. What stays in `files` is
        # what is small enough that a second and third round trip would cost
        # more than the single-connection upload does.
        #
        # ...unless the run is detached, in which case EVERY file goes up this
        # way however small. The server answers 202 before the tool starts, so
        # there is no point in the request at which it could stage a multipart
        # body, and it refuses one -- correctly, and with a message naming
        # POST /uploads. Sending an 80 kB landmark file on a detached run was a
        # refusal, not a slow path: the size threshold and the delivery mode
        # were decided independently of each other and could disagree.
        files, upload_references = self._upload_large_inputs(
            files, progress_cb, always=detached)
        if upload_references:
            data[_UPLOADS_FIELD] = json.dumps(upload_references)

        self._raise_if_cancelled(cancel_event, tool_name)

        if progress_cb:
            # Not the tool's name: a module's tool can be a router (AREG sends
            # an "IOS to CBCT" run to AREG_IOSCBCT), and naming the router read
            # as a step of its own. What is happening is the request leaving.
            progress_cb("Sending the request to the server...")

        post_headers = {**headers, **_RESULT_DELIVERY_HEADER}
        if detached:
            post_headers.update(_RUN_DELIVERY_HEADER)
        if run_id:
            # A header, so a server that has never heard of it ignores an
            # unknown header and answers exactly as it always did. Sent even
            # without an `event_cb`: the id is also what DELETE /runs/{id}
            # addresses, and a caller may well want to be able to cancel a run
            # it is not watching.
            post_headers[RUN_ID_HEADER] = run_id
        if batch:
            # `{"id", "index", "total"}`: which batch of which cohort this is.
            post_headers[BATCH_ID_HEADER] = str(batch["id"])
            post_headers[BATCH_INDEX_HEADER] = str(batch["index"])
            post_headers[BATCH_TOTAL_HEADER] = str(batch["total"])

        # Debug visibility only: argument/file *names*, never the token or the
        # argument/file contents. Silent unless the caller has raised this
        # logger's level (see ARCHITECTURE.md "How to inspect a request").
        logger.debug(
            "POST %s/run/%s | arg keys=%s | file args=%s | pre-uploaded=%s",
            self._server_url,
            tool_name,
            sorted(data.keys()),
            {name: os.path.basename(path) for name, path in files.items()},
            sorted(upload_references),
        )

        # Started BEFORE the POST, necessarily: from the next line on, this
        # thread is inside the request for the whole inference and cannot poll
        # anything. The cost is that the run is not registered server-side
        # until the request lands - and on the multipart path, landing means
        # the upload finishing - so the first attempts can legitimately answer
        # 404. That gap is what watch_run's startup window absorbs; it is not
        # an error condition. Torn down in the `finally` below whatever the
        # outcome, a watcher left retrying against a run that has already
        # answered being a thread holding a connection open for no one.
        watch_stop = threading.Event()
        # Not for a detached run: that one reads the same stream on this thread
        # and would otherwise have two readers of one run, both delivering every
        # event to the same callback.
        watcher = (None if self._detached_runs and run_id
                   else self._start_watcher(run_id, event_cb, watch_stop, cancel_event))

        try:
            file_handles = []
            try:
                files_payload = {}
                for arg_name, path in files.items():
                    file_handle = open(path, "rb")
                    file_handles.append(file_handle)
                    # The filename (with extension) must travel with the upload:
                    # the server validates extensions (.nii/.nii.gz/...) from it.
                    # Without it, requests defaults to a bare filename and every
                    # upload with an extension check fails server-side.
                    files_payload[arg_name] = (os.path.basename(path), file_handle)

                try:
                    # stream=True: the body is NOT downloaded here but inside
                    # _build_result, chunk by chunk straight to disk. Without it,
                    # requests buffers the entire result archive in RAM before a
                    # single byte can be written -- the larger a run's output, the
                    # closer that gets to taking Slicer down with it. The read
                    # timeout then applies between chunks, not to the whole
                    # download, so a big-but-flowing response can never time out
                    # merely for being big.
                    response = self._session.post(
                        f"{self._server_url}/run/{tool_name}",
                        headers=post_headers,
                        data=data,
                        files=files_payload or None,
                        timeout=self._timeout,
                        verify=tls.ca_bundle(),
                        stream=True,
                    )
                except requests.RequestException as exc:
                    raise ServerToolError(f"Network error while calling '{tool_name}': {exc}") from exc
            finally:
                for file_handle in file_handles:
                    file_handle.close()

            logger.debug(
                "Response from %s: status=%s content-type=%s",
                tool_name,
                response.status_code,
                response.headers.get("Content-Type"),
            )

            # Before the body, not after. A cancelled run's answer may still be
            # a several-hundred-megabyte archive, and pulling it down for a
            # panel that has already closed is the one expensive thing this
            # side can still avoid doing.
            self._raise_if_cancelled(cancel_event, tool_name, response=response)

            if detached:
                # 202 and nothing else: the run has not started yet. Everything
                # from here arrives on the stream, read on THIS thread -- the
                # caller has nothing else to do, and reading it here is what
                # makes the wait resumable, since watch_run reconnects and
                # dedupes on `seq` where a dropped POST simply lost the answer.
                return self._collect_detached(
                    tool_name, run_id, response, schema, output_dir,
                    progress_cb, event_cb, cancel_event,
                )

            if progress_cb:
                progress_cb("Processing response...")

            return self._build_result(tool_name, response, schema, output_dir,
                                      progress_cb, run_id=run_id)
        finally:
            # Whatever happened - a result, a 499, a dropped connection - the
            # run this watcher was reading is over. Left running, it would keep
            # reconnecting to a stream that will never say anything again.
            watch_stop.set()
            if watcher is not None:
                # Joined, but briefly. The thread is a daemon holding nothing
                # the caller needs; what makes the wait worth a few
                # milliseconds is the pooled connection it owns. It is never
                # waited on for longer, since it can legitimately be blocked in
                # a read for _RUN_EVENTS_READ_TIMEOUT.
                watcher.join(timeout=0.5)

    # ------------------------------------------------------------------
    # Run progress and cancellation
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_if_cancelled(cancel_event, tool_name: str, response=None) -> None:
        """Stop doing work for a run nobody is waiting for any more.

        Raises RunCancelled rather than returning a sentinel, because every
        caller of `run()` already has an error path and none of them has a
        "the caller changed their mind" path. `RunCancelled` is a class of its
        own precisely so a panel can close quietly on it instead of opening
        the error dialog a failure deserves.
        """
        if cancel_event is None or not cancel_event.is_set():
            return
        if response is not None:
            # The body was never read (stream=True), so this releases the
            # connection instead of leaving it draining an archive nobody will
            # look at.
            response.close()
        raise RunCancelled(f"'{tool_name}' was cancelled.", 499)

    def _start_watcher(self, run_id, event_cb, watch_stop, cancel_event, after_seq=-1):
        """The second connection, on its own daemon thread, or None.

        None whenever there is nothing to watch (no id) or nobody to tell (no
        callback), so the ordinary scripted call pays neither a thread nor a
        connection for a feature it is not using.
        """
        if not run_id or event_cb is None:
            return None

        stop = _AnyEvent([watch_stop, cancel_event])

        def watch():
            try:
                self.watch_run(run_id, event_cb, stop_event=stop, after_seq=after_seq)
            except Exception:
                # Never propagates, and never fails a run. This thread exists
                # to make a wait legible; a run that completed must not be
                # reported as failed because the thread narrating it hit
                # something. Logged at debug: an exception here says nothing
                # the user can act on.
                logger.debug("Run watcher stopped on an error", exc_info=True)

        thread = threading.Thread(target=watch, name="sadt-run-watch", daemon=True)
        thread.start()
        return thread

    def watch_run(self, run_id: str, on_event: Callable[[dict], None], stop_event=None,
                  after_seq: int = -1) -> bool:
        """Stream a run's progress events, calling `on_event` for each.

        Blocks until the run reaches a terminal state, until `stop_event` is
        set, or until it is clear no events will come. Returns whether any
        event was delivered - False is how a caller learns it is talking to a
        server that predates all of this.

        Three things about the loop are load-bearing:

        - **A 404 is tolerated for a startup window** (see
          _RUN_EVENTS_STARTUP_GRACE_SECONDS). The run only exists server-side
          once the POST arrives, and on the multipart path the POST *is* the
          upload. Giving up on the first 404 would go quiet on exactly the
          long uploads this exists for. Once ANY event has arrived the window
          is over: a 404 then means the run was reaped.
        - **It reconnects.** A stream that ends without a terminal event -
          a dropped connection, a proxy's idle timeout, a read timeout - is
          resumed rather than abandoned, which matters most on the runs that
          last hours.
        - **Which makes deduplication mandatory, not decorative.** Every
          connection replays the run's events from the beginning by design
          ("a watcher that attaches late is never behind"), so `seq` is what
          keeps a reconnect from re-announcing an hour of progress.

        `after_seq` skips what an earlier watcher of the same run already
        delivered: a run carried on after a review keeps its id and its event
        file, and replaying it from the start would announce everything before
        the stop a second time.

        Nothing here is ever fatal to a run. It raises only what the caller
        chooses to let escape; `_start_watcher` lets nothing.
        """
        url = f"{self._server_url}/runs/{quote(run_id, safe='')}/events"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "text/event-stream",
            # Compressing a stream is how a proxy ends up buffering it, and a
            # buffered progress stream is no progress stream at all.
            "Accept-Encoding": "identity",
        }
        started = time.monotonic()
        delivered_seq = after_seq
        delivered_any = False
        # Where this run's stream has got to, for the watcher of its resume.
        seen = self.__dict__.setdefault("_delivered_seq", {})

        def stopped() -> bool:
            return stop_event is not None and stop_event.is_set()

        def pause() -> None:
            if stop_event is not None:
                stop_event.wait(_RUN_EVENTS_RETRY_SECONDS)
            else:
                time.sleep(_RUN_EVENTS_RETRY_SECONDS)

        def within_startup_window() -> bool:
            return (
                not delivered_any
                and (time.monotonic() - started) < _RUN_EVENTS_STARTUP_GRACE_SECONDS
            )

        while not stopped():
            try:
                response = self._session.get(
                    url,
                    headers=headers,
                    # The tool's log lines for the requester, interleaved as
                    # `"kind": "log"` events on the same `seq`. Asked for every
                    # time: an older server ignores the parameter and sends
                    # none, and a server that has them sends none without it.
                    params=_RUN_LOGS_QUERY,
                    stream=True,
                    # (connect, read). The read half is long on purpose: the
                    # contract has no heartbeat, so silence is the normal state
                    # of a running inference.
                    timeout=(_TOOLS_FETCH_TIMEOUT, _RUN_EVENTS_READ_TIMEOUT),
                    verify=tls.ca_bundle(),
                )
            except requests.RequestException:
                if stopped() or not within_startup_window():
                    return delivered_any
                pause()
                continue

            with response:
                if response.status_code == 404:
                    # Three different things, and only one is worth waiting
                    # for: the run is not registered YET (keep trying), the
                    # server has no such endpoint (stop, silently - this is an
                    # older deployment and the panel keeps its elapsed timer),
                    # or the run has been reaped (stop, there is nothing left).
                    if not within_startup_window():
                        return delivered_any
                    pause()
                    continue
                if not response.ok:
                    # 401, 5xx, anything else: retrying would only repeat it.
                    logger.debug(
                        "Run event stream refused: HTTP %d", response.status_code
                    )
                    return delivered_any

                try:
                    for frame in _sse_data_frames(
                        response.iter_lines(decode_unicode=True)
                    ):
                        if stopped():
                            return delivered_any
                        try:
                            event = normalise_run_event(json.loads(frame))
                        except ValueError:
                            # A frame we cannot read is one frame, not the end
                            # of the stream: the run keeps going and so does
                            # this. Its content is never logged - a progress
                            # message is written by a tool and may name a file.
                            logger.debug("Unreadable run event frame, skipped")
                            continue
                        # One dedupe for both kinds: a log line and a progress
                        # event share the run's `seq`, so a reconnect's replay
                        # re-announces neither.
                        if event is None or event["seq"] <= delivered_seq:
                            continue
                        delivered_seq = event["seq"]
                        seen[run_id] = delivered_seq
                        delivered_any = True
                        on_event(event)
                        # A log line never ends the watch, whatever state a
                        # server put on it: only progress can be terminal.
                        if (not is_log_event(event)
                                and event["state"] in TERMINAL_RUN_STATES):
                            return True
                except requests.RequestException:
                    # A read timeout or a dropped connection mid-stream. The
                    # run is very probably still going; reconnect and let the
                    # replay-plus-dedupe pick up where this left off.
                    pass

            if stopped():
                return delivered_any
            pause()

        return delivered_any

    def cancel_run(self, run_id: str) -> bool:
        """Ask the server to stop a run: DELETE /runs/{id}. Never raises.

        Returns whether the server acknowledged. False covers three cases a
        caller cannot act on differently anyway - an older server with no such
        route, a run already finished and reaped, and an unreachable server -
        and in all three the local side has already released the panel.

        Retried once, like _release_result and for the same reason: this is the
        difference between a two-hour segmentation stopping now and it holding
        the card until it finishes for nobody, and a single dropped packet
        should not decide that.
        """
        url = f"{self._server_url}/runs/{quote(run_id, safe='')}"
        headers = {"Authorization": f"Bearer {self._token}"}
        for attempt in range(2):
            try:
                response = self._session.delete(
                    url, headers=headers, timeout=_TOOLS_FETCH_TIMEOUT, verify=tls.ca_bundle()
                )
            except requests.RequestException as exc:
                logger.debug("Could not cancel a run (attempt %d): %s", attempt, exc)
                continue
            if response.ok:
                logger.info("Cancelled a run on %s", self._server_url)
                return True
            if response.status_code == 404:
                # Nothing to cancel: no such endpoint, or the run is already
                # over. Neither is worth a retry.
                return False
            logger.debug("Server refused to cancel a run: HTTP %d", response.status_code)
        return False

    def resume_run(
        self,
        tool_name: str,
        run_id: str,
        corrections: Optional[dict] = None,
        output_dir: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        rewind_to: Optional[str] = None,
        event_cb: Optional[Callable[[dict], None]] = None,
        cancel_event=None,
    ) -> ToolResult:
        """Carry a stopped run on: POST /runs/{id}/resume.

        `event_cb` and `cancel_event` are `run`'s: the steps after a review
        report their progress and their log lines exactly as the ones before
        it did. Without a watcher here the panel went silent at the first stop,
        and whatever the tool said after it -- a registration's residuals,
        computed last -- never reached the person who started the run.

        `rewind_to` sends it BACKWARDS instead: the run is armed again at a
        checkpoint it already cleared and stops there, with that step's
        result untouched for a reader to correct. The same route otherwise --
        one more field, one different path -- because what comes back is the
        same thing either way: a finished run, or another checkpoint.

        `corrections` is {step name: local path}, the step names being exactly
        the `produced` entries of the checkpoint -- the server matches them
        against the folders the run actually wrote and answers 400 for
        anything else. An empty mapping is legal and means "carry on with what
        you produced", which is what a reader who changed nothing asks for.

        Blocking, and it has to be: the server offers no detached resume, and
        it ignores `X-Result-Delivery` on this route, so the remaining work
        happens inside this request and its answer streams back in the body.
        The answer is whatever a finished run answers -- or ANOTHER checkpoint,
        when a second one was armed and is reached, which is why this returns
        the same ToolResult as `run` and is meant to be called again from it.
        """
        if not run_id:
            raise ServerToolError("A run id is required to carry a stopped run on.")
        schema = self.get_tool_schema(tool_name)
        route = "rewind" if rewind_to else "resume"
        url = f"{self._server_url}/runs/{quote(run_id, safe='')}/{route}"
        headers = {"Authorization": f"Bearer {self._token}"}

        if progress_cb:
            progress_cb(
                f"Taking '{tool_name}' back to {rewind_to}..." if rewind_to
                else f"Sending your corrections to '{tool_name}'...")

        watch_stop = threading.Event()
        watcher = self._start_watcher(
            run_id, event_cb, watch_stop, cancel_event,
            after_seq=self.__dict__.get("_delivered_seq", {}).get(run_id, -1))
        try:
            return self._resume_request(tool_name, run_id, url, headers, schema,
                                        corrections, output_dir, progress_cb, rewind_to)
        finally:
            watch_stop.set()
            if watcher is not None:
                watcher.join(timeout=0.5)

    def _resume_request(self, tool_name, run_id, url, headers, schema, corrections,
                        output_dir, progress_cb, rewind_to) -> ToolResult:
        """The resume POST itself and its answer; see `resume_run`."""

        # Straight multipart, with no /uploads staging: the server reads this
        # body as a form and has no reference field for it. A correction is a
        # zip of one step's folder, so the size is the reader's edit rather
        # than the cohort, which is what makes that acceptable.
        handles = []
        try:
            payload = {}
            if rewind_to:
                # A plain form field beside the file parts. The server reads
                # the whole body as a form, so the two travel together and a
                # rewind carrying corrections is one request.
                payload["to"] = (None, rewind_to)
            for slot, path in (corrections or {}).items():
                handle = open(path, "rb")
                handles.append(handle)
                payload[slot] = (os.path.basename(path), handle)
            try:
                response = self._session.post(
                    url,
                    headers=headers,
                    files=payload or None,
                    timeout=self._timeout,
                    verify=tls.ca_bundle(),
                    stream=True,
                )
            except requests.RequestException as exc:
                raise ServerToolError(
                    f"Network error while carrying '{tool_name}' on: {exc}") from exc
        finally:
            for handle in handles:
                handle.close()

        logger.info("POST %s -> %s (%d correction(s))",
                    url, response.status_code, len(corrections or {}))
        return self._build_result(tool_name, response, schema, output_dir,
                                  progress_cb, run_id=run_id)

    def _stopped_result(self, tool_name, payload, output_dir, progress_cb,
                        run_id) -> ToolResult:
        """The answer of a run that stopped at a quality-control checkpoint.

        The archive is fetched here rather than left as a reference, because a
        reference is single-use and the server releases it on the first
        `DELETE` -- and what happens next is a person looking at scans, which
        is not a wait to hold server-side storage through.
        """
        reference = payload.get("result_ref")
        fetched = (self._download_reference(tool_name, reference, output_dir, progress_cb)
                   if reference else None)
        # `path` stays on the checkpoint and not on the ToolResult beside it:
        # what came down is not the run's answer, and two fields holding one
        # string is how they come to disagree.
        return ToolResult(
            kind="checkpoint",
            checkpoint=RunCheckpoint(
                run_id=run_id or "",
                stopped_after=str(payload.get("stopped_after") or ""),
                produced=tuple(str(name) for name in payload.get("produced") or ()),
                path=fetched.path if fetched is not None else None,
            ),
        )

    # ------------------------------------------------------------------
    # Bulk transfer (see transfer.py for why it is not one request)
    # ------------------------------------------------------------------

    def _upload_large_inputs(self, files: dict, progress_cb, always=False) -> tuple:
        """Split `files` into what still travels inside the /run request and
        what has already been sent through the upload endpoints.

        Returns `(remaining_files, {argument name: upload id})`. Falls back
        wholesale the moment a server turns out not to have the endpoints, so
        this extension keeps working against a deployment that has not been
        updated, that fallback is the reason the return is a pair rather than
        an in-place mutation.

        `always` sends every file this way whatever its size. A detached run
        needs it: the request it would otherwise ride in is answered 202 before
        the tool starts, and a server cannot stage a body it has already replied
        to. The fallback still applies -- a server with no upload endpoints
        cannot serve a detached run either, and the caller ends up on the
        blocking path with its files in the request, which is what it wants.
        """
        if self._chunked_uploads is False:
            return files, {}

        remaining = dict(files)
        references = {}
        minimum = 1 if always else max(self._chunk_bytes * 2, 1)
        for arg_name, path in files.items():
            if not transfer.should_chunk(path, minimum):
                continue
            try:
                references[arg_name] = transfer.upload_file(
                    self._session,
                    self._server_url,
                    {"Authorization": f"Bearer {self._token}"},
                    path,
                    parallelism=self._parallelism,
                    chunk_bytes=self._chunk_bytes,
                    compress=self._compress_uploads,
                    progress_cb=progress_cb,
                )
            except transfer.UnsupportedByServer:
                logger.info(
                    "%s has no chunked-upload endpoints; falling back to a single request",
                    self._server_url,
                )
                self._chunked_uploads = False
                # Whatever went up before this file did is still valid and is
                # still referenced; only the rest reverts to multipart.
                break
            self._chunked_uploads = True
            remaining.pop(arg_name)
        return remaining, references

    def _collect_detached(self, tool_name, run_id, response, schema, output_dir,
                          progress_cb, event_cb, cancel_event) -> ToolResult:
        """Wait on the event stream for the verdict, then fetch what it names.

        A server that does not know the header answers the ordinary 200 with
        the ordinary body, and that is handled here rather than guarded against
        -- which is what lets one client speak to both.
        """
        if response.status_code != 202:
            return self._build_result(tool_name, response, schema, output_dir,
                                      progress_cb, run_id=run_id)
        response.close()

        if progress_cb:
            progress_cb(f"'{tool_name}' accepted; waiting for it to finish...")

        terminal = self._await_terminal(run_id, event_cb, cancel_event)
        if terminal is None:
            if cancel_event is not None and cancel_event.is_set():
                raise RunCancelled(f"'{tool_name}' was cancelled.", 499)
            raise ServerToolError(
                f"Lost track of '{tool_name}': the server accepted the run, but "
                "its event stream ended without saying how it finished. The run "
                f"id was {run_id}."
            )

        state = terminal.get("state")
        if state == "cancelled":
            raise RunCancelled(f"'{tool_name}' was cancelled.", 499)
        if state != "done":
            message = terminal.get("message")
            raise ServerToolError(
                str(message)[:_SERVER_MESSAGE_MAX_LEN] if message
                else f"'{tool_name}' failed on the server."
            )

        # The terminal event carries either a pointer to fetch, or a small
        # answer inline for a tool whose output is text.
        answer = terminal.get("result") or {}
        reference = answer.get("result_ref")
        if reference:
            return self._download_reference(tool_name, reference, output_dir, progress_cb)
        if "result" in answer:
            return ToolResult(kind="text", text=answer["result"])
        raise ServerToolError(
            f"'{tool_name}' finished, but said nothing about where its result is."
        )

    def _await_terminal(self, run_id, event_cb, cancel_event):
        """Read the stream to its end and hand back the event that ended it.

        `watch_run` already reconnects on a dropped stream and dedupes on
        `seq`, so this wait survives what a blocking POST could not: the
        connection can go away and come back without losing the answer.
        """
        holder = {}

        def capture(event):
            if not is_log_event(event) and event.get("state") in TERMINAL_RUN_STATES:
                holder["event"] = event
            if event_cb is not None:
                event_cb(event)

        self.watch_run(run_id, capture, stop_event=cancel_event)
        return holder.get("event")

    def _download_reference(
        self,
        tool_name: str,
        reference: dict,
        output_dir: Optional[str],
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> ToolResult:
        """Fetch a result the server kept for us, over parallel byte ranges."""
        if not output_dir:
            raise ServerToolError("An output directory is required to save the returned file.")
        result_id = reference.get("result_id")
        if not result_id:
            raise ServerToolError(f"Malformed result reference from '{tool_name}'.")

        os.makedirs(output_dir, exist_ok=True)
        # basename, always: the name is the server's to choose, and a path
        # separator in it would otherwise write outside the output folder.
        filename = os.path.basename(reference.get("filename") or "") or f"{tool_name}_result.bin"
        dest_path = os.path.join(output_dir, filename)
        size = int(reference.get("size") or 0)

        headers = {"Authorization": f"Bearer {self._token}"}
        url = f"{self._server_url}/results/{result_id}"
        try:
            transfer.download_ranged(
                self._session,
                url,
                dest_path,
                size,
                headers=headers,
                parallelism=self._parallelism,
                chunk_bytes=self._chunk_bytes,
                progress_cb=progress_cb,
            )
            self._verify_archive(tool_name, dest_path)
        finally:
            # In a `finally`, and this is the point: the server keeps the
            # result until somebody says it can go, so every way out of this
            # method has to say it -- a download that failed halfway and a
            # result archive that failed its integrity check are exactly the
            # cases where a `return`-only cleanup would leave patient data
            # sitting on the server until the reaper got to it. Neither is
            # retryable from here (the reference is single-use), so there is
            # nothing to keep it for.
            self._release_result(url, headers, result_id)

        logger.info(
            "GET %s -> %d byte(s) saved to %s (ranged, %d stream(s))",
            url, size, dest_path, self._parallelism,
        )
        return ToolResult(kind="file", path=dest_path)

    def _release_result(self, url: str, headers: dict, result_id: str) -> None:
        """Tell the server it can delete the stored result.

        Retried once, because this is the difference between the file going
        away now and it lingering until the server's idle reaper collects it,
        and a single dropped packet should not decide that. Still best effort
        in the end: it must never turn a finished run into a failed one, and
        the reaper is the guarantee behind it -- this is what makes that
        guarantee almost never the thing that has to fire.
        """
        for attempt in range(2):
            try:
                response = self._session.delete(
                    url, headers=headers, timeout=_TOOLS_FETCH_TIMEOUT, verify=tls.ca_bundle()
                )
                if response.ok or response.status_code == 404:
                    return
                logger.debug(
                    "server refused to release result %s: HTTP %d", result_id, response.status_code
                )
            except requests.RequestException as exc:
                logger.debug("could not release result %s (attempt %d): %s", result_id, attempt, exc)
        logger.warning(
            "Result %s could not be released; the server will reap it after its idle timeout.",
            result_id,
        )

    def _build_result(
        self,
        tool_name: str,
        response,
        schema: dict,
        output_dir: Optional[str],
        progress_cb: Optional[Callable[[str], None]] = None,
        run_id: Optional[str] = None,
    ) -> ToolResult:
        # run() sends the request with stream=True, so the body has not been
        # read yet: .json()/.text below consume it for the small responses,
        # the iter_content loop consumes it for file results, and close() in
        # the finally releases the connection on every path.
        try:
            if not response.ok:
                raise error_for_status(response.status_code, self._server_message(response))

            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ServerToolError(f"Malformed response from the tool server: {exc}") from exc
                # The run STOPPED rather than finished. Checked before the
                # reference below, because a quality-control payload carries
                # one too and downloading it as if it were the answer would
                # lose the one thing that says the run is still alive.
                if payload.get("quality_control"):
                    return self._stopped_result(
                        tool_name, payload, output_dir, progress_cb, run_id)
                # A file result the server agreed to hand over by reference
                # (see _RESULT_DELIVERY_HEADER): the bytes are still on the
                # server and come down next, in parallel. Anything else is a
                # "text" tool's answer, exactly as before.
                if payload.get("result_ref"):
                    return self._download_reference(
                        tool_name, payload["result_ref"], output_dir, progress_cb
                    )
                return ToolResult(kind="text", text=payload.get("result"))

            if not output_dir:
                raise ServerToolError("An output directory is required to save the returned file.")

            os.makedirs(output_dir, exist_ok=True)
            dest_path = os.path.join(
                output_dir, self._result_filename(tool_name, response, schema, content_type)
            )
            expected_bytes = self._expected_length(response)
            received = 0
            with open(dest_path, "wb") as fh:
                for chunk in response.iter_content(_DOWNLOAD_CHUNK_BYTES):
                    fh.write(chunk)
                    received += len(chunk)
                    if progress_cb:
                        progress_cb(_download_message(received, expected_bytes))
            self._verify_download(tool_name, response, dest_path, received)

            # INFO on purpose (the request-shape logs above are DEBUG): this
            # is the one line that decides, after the fact, whether a
            # missing-results report is a transfer problem or a server one.
            # Only sizes and headers -- never the file's contents.
            logger.info(
                "POST %s/run/%s -> %d byte(s) saved to %s (Content-Type: %s, Content-Disposition: %s)",
                self._server_url,
                tool_name,
                received,
                dest_path,
                content_type or "<none>",
                response.headers.get("Content-Disposition") or "<none>",
            )
            return ToolResult(kind="file", path=dest_path)
        finally:
            response.close()

    @staticmethod
    def _expected_length(response) -> Optional[int]:
        """The download's total size, when it can be trusted.

        None whenever the body is transfer-compressed: Content-Length then
        counts wire bytes while what lands on disk is the decompressed stream,
        so using it would report a progress percentage running past 100.
        """
        if response.headers.get("Content-Encoding"):
            return None
        raw = response.headers.get("Content-Length")
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @classmethod
    def _verify_download(cls, tool_name: str, response, dest_path: str, received: int) -> None:
        """A file result must arrive complete or fail loudly, whatever its size.

        Without this, a connection dropped mid-body leaves a truncated file on
        disk; for a .zip the base widget then unpacks whatever central
        directory survives, silently delivering a SUBSET of the results -- the
        worst possible failure for medical data. The partial file is removed
        before raising, so no later step can pick it up by accident.
        """
        expected_bytes = cls._expected_length(response)
        if expected_bytes is not None and received != expected_bytes:
            os.remove(dest_path)
            raise ServerToolError(
                f"Truncated result from '{tool_name}': received {received} of "
                f"{expected_bytes} bytes. Nothing was kept; run the tool again."
            )
        cls._verify_archive(tool_name, dest_path)

    @staticmethod
    def _verify_archive(tool_name: str, dest_path: str) -> None:
        """CRC-check every member of a result .zip.

        Catches corruption that a matching byte count cannot (and truncation
        too, when the server never sent a Content-Length). Reads the archive
        once from local disk -- seconds, next to an inference measured in
        minutes -- and it is what stands between a half-transferred archive and
        the base widget unpacking whatever central directory survived, silently
        delivering a SUBSET of the results.
        """
        if not dest_path.lower().endswith(".zip"):
            return
        try:
            with zipfile.ZipFile(dest_path) as archive:
                corrupt = archive.testzip()
        except zipfile.BadZipFile as exc:
            os.remove(dest_path)
            raise ServerToolError(
                f"The result archive from '{tool_name}' is unreadable "
                f"(incomplete transfer?): {exc}. Nothing was kept; run the tool again."
            ) from exc
        if corrupt is not None:
            os.remove(dest_path)
            raise ServerToolError(
                f"The result archive from '{tool_name}' failed its integrity check "
                f"at '{corrupt}'. Nothing was kept; run the tool again."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_filename(tool_name: str, response, schema: dict, content_type: str) -> str:
        """Prefer the server-provided filename (Content-Disposition); otherwise
        derive one from the schema's output_kind so file loaders that key off the
        extension (e.g. slicer.util.loadSegmentation expects .nii/.nii.gz) work."""
        content_disposition = response.headers.get("Content-Disposition", "")
        match = _CONTENT_DISPOSITION_FILENAME_RE.search(content_disposition)
        if match:
            # basename: the header is the server's to write, and a path
            # separator in it would place the result outside output_dir.
            name = os.path.basename(match.group(1).strip())
            if name:
                return name

        if schema.get("output_kind") == "segmentation":
            extension = ".nii.gz"
        else:
            # Mirror the server's own mimetypes.guess_type(): derive a real
            # extension from Content-Type instead of a generic .bin/.gz guess.
            # This keeps extension-based decisions downstream (e.g.
            # slicer_io.is_extractable_archive) correct even without a
            # Content-Disposition header.
            bare_content_type = content_type.split(";", 1)[0].strip()
            extension = mimetypes.guess_extension(bare_content_type) if bare_content_type else None
            if not extension:
                extension = ".gz" if "gzip" in content_type else ".bin"
        return f"{tool_name}_result{extension}"

    @staticmethod
    def _server_message(response) -> Optional[str]:
        """For 400/422 the server's own message must be propagated verbatim. Try
        JSON's "detail"/"message" first (FastAPI-style errors), then fall back to
        the raw response body so a plain-text error is never silently dropped."""
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            message = payload.get("detail") or payload.get("message")
            if message:
                # Capped like the raw body below: it ends up in a dialog, and
                # a server is not owed an unbounded one.
                if not isinstance(message, str):
                    message = json.dumps(message)
                return message[:_SERVER_MESSAGE_MAX_LEN]

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            return None
        return text[:_SERVER_MESSAGE_MAX_LEN]

    @staticmethod
    def _stringify(args: dict) -> dict:
        """Every scalar becomes a string; the server does the type coercion.

        A "multichoice" argument arrives here as the *complete* {option:
        checked} dict (see formgen.MultiChoiceGroup) and is sent as JSON. Two
        things about that are load-bearing:

        - The whole dict travels, unchecked options included. Server-side, what
          is sent *is* the selection: an option left out counts as unchecked
          whatever its declared default, and omitting the argument entirely is
          what applies the defaults. So "everything unchecked" and "argument
          absent" are different requests, and only the real state of the boxes
          can tell them apart.
        - JSON, not the `a,b` shortcut. The server also accepts a
          comma-separated list of the checked options, but that spelling is for
          curl: it breaks the moment an option name contains a comma.
        """
        stringified = {}
        for key, value in args.items():
            if isinstance(value, bool):
                stringified[key] = "true" if value else "false"
            elif isinstance(value, (dict, list, tuple)):
                # dict: the multichoice state above. list/tuple: a "vec2"
                # argument's [x, y] pair (formgen.JoystickInput).
                stringified[key] = json.dumps(value)
            else:
                stringified[key] = str(value)
        return stringified

    @staticmethod
    def _validate_against_schema(schema: dict, args: dict, files: dict) -> None:
        """Catch obvious mistakes before paying a network round-trip.

        Mirrors the server's own checks (unexpected/missing arguments); a real
        request can still fail server-side (e.g. disallowed file extension).
        """
        tool_name = schema.get("name", "?")
        arguments = schema.get("arguments", {})

        for name in args:
            if name not in arguments:
                raise ServerToolError(f"Unexpected argument '{name}' for tool '{tool_name}'.")

        for name in files:
            if name not in arguments:
                raise ServerToolError(f"Unexpected file argument '{name}' for tool '{tool_name}'.")
            if not is_file_type(arguments[name].get("type", "")):
                raise ServerToolError(f"Argument '{name}' for tool '{tool_name}' is not a file argument.")

        for name, spec in arguments.items():
            if is_file_type(spec.get("type", "")):
                # A `server_selectable` file argument has two valid shapes: an
                # upload, or the NAME of a file the server hosts, sent as a
                # plain form value under the same field name. Requiring an
                # upload here would reject the second - the very shape that
                # keeps a hosted test cohort from travelling.
                satisfied = name in files or (spec.get("server_selectable") and name in args)
                if spec.get("required") and not satisfied:
                    raise ServerToolError(f"Missing required file argument '{name}' for tool '{tool_name}'.")
            elif spec.get("required") and name not in args:
                raise ServerToolError(f"Missing required argument '{name}' for tool '{tool_name}'.")
