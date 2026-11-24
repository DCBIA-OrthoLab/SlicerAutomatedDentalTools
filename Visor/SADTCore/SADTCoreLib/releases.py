"""Follow SADT-VISOR's tools releases, and fetch the code of one of them.

Every push to SADT-VISOR's `deploy` that changes what a tool is cuts a tag
`tools-YYYY.MM.DD-N`. That tag is the version a clinician sees, the folder the
tools are installed in, and what they are updated to. This module answers three
questions, with the standard library only and without git on the machine:

* which `tools-*` tags exist (`list_versions`): one HTTPS request to the
  repository's ref advertisement, the request `git ls-remote` makes. It is not
  GitHub's REST API, so the 60-requests-an-hour limit an office shares behind
  one address does not apply, and asking every 15 minutes costs a few KB;
* whether this client can drive a release (`api_level`): the release's
  `scripts/run_tool.py` declares `SUPERVISOR_API`; a release without it, or at
  a level this client does not know, is never offered;
* the code of a release (`fetch_source`): the tag's source archive, ~3 MB,
  unpacked once per version under the tools folder. It holds every tool's
  `pyproject.toml` and `uv.lock`, which is what an install reads.

What was learnt is kept in `<tools folder>/releases.json`, shared by every
Slicer using that folder, so a Slicer started offline still knows the newest
version it saw.
"""

import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.request
from typing import Optional

REPOSITORY = "DCBIA-OrthoLab/SADT-VISOR"
TAG_PREFIX = "tools-"
# The SUPERVISOR_API levels this client drives (see run_tool.py).
KNOWN_API_LEVELS = (2,)
# How often a Slicer asks for new releases.
CHECK_INTERVAL_SECONDS = 15 * 60
STATE_FILE = "releases.json"
SOURCES_DIR = ".sources"

_REFS_URL = "https://github.com/{repo}.git/info/refs?service=git-upload-pack"
_RAW_URL = "https://raw.githubusercontent.com/{repo}/{tag}/scripts/run_tool.py"
_ARCHIVE_URL = "https://codeload.github.com/{repo}/tar.gz/refs/tags/{tag}"
_HEADERS = {"User-Agent": "sadt-slicer-client"}
_LEVEL = re.compile(rb"^SUPERVISOR_API = (\d+)\s*$", re.M)


class ReleaseError(RuntimeError):
    """A release could not be listed or fetched; the message says why."""


def version_key(version: str):
    """`2026.11.24-2` sorted as numbers, so -10 comes after -9."""
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 -- fixed https URLs
            return response.read()
    except Exception as error:  # noqa: BLE001 -- offline, DNS, HTTP: all mean "not now"
        raise ReleaseError(f"{url.split('?')[0]}: {error}") from error


def parse_refs(advertisement: bytes) -> list:
    """The `tools-*` versions named in a smart-HTTP ref advertisement, newest first."""
    found = set()
    for match in re.finditer(rb"refs/tags/" + re.escape(TAG_PREFIX.encode()) + rb"([0-9][0-9.\-]*[0-9])(\^\{\})?",
                             advertisement):
        found.add(match.group(1).decode())
    return sorted(found, key=version_key, reverse=True)


def list_versions(timeout: float = 10.0) -> list:
    """Every `tools-*` release of SADT-VISOR, newest first, without the prefix."""
    return parse_refs(_get(_REFS_URL.format(repo=REPOSITORY), timeout))


def parse_level(run_tool_source: bytes) -> Optional[int]:
    found = _LEVEL.search(run_tool_source)
    return int(found.group(1)) if found else None


def api_level(version: str, timeout: float = 10.0) -> Optional[int]:
    """The SUPERVISOR_API of a release, or None when it declares none (too old)."""
    tag = TAG_PREFIX + version
    return parse_level(_get(_RAW_URL.format(repo=REPOSITORY, tag=tag), timeout))


# ---- what this tools folder has learnt -----------------------------------

def read_state(root: str) -> dict:
    try:
        with open(os.path.join(root, STATE_FILE), encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(root: str, state: dict) -> None:
    os.makedirs(root, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=root, prefix=".releases-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=1, sort_keys=True)
    os.replace(temporary, os.path.join(root, STATE_FILE))


def latest_known(root: str) -> Optional[str]:
    """The newest release this client can drive, as last seen. None before any check."""
    return read_state(root).get("latest")


def check(root: str, timeout: float = 10.0, levels=KNOWN_API_LEVELS) -> dict:
    """Ask GitHub for the releases, and remember the newest one this client drives.

    The level of a release never changes, so it is asked once per release and
    kept. Returns the state written: `latest`, `checked` (epoch seconds),
    `error` (empty when the check succeeded).
    """
    state = read_state(root)
    known = dict(state.get("levels", {}))
    try:
        versions = list_versions(timeout)
        latest = None
        for version in versions:
            if version not in known:
                known[version] = api_level(version, timeout)
            if known[version] in levels:
                latest = version
                break
        state.update(latest=latest or state.get("latest"), levels=known,
                     checked=time.time(), error="")
    except ReleaseError as error:
        state.update(levels=known, checked=time.time(), error=str(error))
    _write_state(root, state)
    return state


def describe_check(before: str, state: dict) -> str:
    """One line for a user who asked: what the check found, against `before`."""
    when = time.strftime("%H:%M", time.localtime(float(state.get("checked") or time.time())))
    latest = state.get("latest")
    if state.get("error"):
        seen = f" The newest release seen before is {latest}." if latest else ""
        return f"Could not reach GitHub at {when} (offline?).{seen}"
    if latest and version_key(latest) > version_key(before):
        return f"Tools {latest} is available (this Slicer had {before}), asked at {when}."
    return f"Up to date: {before} is the newest tools release, asked at {when}."


def due(root: str, interval: float = CHECK_INTERVAL_SECONDS) -> bool:
    """Whether this tools folder has not been checked for `interval` seconds."""
    return time.time() - float(read_state(root).get("checked") or 0) >= interval


# ---- the code of a release -----------------------------------------------

def source_dir(root: str, version: str) -> str:
    return os.path.join(root, SOURCES_DIR, TAG_PREFIX + version)


def fetch_source(root: str, version: str, timeout: float = 60.0) -> str:
    """The unpacked code of a release, downloaded the first time it is needed.

    Unpacked beside its final place and renamed into it, so a download cut
    short never leaves a half tree that later reads as complete.
    """
    target = source_dir(root, version)
    if os.path.isdir(os.path.join(target, "tools")):
        return target
    data = _get(_ARCHIVE_URL.format(repo=REPOSITORY, tag=TAG_PREFIX + version), timeout)
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(dir=parent, prefix=".unpacking-")
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            _safe_extract(archive, staging)
        tops = os.listdir(staging)
        if len(tops) != 1 or not os.path.isdir(os.path.join(staging, tops[0], "tools")):
            raise ReleaseError(f"the archive of {TAG_PREFIX}{version} is not an SADT-VISOR tree")
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.replace(os.path.join(staging, tops[0]), target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def _safe_extract(archive: tarfile.TarFile, destination: str) -> None:
    """Extract, refusing a member that would land outside `destination`."""
    base = os.path.realpath(destination)
    for member in archive.getmembers():
        path = os.path.realpath(os.path.join(destination, member.name))
        if path != base and not path.startswith(base + os.sep):
            raise ReleaseError(f"unsafe path in the release archive: {member.name}")
        if member.issym() or member.islnk():
            link = os.path.realpath(os.path.join(os.path.dirname(path), member.linkname))
            if not link.startswith(base + os.sep):
                raise ReleaseError(f"unsafe link in the release archive: {member.name}")
    try:
        archive.extractall(destination, filter="data")  # noqa: S202 -- checked above too
    except TypeError:  # Python without extraction filters
        archive.extractall(destination)  # noqa: S202 -- every member checked above
