"""Where the SADT-VISOR tools live on this machine, and how they get there.

One folder per user, SHARED by every Slicer installed: a tool carries its own
venv and does not depend on Slicer's Python, so the copy of AMASSS (~8 GB)
one Slicer downloaded serves every other. Layout:

    <root>/
        README.txt                what this folder is, and that it is safe to delete
        users.json                which Slicer installs use it, and when they last did
        tools-<version>/
            scripts/              run_tool.py, describe.py: SADT-VISOR's supervisor
            tools/<Tool>/         the tool's source and its .venv (uv sync --frozen)
            tools/<Tool>/.sadt-install.json   version, calls, size, when
        DATA/<slug>/models/...    the models, shared across tool versions

The root is, in order: `SADT_TOOLS_ROOT` (development), the location this Slicer
was told to use (its own settings, read by the caller), or the per-user default
for the platform. Installing a tool installs the tools it calls through its
supervisor, read off its source by describe.py (`calls`), recursively.

No slicer, no qt: the panels drive it, the unit tests run it bare.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# The SADT-VISOR tools release this extension was built with: the release tag
# without its `tools-` prefix (SADT-VISOR's tools-release workflow tags them
# tools-YYYY.MM.DD-N). Every install lands in its own `tools-<version>` folder,
# so two versions can coexist. Following releases (releases.py), the client
# moves past it to the newest release it can drive; this one is what it runs
# before it has ever asked, and it must be a release carrying SUPERVISOR_API.
TOOLS_VERSION = "2026.11.24-2"

ROOT_VARIABLE = "SADT_TOOLS_ROOT"
# Where the pilot takes the tools and the models FROM: an SADT-VISOR checkout
# and a DATA folder. A release download replaces these two, nothing else.
SOURCE_VARIABLE = "SADT_VISOR_SOURCE"
DATA_SOURCE_VARIABLE = "SADT_DATA_SOURCE"

INSTALL_RECORD = ".sadt-install.json"
# A tool being replaced waits here until its successor is complete.
PREVIOUS_SUFFIX = ".previous"

# The uv every install runs, fetched into the tools folder on first use: a
# clinician installs nothing by hand, and every machine builds the tools with
# the same uv. The checksums are those uv publishes beside each release.
UV_VERSION = "0.12.3"
UV_RELEASES = {
    ("Darwin", "arm64"): ("aarch64-apple-darwin.tar.gz",
                          "546f7f8a6c70ff13a3a9d2bc958db3427298cebf3e0cb756f9177133b7068843"),
    ("Darwin", "x86_64"): ("x86_64-apple-darwin.tar.gz",
                           "4c9f52262a14da336e4a42ed24992d12d0c956acde87619e4611d321dffa602b"),
    ("Linux", "x86_64"): ("x86_64-unknown-linux-gnu.tar.gz",
                          "600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101"),
    ("Linux", "aarch64"): ("aarch64-unknown-linux-gnu.tar.gz",
                           "bb66cb52e7b1823aed1183630d8d8e5c958840d584a4c55ec10a4cfc168dcca2"),
    ("Windows", "x86_64"): ("x86_64-pc-windows-msvc.zip",
                            "b23350c79e8ad0192b8124af13a0f17e8d4e4549524785e1aef389ae5a06990e"),
    ("Windows", "arm64"): ("aarch64-pc-windows-msvc.zip",
                           "4343217d668727b8a8eb5cad92389a1d2eeead93c89940d1b955ba1bb15462eb"),
}
UV_URL = "https://github.com/astral-sh/uv/releases/download/{version}/uv-{asset}"

# How a tool's environment is made: from its lock, without its development
# tools, and WITH its optional extras, as the VISOR server does. A tool may keep
# its engine in an extra (Crown_Seg's `segmentation` holds shapeaxi); without
# `--all-extras` it installs, shows its panel, and fails every run.
UV_SYNC = ("sync", "--frozen", "--no-dev", "--all-extras", "--quiet")
FOLDER_NAME = "sadt-tools"

README = """SADT tools
==========

This folder holds the tools of the SlicerAutomatedDentalTools extension for
3D Slicer (SADT-VISOR tools): each tool's code, its own Python environment,
and the models it uses. It is shared by every Slicer installed for this user.

It is safe to delete. A tool that is needed again is downloaded again.

To see what is here, how much space it takes, and which Slicer uses it, or to
remove tools one by one: in Slicer, open Automated Dental Tools > SADT Tools
Manager.

Used by (updated each time a Slicer uses it): see users.json.
"""


def default_root() -> str:
    """The per-user folder for the platform: shared by every Slicer of this user."""
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif system == "Darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, FOLDER_NAME)


def resolve_root(slicer_choice: str = "") -> tuple:
    """`(root, why)`: where the tools are, and what decided it, for the panel to say."""
    if os.environ.get(ROOT_VARIABLE):
        return os.environ[ROOT_VARIABLE], "set by SADT_TOOLS_ROOT (development)"
    if slicer_choice:
        return slicer_choice, "chosen for this Slicer"
    return default_root(), "default, shared by every Slicer of this user"


def folder_size(path: str) -> int:
    """Bytes under `path`, counting each hard-linked file once and following no link."""
    total, seen = 0, set()
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                    continue
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += stat.st_size
    return total


def free_space(path: str) -> int:
    """Bytes free on the disk `path` is (or will be) on: its nearest existing parent."""
    probe = os.path.abspath(path)
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def needed_on_disk(estimate: dict) -> int:
    """What an install takes on disk: environments unpack to ~2.6x their download."""
    return int(estimate["environment"] * 2.6) + estimate["models"] + estimate.get("testfiles", 0)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def version_key(version: str) -> tuple:
    """`2026.11.19-10` sorts after `2026.11.19-2`: compared as numbers, not text."""
    import re

    return tuple(int(part) for part in re.findall(r"\d+", version or ""))


def no_window() -> dict:
    """Popen keywords that keep a console window from opening on Windows.

    Slicer is a GUI program: without this every `uv sync`, download and tool
    run would open a console, and closing it would kill the step.
    """
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


import threading

# The same rule inside one Slicer, between its own threads: an install running
# on a worker while the Manager starts a removal on another. Re-entrant, so an
# install may prune the cache it holds the lock for.
_THREAD_LOCK = threading.RLock()


class StoreBusy(RuntimeError):
    """Another Slicer (or another window of this one) is changing the tools folder."""


class StoreLock:
    """One install or removal at a time in a tools folder, across every Slicer.

    A lock FILE holding the owner's process id, created exclusively: it works
    the same on every platform and every file system. One left behind by a
    Slicer that crashed is recognised by its dead process and taken over.
    """

    NAME = ".lock"

    def __init__(self, root: str):
        self.path = os.path.join(root, self.NAME)
        self._held = False

    def holder(self) -> Optional[int]:
        try:
            with open(self.path, encoding="utf-8") as handle:
                pid = int(handle.read().strip() or 0)
        except (OSError, ValueError):
            return None
        return pid if _process_alive(pid) and pid != os.getpid() else None

    def __enter__(self):
        if not _THREAD_LOCK.acquire(blocking=False):
            raise StoreBusy("Tools are already being installed or removed in this Slicer. "
                            "Try again when it has finished.")
        try:
            return self._take_file()
        except BaseException:
            _THREAD_LOCK.release()
            raise

    def _take_file(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        for _attempt in range(2):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    with open(self.path, encoding="utf-8") as existing:
                        pid = int(existing.read().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid == os.getpid():
                    return self  # re-entered by this process: already ours
                if _process_alive(pid):
                    raise StoreBusy(
                        "Another Slicer is installing or removing tools in this folder. "
                        "Try again when it has finished.")
                try:
                    os.remove(self.path)  # left by a Slicer that is gone
                except OSError:
                    pass
                continue
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(str(os.getpid()))
            self._held = True
            return self
        raise StoreBusy("The tools folder is locked; try again in a moment.")

    def __exit__(self, *_exc):
        if self._held:
            self._held = False
            try:
                os.remove(self.path)
            except OSError:
                pass
        _THREAD_LOCK.release()
        return False


class InstallError(RuntimeError):
    """A tool could not be installed; the message says why, for a clinician."""


class ToolStore:
    """The tools folder of one root, at one tools version."""

    def __init__(self, root: str, version: str = TOOLS_VERSION, source: str = "",
                 data_source: str = ""):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.version = version
        # Where tools and models are installed FROM (the pilot: a checkout and a
        # DATA folder); also what an installed tool is compared with for updates.
        self.source = source
        self.data_source = data_source
        # Installed from a SADT-VISOR release (its code downloaded under
        # `.sources/`) rather than from a checkout; and kept on that release.
        self.from_release = False
        self.pinned = False

    # ---- layout ------------------------------------------------------------

    @property
    def version_dir(self) -> str:
        return os.path.join(self.root, f"tools-{self.version}")

    @property
    def scripts_dir(self) -> str:
        return os.path.join(self.version_dir, "scripts")

    @property
    def tools_dir(self) -> str:
        return os.path.join(self.version_dir, "tools")

    @property
    def data_dir(self) -> str:
        return os.path.join(self.root, "DATA")

    @property
    def uv_cache(self) -> str:
        return os.path.join(self.root, ".uv-cache")

    @property
    def private_cache(self) -> bool:
        """The clinician's setup: packages cached inside this folder.

        A developer pointing the tools at a local DATA mirror keeps uv's own
        cache instead: installs stay instant and nothing is stored twice.
        """
        return not self.data_source

    def lock(self) -> StoreLock:
        return StoreLock(self.root)

    def ensure_layout(self) -> None:
        os.makedirs(self.tools_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        readme = os.path.join(self.root, "README.txt")
        if not os.path.isfile(readme):
            with open(readme, "w", encoding="utf-8") as handle:
                handle.write(README)

    # ---- who uses it -------------------------------------------------------

    def register_user(self, slicer_home: str, slicer_version: str) -> None:
        """Note that this Slicer uses the folder, so the Manager can say who does."""
        try:
            self.ensure_layout()
            path = os.path.join(self.root, "users.json")
            users = self.users()
            users[slicer_home] = {"version": slicer_version, "last_used": time.strftime("%Y-%m-%d")}
            temporary = f"{path}.{os.getpid()}.tmp"  # two Slicers starting at once
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(users, handle, indent=2, sort_keys=True)
            os.replace(temporary, path)
        except OSError as error:
            logger.warning("Could not record this Slicer in %s: %s", self.root, error)

    def users(self) -> dict:
        try:
            with open(os.path.join(self.root, "users.json"), encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    # ---- what is installed -------------------------------------------------

    def tool_dir(self, name: str, version_dir: Optional[str] = None) -> Optional[str]:
        """`tools/<name>` or `tools/<group>/<name>`, the way run_tool.py finds it."""
        tools = os.path.join(version_dir or self.version_dir, "tools")
        direct = os.path.join(tools, name)
        if os.path.isfile(os.path.join(direct, "pyproject.toml")):
            return direct
        if os.path.isdir(tools):
            for group in sorted(os.listdir(tools)):
                nested = os.path.join(tools, group, name)
                if os.path.isfile(os.path.join(nested, "pyproject.toml")):
                    return nested
        return None

    @staticmethod
    def _record(tool_dir: str) -> Optional[dict]:
        try:
            with open(os.path.join(tool_dir, INSTALL_RECORD), encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def is_installed(self, name: str) -> bool:
        folder = self.tool_dir(name)
        return bool(folder and self._record(folder))

    def installed(self) -> list:
        """Every installed tool of every version: dicts for the Manager's table."""
        rows = []
        if not os.path.isdir(self.root):
            return rows
        for entry in sorted(os.listdir(self.root)):
            if not entry.startswith("tools-"):
                continue
            version = entry[len("tools-"):]
            tools = os.path.join(self.root, entry, "tools")
            if not os.path.isdir(tools):
                continue
            for root, dirs, files in os.walk(tools):
                if INSTALL_RECORD in files:
                    record = self._record(root) or {}
                    rows.append({
                        "name": os.path.basename(root),
                        "version": version,
                        "current": version == self.version,
                        "path": root,
                        "calls": record.get("calls", []),
                        "installed_at": record.get("installed_at", ""),
                        # Measured once, at the end of the install: the Manager
                        # reads it rather than walking 8 GB of environment.
                        "size": record.get("size"),
                    })
                    dirs[:] = []
                else:
                    dirs[:] = [d for d in dirs
                               if not d.startswith(".") and not d.endswith(PREVIOUS_SUFFIX)]
        return rows

    def recover(self) -> list:
        """Finish what a Slicer that died during an update left: a tool moved
        aside (`<tool>.previous`) whose successor never got its record is put
        back; one whose successor did is deleted. Returns the tools put back.

        Called at startup; skipped when another Slicer holds the folder.
        """
        restored = []
        if not os.path.isdir(self.root):
            return restored
        try:
            with self.lock():
                migrate_model_sizes(self.data_dir)
                for entry in os.listdir(self.root):
                    tools = os.path.join(self.root, entry, "tools")
                    if not entry.startswith("tools-") or not os.path.isdir(tools):
                        continue
                    for root, dirs, _files in os.walk(tools):
                        for name in [d for d in dirs if d.endswith(PREVIOUS_SUFFIX)]:
                            aside = os.path.join(root, name)
                            target = aside[: -len(PREVIOUS_SUFFIX)]
                            if os.path.isfile(os.path.join(target, INSTALL_RECORD)):
                                _rmtree(aside)
                            else:
                                if os.path.isdir(target):
                                    _rmtree(target)
                                os.replace(aside, target)
                                restored.append(os.path.basename(target))
                        dirs[:] = [d for d in dirs if not d.startswith(".")
                                   and not d.endswith(PREVIOUS_SUFFIX)]
        except StoreBusy:
            pass
        return restored

    def versions(self) -> list:
        """The tools versions present in this folder, oldest first."""
        if not os.path.isdir(self.root):
            return []
        return sorted((entry[len("tools-"):] for entry in os.listdir(self.root)
                       if entry.startswith("tools-") and os.path.isdir(os.path.join(self.root, entry))),
                      key=version_key)

    def installed_version(self, name: str) -> Optional[str]:
        """The version `name` runs at: this extension's if installed there, else the newest one."""
        if self.is_installed(name):
            return self.version
        for version in reversed(self.versions()):
            folder = self.tool_dir(name, os.path.join(self.root, f"tools-{version}"))
            if folder and self._record(folder):
                return version
        return None

    def update_available(self, name: str) -> bool:
        """Installed, but at an older version than this extension's, or changed since."""
        version = self.installed_version(name)
        if version is None:
            return False
        if version == self.version:
            return bool(self.source and self.outdated(name, self.source))
        # A NEWER version (another Slicer's) is not an update for this one.
        return version_key(version) < version_key(self.version)

    def is_older(self, version: str) -> bool:
        """Whether `version` is older than this extension's: what "old" means here."""
        return version_key(version) < version_key(self.version)

    def needed_by(self, name: str) -> list:
        """The installed tools of this version that call `name`, directly or not."""
        rows = [row for row in self.installed() if row["current"]]
        callers = {row["name"]: set(row["calls"]) for row in rows}
        needing = set()
        changed = True
        while changed:
            changed = False
            for tool, calls in callers.items():
                if tool in needing or tool == name:
                    continue
                if name in calls or calls & needing:
                    needing.add(tool)
                    changed = True
        return sorted(needing)

    # ---- removing ----------------------------------------------------------

    def _needs(self) -> dict:
        """`{tool: tools it brings along}` for the installed tools of this version:
        what it calls, and the other engines of its module (ALI_IOS with ALI_CBCT)."""
        from . import local_schema

        siblings = {}
        for facade in local_schema.FACADES:
            for engine in local_schema.facade_targets(facade):
                siblings[engine] = set(local_schema.facade_targets(facade)) - {engine}
        return {row["name"]: set(row["calls"]) | siblings.get(row["name"], set())
                for row in self.installed() if row["current"]}

    @staticmethod
    def _closure(start, needs: dict) -> set:
        found, pending = set(), list(start)
        while pending:
            tool = pending.pop()
            for needed in needs.get(tool, ()):
                if needed not in found:
                    found.add(needed)
                    pending.append(needed)
        return found

    def removable_with(self, name: str) -> list:
        """The tools `name` brought along that nothing else installed still needs."""
        needs = self._needs()
        candidates = self._closure([name], needs) - {name}
        remaining = set(needs) - candidates - {name}
        still_needed = remaining | self._closure(remaining, needs)
        return sorted(tool for tool in candidates if tool not in still_needed and tool in needs)

    def data_of(self, tools, keep=()) -> list:
        """The models and test files of `tools` that no tool in `keep` uses.

        `[{"path", "name", "kind", "used_by"}]`; `used_by` names the tools of
        `keep` that use an entry, which is then not offered for removal.
        """
        from . import local_schema

        rows, seen = [], set()
        for tool in tools:
            slug = local_schema.data_slug(tool, self.data_dir)
            for kind in ("models", "testfiles"):
                folder = os.path.join(self.data_dir, slug, kind)
                for entry in model_entries(tool, folder):
                    path = os.path.join(folder, entry)
                    if path in seen or not os.path.exists(path):
                        continue
                    seen.add(path)
                    users = sorted(other for other in keep
                                   if local_schema.data_slug(other, self.data_dir) == slug
                                   and entry in model_entries(other, folder))
                    rows.append({"path": path, "name": entry, "used_by": users,
                                 "kind": "model" if kind == "models" else "test files"})
        return rows

    def remove_data(self, path: str) -> None:
        """One model or test file (the Manager's Remove, and a tool's removal)."""
        self.remove_model(path)

    def prune_cache(self) -> int:
        """Delete the packages of the private cache no installed tool uses any more.

        uv unpacks each package once into `archive-v0/<id>/` and hard-links its
        files into every environment that needs it. A package whose files are
        linked nowhere else (one link: the cache's own) belongs to no tool any
        more; deleting it is what gives a removed tool's space back. Returns the
        bytes freed. uv downloads a pruned package again if a tool needs it.
        """
        archives = os.path.join(self.uv_cache, "archive-v0")
        if not os.path.isdir(archives):
            return 0
        # Under the lock: a package uv has just unpacked for an install still
        # running has one link too, until uv links it into the new environment.
        with self.lock():
            return self._prune(archives)

    def _prune(self, archives: str) -> int:
        freed = 0
        for entry in os.listdir(archives):
            folder = os.path.join(archives, entry)
            if not os.path.isdir(folder) or os.path.islink(folder):
                continue
            files, orphan = [], True
            for root, _dirs, names in os.walk(folder):
                for name in names:
                    try:
                        stat = os.lstat(os.path.join(root, name))
                    except OSError:
                        continue
                    if stat.st_nlink > 1:
                        orphan = False
                        break
                    files.append(stat.st_size)
                if not orphan:
                    break
            if orphan:
                shutil.rmtree(folder, ignore_errors=True)
                freed += sum(files)
        return freed

    def _cache_inodes(self) -> dict:
        """`{(dev, inode): links in the private cache}`: what a removal also frees there."""
        found = {}
        for root, _dirs, names in os.walk(self.uv_cache):
            for name in names:
                try:
                    stat = os.lstat(os.path.join(root, name))
                except OSError:
                    continue
                key = (stat.st_dev, stat.st_ino)
                found[key] = found.get(key, 0) + 1
        return found

    def reclaim_with_cache(self, paths) -> dict:
        """`reclaim`, counting the cache copy a prune deletes along with the tool."""
        return self.reclaim(paths, extra=self._cache_inodes() if self.private_cache else None)

    @staticmethod
    def reclaim(paths, extra=None) -> dict:
        """What deleting `paths` gives back, in bytes: `{"freed", "shared"}`.

        A file hard-linked from elsewhere (uv's package cache, the DATA folder it
        was installed from, another tool) stays on disk after the delete: it is
        counted as shared, not freed. Exact, file by file.
        """
        seen, freed, shared = set(), 0, 0
        links = {}
        for path in ([paths] if isinstance(paths, str) else paths):
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        stat = os.lstat(os.path.join(root, name))
                    except OSError:
                        continue
                    key = (stat.st_dev, stat.st_ino)
                    links[key] = links.get(key, 0) + 1
                    if key in seen:
                        continue
                    seen.add(key)
                    links[(key, "size")] = (stat.st_size, stat.st_nlink)
        for key in seen:
            size, nlink = links[(key, "size")]
            # A link the prune removes too (the private cache's) is no reason to stay.
            if links[key] + (extra or {}).get(key, 0) >= nlink:
                freed += size
            else:
                shared += size
        return {"freed": freed, "shared": shared}

    def remove_tool(self, path: str, prune: bool = True) -> None:
        """Delete one installed tool (its folder, venv included)."""
        path = os.path.abspath(path)
        if not path.startswith(self.root + os.sep) or not os.path.isfile(
                os.path.join(path, INSTALL_RECORD)):
            raise InstallError(f"{path} is not an installed tool of {self.root}.")
        with self.lock():
            # The record first: a removal cut short (a file Windows keeps open,
            # Slicer closed) then reads as "not installed", never as a tool
            # that is installed but has lost half of its environment.
            os.remove(os.path.join(path, INSTALL_RECORD))
            _rmtree(path)
            if prune:
                self.prune_cache()

    def remove_unused_versions(self) -> list:
        """Delete every `tools-<version>` OLDER than this one. Returns what was removed.

        Never a newer one: that belongs to a newer Slicer sharing the folder.
        """
        removed = []
        if not os.path.isdir(self.root):
            return removed
        with self.lock():
            for version in self.versions():
                if self.is_older(version):
                    _rmtree(os.path.join(self.root, f"tools-{version}"))
                    removed.append(f"tools-{version}")
            self.prune_cache()
        return removed

    def remove_everything(self) -> None:
        """Delete the whole folder: tools, environments and models."""
        if os.path.isdir(self.root):
            with self.lock():
                for entry in os.listdir(self.root):
                    if entry != StoreLock.NAME:
                        path = os.path.join(self.root, entry)
                        if os.path.isdir(path) and not os.path.islink(path):
                            _rmtree(path)
                        else:
                            os.remove(path)
            shutil.rmtree(self.root, ignore_errors=True)

    # ---- installing --------------------------------------------------------

    def plan(self, name, source: str) -> list:
        """`name` (or a list of names) and every tool they call, recursively, in
        install order: callees first."""
        from . import local_schema

        order, visiting = [], set()
        # A module is installed whole: an engine (ALI_CBCT) brings its module's
        # other engines (ALI_IOS) and theirs, so the module's panel (ALI) offers
        # every mode as soon as any part of it is here.
        siblings = {}
        for facade in local_schema.FACADES:
            for engine in local_schema.facade_targets(facade):
                siblings[engine] = local_schema.facade_targets(facade)

        def visit(tool):
            if tool in order or tool in visiting:
                return
            visiting.add(tool)
            folder = _source_tool_dir(source, tool)
            if folder is None:
                raise InstallError(f"There is no tool called {tool} in the tools source.")
            for callee in _calls(source, folder):
                visit(callee)
            visiting.discard(tool)
            order.append(tool)
            for sibling in siblings.get(tool, ()):
                visit(sibling)

        for wanted in ([name] if isinstance(name, str) else list(name)):
            for tool in local_schema.facade_targets(wanted) or [wanted]:
                visit(tool)
        return order

    def missing(self, name, source: str) -> list:
        """What installing `name` has to (re)install: absent, or changed since."""
        return [tool for tool in self.plan(name, source)
                if not self.is_installed(tool) or self.outdated(tool, source)]

    def models_missing(self, tool: str) -> bool:
        """Installed, but a model it needs is not in DATA (removed in the Manager)."""
        manifest = self._manifest()
        key = manifest_key(tool, manifest)
        if key is None:
            return False
        return any(not os.path.exists(os.path.join(self.data_dir, key, "models", _entry_target(entry)))
                   for entry in manifest.get(key, {}).get("models", []))

    def chain_models_missing(self, tool: str) -> list:
        """The tools of `tool`'s chain -- itself and every installed tool it
        calls, recursively -- that lack a model.

        Read from the install records, so it needs no source: a run of AREG
        calls AMASSS, and AMASSS's model removed in the Manager made AREG's
        panel look ready and its run fail half way.
        """
        lacking, seen, pending = [], set(), [tool]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if self.models_missing(name):
                lacking.append(name)
            folder = self.tool_dir(name)
            record = self._record(folder) if folder else None
            pending.extend((record or {}).get("calls", []))
        return lacking

    def incomplete(self, name, source: str) -> list:
        """What installing `name` has to do anything for: missing, or lacking a model."""
        missing = self.missing(name, source)
        return [tool for tool in self.plan(name, source)
                if tool in missing or self.models_missing(tool)]

    def outdated(self, name: str, source: str) -> bool:
        """Installed, but its code or its locked dependencies changed in the source.

        Compared by fingerprint, tool by tool: a release that did not touch a
        tool does not make anyone download its 8 GB again.
        """
        folder = self.tool_dir(name)
        record = self._record(folder) if folder else None
        origin = _source_tool_dir(source, name) if source else None
        if not record or not origin:
            return False
        if record.get("fingerprint") != fingerprint(origin):
            return True
        # A shared library it runs on (ALI/common) changed in the source.
        for dependency in _local_dependencies(origin, source):
            installed = os.path.join(self.version_dir, os.path.relpath(dependency, source))
            if fingerprint(installed) != fingerprint(dependency):
                return True
        return False

    def install(self, name, source: str, data_source: str,
                progress: Optional[Callable[[str], None]] = None,
                environment: Optional[dict] = None, testfiles: bool = False) -> list:
        """Install `name` (or a list of names) and what they call. Returns what was installed.

        One install at a time per tools folder (StoreLock). A tool being
        replaced (an update) is kept aside until its successor is complete, and
        put back if anything fails: an update cut short by the network leaves
        the tool that worked, not nothing.
        """
        say = progress or (lambda message: None)
        self.ensure_layout()
        with self.lock():
            return self._install_locked(name, source, data_source, say, environment, testfiles)

    def _install_locked(self, name, source, data_source, say, environment, testfiles) -> list:
        self._install_scripts(source)
        if self.missing(name, source):
            self.uv(say=say)
        installed = []
        estimate = self.estimate(name, source, data_source, testfiles=testfiles)
        track = _Progress(say, estimate)
        missing = self.missing(name, source)
        for tool in missing:
            track.start(tool)
            say(f"Installing {tool}...")
            origin = _source_tool_dir(source, tool)
            relative = os.path.relpath(origin, os.path.join(source, "tools"))
            target = os.path.join(self.tools_dir, relative)
            aside = target + PREVIOUS_SUFFIX
            if os.path.isdir(aside):
                _rmtree(aside)
            if os.path.isdir(target):
                if os.path.isfile(os.path.join(target, INSTALL_RECORD)):
                    os.replace(target, aside)
                else:
                    _rmtree(target)  # what an interrupted install left
            try:
                shutil.copytree(origin, target, ignore=shutil.ignore_patterns(
                    ".venv", "__pycache__", ".pytest_cache", "tests", "*.egg-info"))
                self._copy_local_sources(origin, target, source)
                say(f"Creating the environment of {tool} (uv sync)...")
                self._uv_sync(target, environment, track=track)
                track.done("environment")
                say(f"Fetching the models of {tool}...")
                python = _venv_python(target)
                self._install_data(tool, data_source, python=python, say=say, track=track)
                track.done("models")
                if testfiles:
                    say(f"Fetching the test files of {tool}...")
                    self._install_data(tool, data_source, kind="testfiles", python=python,
                                       say=say, track=track)
                say(f"Measuring {tool}...")
                record = {"version": self.version,
                          "calls": _calls(source, origin),
                          "installed_at": time.strftime("%Y-%m-%d %H:%M"),
                          "size": folder_size(target),
                          "fingerprint": fingerprint(origin)}
                with open(os.path.join(target, INSTALL_RECORD), "w", encoding="utf-8") as handle:
                    json.dump(record, handle, indent=2)
            except BaseException:
                if os.path.isdir(aside):
                    if os.path.isdir(target):
                        _rmtree(target)
                    os.replace(aside, target)
                raise
            if os.path.isdir(aside):
                _rmtree(aside)
            installed.append(tool)
        for tool in self.plan(name, source):
            if tool in installed:
                continue
            python = _venv_python(self.tool_dir(tool) or "")
            if self.models_missing(tool):
                # A model removed in the Manager comes back through Install.
                say(f"Fetching the models of {tool}...")
                self._install_data(tool, data_source, python=python, say=say)
            if testfiles:
                # Tools already installed get their test files too, when asked for.
                self._install_data(tool, data_source, kind="testfiles", python=python, say=say)
        if installed:
            self.prune_cache()  # what the tools just replaced no longer uses
        track.finish()
        say("Done.")
        return installed

    def estimate(self, name, source: str, data_source: str, testfiles: bool = False) -> dict:
        """What installing `name` would download, per missing tool, and in all.

        `{"tools": [{"name", "environment", "models"}], "environment", "models", "total"}`
        in bytes. Environments from each tool's lock, less the packages the
        private uv cache already holds (installed by hard link, so nearly free);
        models from the size of the files that would be fetched. A model folder
        already installed, or shared with a tool already counted, counts once.
        """
        from . import local_schema

        rows, counted = [], set()
        missing = self.missing(name, source)
        # A package already in the private cache, or brought by a tool counted
        # above, is hard-linked: torch is paid once, not once per tool.
        present = cached_packages(self.uv_cache) if self.private_cache else set()
        for tool in self.incomplete(name, source):
            origin = _source_tool_dir(source, tool)
            environment = 0
            if tool in missing:
                for package, size in _lock_packages(origin).items():
                    if package not in present:
                        present.add(package)
                        environment += size
            models = 0
            if not data_source:
                models = self._manifest_download(tool, "models", counted)
            elif data_source:
                slug = local_schema.data_slug(tool, data_source)
                origin_models = os.path.join(data_source, slug, "models")
                for entry in model_entries(tool, origin_models):
                    if (slug, entry) in counted or os.path.exists(
                            os.path.join(self.data_dir, slug, "models", entry)):
                        continue
                    counted.add((slug, entry))
                    models += _source_size(os.path.join(origin_models, entry))
            rows.append({"name": tool, "environment": environment, "models": models})
        environment = sum(row["environment"] for row in rows)
        models = sum(row["models"] for row in rows)
        tests = self.testfiles_size(name, source, data_source) if testfiles else 0
        return {"tools": rows, "environment": environment, "models": models,
                "testfiles": tests, "total": environment + models + tests}

    def testfiles_size(self, name, source: str, data_source: str) -> int:
        """What the test files of `name` and what it needs add, when not here yet."""
        from . import local_schema

        total, counted = 0, set()
        if not data_source:
            return sum(self._manifest_download(tool, "testfiles", counted)
                       for tool in self.plan(name, source))
        for tool in self.plan(name, source):
            slug = local_schema.data_slug(tool, data_source)
            origin = os.path.join(data_source, slug, "testfiles")
            for entry in model_entries(tool, origin):
                if (slug, entry) in counted or os.path.exists(
                        os.path.join(self.data_dir, slug, "testfiles", entry)):
                    continue
                counted.add((slug, entry))
                total += _source_size(os.path.join(origin, entry))
        return total

    def _install_scripts(self, source: str) -> None:
        os.makedirs(self.scripts_dir, exist_ok=True)
        # The supervisor, and SADT-VISOR's own downloader with the manifest it
        # reads: models and test files come from the same releases as VISOR's.
        for name in ("run_tool.py", "describe.py", "fetch_data.py", "data-manifest.yml"):
            shutil.copy2(os.path.join(source, "scripts", name), os.path.join(self.scripts_dir, name))

    def _copy_local_sources(self, origin: str, target: str, source: str) -> None:
        """Bring along the local path dependencies a tool's lock points at, recursively.

        `../common`, `../../naming`, the testkit: the lock names them by a path
        relative to the tool, so they are mirrored at the same place relative to
        the version folder, which mirrors the SADT-VISOR checkout.
        """
        pending, done = [origin], set()
        while pending:
            folder = pending.pop()
            if folder in done:
                continue
            done.add(folder)
            try:
                with open(os.path.join(folder, "pyproject.toml"), encoding="utf-8") as handle:
                    text = handle.read()
            except OSError:
                continue
            for line in text.splitlines():
                if "path = " not in line or '"' not in line.split("path = ", 1)[1]:
                    continue
                relative = line.split("path = ", 1)[1].split('"')[1]
                dependency = os.path.normpath(os.path.join(folder, relative))
                inside = os.path.relpath(dependency, source)
                if inside.startswith(".."):
                    continue
                destination = os.path.join(self.version_dir, inside)
                # Copied again every time: a shared library (ALI/common) changed
                # in the source, or a copy cut short, must not stay as it was.
                if os.path.isdir(destination):
                    _rmtree(destination)
                shutil.copytree(dependency, destination, ignore=shutil.ignore_patterns(
                    ".venv", "__pycache__", ".pytest_cache", "tests"))
                pending.append(dependency)

    @property
    def uv_path(self) -> str:
        return os.path.join(self.root, ".uv-bin", "uv.exe" if sys.platform == "win32" else "uv")

    def uv(self, say=None) -> str:
        """The uv of this tools folder, downloaded the first time it is needed.

        A machine without a uv release for its platform, or without internet
        on that first time, falls back on a uv found on the PATH.
        """
        if os.path.isfile(self.uv_path):
            return self.uv_path
        release = UV_RELEASES.get(_uv_platform())
        try:
            if release is None:
                raise InstallError(f"No uv release for {platform.system()} {platform.machine()}.")
            if say is not None:
                say("Downloading the installer of the tools (uv, 20 MB)...")
            _download_uv(release, self.uv_path)
            return self.uv_path
        except (InstallError, OSError) as error:
            found = shutil.which("uv")
            if found:
                return found
            raise InstallError(f"The installer of the tools (uv) could not be downloaded: {error}"
                               + _network_hint(str(error))) from error

    def _uv_sync(self, target: str, environment: Optional[dict], track=None) -> None:
        uv = self.uv()
        env = dict(environment or os.environ)
        for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(variable, None)
        if self.private_cache:
            # Inside this folder: what Remove deletes is really given back, and
            # Remove everything leaves nothing behind. Still one cache for every
            # tool, so torch is stored once however many tools use it.
            env["UV_CACHE_DIR"] = self.uv_cache
            # The Python the tools run on, too: nothing of theirs outside this folder.
            env["UV_PYTHON_INSTALL_DIR"] = os.path.join(self.root, ".uv-python")
            # Hard links from the cache on every platform (uv clones on macOS by
            # default): the link count is how prune_cache tells a package still
            # in use from one no tool needs any more.
            env["UV_LINK_MODE"] = "hardlink"
        import tempfile as _tempfile

        with _tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as errors:
            process = subprocess.Popen(
                [uv, *UV_SYNC],
                cwd=target, env=env, stdout=subprocess.DEVNULL, stderr=errors, text=True,
                encoding="utf-8", errors="replace", **no_window(),
            )
            # uv says nothing while it works. It downloads every package into its
            # cache first and fills the environment at the end, so what the
            # private cache has grown by is the sign of life, then the environment.
            venv = os.path.join(target, ".venv")
            cache = self.uv_cache if self.private_cache else ""
            baseline = folder_size(cache) if cache and os.path.isdir(cache) else 0
            while process.poll() is None:
                if track is not None:
                    grown = (folder_size(cache) - baseline) if cache and os.path.isdir(cache) else 0
                    filled = folder_size(venv) if os.path.isdir(venv) else 0
                    track.environment(max(grown, filled))
                time.sleep(1.5)
            errors.seek(0)
            message = errors.read()
        if process.returncode != 0:
            raise InstallError(
                f"The environment of {os.path.basename(target)} could not be created."
                + _network_hint(message) + "\n\n" + message.strip()[-800:])

    def _install_data(self, tool: str, data_source: str, kind: str = "models",
                      python: str = "", say=None, track=None) -> None:
        """The tool's models (or test files).

        Downloaded from SADT-VISOR's data releases by SADT-VISOR's own
        `fetch_data.py`, reading `data-manifest.yml`: the same files, URLs and
        sha256 checks as a VISOR server. A local DATA folder (`data_source`) is
        only a development mirror: its files are hard-linked instead.
        """
        if data_source:
            self._link_data(tool, data_source, kind)
            return
        key = manifest_key(tool, self._manifest())
        if key is None or not self._manifest().get(key, {}).get(kind):
            return
        fetch = os.path.join(self.scripts_dir, "fetch_data.py")
        command = [python or sys.executable, fetch, "--tool", key, "--kind", kind,
                   "--data-dir", self.data_dir,
                   "--manifest", os.path.join(self.scripts_dir, "data-manifest.yml"),
                   "--progress", "always"]
        env = dict(os.environ)
        for variable in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(variable, None)
        env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env,
                                   **no_window())
        tail = []
        for line in process.stdout:
            line = line.strip()
            if line:
                tail = (tail + [line])[-20:]
                if track is not None:
                    track.download_line(line)
                elif say:
                    say(f"{tool}: {line[:160]}")
        if process.wait() != 0:
            raise InstallError(f"The {kind} of {tool} could not be downloaded."
                               + _network_hint("\n".join(tail)) + "\n\n" + "\n".join(tail[-8:]))
        folder = os.path.join(self.data_dir, key, kind)
        if os.path.isdir(folder):
            for entry in os.listdir(folder):
                if not entry.startswith(".") and entry not in _model_sizes(folder):
                    _record_model_size(folder, entry)

    def _manifest(self) -> dict:
        """The data manifest of the tools source (or of the installed scripts)."""
        for folder in (os.path.join(self.source, "scripts") if self.source else "", self.scripts_dir):
            path = os.path.join(folder, "data-manifest.yml") if folder else ""
            if path and os.path.isfile(path):
                return _parse_manifest(folder, path)
        return {}

    def _manifest_download(self, tool: str, kind: str, counted: set) -> int:
        """Bytes the manifest would download for `tool`'s `kind`, not already here."""
        manifest = self._manifest()
        key = manifest_key(tool, manifest)
        if key is None:
            return 0
        total = 0
        for entry in manifest.get(key, {}).get(kind, []):
            where = _entry_target(entry)
            if (key, where) in counted or os.path.exists(os.path.join(self.data_dir, key, kind, where)):
                continue
            counted.add((key, where))
            total += int(entry.get("size") or 0)
        return total

    def _link_data(self, tool: str, data_source: str, kind: str) -> None:
        """Development mirror: the tool's files hard-linked from a local DATA folder."""
        # The server's own rule for which DATA folder serves which tool
        # (ALI_CBCT's models are under DATA/ALI): one copy of it, in local_schema.
        from . import local_schema

        slug = local_schema.data_slug(tool, data_source)
        origin = os.path.join(data_source, slug, kind)
        target = os.path.join(self.data_dir, slug, kind)
        for entry in model_entries(tool, origin):
            source_path, target_path = os.path.join(origin, entry), os.path.join(target, entry)
            if os.path.exists(target_path):
                continue
            os.makedirs(target, exist_ok=True)
            if os.path.isdir(source_path):
                shutil.copytree(source_path, target_path, copy_function=_link_or_copy)
            else:
                _link_or_copy(source_path, target_path)
            _record_model_size(target, entry)

    def models(self) -> list:
        """Every model and test file installed, for the Manager:
        `{"folder", "name", "kind", "path", "used_by", "size"}`."""
        rows = []
        if not os.path.isdir(self.data_dir):
            return rows
        from . import local_schema

        # Any version: an old tool still running until its update uses its models.
        installed = sorted({row["name"] for row in self.installed()})
        for slug, kind in sorted((slug, kind) for slug in os.listdir(self.data_dir)
                                 for kind in ("models", "testfiles")):
            models_dir = os.path.join(self.data_dir, slug, kind)
            if not os.path.isdir(models_dir):
                continue
            for entry in sorted(os.listdir(models_dir)):
                if entry.startswith("."):
                    continue
                users = [tool for tool in installed
                         if local_schema.data_slug(tool, self.data_dir) == slug
                         and entry in model_entries(tool, models_dir)]
                rows.append({"folder": slug, "name": entry,
                             "kind": "model" if kind == "models" else "test files",
                             "path": os.path.join(models_dir, entry), "used_by": users,
                             "size": _model_sizes(models_dir).get(entry)})
        return rows

    def remove_model(self, path: str) -> None:
        path = os.path.abspath(path)
        if not path.startswith(self.data_dir + os.sep):
            raise InstallError(f"{path} is not a model of {self.root}.")
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        sizes = _model_sizes(os.path.dirname(path))
        if sizes.pop(os.path.basename(path), None) is not None:
            _write_model_sizes(os.path.dirname(path), sizes)


# ---- how much an install downloads ---------------------------------------

_ENTRY = None  # compiled lazily


def _platform_tags():
    """What a wheel's file name must say to install on this machine."""
    system = platform.system()
    machine = _native_machine()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "aarch64"}.get(machine, machine)
    if system == "Windows":
        return ("win_amd64",) if arch == "x86_64" else ("win_arm64",)
    if system == "Darwin":
        return ("macosx", arch, "universal2")
    return ("linux", arch)


def _wheel_fits(filename: str, python_tag: str) -> bool:
    stem = filename[:-4]
    parts = stem.split("-")
    if len(parts) < 5:
        return False
    pythons, abi, plat = parts[-3], parts[-2], parts[-1]
    if not any(tag in pythons for tag in (python_tag, "py3", "py2.py3")) and abi != "abi3":
        return False
    if plat == "any":
        return True
    tags = _platform_tags()
    system = platform.system()
    if system == "Darwin":
        return "macosx" in plat and (tags[1] in plat or "universal2" in plat)
    if system == "Windows":
        return plat == tags[0]
    return "linux" in plat and tags[1] in plat


def lock_packages(lock_path: str, python_tag: str = "cp311") -> dict:
    """`{"name-version": bytes}`: what `uv sync` would download for this lock.

    Read off the lock itself, which records every wheel's size: for each
    package, the first wheel that fits this platform and Python, else its
    source archive. Local packages (the tool, `../common`) download nothing.
    Keys are normalised as in a `.dist-info` folder name, so they can be
    matched against what the uv cache already holds.
    """
    import re

    global _ENTRY
    if _ENTRY is None:
        _ENTRY = re.compile(r'url = "([^"]+)"[^}]*?size = (\d+)')
    try:
        with open(lock_path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return {}
    packages = {}
    for block in text.split("\n[[package]]\n")[1:]:
        head = block.split("\n[package.")[0]
        if any(marker in head for marker in
               ("editable = ", "virtual = ", "directory = ", "path = ")):
            continue
        name = re.search(r'^name = "([^"]+)"', head, re.M)
        version = re.search(r'^version = "([^"]+)"', head, re.M)
        sdist, chosen = 0, 0
        for url, size in _ENTRY.findall(block):
            filename = url.rsplit("/", 1)[-1].split("#")[0]
            if filename.endswith(".whl"):
                if not chosen and _wheel_fits(filename, python_tag):
                    chosen = int(size)
            elif not sdist:
                sdist = int(size)
        if name is None:
            continue
        key = _distribution_key(name.group(1), version.group(1) if version else "")
        packages[key] = chosen or sdist
    return packages


def _distribution_key(name: str, version: str) -> str:
    import re

    return f"{re.sub(r'[-_.]+', '_', name).lower()}-{version.lower()}"


def cached_packages(cache_dir: str) -> set:
    """`name-version` of every package already unpacked in a uv cache.

    uv installs those by hard link: a tool that needs one costs no download
    and next to no disk.
    """
    found = set()
    archives = os.path.join(cache_dir, "archive-v0")
    try:
        entries = os.listdir(archives)
    except OSError:
        return found
    for entry in entries:
        try:
            names = os.listdir(os.path.join(archives, entry))
        except OSError:
            continue
        for name in names:
            if name.endswith(".dist-info"):
                distribution, _, version = name[:-len(".dist-info")].partition("-")
                found.add(_distribution_key(distribution, version))
    return found


def _python_tag(tool_dir: str) -> str:
    """`cp311` from a tool's `requires-python = ">=3.11,<3.12"`."""
    import re

    try:
        with open(os.path.join(tool_dir, "pyproject.toml"), encoding="utf-8") as handle:
            match = re.search(r'requires-python\s*=\s*"[^"]*?(\d+)\.(\d+)', handle.read())
    except OSError:
        match = None
    return f"cp{match.group(1)}{match.group(2)}" if match else "cp311"


# The measured size of every model and test file, in ONE file at the root of
# DATA (beside fetch_data.py's own `.fetched.json`), keyed "<slug>/<kind>/<entry>".
# Never inside a tool's models folder: tools read those folders, and ASO took a
# `.sizes.json` there for a landmarks file -- the folder of reference bundles
# then looked like one bundle, and every run oriented onto Frankfurt.
MODEL_SIZES = ".sizes.json"


def _sizes_location(models_dir: str) -> tuple:
    """`(file, prefix)`: where the sizes of `models_dir` are kept, and their key prefix."""
    models_dir = os.path.abspath(models_dir)
    data_dir = os.path.dirname(os.path.dirname(models_dir))
    prefix = os.path.relpath(models_dir, data_dir).replace(os.sep, "/") + "/"
    return os.path.join(data_dir, MODEL_SIZES), prefix


def _all_sizes(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            sizes = json.load(handle)
        return sizes if isinstance(sizes, dict) else {}
    except (OSError, ValueError):
        return {}


def _model_sizes(models_dir: str) -> dict:
    path, prefix = _sizes_location(models_dir)
    sizes = {key[len(prefix):]: value for key, value in _all_sizes(path).items()
             if key.startswith(prefix)}
    # Written inside the folder by earlier pilots; read until migrated.
    legacy = _all_sizes(os.path.join(models_dir, MODEL_SIZES))
    return {**legacy, **sizes}


def _write_model_sizes(models_dir: str, sizes: dict) -> None:
    path, prefix = _sizes_location(models_dir)
    every = {key: value for key, value in _all_sizes(path).items() if not key.startswith(prefix)}
    every.update({prefix + entry: value for entry, value in sizes.items()})
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(every, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)
    legacy = os.path.join(models_dir, MODEL_SIZES)
    if os.path.isfile(legacy):
        os.remove(legacy)


def migrate_model_sizes(data_dir: str) -> int:
    """Move the sizes earlier pilots wrote inside the models folders to DATA's root."""
    moved = 0
    if not os.path.isdir(data_dir):
        return moved
    for slug in os.listdir(data_dir):
        for kind in ("models", "testfiles"):
            folder = os.path.join(data_dir, slug, kind)
            if os.path.isfile(os.path.join(folder, MODEL_SIZES)):
                _write_model_sizes(folder, _model_sizes(folder))
                moved += 1
    return moved


def _record_model_size(models_dir: str, entry: str) -> None:
    """Measure a model once, when it is installed, as a tool's size is."""
    path = os.path.join(models_dir, entry)
    sizes = _model_sizes(models_dir)
    sizes[entry] = folder_size(path) if os.path.isdir(path) else os.path.getsize(path)
    _write_model_sizes(models_dir, sizes)


class _Progress:
    """What an install has done, as a fraction of what it announced, for the dialog.

    Weighted by the estimate: a tool's environment and its models count for
    their download size. Events are dicts `{"message", "fraction", "detail"}`
    handed to the same callback as the plain messages.
    """

    # An environment takes about this many times its download once unpacked.
    UNPACKED = 2.6

    def __init__(self, say, estimate):
        self.say = say
        self.rows = {row["name"]: row for row in estimate["tools"]}
        self.total = max(1, sum(row["environment"] + row["models"] for row in estimate["tools"])
                         + estimate.get("testfiles", 0))
        self.done_bytes = 0
        self.tool = None
        self.step = 0
        self.file_done = 0
        self.file_current = 0

    def _emit(self, message, detail):
        self.say({"message": message, "detail": detail,
                  "fraction": min(0.99, (self.done_bytes + self.step) / self.total)})

    def start(self, tool):
        self.tool, self.step = tool, 0

    def environment(self, written):
        expected = self.rows.get(self.tool, {}).get("environment", 0)
        self.step = min(int(expected * 0.97), int(written / self.UNPACKED))
        self._emit(f"Creating the environment of {self.tool}",
                   f"{human_size(written)} written so far")

    def download_line(self, line):
        """fetch_data.py's output: `+ <file>` starts one, `a / b (p%)` advances it."""
        import re

        if line.startswith("+ "):
            self.file_done += self.file_current
            self.file_current = 0
            return
        match = re.match(r"([\d.]+ [KMGT]?B) / ([\d.]+ [KMGT]?B)", line)
        if not match:
            return
        current, total = _bytes(match.group(1)), _bytes(match.group(2))
        self.file_current = total if current >= total else current
        self.step = self.rows.get(self.tool, {}).get("environment", 0) + self.file_done + self.file_current
        self._emit(f"Downloading the models of {self.tool}",
                   f"{human_size(self.file_done + current)} downloaded")

    def done(self, what):
        row = self.rows.get(self.tool, {})
        if what == "environment":
            self.step = row.get("environment", 0)
        else:
            self.done_bytes += row.get("environment", 0) + row.get("models", 0)
            self.step, self.file_done, self.file_current = 0, 0, 0
        self._emit(f"{self.tool}: {what} ready", "")

    def finish(self):
        self.say({"message": "Done", "detail": "", "fraction": 1.0})


def _bytes(text: str) -> int:
    number, unit = text.split()
    factor = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}[unit]
    return int(float(number) * factor)


_MANIFESTS = {}


def _parse_manifest(scripts_dir: str, path: str) -> dict:
    """data-manifest.yml as fetch_data.py reads it, by fetch_data.py's own parser."""
    import importlib.util

    key = (path, _stamp(path))
    if key not in _MANIFESTS:
        spec = importlib.util.spec_from_file_location(
            "sadt_fetch_data", os.path.join(scripts_dir, "fetch_data.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MANIFESTS[key] = module._parse_manifest(path)
    return _MANIFESTS[key]


def manifest_key(tool: str, manifest: dict):
    """The manifest entry serving `tool`: its own, or the one that `provides` it (ALI)."""
    if tool in manifest:
        return tool
    for key, entry in manifest.items():
        if tool in (entry.get("provides") or []):
            return key
    return None


def _venv_python(tool_dir: str) -> str:
    for relative in (("bin", "python"), ("Scripts", "python.exe")):
        candidate = os.path.join(tool_dir, ".venv", *relative)
        if os.path.isfile(candidate):
            return candidate
    return ""


_FINGERPRINTS = {}


def _local_dependencies(tool_dir: str, source: str) -> list:
    """The folders of the local path dependencies of a tool (../common, ...), recursively."""
    found, pending = [], [tool_dir]
    while pending:
        folder = pending.pop()
        try:
            with open(os.path.join(folder, "pyproject.toml"), encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        for line in text.splitlines():
            if "path = " not in line or '"' not in line.split("path = ", 1)[1]:
                continue
            dependency = os.path.normpath(os.path.join(folder, line.split("path = ", 1)[1].split('"')[1]))
            if source and os.path.relpath(dependency, source).startswith(".."):
                continue
            if dependency not in found and dependency != tool_dir:
                found.append(dependency)
                pending.append(dependency)
    return sorted(found)


def fingerprint(tool_dir: str) -> str:
    """sha256 of what a tool IS: its code (src/), its pyproject and its uv.lock.

    Two installs with the same fingerprint run the same code with the same
    dependencies, whatever release they came from. The local libraries it
    depends on (ALI/common) are compared on their own (`outdated`).
    """
    import hashlib

    paths = [os.path.join(tool_dir, "pyproject.toml"), os.path.join(tool_dir, "uv.lock")]
    for root, dirs, files in os.walk(os.path.join(tool_dir, "src")):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        paths += [os.path.join(root, f) for f in sorted(files) if not f.endswith((".pyc", ".pyo"))]
    key = (tool_dir, tuple(_stamp(path) for path in paths))
    if key not in _FINGERPRINTS:
        digest = hashlib.sha256()
        for path in paths:
            digest.update(os.path.relpath(path, tool_dir).replace(os.sep, "/").encode() + b"\0")
            try:
                with open(path, "rb") as handle:
                    digest.update(handle.read())
            except OSError:
                pass
        _FINGERPRINTS[key] = digest.hexdigest()
    return _FINGERPRINTS[key]


def model_entries(tool: str, models_dir: str) -> list:
    """The entries of a DATA models folder that belong to `tool`.

    Several tools share one DATA folder (ALI_CBCT and ALI_IOS keep theirs under
    DATA/ALI): an entry named after the tool is its own, and when none is, the
    whole folder is the tool's (ASO, AMASSS). An entry named after none of the
    tools sharing the folder (Crown_Seg_Model, which ALI's manifest stages for
    on-the-fly segmentation) belongs to all of them.
    """
    if not os.path.isdir(models_dir):
        return []
    entries = sorted(name for name in os.listdir(models_dir) if not name.startswith("."))

    def norm(text):
        return text.lower().replace("_", "")

    key = norm(tool)
    own = [name for name in entries if norm(name).startswith(key)]
    if not own:
        return entries
    from . import local_schema

    slug_dir = os.path.dirname(os.path.abspath(models_dir))
    data_dir, slug = os.path.dirname(slug_dir), os.path.basename(slug_dir)
    keys = [norm(other) for other in local_schema.DEPLOYMENT
            if other not in local_schema.FACADES
            and local_schema.data_slug(other, data_dir) == slug] + [key]
    unclaimed = [name for name in entries
                 if not any(norm(name).startswith(other) for other in keys)]
    return sorted(own + unclaimed)


_NETWORK_WORDS = ("connect", "tunnel", "timed out", "timeout", "resolve", "dns", "network",
                  "unreachable", "urlopen", "temporary failure", "name or service")


def _entry_target(entry: dict) -> str:
    """Where a manifest entry lands under DATA/<tool>/<kind>/: fetch_data.py's
    `_target_path` rule (`dest`, else the name, without `.zip` once extracted)."""
    destination = entry.get("dest")
    if destination:
        return os.path.join(*destination.split("/"))
    name = entry["name"]
    if entry.get("extract") and name.lower().endswith(".zip"):
        name = name[:-4]
    return name


def _native_machine() -> str:
    """The processor's own architecture, not the one this Python was built for.

    Slicer for macOS is an Intel build: on Apple Silicon it runs under Rosetta
    and `platform.machine()` says x86_64. The tools must not follow it: Intel
    builds of torch stopped at 2.2 and know nothing of the Apple GPU (MPS).
    """
    machine = platform.machine().lower()
    if platform.system() == "Darwin" and machine == "x86_64":
        try:
            arm = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            arm = ""
        if arm == "1":
            return "arm64"
    return machine


def _uv_platform() -> tuple:
    machine = _native_machine()
    machine = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "aarch64"}.get(
        machine, machine)
    system = platform.system()
    if system == "Linux" and machine == "arm64":
        machine = "aarch64"
    if system in ("Darwin", "Windows") and machine == "aarch64":
        machine = "arm64"
    return system, machine


def _download_uv(release: tuple, destination: str, url: str = "") -> None:
    """Fetch one uv release archive, check its sha256, and put its `uv` at `destination`."""
    import hashlib
    import tarfile
    import tempfile
    import zipfile

    asset, expected = release
    url = url or UV_URL.format(version=UV_VERSION, asset=asset)
    folder = os.path.dirname(destination)
    os.makedirs(folder, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        archive = os.path.join(scratch, asset)
        with _open_url(url) as response, open(archive, "wb") as handle:
            shutil.copyfileobj(response, handle)
        digest = hashlib.sha256()
        with open(archive, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise InstallError(f"The uv download is damaged (sha256 {digest.hexdigest()}).")
        wanted = os.path.basename(destination)
        found = None
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.namelist():
                    if member.rsplit("/", 1)[-1] == wanted:
                        found = os.path.join(scratch, "uv-extracted")
                        with bundle.open(member) as source, open(found, "wb") as target:
                            shutil.copyfileobj(source, target)
                        break
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                for member in bundle.getmembers():
                    if member.isfile() and member.name.rsplit("/", 1)[-1] == wanted:
                        found = os.path.join(scratch, "uv-extracted")
                        with bundle.extractfile(member) as source, open(found, "wb") as target:
                            shutil.copyfileobj(source, target)
                        break
        if found is None:
            raise InstallError(f"No {wanted} in the uv download.")
        os.chmod(found, 0o755)
        os.replace(found, destination)  # whole or not at all


def _open_url(url: str):
    """urlopen, with certifi's certificates when the Python's own store is empty."""
    import ssl
    import urllib.error
    import urllib.request

    try:
        return urllib.request.urlopen(url, timeout=60)
    except urllib.error.URLError as error:
        if not isinstance(getattr(error, "reason", None), ssl.SSLError):
            raise
        try:
            import certifi
        except ImportError:
            raise error
        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, timeout=60, context=context)


def _network_hint(output: str) -> str:
    """A sentence a clinician can act on, when the output reads like no network."""
    lowered = (output or "").lower()
    if any(word in lowered for word in _NETWORK_WORDS):
        return (" This computer could not reach the internet (GitHub and the Python package"
                " index): check the connection, or the proxy, and try again.")
    return ""


def _rmtree(path: str) -> None:
    """shutil.rmtree that also deletes read-only files (Windows marks some so)."""
    import stat

    def retry(function, failed, _info):
        try:
            os.chmod(failed, stat.S_IWRITE)
            function(failed)
        except OSError:
            raise

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=lambda f, p, e: retry(f, p, e))
    else:
        shutil.rmtree(path, onerror=retry)


def _link_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _src_of(tool_dir: str) -> "os.PathLike":
    from pathlib import Path

    return Path(tool_dir) / "src"


def _source_tool_dir(source: str, name: str) -> Optional[str]:
    tools = os.path.join(source, "tools")
    direct = os.path.join(tools, name)
    if os.path.isfile(os.path.join(direct, "pyproject.toml")):
        return direct
    if os.path.isdir(tools):
        for group in sorted(os.listdir(tools)):
            nested = os.path.join(tools, group, name)
            if os.path.isfile(os.path.join(nested, "pyproject.toml")):
                return nested
    return None


_DESCRIBE_MODULES = {}

# What reading a source costs once is not paid again: the install dialog asks
# for every tool's plan and download on each tick. Keyed by path and mtime, so
# an edited source is read again.
_CALLS = {}
_LOCK_SIZES = {}
_SOURCE_SIZES = {}


def _stamp(path: str):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _calls(source: str, tool_dir: str) -> list:
    """The tools `tool_dir` calls through its supervisor (describe.py), read once."""
    src = os.path.join(tool_dir, "src")
    key = (src, _stamp(src))
    if key not in _CALLS:
        _CALLS[key] = list(_describe_module(source).supervised_calls(_src_of(tool_dir)))
    return _CALLS[key]


def _lock_packages(tool_dir: str) -> dict:
    lock = os.path.join(tool_dir, "uv.lock")
    key = (lock, _stamp(lock), "packages")
    if key not in _LOCK_SIZES:
        _LOCK_SIZES[key] = lock_packages(lock, _python_tag(tool_dir))
    return _LOCK_SIZES[key]


def _source_size(path: str) -> int:
    key = (path, _stamp(path))
    if key not in _SOURCE_SIZES:
        _SOURCE_SIZES[key] = folder_size(path) if os.path.isdir(path) else os.path.getsize(path)
    return _SOURCE_SIZES[key]


def _describe_module(source: str):
    """SADT-VISOR's describe.py from the source, for its `supervised_calls`. Loaded once."""
    import importlib.util

    path = os.path.join(source, "scripts", "describe.py")
    if path in _DESCRIBE_MODULES:
        return _DESCRIBE_MODULES[path]
    if not os.path.isfile(path):
        raise InstallError(f"No describe.py in {source}/scripts: is it an SADT-VISOR checkout?")
    spec = importlib.util.spec_from_file_location("sadt_describe_source", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sadt_describe_source", module)
    spec.loader.exec_module(module)
    _DESCRIBE_MODULES[path] = module
    return module
