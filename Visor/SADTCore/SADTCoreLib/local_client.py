"""The client the panels use when SADT-VISOR's tools run on THIS machine.

Same interface as `ToolServerClient` -- the members `base_widget.py` calls --
so every panel built on `ServerToolWidgetBase` works unchanged. Where the
remote client speaks HTTP to a VISOR server, this one:

- reads a tool's schema from SADT-VISOR's `scripts/describe.py`, run with the
  tool's own interpreter, and translates it into the schema the server would
  publish (`local_schema`);
- runs the tool through SADT-VISOR's own supervisor, `scripts/run_tool.py`, in
  the tool's own venv, so chaining, progress, errors and cancellation behave
  exactly as from a terminal;
- answers the "hosted data" questions from a local DATA folder laid out as the
  server's: `DATA/<tool>/models/<name>` and `DATA/<tool>/testfiles/<name>`.

Where things are, until the extension downloads them itself:

    SADT_VISOR_SCRIPTS  the folder holding run_tool.py and describe.py
    SADT_TOOLS_DIR      the folder of tools, each with its built .venv
    SADT_DATA_DIR       the DATA root

Like client.py, this module imports neither slicer nor qt: the environment a
tool starts from is handed in by whoever builds the client.
"""

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from typing import Callable, Optional

from . import local_schema

# The tools running in THIS Slicer, by engine name: what the Manager must not
# remove or replace under a run. Module-level, because every panel has a client.
_RUNNING = {}
_RUNNING_LOCK = threading.Lock()


# Sizes of test-file and model folders, measured once per process: they are
# immutable once installed, and AREG's take 0.7 s to walk on every panel visit.
_FOLDER_BYTES = {}


def _folder_bytes(path: str) -> int:
    try:
        key = (path, os.stat(path).st_mtime_ns)
    except OSError:
        return 0
    if key not in _FOLDER_BYTES:
        size = 0
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    size += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        _FOLDER_BYTES[key] = size
    return _FOLDER_BYTES[key]


def running_tools() -> list:
    with _RUNNING_LOCK:
        return sorted(_RUNNING)
from .tools_store import no_window
from .client import ToolResult
from .errors import RunCancelled, ServerToolError

logger = logging.getLogger(__name__)

SCRIPTS_VARIABLE = "SADT_VISOR_SCRIPTS"
TOOLS_VARIABLE = "SADT_TOOLS_DIR"
DATA_VARIABLE = "SADT_DATA_DIR"

# How often a running tool's progress file and the cancel flag are looked at.
POLL_SECONDS = 0.25

# What a describe.py run may take: it imports the tool's package, which must
# not import anything heavy at module level (SADT-VISOR's own rule).
DESCRIBE_TIMEOUT_SECONDS = 120


def _normalised(name: str) -> str:
    """The name lookup the remote client uses: case and separators ignored."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _load_supervisor(scripts_dir: str):
    """SADT-VISOR's `run_tool` module of one tools version, under a private name.

    One module per scripts folder: two tools versions installed side by side
    each run under their own supervisor, with their own TOOLS_DIR.
    """
    key = "sadt_run_tool_" + "".join(ch if ch.isalnum() else "_" for ch in os.path.abspath(scripts_dir))
    loaded = sys.modules.get(key)
    if loaded is not None:
        return loaded
    path = os.path.join(scripts_dir, "run_tool.py")
    if not os.path.isfile(path):
        raise ServerToolError(f"There is no run_tool.py in {scripts_dir}.")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return module


class LocalToolClient:
    """Runs SADT-VISOR tools in their own venvs, behind the remote client's interface."""

    is_local = True

    def __init__(self, scripts_dir=None, tools_dir=None, data_dir=None,
                 environment: Optional[Callable[[], dict]] = None, store=None):
        # The installed tools (tools_store): where they are unless a developer
        # points the three variables at a checkout instead.
        self.store = store
        self._scripts_dir = (scripts_dir or os.environ.get(SCRIPTS_VARIABLE, "")
                             or (store.scripts_dir if store else ""))
        self._tools_dir = (tools_dir or os.environ.get(TOOLS_VARIABLE, "")
                           or (store.tools_dir if store else ""))
        self._data_dir = (data_dir or os.environ.get(DATA_VARIABLE, "")
                          or (store.data_dir if store else ""))
        # Folders named explicitly (a developer's checkout): no versions to choose from.
        self._fixed = bool(scripts_dir or tools_dir or os.environ.get(SCRIPTS_VARIABLE)
                           or os.environ.get(TOOLS_VARIABLE))
        # The environment a tool starts from. Inside Slicer: the one Slicer was
        # launched with, from before it pointed Python at its own library.
        self._environment = environment or (lambda: dict(os.environ))
        self._tools_cache = None
        self._raw_cache = {}
        self._jobs = {}
        self._jobs_lock = threading.Lock()
        # Read by the settings panel, which this extension does not ship.
        self.server_url = "local"
        self.token = ""
        self.timeout = None

    # ---- configuration -----------------------------------------------------

    def configure(self, server_url=None, token=None, timeout=None):
        """Called at startup with the remote client's saved settings: nothing to use here."""
        self._tools_cache = None
        self._raw_cache = {}

    def _require(self, path: str, variable: str, what: str) -> str:
        if not path or not os.path.isdir(path):
            if self.store is not None and not os.environ.get(variable):
                # The clinician's case: nothing installed yet, not a setup error.
                raise ServerToolError("This tool is not installed on this computer yet.")
            raise ServerToolError(
                f"{what} not found. Set {variable} to its folder before starting Slicer.")
        return path

    def _dirs(self, tool=None) -> tuple:
        """`(scripts, tools)` of the tools version `tool` runs at.

        This extension's version when the tool is installed there; otherwise
        the newest version it is installed at, so a tool keeps running until
        its update is installed.
        """
        if self.store is None or self._fixed:
            return self._scripts_dir, self._tools_dir
        version = None
        if tool:
            engines = local_schema.facade_targets(tool) or [tool]
            version = self.store.installed_version(engines[0])
        folder = os.path.join(self.store.root, f"tools-{version or self.store.version}")
        return os.path.join(folder, "scripts"), os.path.join(folder, "tools")

    def tool_version(self, tool_name: str):
        """The tools version `tool_name` runs at here, or None when not installed."""
        if self.store is None or self._fixed:
            return None
        engines = local_schema.facade_targets(tool_name) or [tool_name]
        versions = {self.store.installed_version(engine) for engine in engines}
        if None in versions:
            return None
        from .tools_store import version_key

        return max(versions, key=version_key)

    def update_available(self, tool_name: str) -> bool:
        """Installed, but at an older tools version than this extension's."""
        if self.store is None or self._fixed:
            return False
        engines = local_schema.facade_targets(tool_name) or [tool_name]
        return any(self.store.update_available(engine) for engine in engines)

    def _supervisor(self, tool=None):
        scripts, tools = self._dirs(tool)
        module = _load_supervisor(
            self._require(scripts, SCRIPTS_VARIABLE, "The SADT-VISOR supervisor"))
        module.TOOLS_DIR = module.Path(
            self._require(tools, TOOLS_VARIABLE, "The SADT-VISOR tools folder"))
        return module

    def has_tool(self, tool_name: str) -> bool:
        """Whether `tool_name` (or, for a facade, one of its engines) is installed here."""
        try:
            folders = self._tool_folders(tool_name)
        except ServerToolError:
            return False
        # A facade is installed only when ALL its engines are: otherwise its panel
        # would offer only some modes (ALI without IOS) and say nothing.
        targets = local_schema.facade_targets(tool_name) or [tool_name]
        if not all(self._resolve_name_quiet(target, folders) for target in targets):
            return False
        # A model removed in the Manager: the panel offers Install, which
        # fetches it again, rather than a form whose run would fail.
        # The tools they call count too: their runs fail the same way.
        if self.store is not None and not self._fixed:
            return not any(self.store.chain_models_missing(target) for target in targets)
        return True

    def _resolve_name_quiet(self, name, known):
        try:
            return self._resolve_name(name, known)
        except ServerToolError:
            return None

    def health(self) -> bool:
        return all(os.path.isdir(path) for path in self._dirs())

    def batch_policy(self):
        # One cohort is one run here: nothing is split, so nothing to schedule.
        return None

    # ---- tools and schemas -------------------------------------------------

    def _tool_folders(self, tool=None) -> dict:
        """`{name: folder}` for every built tool: `tools/<name>` or `tools/<group>/<name>`.

        A tool is a folder whose pyproject.toml declares `[tool.sadt] tool = true`
        and that has a venv; `_template` and shared libraries are not tools.
        """
        root = self._require(self._dirs(tool)[1], TOOLS_VARIABLE, "The SADT-VISOR tools folder")
        found = {}
        candidates = []
        for entry in sorted(os.listdir(root)):
            folder = os.path.join(root, entry)
            if not os.path.isdir(folder) or entry.startswith((".", "_")):
                continue
            candidates.append(folder)
            for sub in sorted(os.listdir(folder)):
                if os.path.isdir(os.path.join(folder, sub)) and not sub.startswith((".", "_")):
                    candidates.append(os.path.join(folder, sub))
        for folder in candidates:
            pyproject = os.path.join(folder, "pyproject.toml")
            if not os.path.isfile(pyproject) or self._python_of(folder) is None:
                continue
            # In the tools folder, a tool is installed once its install record
            # is written, the very last step: one whose models never arrived,
            # or whose removal was cut short, is not a tool to run.
            if (self.store is not None and not self._fixed
                    and not os.path.isfile(os.path.join(folder, ".sadt-install.json"))):
                continue
            with open(pyproject, encoding="utf-8") as handle:
                text = handle.read()
            if "[tool.sadt]" not in text or "tool = true" not in text:
                continue
            found[os.path.basename(folder)] = folder
        return found

    @staticmethod
    def _python_of(folder: str):
        for relative in (("bin", "python"), ("Scripts", "python.exe")):
            candidate = os.path.join(folder, ".venv", *relative)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _clean_environment(self) -> dict:
        environment = dict(self._environment())
        for variable in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
            environment.pop(variable, None)
        return environment

    def _describe(self, name: str, folder: str) -> dict:
        """describe.py's raw schema for one tool, run with the tool's own interpreter."""
        scripts = self._require(self._dirs(name)[0], SCRIPTS_VARIABLE, "The SADT-VISOR supervisor")
        try:
            completed = subprocess.run(
                [self._python_of(folder), os.path.join(scripts, "describe.py"), folder],
                capture_output=True, text=True, timeout=DESCRIBE_TIMEOUT_SECONDS,
                env=self._clean_environment(), cwd=folder, encoding="utf-8", errors="replace",
                **no_window(),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ServerToolError(f"Could not read the description of {name}: {error}")
        if completed.returncode != 0:
            raise ServerToolError(
                f"Could not read the description of {name}: {completed.stderr.strip()[-500:]}")
        return json.loads(completed.stdout)

    def _raw(self, name: str, folders: dict, force_refresh: bool) -> dict:
        # Keyed by folder: the same tool at two versions is two schemas.
        key = folders[name]
        if force_refresh or key not in self._raw_cache:
            self._raw_cache[key] = self._describe(name, folders[name])
        return self._raw_cache[key]

    def _schema_for(self, name: str, folders: dict, force_refresh: bool) -> dict:
        # The panel only catches ServerToolError: a schema that cannot be
        # translated must reach it as one, with its reason.
        try:
            return self._translated(name, folders, force_refresh)
        except local_schema.LocalSchemaError as error:
            raise ServerToolError(f"{name}: {error}")

    def _translated(self, name: str, folders: dict, force_refresh: bool) -> dict:
        targets = local_schema.facade_targets(name)
        if targets:
            absent = [target for target in targets if target not in folders]
            if absent:
                raise ServerToolError(
                    f"{name} is not fully installed on this computer: {', '.join(absent)} "
                    + ("are" if len(absent) > 1 else "is") + " missing.")
            raws = {target: self._raw(target, folders, force_refresh) for target in targets}
            return local_schema.facade_schema(name, raws)
        return local_schema.panel_schema(name, self._raw(name, folders, force_refresh))

    def list_tools(self, force_refresh: bool = False) -> dict:
        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache
        folders = self._tool_folders()
        names = list(folders) + [facade for facade in local_schema.FACADES
                                 if any(t in folders for t in local_schema.facade_targets(facade))]
        tools = {}
        for name in names:
            try:
                tools[name] = self._schema_for(name, folders, force_refresh)
            except ServerToolError as error:
                logger.warning("%s", error)
        self._tools_cache = tools
        return tools

    def _resolve_name(self, tool_name: str, known) -> str:
        if tool_name in known:
            return tool_name
        wanted = _normalised(tool_name)
        for name in known:
            if _normalised(name) == wanted:
                return name
        raise ServerToolError(f"{tool_name} is not installed on this computer yet.")

    def get_tool_schema(self, tool_name: str, force_refresh: bool = False) -> dict:
        """One tool's schema, described on demand rather than every tool at once."""
        folders = self._tool_folders(tool_name)
        known = list(folders) + list(local_schema.FACADES)
        name = self._resolve_name(tool_name, known)
        return self._schema_for(name, folders, force_refresh)

    # ---- hosted data: the local DATA folder --------------------------------

    def _data_root_of(self, tool_name: str) -> str:
        data = self._require(self._data_dir, DATA_VARIABLE, "The SADT-VISOR data folder (DATA)")
        return os.path.join(data, local_schema.data_slug(tool_name, data))

    @staticmethod
    def _entries(folder: str) -> list:
        entries = []
        if not os.path.isdir(folder):
            return entries
        for name in sorted(os.listdir(folder)):
            if name.startswith("."):
                continue
            path = os.path.join(folder, name)
            if os.path.isdir(path):
                entries.append({"name": name, "kind": "folder", "size": _folder_bytes(path)})
            else:
                entries.append({"name": name, "kind": "file", "size": os.path.getsize(path)})
        return entries

    def _listing(self, base: str) -> dict:
        models = self._entries(os.path.join(base, "models"))
        testfiles = self._entries(os.path.join(base, "testfiles"))
        return {
            "models": [entry["name"] for entry in models],
            "testfiles": [entry["name"] for entry in testfiles],
            "entries": {"models": models, "testfiles": testfiles},
        }

    def list_tool_data(self, tool_name: str) -> dict:
        root = self._data_root_of(tool_name)
        data = self._listing(root)
        scopes = set()
        try:
            schema = self.get_tool_schema(tool_name)
            for argument in (schema.get("arguments") or {}).values():
                scopes.add(argument.get("selectable_scope"))
                for per_mode in (argument.get("selectable_scope_when") or {}).values():
                    scopes.update(per_mode.values())
            scopes.discard(None)
        except ServerToolError:
            pass
        data["scoped"] = {}
        for scope in sorted(scopes):
            scoped = {"models": [], "testfiles": [], "entries": {"models": [], "testfiles": []}}
            for kind in ("models", "testfiles"):
                entries = self._entries(os.path.join(root, kind, scope))
                scoped[kind] = [entry["name"] for entry in entries]
                scoped["entries"][kind] = entries
            data["scoped"][scope] = scoped
        return data

    def local_testfile_path(self, tool_name, filename, scope=""):
        """Where an installed test file already is, to be used in place; None if absent."""
        try:
            path = os.path.join(self._data_root_of(tool_name), "testfiles", scope or "", filename)
        except ServerToolError:
            return None
        return path if os.path.exists(path) else None

    def download_testfile(self, tool_name, filename, destination, progress_cb=None, scope=""):
        """Copy a test file from DATA; a folder is handed over as the .zip the panel unpacks."""
        source = os.path.join(self._data_root_of(tool_name), "testfiles", scope or "", filename)
        if not os.path.exists(source):
            raise ServerToolError(f"There is no test file {filename!r} for {tool_name}.")
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        if progress_cb:
            progress_cb(f"Copying the test file {filename}...")
        if os.path.isdir(source):
            with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as archive:
                for root, _, files in os.walk(source):
                    for name in files:
                        path = os.path.join(root, name)
                        archive.write(path, os.path.relpath(path, source))
        else:
            shutil.copyfile(source, destination)
        return destination

    # ---- running -----------------------------------------------------------

    def run(self, tool_name, args=None, files=None, output_dir=None, progress_cb=None,
            run_id=None, event_cb=None, cancel_event=None, batch=None, **_):
        """Run the tool and wait for it, on the caller's (worker) thread."""
        run_tool = self._supervisor(tool_name)
        schema = self.get_tool_schema(tool_name)
        name = self._resolve_name(tool_name, list(self._tool_folders(tool_name)) + list(local_schema.FACADES))
        values = dict(args or {})
        values.update(files or {})
        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix=f"sadt_{name}_")
        try:
            os.makedirs(output_dir, exist_ok=True)
            probe = tempfile.NamedTemporaryFile(dir=output_dir, prefix=".sadt-write-test-")
            probe.close()
        except OSError as error:
            raise ServerToolError(
                f"The results cannot be written to {output_dir}: "
                f"{error.strerror or error}. Choose another output folder.")
        data_dir = self._require(self._data_dir, DATA_VARIABLE, "The SADT-VISOR data folder (DATA)")

        try:
            target, values = local_schema.resolve_facade(name, values)
            params = local_schema.wire_arguments(schema, values, data_dir, target)
        except local_schema.LocalSchemaError as error:
            raise ServerToolError(str(error))
        params["output_dir"] = output_dir

        before = self._snapshot(output_dir)
        job_dir = tempfile.mkdtemp(prefix=f"sadt_job_{target}_")
        if progress_cb:
            progress_cb(f"Starting {target} on this machine...")
        job = run_tool.start(target, params, job_dir, env=self._environment(), data_dir=data_dir)
        with _RUNNING_LOCK:
            _RUNNING[target] = _RUNNING.get(target, 0) + 1
        if run_id:
            with self._jobs_lock:
                self._jobs[run_id] = job
        logger.info("%s started, log in %s", target, job.log_file)
        sequence = 0
        try:
            while True:
                for event in job.new_progress():
                    sequence += 1
                    if event_cb:
                        event_cb({"seq": sequence, "state": "running", "phase": "running",
                                  "fraction": event.get("fraction"),
                                  "message": event.get("message", ""), "depth": 0})
                if job.poll() is not None:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    job.cancel()
                    raise RunCancelled("The run was cancelled.", 499)
                time.sleep(POLL_SECONDS)
            for event in job.new_progress():
                sequence += 1
                if event_cb:
                    event_cb({"seq": sequence, "state": "running", "phase": "running",
                              "fraction": event.get("fraction"),
                              "message": event.get("message", ""), "depth": 0})
            if cancel_event is not None and cancel_event.is_set():
                raise RunCancelled("The run was cancelled.", 499)
            try:
                job.result()
            except run_tool.ToolFailed as error:
                raise ServerToolError(f"{type(error).__name__}: {error}")
        finally:
            with _RUNNING_LOCK:
                _RUNNING[target] -= 1
                if not _RUNNING[target]:
                    del _RUNNING[target]
            if run_id:
                with self._jobs_lock:
                    self._jobs.pop(run_id, None)

        produced = self._produced(output_dir, before)
        # `path` is never opened: its folder is where the panel looks for the
        # results, and not ending in .zip tells it there is nothing to unpack.
        return ToolResult(kind="file", path=os.path.join(output_dir, f"{name}_result"),
                          extracted=produced)

    @staticmethod
    def _snapshot(folder: str) -> dict:
        seen = {}
        for root, _, files in os.walk(folder):
            for name in files:
                path = os.path.join(root, name)
                try:
                    seen[path] = os.path.getmtime(path)
                except OSError:
                    pass
        return seen

    def _produced(self, folder: str, before: dict) -> list:
        """The files this run wrote or rewrote: earlier runs may share the folder."""
        after = self._snapshot(folder)
        return sorted(path for path, mtime in after.items() if before.get(path) != mtime)

    def cancel_run(self, run_id: str) -> bool:
        with self._jobs_lock:
            job = self._jobs.get(run_id)
        if job is None:
            return False
        try:
            job.cancel()
        except Exception:  # noqa: BLE001 -- a cancel must never raise
            return False
        return True

    def resume_run(self, *args, **kwargs):
        raise ServerToolError("Resuming a run after a review is not available on this machine yet.")
