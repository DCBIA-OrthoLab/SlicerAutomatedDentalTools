"""Generic Slicer widget for any tool exposed by the tool server.

A concrete module declares TOOL_NAME, optionally overrides what its tool's
schema cannot state (FILE_INPUTS, RESULT_KIND) and optionally overrides a few
hooks; everything else — Slicer lifecycle, schema-driven GUI, theme, async
call, error handling, temp-file cleanup — is inherited from here.

See ARCHITECTURE.md, "How to add a new module in 5 minutes".
"""

import logging
import fnmatch
import glob
import importlib
import json
import os
import re
import shutil
import threading
import time
import zipfile

import ctk
import qt
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin

from . import (config, console, design, digest, formgen, mode_switch, is_file_type, is_log_event,
               new_run_id, slicer_io, testfile_entries)
from .errors import RunCancelled, ServerToolError, plain_text
from .worker import BackgroundJob

logger = logging.getLogger("SADTCore.base_widget")


def _client_is_local() -> bool:
    """Whether the extension's client runs the tools on this machine."""
    from . import get_client

    return bool(getattr(get_client(), "is_local", False))

# "auto" is the schema-driven default: the argument's `types` decide whether it
# gets a file picker, a folder picker, or the choice between both (and which
# extensions the file picker offers). The explicit modes are for what the
# schema cannot express — picking a volume from the MRML scene — for forcing
# one selection kind, or ("none") for not offering an argument at all.
_FILE_INPUT_MODES = ("auto", "single_file", "folder_zip", "file_or_folder", "volume_node", "none")
_RESULT_KINDS = ("text", "segmentation", "labelmap", "volume", "model", "save_as")

# The box holding the output folder picker, which no schema argument owns. A
# tool may still put arguments of its own in it by declaring section="Outputs"
# (ASO's output_suffix does), which is why it is a plain name rather than a
# separate widget.
_OUTPUTS_SECTION = "Outputs"

# Where the server files what a supervised chain produced, inside the result
# archive. Its own, unimportable: `execution/runner.INTERMEDIATE_DIRNAME`.
_INTERMEDIATE_DIRNAME = "intermediate"

# Where what a STOPPED run produced is unpacked, under the run's output
# folder. One subdirectory per stop below it -- see `_unpackCheckpoint`.
# Apart from the results themselves, because the two are not the same thing: a
# checkpoint holds a copy of a step's output for a reader to correct, and the
# run is still going to write its real answer beside it.
_CHECKPOINT_DIRNAME = "quality_control"
# Where a stopped run's archive puts each step's output: `intermediate/<NN>_<Tool>/`.
_STEPS_DIRNAME = "intermediate"

# How a level reads in the console. Fixed English words rather than translated
# ones: they are a column a reader scans for "ERROR", and a fixed width keeps
# the messages after them aligned.
_LOG_LEVEL_WORDS = {"debug": "DEBUG", "info": "INFO", "warning": "WARNING", "error": "ERROR",
                    "metric": "METRIC"}


def format_run_log_line(event, prefix: str = "", now=None) -> str:
    """One console line: `HH:MM:SS  [source]  LEVEL  message`.

    `prefix` names the run ("Run 2") when several are in flight, and is left
    out for a lone run for the reason _describeRun leaves it out. The time is
    the server's, when it sent a usable one, and the moment the line arrived
    otherwise -- it is a time of day for a reader, not a key anything sorts on.
    `[source]` is the tool of a chain that wrote the line, absent for the tool
    the user asked for.
    """
    at = event.get("at")
    if at is None:
        at = time.time() if now is None else now
    try:
        stamp = time.strftime("%H:%M:%S", time.localtime(at))
    except (OverflowError, OSError, ValueError):
        stamp = "--:--:--"
    level = event.get("level") or "info"
    parts = []
    if prefix:
        parts.append(prefix + ":")
    parts.append(stamp)
    if event.get("source"):
        parts.append("[{}]".format(event["source"]))
    parts.append(_LOG_LEVEL_WORDS.get(level, "INFO").ljust(7))
    parts.append(event.get("message") or "")
    return "  ".join(parts)


_DIGITS = re.compile(r"\d+")


def step_key(text: str) -> str:
    """What makes two step lines "the same step" for the console narrative.

    Numbers are blanked, so "scan 3 of 12" follows "scan 2 of 12" on the panel
    without adding a console line per scan: the console tells the run's story
    one step at a time, the panel keeps the live count.
    """
    return _DIGITS.sub("#", text or "").strip()


def format_duration(seconds: float) -> str:
    """`seconds` the way the end-of-run line says it: "48 s", "4 min 58 s"."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return "{} s".format(seconds)
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return "{} min {:02d} s".format(minutes, seconds)
    hours, minutes = divmod(minutes, 60)
    return "{} h {:02d} min".format(hours, minutes)


# Result kinds drawn by their own display node rather than by a slice
# layer. A volume is not one: it is shown by being put in a layer, and
# its display node governs window/level, not whether it appears.
_SELF_DISPLAYING_KINDS = ("markups", "model", "segmentation")
# Sections a panel opens folded. By NAME, because that is what a tool
# declares -- there is no "advanced" flag on an argument, and inferring it
# from "every argument here is optional" would fold a section a tool meant
# to be read.
_COLLAPSED_SECTIONS = ("Advanced",)

# Characters a hosted test file's name may not contribute to a path built from
# it. The name comes from the server's own listing rather than from a user, but
# it is joined onto a local directory, and a "/" or a ".." in one would place
# the download somewhere nobody asked for.
_UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    """A single, harmless path component for a hosted entry's name."""
    cleaned = _UNSAFE_NAME_CHARACTERS.sub("_", os.path.basename(name)).strip("._-")
    return cleaned or "test_file"


def _process_is_running(pid: int) -> bool:
    """Is process `pid` alive? Asks the system and changes nothing.

    Not `os.kill(pid, 0)` on Windows: there signal 0 is CTRL_C_EVENT, so the
    call sends a console Ctrl+C instead of probing, and fails for any process
    outside this console -- which read as "dead" and let the sweep delete the
    test files of a Slicer that was still running.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        process_query_limited_information = 0x1000
        still_active = 259
        error_access_denied = 5
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            # Denied means it exists, under another account; anything else
            # (an invalid parameter) means there is no such process.
            return ctypes.get_last_error() == error_access_denied
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _merged_report(first, second):
    """Two run reports folded into one, without knowing what a tool puts in them.

    Every tool writes its own shape, so the rule is on the JSON types rather
    than on any field: lists concatenate (the per-scan entries), numbers add up
    (the counters in `summary`), objects merge key by key, and anything else
    keeps the first batch's value -- a tool name, a model bundle, a flag, all of
    which are properties of the run and not of the batch.

    Best effort, like every other use of a report: the results are on disk and
    in the scene whatever this produces.
    """
    if isinstance(first, dict) and isinstance(second, dict):
        merged = dict(first)
        for key, value in second.items():
            merged[key] = _merged_report(first[key], value) if key in first else value
        return merged
    if isinstance(first, list) and isinstance(second, list):
        return first + second
    # Before the number case: a bool IS an int in Python, and adding two
    # `"gpu_resampling": true` gives 2.
    if isinstance(first, bool) or isinstance(second, bool):
        return first
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return first + second
    return first


class _CohortView:
    """The widgets a cohort's progress is written into.

    Kept so the one-second tick writes VALUES rather than rebuilding Qt objects:
    a panel that recreates its own widgets every second flickers, and cannot be
    interacted with at all. A plain Python object, so holding references on it
    is allowed -- PythonQt refuses new attributes on a C++ one.
    """

    def __init__(self, frame, total, bar, rows, cohort, remainder=None):
        self.frame = frame
        self.total = total
        self.bar = bar
        self.rows = rows  # {run number: (label, bar)}, the batches listed
        self.cohort = cohort
        self.remainder = remainder  # the "+ N more" line, or None


def _batch_dirname(index) -> str:
    """What one batch of a cohort calls its folder.

    One function because two things are named by it and they have to agree:
    the folder a batch WRITES into while the cohort runs, and the folder its
    review is unpacked in. A reader who saw `batch_02` on a progress line
    finds `batch_02` in both places.
    """
    return "batch_%02d" % int(index)


# What `_mergeCohortFolders` is allowed to fold back up. Matched rather than
# remembered, so a cohort interrupted by a crashed Slicer is still merged the
# next time -- and so nothing the clinician put in that folder themselves can
# be moved by us.
_BATCH_DIRNAME = re.compile(r"^batch_\d{2,}$")


class _Cohort:
    """The few things several batches of one Apply have in common.

    Deliberately thin. A batch is an ordinary run in every way that matters --
    it queues, reports, cancels and fails on its own -- and this holds only what
    genuinely cannot be answered one run at a time: how many there are, how many
    have ended, and the report so far.
    """

    def __init__(self, total: int, total_scans: int = 0):
        self.total = total
        # The folder the clinician chose, under which every batch has one of
        # its own. Set when the batches are queued.
        self.root = None
        # In SCANS, not batches: a batch is how the transfer was cut up and
        # nobody has twenty batches of work to do. Known before anything is
        # sent, which is what lets the panel answer "how many of my scans are
        # done" with a number rather than an impression.
        self.total_scans = total_scans
        self.scans_done = 0
        self.scans_failed = 0
        self.finished = 0
        # The merge of every batch's report so far, written back to disk each
        # time so `_readRunReport` answers for the cohort and no module has to
        # know this feature exists.
        self.report = None
        # Sent with every batch, so the server can tell this cohort's batches
        # apart from another's and order them. Random rather than a counter:
        # two workstations, or two panels, must never mint the same one.
        self.batch_id = new_run_id()

    @property
    def complete(self) -> bool:
        return self.finished >= self.total

    def progress(self, running) -> float:
        """0..1 for the cohort's bar, counting what is in flight.

        The scans of the finished batches are exact; a running batch
        contributes its own reported fraction of its own size. The bar is the
        impression and the count below it is the fact -- which is why the count
        never includes a batch that has not finished.
        """
        if self.total_scans <= 0:
            return 0.0
        done = float(self.scans_done + self.scans_failed)
        for run in running:
            done += (run.fraction or 0.0) * run.scan_count
        return max(0.0, min(1.0, done / self.total_scans))


class _Run:
    """One tool execution: its own inputs, its own scratch directory, its own thread.

    A panel holds a LIST of these rather than a single job, because the wait a
    clinician actually has is a cohort, and the upload of the next patient has
    no reason to sit behind the inference of the previous one.

    What it does NOT buy is parallel inference: the server serialises the card
    (`MAX_CONCURRENT_GPU_JOBS`), so four runs of a segmentation still segment one
    at a time. The gain is the overlap of transfer with compute, which on a
    cohort is most of the wall clock, and genuine parallelism across tools that
    do not both want the GPU.
    """

    def __init__(self, number, label, args, files, output_dir, workspace,
                 cohort=None, cohort_index=None, scan_count=0):
        self.number = number
        self.label = label
        self.args = args
        self.files = files
        self.output_dir = output_dir
        self.workspace = workspace
        # The cohort this run is one batch of, and which batch, 1-based. Both
        # None when the input travelled whole -- which is every run that is not
        # a divided cohort, and the state in which this feature is not
        # observable at all.
        self.cohort = cohort
        self.cohort_index = cohort_index
        # The files and folders the reader chose for this run, as they are on
        # this machine. A stop hands them to the reviewer beside what the steps
        # wrote: a step that writes only landmarks leaves its scan behind, and
        # this side is the only one that has it.
        self.inputs = ()
        # Entries in this batch -- top-level ones, so a per-patient folder
        # counts as the one patient it is. 0 for a run that was not divided.
        self.scan_count = scan_count
        self.job = None
        self.phase = ""
        self.started_at = None  # None while the run is still queued
        # The step the console last narrated, as `step_key` reads it, so a
        # step is written once however many progress events repeat it.
        self.narrated_step = None
        # The checkpoint this run is stopped at, or None. A stopped run holds
        # no thread and no card -- the server is keeping its work for it while
        # somebody reads what it produced -- but it is emphatically not over,
        # and `running` answers True for it so the admission pump never starts
        # a second copy of a run that is merely waiting on a person.
        self.paused = None
        # The folder the reviewer was opened on at the last stop, so
        # withdrawing a paused run can take its review down with it.
        self.review_folder = None
        # What the checkpoint folder held the moment it was unpacked, as
        # {relative path: digest}. It is what "the reader changed this file"
        # is measured against -- see `_corrections` -- and it is replaced on
        # every unpack, because a run that stops twice reviews the same folder
        # twice and the second pass must diff against what the FIRST resume
        # produced, not against what the run started from.
        self.checkpoint_digests = {}

        # Minted here, before anything is sent, because the id has to be known
        # to both sides while the request is still in flight -- which is the
        # whole point: a server-assigned id could only travel in the response,
        # and the response is the last thing that happens. It is also a
        # capability (it plus the token is what authorises reading this run's
        # progress and cancelling it), hence a CSPRNG rather than the run
        # number sitting right above it.
        self.run_id = new_run_id()
        # Set when the user withdraws this run. Shared by the worker thread and
        # the progress watcher, so cancelling closes both.
        self.cancel_event = threading.Event()

        # The last thing the SERVER said about this run, kept apart from
        # `phase` (which is what the CLIENT is doing). Both are rendered, and
        # neither can stand in for the other: only the client knows it is
        # uploading, and only the server knows it has been queued for the GPU
        # for four minutes.
        self.server_phase = ""
        self.server_message = ""
        self.fraction = None  # 0.0..1.0, or None for "the tool did not say"
        self.depth = 0  # 0 is the tool that was asked for; deeper is a chain
        # Which tool of a supervised chain wrote the last progress event, when
        # the server said: the opening marker of a nested call names the
        # callee and its `call` id, and every record written inside that call
        # carries the same id. {call id: tool name}, filled as markers arrive.
        self.nested_tool = None
        self.call_tools = {}

    @property
    def running(self) -> bool:
        return self.job is not None or self.paused is not None

    def clear_server_progress(self) -> None:
        """Forget what the server last said.

        Called when the CLIENT reports something of its own, because by then
        the server's last word is behind us: "Downloading results" comes after
        "packaging", and showing both would leave the older of the two on the
        panel. The newest information wins, whichever side it came from.
        """
        self.server_phase = ""
        self.server_message = ""
        self.fraction = None

    def cancel(self) -> None:
        """Withdraw this run locally: nothing further is delivered, the worker
        and its watcher are told to stop, and the scratch directory goes.

        This does NOT stop the run on the server -- that is
        `client.cancel_run`, which the panel issues alongside (see
        ServerToolWidgetBase._cancelRun). Kept separate on purpose: this half
        must work with no server at all, and the queued half of a cancel has
        no server-side existence to withdraw.
        """
        self.cancel_event.set()
        if self.job:
            self.job.cancel()
            self.job = None
        self.close()

    def close(self) -> None:
        """Drop the scratch directory. Per run, never shared: two runs writing
        into one temp folder would overwrite each other's inputs."""
        if self.workspace:
            self.workspace.__exit__(None, None, None)
            self.workspace = None


# A timepoint word in a case's name: `P001_T1` and `P001_T2` are one patient.
_TIMEPOINT_WORD = re.compile(r"(?i)(^|[_\-.\s])T[0-9](?=$|[_\-.\s])")


def _patients_of(paths) -> set:
    """The patients a run's results cover, read off its `scene.json` files.

    One `scene.json` per case is the tools' convention (VISU's
    `index.SCENE_SUFFIX`): `<case>/scene.json`, or `<case>_scene.json` when
    several share a folder. The case is that name with its timepoint removed.
    """
    patients = set()
    for path in paths:
        name = os.path.basename(path)
        if not name.endswith("scene.json"):
            continue
        case = name[: -len("scene.json")].rstrip("_-. ")
        if not case:
            case = os.path.basename(os.path.dirname(path))
        patients.add(_TIMEPOINT_WORD.sub("", case).strip("_-. ").lower())
    return patients


class ServerToolWidgetBase(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Only TOOL_NAME is required. Everything the tool's own schema already
    states — which arguments are file inputs, what each picker looks like, what
    comes back — is derived from it (see formgen.file_input_modes and
    formgen.result_kind_for); the two attributes below are *overrides*, for the
    handful of things the server cannot know."""

    # -- declared by subclasses --------------------------------------
    TOOL_NAME = None
    # {schema_argument_name: mode} merged over the schema's own file arguments.
    # Only what the schema cannot say: "volume_node", a forced picker kind, or
    # "none" to leave an optional file argument out. See _FILE_INPUT_MODES.
    FILE_INPUTS = {}
    # None -> derived from the tool's output_kind. Declare one only when that
    # is ambiguous: output_kind "file" says a file comes back, not whether to
    # load it into the scene ("volume"/"model") or save it ("save_as").
    RESULT_KIND = None
    AUTO_UI = True

    # There is deliberately no TEST_DATA attribute any more. A tool's test data
    # is what the SERVER hosts for it (GET /tools/{tool}/data), offered in the
    # input row's own dropdown and downloaded on selection -- so every tool
    # gets it, and no module declares anything. What was here instead was a
    # per-module dict of hardcoded GitHub release URLs, which four modules had
    # and eleven did not, and which the server knew nothing about.

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

        if not self.TOOL_NAME:
            raise ValueError(f"{type(self).__name__} must set TOOL_NAME.")
        for arg_name, mode in self.FILE_INPUTS.items():
            if mode not in _FILE_INPUT_MODES:
                raise ValueError(f"{type(self).__name__}: unknown file input mode '{mode}' for '{arg_name}'.")
        if self.RESULT_KIND is not None and self.RESULT_KIND not in _RESULT_KINDS:
            raise ValueError(f"{type(self).__name__}: unknown RESULT_KIND '{self.RESULT_KIND}'.")

        # `self.client` is a property: the extension's client of the moment.
        self._argWidgets = {}
        self._schema = None
        # Active runs, queued and running, in the order they were asked for.
        # A list rather than one `_job`, so a second Apply enqueues instead of
        # being refused; `_pumpRuns` decides how many of them run at once.
        self._runs = []
        self._runsStarted = 0  # only ever grows: it numbers runs for the user
        self._inputWidgets = {}  # {schema_argument_name: widget}
        self._inputModes = {}  # {schema_argument_name: mode}, "auto" already resolved
        self._outputFolderWidget = None
        # The folder the PANEL proposed, so a path the user typed is never
        # replaced by the next suggestion. See _suggestOutputFolder.
        self._suggestedOutput = None
        # What the LAST run wrote, from the archive itself. See _loadResults.
        self._producedFiles = []
        # Where those files were unpacked. Kept so a member's path
        # INSIDE the archive can be read back -- which is how the
        # chain's own results are told apart from the run's.
        self._producedRoot = ""
        # Schema-driven panel layout, all rebuilt wholesale by _buildForm.
        self._sectionBoxes = {}  # {section name: ctkCollapsibleButton}
        self._sectionLayouts = {}  # {section name: QFormLayout}
        self._rows = {}  # {schema_argument_name: (label, field)} — hidden together
        self._rowSections = {}  # {schema_argument_name: section name}
        self._sectionsWithOwnRows = set()  # sections holding a row no argument owns
        self._hiddenArgs = set()  # arguments whose `visible_when` is not satisfied
        # Set by a build, consumed by the first enter() after it. See
        # _collapseAdvancedSections for why a fold has to happen twice.
        self._collapsePending = False
        self._statusBadge = None
        self._statusJob = None
        self._downloadJob = None  # one test-file fetch at a time
        # Where a downloaded test file lands, and what is already there.
        # Created on first use and never in Documents: a 648 MB cohort a user
        # clicked once must not still be on their disk next month.
        self._testFileRoot = None
        self._testFileCache = {}  # {(scope, hosted name): local path already fetched}
        # {argument: path already put in the scene}, so re-picking the same
        # file does not stack a second copy of it on the first.
        self._scenePreviews = {}
        self._sceneVolumes = {}  # {display name: vtkMRMLScalarVolumeNode}
        self._schemaError = None  # set while the panel could not be built from a schema
        self._rootLayout = None
        self._formWidget = None  # the schema-driven part, replaced wholesale on a rebuild
        self.applyButton = None
        self.cancelButton = None
        self.reviewButton = None
        self.uiWidget = None
        self._progressLabel = None
        self._progressBar = None  # determinate, and only while a tool reports a fraction
        # One Cancel per run, rebuilt whenever the set of runs changes. The
        # host widget stays put in the layout; only its single child is
        # replaced, the same swap _buildForm makes for the schema-driven part.
        self._runControlsLayout = None
        # Set while a cohort is in flight; see _buildCohortView.
        self._cohortView = None
        self._runControlsWidget = None
        self._elapsedTimer = None  # ticks once a second while any run is active

    # ------------------------------------------------------------------
    # Slicer lifecycle
    # ------------------------------------------------------------------

    @property
    def client(self):
        """The extension's client of the moment, unless one was set (the tests do).

        Read on every use rather than kept: the SADT Tools Manager replaces the
        client when the tools folder moves or a tool is installed, and a panel
        holding the old one would keep looking in the old place.
        """
        override = self.__dict__.get("_clientOverride")
        if override is not None:
            return override
        from . import get_client

        return get_client()

    @client.setter
    def client(self, value) -> None:
        self.__dict__["_clientOverride"] = value

    def _onInstallUpdate(self) -> None:
        from . import tools_ui

        tools_ui.confirm_and_install(self.parent, self.TOOL_NAME,
                                     on_done=lambda _installed: self._onRetryButton())

    def _linkBadgeToManager(self) -> None:
        """The green "Local run" says where the tools are, and opens the Manager on a click."""
        store = getattr(self.client, "store", None)
        where = store.root if store is not None else "this computer"
        self._statusBadge.setToolTip(
            _("The tools run on this computer. They are stored in {where}. "
              "Click to see, move or remove them (SADT Tools Manager).").format(where=where))
        self._statusBadge.setOpenExternalLinks(False)

        def clicked(link):
            if link == "check":
                self._checkForUpdates()
                return
            from . import tools_ui

            tools_ui.open_manager()

        self._statusBadge.linkActivated.connect(clicked)

    def _releaseCheckBadge(self, store) -> dict:
        """The badge's "check" link and what the last check said, for a local panel."""
        from . import release_watch

        if not self._isLocal() or store is None:
            return {}
        if getattr(store, "pinned", False):
            return {"note": _("kept for a study")}
        if not release_watch.follows_releases(store):
            return {"note": _("from a checkout")}
        if release_watch.checking():
            return {"note": _("checking...")}
        return {"note": getattr(self, "_releaseNote", ""), "check": True}

    def _checkForUpdates(self) -> None:
        """The badge's "check": ask for a new tools release now, then redraw the badge.

        Only ASKS: when there is one, the Update button appears, and installing
        stays that button's confirmation (size, the tools it brings, test files).
        """
        import time

        from . import release_watch

        def done(_before, state):
            if state.get("error"):
                self._releaseNote = _("could not reach GitHub")
            else:
                self._releaseNote = _("checked {time}").format(
                    time=time.strftime("%H:%M", time.localtime(float(state.get("checked") or time.time()))))
            try:
                self._refreshServerStatus()  # reads the version again, from a fresh client
            except (RuntimeError, AttributeError):
                pass  # the panel was closed meanwhile

        if release_watch.check_now(done):
            self._onStatusChecked(True)  # shows "checking..."

    def _isLocal(self) -> bool:
        """Whether the tools run on this machine (the local client) rather than on a server."""
        return bool(getattr(self.client, "is_local", False))

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        self.uiWidget = qt.QWidget()
        self.layout.addWidget(self.uiWidget)
        # The bar that opens the classic version of this tool, above everything
        # else. Added again by every setup(), so a Reload keeps it.
        mode_switch.add_visor_bar(self)
        rootLayout = qt.QVBoxLayout(self.uiWidget)

        self._statusBadge = design.status_badge(local=self._isLocal())
        if self._isLocal():
            # The badge and, on the same line, "Update now" when this tool runs
            # at an older tools version than the extension's (or changed since).
            badgeRow = qt.QHBoxLayout()
            badgeRow.addWidget(self._statusBadge)
            badgeRow.addStretch(1)
            self._updateButton = design.primary_button(_("Update now"))
            self._updateButton.setVisible(False)
            self._updateButton.clicked.connect(lambda: self._onInstallUpdate())
            badgeRow.addWidget(self._updateButton)
            rootLayout.addLayout(badgeRow)
            self._linkBadgeToManager()
        else:
            rootLayout.addWidget(self._statusBadge)

        # The schema-driven part lives in its own container so it can be thrown
        # away and rebuilt in place — see _buildForm.
        self._rootLayout = rootLayout
        # force_refresh: the client caches GET /tools on a singleton that
        # OUTLIVES this widget, so "Reload" rebuilt the panel from the response
        # fetched when Slicer started. A tool whose schema changed since -- a
        # new field, a hidden one, a different layout -- kept rendering the old
        # one, and only restarting Slicer showed the change. Setup runs once per
        # module load, so this costs one request per reload.
        self._buildForm(force_refresh=True)

        # In a host widget of its own, so it can be hidden with Apply while the
        # tool is not installed (see _showRunControls).
        self._extraHost = qt.QWidget()
        extraLayout = qt.QVBoxLayout(self._extraHost)
        extraLayout.setContentsMargins(0, 0, 0, 0)
        rootLayout.addWidget(self._extraHost)
        self.addExtraWidgets(extraLayout)
        # After the module's own widgets, so a module that adds some keeps them
        # above this one -- which is where every module that hand-built this box
        # had put it.
        self._addLoadResultsCheckBox(extraLayout)

        self.applyButton = design.primary_button(_("Apply"))
        self.cancelButton = design.danger_button(_("Cancel"))
        self.cancelButton.setVisible(False)
        # Shown only while a run of this panel waits at a checkpoint: the
        # reader may have wandered off VISU, and this is the way back to the
        # one place the run can go on from.
        self.reviewButton = design.secondary_button(_("Review the paused run in VISU"))
        self.reviewButton.setVisible(False)
        self.reviewButton.connect("clicked()", self._goToReview)
        rootLayout.addWidget(self.reviewButton)
        rootLayout.addWidget(self.applyButton)
        rootLayout.addWidget(self.cancelButton)

        self._progressLabel = design.progress_label()
        rootLayout.addWidget(self._progressLabel)

        self._progressBar = design.progress_bar()
        rootLayout.addWidget(self._progressBar)

        runControlsHost = qt.QWidget()
        self._runControlsLayout = qt.QVBoxLayout(runControlsHost)
        self._runControlsLayout.setContentsMargins(0, 0, 0, 0)
        self._runControlsLayout.setSpacing(design.SPACING_XS)
        rootLayout.addWidget(runControlsHost)

        self.applyButton.clicked.connect(self.onApplyButton)
        self.cancelButton.clicked.connect(self.onCancelButton)
        self._showRunControls()

        # Without a trailing stretch, QVBoxLayout spreads its (Preferred-policy)
        # widgets across the whole module panel height instead of packing them
        # at the top — the same reason every hand-written .ui file in this repo
        # ends with a vertical spacer.
        rootLayout.addStretch(1)

        design.apply(self.uiWidget)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        # A volume loaded or removed while the module is open must appear in
        # (or leave) the input dropdowns without the user having to switch
        # modules and back; enter() alone cannot see it happen.
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self._onSceneNodesChanged)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self._onSceneNodesChanged)

        self._checkCanApply()

        # Also kick off the health check here, not only in enter(): a module
        # reload re-instantiates the widget and calls setup() but never enter()
        # (see slicer.util.reloadScriptedModule), which would leave the freshly
        # created badge stuck on "checking..." until the user leaves the module
        # and comes back.
        self._refreshServerStatus()

    def cleanup(self) -> None:
        mode_switch.remove_visor_bar(self)
        self.removeObservers()
        # Server-side too, not only locally. This panel is going away (a module
        # reload, Slicer closing), so nothing here can ever collect these
        # results -- leaving an inference holding the card for an hour on
        # behalf of a widget that no longer exists is pure waste, and the GPU
        # is shared with every other client.
        # A PAUSED run is the exception, and the reasoning above is what says
        # so: it is holding nothing. It is not on the card, it is not in a
        # worker thread, it is a job directory waiting for a person to finish
        # looking -- and the server's idle TTL already bounds that, exactly as
        # it bounds an abandoned transfer. Cancelling it threw away the review
        # the moment the module was reloaded, which is precisely what one does
        # while working on a tool: the reader pressed Continue and the resume
        # came back 500, the run having been cancelled underneath them.
        running = [run.run_id for run in self._runs
                   if run.started_at is not None and run.paused is None]
        for run in list(self._runs):
            run.cancel()
        self._runs = []
        self._requestServerCancel(running)
        if self._statusJob:
            self._statusJob.cancel()
            self._statusJob = None
        if self._downloadJob:
            self._downloadJob.cancel()
            self._downloadJob = None
        self._removeOwnTestFiles()

    def enter(self) -> None:
        if self.uiWidget:
            design.apply(self.uiWidget)
        # The fold the build asked for, applied now that the panel has a parent
        # and a screen. See _collapseAdvancedSections.
        if self._collapsePending:
            self._collapsePending = False
            self._collapseAdvancedSections()
        # The hosted-file lists are re-read here, not only at setup(). They are
        # server-side state that changes independently of the schema — a model
        # dropped into DATA/<tool>/models/ does not touch /tools — so nothing
        # in the schema-rebuild path (which only fires when the schema fetch
        # FAILED) can ever notice one. Without this, a bundle added while
        # Slicer is open is invisible until Slicer is restarted, with no
        # affordance on the panel saying so: the user sees a dropdown that is
        # simply missing the entry they were told to pick.
        # Swept on ENTER, not only when someone picks a test file. Slicer
        # never removes what `tempDirectory()` creates -- its own docstring
        # says so -- and a user who downloaded a 648 MB cohort once and did not
        # come back would keep it for good. Opening any tool is now enough.
        self._sweepLeftoverTestFiles()
        # A server that could not be reached for the schema will not answer
        # for the hosted files a moment later either; asking anyway only adds
        # a second warning about the same outage.
        if self._refreshSchema() is not False:
            self._refreshServerSelectables()
        self._refreshSceneVolumes()
        self._refreshServerStatus()

    def exit(self) -> None:
        pass

    def onSceneStartClose(self, caller, event) -> None:
        pass

    def onSceneEndClose(self, caller, event) -> None:
        # The scene is empty now: the dropdowns must stop offering volumes
        # that no longer exist.
        if self.uiWidget:
            self._refreshSceneVolumes()

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _buildForm(self, force_refresh: bool = False) -> None:
        """Build the schema-driven part of the panel into a fresh container,
        replacing the previous one if there was any.

        Called once from setup(), and again by _onStatusChecked when a server
        that was unreachable at setup() time comes back: the panel is built
        from the schema, so a failed fetch leaves nothing but an error label,
        and nothing else would ever clear it — the module would stay broken for
        the rest of the Slicer session even though the server is back.

        Replacing the whole container rather than clearing a layout keeps this
        simple and total: no widget of the previous attempt survives, including
        the error label and any stale server-side dropdown.
        """
        formWidget = qt.QWidget()
        formLayout = qt.QVBoxLayout(formWidget)
        formLayout.setContentsMargins(0, 0, 0, 0)

        try:
            if self.AUTO_UI:
                self._buildAutoUI(formLayout, force_refresh=force_refresh)
            else:
                self.buildCustomUI(formLayout)
        except Exception as exc:
            # Never leave the user with a silently blank/half-built panel: a bad
            # CTK/Qt call, a module misconfiguration, etc. must be visible right
            # here, not just in the Python console.
            logger.exception("Failed to build the UI for tool '%s'", self.TOOL_NAME)
            formLayout.addWidget(
                design.warning_label(_("Could not build this module's UI: {error}").format(error=exc))
            )

        previous = self._formWidget
        if previous is None:
            self._rootLayout.addWidget(formWidget)
        else:
            self._rootLayout.insertWidget(self._rootLayout.indexOf(previous), formWidget)
            # Hide and unparent so the old panel leaves the layout now, but let
            # Qt destroy it later: this can run from a signal emitted by one of
            # its own children (the Retry button below).
            previous.setVisible(False)
            previous.setParent(None)
            previous.deleteLater()
        self._formWidget = formWidget

        if previous is not None:
            # A rebuild: the stylesheet was applied to widgets that no longer
            # exist, and the Apply button's state was computed from them.
            design.apply(self.uiWidget)
            self._checkCanApply()

        # Now that the form has a parent. A rebuild triggered while the module
        # is on screen -- a server that was down at setup() coming back -- never
        # sees another enter(), so the flag set inside the build would not be
        # consumed and this is the only chance to fold.
        self._collapseAdvancedSections()
        self._showRunControls()

    def _showRunControls(self) -> None:
        """Apply, its options and its progress only for a tool that can run."""
        installed = not getattr(self, "_notInstalled", False)
        for widget in (getattr(self, "applyButton", None), self._loadResultsCheckBox,
                       getattr(self, "_extraHost", None)):
            if widget is not None:
                widget.setVisible(installed)

    def _onRetryButton(self) -> None:
        """Rebuild from a fresh /tools fetch. Safe to call from the button's own
        handler: _buildForm hides the old container and defers its destruction
        with deleteLater(), so the button outlives the click it is handling."""
        self._buildForm(force_refresh=True)
        self._refreshServerStatus()

    def _buildAutoUI(self, rootLayout, force_refresh: bool = False) -> None:
        logger.info("Building AUTO_UI for TOOL_NAME='%s' (FILE_INPUTS overrides=%s, RESULT_KIND=%s)",
                    self.TOOL_NAME, self.FILE_INPUTS, self.RESULT_KIND or "<from output_kind>")

        # The schema is fetched before any widget is built, not after: a file
        # argument's declared `types` decide what its picker looks like (file,
        # folder, or both — and with which extensions), so the widgets cannot
        # be built without it. The failure path below still builds them, from
        # an empty schema, so the panel is never blank.
        # _schemaError is what tells _onStatusChecked this panel is worth
        # rebuilding once the server answers again.
        self._schemaError = None
        try:
            self._schema = self.client.get_tool_schema(self.TOOL_NAME, force_refresh=force_refresh)
            logger.info(
                "Schema for '%s': output_kind=%s, argument keys=%s",
                self.TOOL_NAME,
                self._schema.get("output_kind"),
                sorted(self._schema.get("arguments", {}).keys()),
            )
        except ServerToolError as exc:
            logger.warning("Could not load schema for '%s': %s", self.TOOL_NAME, exc)
            self._schema = {"arguments": {}}
            self._schemaError = exc
        self._notInstalled = bool(
            self._schemaError is not None and self._isLocal()
            and hasattr(self.client, "has_tool") and not self.client.has_tool(self.TOOL_NAME))
        if self._notInstalled:
            # No inputs, no outputs, no Apply: a form for a tool that is not
            # there only invites a run that cannot happen.
            from . import tools_ui

            tools_ui.not_installed_card(self.parent, rootLayout, self.TOOL_NAME,
                                        on_installed=lambda _installed: self._onRetryButton())
            self._inputWidgets = {}
            return

        # One collapsible box per section the schema names, in declaration
        # order. A tool naming none gets exactly one box called "Inputs" — the
        # panel every module has today, unchanged.
        arguments = self._schema.get("arguments", {})
        # DEFAULT_SECTION is always created, even when every argument claims
        # another one: it is where anything without a section of its own goes,
        # including the error path's empty schema. An unused one holds no rows
        # and _applyVisibility hides it, so it costs nothing on screen.
        extraSections = [formgen.DEFAULT_SECTION]
        if self.resultKind == "save_as":
            extraSections.append(_OUTPUTS_SECTION)
        self._sectionBoxes = {}
        self._sectionLayouts = {}
        self._rows = {}
        self._rowSections = {}
        self._sectionsWithOwnRows = set()
        for sectionName in formgen.sections_of(arguments, extraSections):
            box = ctk.ctkCollapsibleButton()
            box.text = _(sectionName)
            # A section in _COLLAPSED_SECTIONS is folded at the END of this
            # method, not here -- see `_collapseAdvancedSections`.
            # A section the schema lays out in columns gets a grid; everything
            # else keeps the one-argument-per-row form. FlexReg's four patch
            # corners are a 2x2 that mirrors the arch, so where a pad sits on
            # screen is where that corner sits in the mouth.
            columns = formgen.section_columns(arguments, sectionName)
            if columns > 1:
                self._sectionLayouts[sectionName] = qt.QGridLayout(box)
            else:
                self._sectionLayouts[sectionName] = qt.QFormLayout(box)
            self._sectionBoxes[sectionName] = box
            rootLayout.addWidget(box)

        inputsLayout = self._sectionLayouts[formgen.DEFAULT_SECTION]

        self._inputWidgets = self._buildInputWidgets(inputsLayout)

        if self._schemaError is not None:
            rootLayout.addWidget(
                design.warning_label(
                    (_("Could not load '{tool}': {error}") if self._isLocal()
                     else _("Could not load '{tool}' from the server: {error}")).format(
                        tool=self.TOOL_NAME, error=self._schemaError
                    )
                )
            )
            # Leaving and re-entering the module also retries (see
            # _onStatusChecked), but a user staring at this error should not
            # have to discover that.
            if self._isLocal():
                # Not installed yet: offer it, with what it needs, rather than a retry.
                from . import tools_ui

                installButton = design.primary_button(
                    _("Install {tool}...").format(tool=self.TOOL_NAME))
                installButton.clicked.connect(lambda: tools_ui.confirm_and_install(
                    self.parent, self.TOOL_NAME, on_done=lambda _installed: self._onRetryButton()))
                rootLayout.addWidget(installButton)
            retryButton = design.primary_button(_("Retry"))
            retryButton.clicked.connect(self._onRetryButton)
            rootLayout.addWidget(retryButton)
        else:
            self._warnAboutFileInputsMismatch(rootLayout)

        self._argWidgets = formgen.build(
            arguments, inputsLayout, sections=self._sectionLayouts, rows=self._rows
        )
        self._rowSections.update(
            {name: formgen.section_of(arguments.get(name, {})) for name in self._rows}
        )
        logger.info("AUTO_UI built %d scalar field(s) for '%s': %s",
                    len(self._argWidgets), self.TOOL_NAME, sorted(self._argWidgets.keys()))
        for widget in self._argWidgets.values():
            formgen.connect_changed(widget, self._checkCanApply)

        self._populateServerSelectables(rootLayout)
        self._refreshSceneVolumes()

        if self.resultKind == "save_as":
            outputsLayout = self._sectionLayouts[_OUTPUTS_SECTION]
            # The SAME row every input on the panel is: what it holds on the
            # left, one blue `Select` on the right. It was the last
            # ctkPathLineEdit in a generated panel -- an editable box with a
            # small grey `...` at its end, sitting under four rows that had
            # stopped looking anything like it. One folder, so no source bar:
            # a choice of one is not a choice.
            self._outputFolderWidget = formgen.FileOrFolderInput(
                modes=("folder",), destination=True)
            outputsLayout.addRow(design.required_label(_("Output folder")),
                                 formgen.row_widget(self._outputFolderWidget))
            formgen.connect_changed(self._outputFolderWidget, self._checkCanApply)
            self._suggestOutputFolder()

            # This row belongs to no schema argument, so it must keep its
            # section on screen even when every argument in it is hidden.
            self._sectionsWithOwnRows.add(_OUTPUTS_SECTION)

        self._wireVisibility(arguments)

        self.configureFields()

        self._collapseAdvancedSections()
        # ... and again the first time this panel is actually on screen.
        self._collapsePending = True

    def _collapseAdvancedSections(self) -> None:
        """Fold the sections meant to start folded, once every row exists.

        Folded shut, not hidden. "Advanced" is the convention's own name for
        what a clinician does not need to decide -- a seed, a tile step, the
        reference volume someone who needs one goes looking for. Open by default
        they sit between the inputs and Apply, so every reader steps over them;
        folded, the ones who want them still find them in one click and nobody
        else meets them at all.

        Done LAST because a ctkCollapsibleButton hides the children it has at
        the moment it is collapsed, and nothing afterwards: a row added to an
        already-folded box is never hidden and draws over the title bar. That is
        what made AutoMatrix's "Reference volume" look crushed into the
        "Advanced" header, and it would have hit every later section the same
        way. configureFields() runs before this, so a module adding its own row
        to one of these sections is folded away with the rest.

        **And done a SECOND time, on the first enter().** Last within the build
        is still not late enough: `_buildForm` assembles the whole panel into a
        detached `QWidget` and only parents it into the module afterwards, so
        this runs on a tree that has never been shown. ctkCollapsibleButton
        folds by hiding its children, and hiding a child of a widget Qt has not
        realised yet does not survive the show that follows -- the box reads as
        folded while its rows are drawn underneath it. Re-folding once the
        module is actually on screen is what makes the state stick.

        Once, not on every enter(): a section the user opened must stay open
        when they leave the module and come back, which is why this is a
        one-shot flag a build sets and the next enter() consumes rather than a
        fold applied every time the panel appears.

        **The fold is forced through a real state change**, and that is the
        whole reason the second attempt works where the first did not.
        `ctkCollapsibleButton::setCollapsed` returns immediately when the value
        it is handed is the one it already holds -- so a box folded during the
        build already believes it is folded, and folding it "again" later runs
        nothing at all: no child pass, no hiding, the rows stay on screen under
        a bar that reads as shut. Stepping through False first is what makes
        CTK run the pass, this time on a widget that has a parent and a screen.
        Both assignments happen inside one turn of the event loop, so nothing
        is painted in between.
        """
        for sectionName in _COLLAPSED_SECTIONS:
            box = self._sectionBoxes.get(sectionName)
            if box is not None:
                box.collapsed = False
                box.collapsed = True

    # ------------------------------------------------------------------
    # Conditional fields (`visible_when`)
    # ------------------------------------------------------------------

    def _wireVisibility(self, arguments: dict) -> None:
        """Re-evaluate every `visible_when` whenever a controlling field
        changes, and once now so the panel opens in the right state.

        Called from _buildAutoUI rather than from setup(), because the panel is
        rebuilt from scratch when a server that was down comes back — the same
        reason configureFields() exists (see ARCHITECTURE.md). Wiring this once
        at setup() would leave the rebuilt panel showing every field of every
        mode again.
        """
        for name in formgen.controlling_arguments(arguments):
            widget = self._argWidgets.get(name)
            if widget is None:
                # check_schema rejects a visible_when naming an argument the
                # tool doesn't declare, so this means the schema fetch failed
                # and there is no form to drive. is_visible() hides what it
                # cannot evaluate, which is already the right answer.
                continue
            formgen.connect_changed(widget, self._applyVisibility)
        self._applyVisibility()

    def _narrowChoices(self, name: str, allowed, groups=None) -> None:
        """Restrict one choice argument to `allowed`, keeping the selection if it
        survives. Falls back to the first option, because a QComboBox cannot be
        empty and index 0 is what it would select anyway."""
        widget = self._argWidgets.get(name)
        if isinstance(widget, formgen.MultiChoiceGroup):
            # A facade publishes the UNION of its engines' options; the mode says
            # which apply, and which tabs they belong in. Only combo boxes were
            # narrowed here, so ALI's intraoral landmarks arrived in the CBCT
            # panel's four anatomical tabs -- every option offered, laid out
            # under a region that does not exist in an intraoral scan.
            declared = self._schemaArgument(name).get("choices") or {}
            widget.rebuild({option: bool(declared.get(option)) for option in allowed}, groups)
            return
        if widget is None or not hasattr(widget, "addItems"):
            return
        current = widget.currentText
        if [widget.itemText(i) for i in range(widget.count)] == list(allowed):
            return
        was = widget.blockSignals(True)
        widget.clear()
        widget.addItems(list(allowed))
        widget.blockSignals(was)
        index = widget.findText(current)
        widget.setCurrentIndex(index if index >= 0 else 0)

    def _applyVisibility(self, *_args) -> None:
        arguments = (self._schema or {}).get("arguments", {})
        controlling = formgen.controlling_arguments(arguments)
        values = formgen.collect(
            {name: self._argWidgets[name] for name in controlling if name in self._argWidgets}
        )

        # Narrow the choice boxes BEFORE deciding what is visible: an option
        # the current mode does not have must not merely fail at the end of a
        # run, and re-selecting here can itself change what the rest of the
        # panel shows.
        for name, spec in arguments.items():
            allowed = formgen.allowed_options(spec, values)
            groups = formgen.allowed_groups(spec, values)
            if allowed is not None:
                self._narrowChoices(name, allowed, groups)
        values = formgen.collect(
            {name: self._argWidgets[name] for name in controlling if name in self._argWidgets}
        )

        # A test-file list whose subfolder follows the mode is refilled when
        # the mode changes, from the listing already read.
        data = getattr(self, "_selectableData", None)
        if data is not None:
            for name, kind in self._serverSelectableArguments().items():
                if arguments.get(name, {}).get("selectable_scope_when"):
                    self._fillServerSelectable(name, kind, data)

        hidden = set()
        for name, spec in arguments.items():
            visible = formgen.is_visible(spec, values)
            if not visible:
                hidden.add(name)
            for widget in self._rows.get(name, ()):
                widget.setVisible(visible)
        self._hiddenArgs = hidden

        # A section whose every row is hidden is an empty titled box; hide it
        # too. This is what turns two mutually exclusive sets of arguments into
        # the old module's two stacked pages, with no client-side notion of a
        # "page" anywhere.
        for sectionName, box in self._sectionBoxes.items():
            owned = [name for name, owner in self._rowSections.items() if owner == sectionName]
            box.setVisible(
                sectionName in self._sectionsWithOwnRows
                or any(name not in hidden for name in owned)
            )

        self._checkCanApply()

    def configureFields(self) -> None:
        """Override to touch up the generated widgets once they all exist —
        a placeholder, an initial value, a connection between two fields.

        Called at the end of every auto-generated panel build, `addExtraWidgets`
        only at the first: the panel is rebuilt from scratch when a server that
        was down at setup() time comes back (see _buildForm), and anything
        applied outside this hook would be lost on that rebuild, leaving a
        subtly different panel from the one the module describes.

        `self._argWidgets` and `self._inputWidgets` are populated by now; both
        are empty when the schema could not be fetched, so read them with
        `.get()`.
        """

    @property
    def resultKind(self) -> str:
        """RESULT_KIND if the module declares one, otherwise derived from the
        tool's own output_kind (see formgen.result_kind_for)."""
        return formgen.result_kind_for((self._schema or {}).get("output_kind"), self.RESULT_KIND)

    def _buildInputWidgets(self, layout) -> dict:
        # Resolved once, here: each mode is needed both to build the widget and,
        # later, to know whether the selection has to be zipped before upload.
        self._inputModes = formgen.file_input_modes(
            (self._schema or {}).get("arguments", {}), self.FILE_INPUTS
        )
        return {
            arg_name: self._buildFileInputWidget(layout, arg_name, mode)
            for arg_name, mode in self._inputModes.items()
        }

    def _schemaArgument(self, arg_name: str) -> dict:
        return (self._schema or {}).get("arguments", {}).get(arg_name, {})

    def _buildFileInputWidget(self, layout, arg_name: str, mode: str):
        spec = self._schemaArgument(arg_name)
        # Same rule as every other row (formgen.label_for): the tool's own
        # wording when it declares one, the prettified name otherwise. Not
        # wrapped in _(): a label coming from the server is not in this
        # module's translation catalog, and the fallback is a schema
        # identifier rather than a phrase anyone wrote.
        label = formgen.label_for(arg_name, spec)
        # A file argument goes in the section its own spec names, like every
        # other argument; `layout` is the fallback for one that names none.
        section = formgen.section_of(spec)
        target = self._sectionLayouts.get(section, layout)
        labelWidget = design.fit_row_label(
            design.required_label(label)
            if formgen.may_be_required(spec)
            else design.optional_label(label)
        )

        if mode == "volume_node":
            widget = slicer.qMRMLNodeComboBox()
            widget.nodeTypes = ["vtkMRMLScalarVolumeNode"]
            widget.noneEnabled = True
            widget.setMRMLScene(slicer.mrmlScene)
            target.addRow(labelWidget, widget)
            field = widget
            widget.currentNodeChanged.connect(self._checkCanApply)
        else:
            widget = formgen.file_widget(spec, mode, arg_name)
            field = formgen.row_widget(widget)
            target.addRow(labelWidget, field)
            formgen.connect_changed(widget, self._checkCanApply)
            # Choosing a scan should show it, the way choosing a hosted test
            # file already does -- a clinician picks a file in order to look at
            # it, and having to open it a second time through Add Data is a step
            # the panel can spare them.
            formgen.connect_changed(
                widget, lambda arg=arg_name: self._previewPickedFile(arg))
            # Picking one of the tool's hosted test files is an action, not a
            # value: this is where it lands, and the download that follows is
            # why formgen hands the choice back instead of acting on it.
            setter = getattr(widget, "setHostedCallback", None)
            if setter is not None:
                setter(lambda name, arg=arg_name: self._onHostedTestFile(arg, name))
            # Same reasoning for a volume already open in Slicer: choosing one
            # means writing it to disk, which a widget factory does not do.

        # Recorded like a scalar row so `visible_when` can hide a file input
        # too, and so a section holding only file inputs is not mistaken for an
        # empty one.
        self._rows[arg_name] = (labelWidget, field)
        self._rowSections[arg_name] = section

        # The server's own wording for this input, now that the schema is known.
        description = spec.get("description")
        if description:
            widget.setToolTip(description)
        return widget

    def _serverSelectableArguments(self) -> dict:
        """`{argument name: "model" | "testfile"}` for every dropdown fed by
        GET /tools/{tool}/data.

        Two widget kinds, one mechanism: a SCALAR server_selectable argument
        (a model, which must never leave the server) is a plain combo box in
        `_argWidgets`; a FILE-typed one is an input row that also offers the
        hosted names, in `_inputWidgets`. Both are filled from the same call.
        """
        arguments = (self._schema or {}).get("arguments", {})
        return {
            name: spec["server_selectable"]
            for name, spec in arguments.items()
            if spec.get("server_selectable")
            and (name in self._argWidgets or name in self._inputWidgets)
        }

    def _fillServerSelectable(self, arg_name: str, kind: str, data: dict) -> list:
        """Put the hosted names into one dropdown and return them.

        **The current selection survives if the server still offers it.** This
        is refilled on every `enter()`, so without it, switching away from the
        module and back would silently reset a chosen model to the first entry
        in the list — the kind of change a user does not look for, and which
        would then run the tool against weights they never picked.
        """
        # An argument may draw from ONE subfolder of the tool's hosted files
        # rather than from all of them: a tool serving several modalities keeps
        # one folder per modality, and AREG's CBCT baseline picker was offering
        # the intraoral meshes, which cannot be a baseline. The server publishes
        # the subfolder on the argument and the list beside the flat one; a
        # deployment that scopes nothing sends no `scoped` section and every
        # argument reads the same list it always did.
        spec = (self._schema or {}).get("arguments", {}).get(arg_name, {})
        scope = self._selectableScope(arg_name)
        source = (data.get("scoped", {}) or {}).get(scope, data) if scope else data
        choices = list(source.get("models" if kind == "model" else "testfiles", []))
        fileInput = self._inputWidgets.get(arg_name)

        if fileInput is not None:
            # A file input needs no "(automatic)" entry: it leads with its own
            # prompt, so it can express "nothing chosen here". The current
            # selection surviving the refill lives inside the widget: its
            # rebuild keeps the entry when the server still offers it.
            #
            # It is fed the ENTRIES, not the bare names: what a user needs
            # before clicking a test file is whether it is one scan or a whole
            # cohort and how many bytes that is. A model dropdown gets the
            # names, because naming a model is all that ever travels for one.
            fileInput.setChoices(
                testfile_entries(source) if kind == "testfile" else choices
            )
            return choices

        entries = list(choices)
        if not spec.get("required"):
            entries.insert(0, formgen.AUTOMATIC_OPTION)

        widget = self._argWidgets[arg_name]
        previous = widget.currentText
        widget.clear()
        widget.addItems(entries)
        if previous in entries:
            widget.setCurrentIndex(entries.index(previous))
        return choices

    def _refreshSchema(self) -> "bool | None":
        """Rebuild the form when the SERVER's schema has changed under it.

        Returns False when the server could not be reached, so `enter()` does
        not ask it for anything else; None when there was nothing to compare.

        The schema is read once, at `setup()`. A panel built while the server
        offered an older version of this tool then keeps that form for the
        whole Slicer session, and visiting another module and coming back does
        not help: `enter()` re-reads the hosted FILES, not the shape of the
        form. On a deployment being worked on that is a field added, renamed or
        hidden minutes ago and simply absent, with nothing on the panel saying
        why -- the user is looking at a form the server stopped publishing.

        Rebuilt ONLY when it actually differs. A rebuild throws the form away,
        so a path somebody typed and has not run yet must survive an ordinary
        trip to another module; paying that on every visit to fix a case that
        almost never happens would be the worse bargain.

        A server that cannot be reached leaves the panel exactly as it is, for
        the reason `_refreshServerSelectables` gives: the form is already
        usable, and a server that went away between two visits must not empty
        it.
        """
        if self._schema is None:
            return None
        try:
            schema = self.client.get_tool_schema(self.TOOL_NAME, force_refresh=True)
        except ServerToolError as exc:
            logger.warning("Could not re-read the schema for '%s': %s",
                           self.TOOL_NAME, exc)
            return False
        if schema == self._schema:
            return True
        logger.info("'%s': the server's schema changed; rebuilding the panel",
                    self.TOOL_NAME)
        self._buildForm(force_refresh=True)
        return True

    def _refreshServerSelectables(self) -> None:
        """Re-read the hosted-file lists and update the dropdowns in place.

        Called from `enter()`. Deliberately quieter than the build-time pass:
        it adds no warning label (there is no root layout to attach one to
        outside a build, and a banner appended on every visit to the module
        would accumulate), and a failure leaves the dropdowns exactly as they
        were — the panel is already usable, so a server that has gone away
        between two visits must not empty a working list.
        """
        selectable = self._serverSelectableArguments()
        if not selectable:
            return
        try:
            data = self.client.list_tool_data(self.TOOL_NAME)
        except ServerToolError as exc:
            logger.warning(
                "Could not refresh server-side data for '%s': %s", self.TOOL_NAME, exc
            )
            return

        self._selectableData = data
        for arg_name, kind in selectable.items():
            self._fillServerSelectable(arg_name, kind, data)
        # The refill can add the very entry that makes a required argument
        # satisfiable, or remove the one that was satisfying it.
        self._checkCanApply()

    def _populateServerSelectables(self, rootLayout) -> None:
        """Fill every server_selectable dropdown (see formgen._make_widget)
        with the file names hosted on the server for this tool, from
        GET /tools/{tool}/data — e.g. SurgMovPred's "model" argument, which
        is picked among the server's models by name, never uploaded.

        Synchronous like the schema fetch just above, and for the same reason:
        the form needs its choices before the first paint, and the call is
        capped at the same short timeout. A failure (or an empty list) shows a
        visible warning instead of leaving a silently empty dropdown.
        """
        selectable = self._serverSelectableArguments()
        if not selectable:
            return

        try:
            data = self.client.list_tool_data(self.TOOL_NAME)
        except ServerToolError as exc:
            logger.warning("Could not list server-side data for '%s': %s", self.TOOL_NAME, exc)
            rootLayout.addWidget(
                design.warning_label(
                    (_("Could not list the models and test files of '{tool}': {error}")
                     if self._isLocal()
                     else _("Could not list the server-side files for '{tool}': {error}")).format(
                        tool=self.TOOL_NAME, error=exc
                    )
                )
            )
            return

        self._selectableData = data
        for arg_name, kind in selectable.items():
            choices = self._fillServerSelectable(arg_name, kind, data)
            fileInput = self._inputWidgets.get(arg_name)
            logger.info("Populated '%s.%s' with %d server-side %s(s)",
                        self.TOOL_NAME, arg_name, len(choices), kind)
            # An empty list only blocks the user when there is no other way to
            # provide the argument. A file-typed one can always be uploaded
            # instead, so warning about it would be noise on every server that
            # simply hosts no test data.
            if not choices and fileInput is None:
                rootLayout.addWidget(
                    design.warning_label(
                        (_("No {kind} available on this machine for '{tool}'.") if self._isLocal()
                         else _("No {kind} available on the server for '{tool}' — ask the server maintainer to add one.")).format(
                            kind=kind, tool=self.TOOL_NAME
                        )
                    )
                )

    def _warnAboutFileInputsMismatch(self, rootLayout) -> None:
        """Catch schema drift early. The set of file inputs is derived from the
        schema and so cannot drift; FILE_INPUTS *overrides* are written by hand
        against a remembered schema, so an override naming an argument the
        server no longer declares as a file surfaces immediately here instead
        of being silently ignored (or failing later with a confusing 422)."""
        declared = {name for name, spec in self._schema.get("arguments", {}).items() if is_file_type(spec.get("type", ""))}
        missing = set(self.FILE_INPUTS) - declared
        if missing:
            message = _(
                "FILE_INPUTS declares {missing} but the server's '{tool}' schema doesn't have "
                "them as file arguments (it has: {declared})."
            ).format(missing=sorted(missing), tool=self.TOOL_NAME, declared=sorted(declared))
            logger.warning(message)
            rootLayout.addWidget(design.warning_label(message))

    def buildCustomUI(self, layout) -> None:
        """Override when AUTO_UI = False."""
        raise NotImplementedError(f"{type(self).__name__} must implement buildCustomUI() since AUTO_UI is False.")

    def addExtraWidgets(self, layout) -> None:
        """Override to add a custom button or field. Called after the auto-generated
        GUI, before Apply/Cancel — this is the supported way to extend a module
        without touching setup()."""

    # ------------------------------------------------------------------
    # Overridable data hooks
    # ------------------------------------------------------------------

    def collectArgs(self) -> dict:
        """Override to transform values before sending.

        An OPTIONAL text field left empty is dropped rather than sent as "".
        The server applies an omitted optional argument's default; it takes a
        present one literally, so sending "" is asking for an empty value, not
        for the default. That is never what an untouched field means — for
        ALI's `prediction_ID` it produced `scan_lm_.mrk.json` instead of
        `scan_lm_Pred.mrk.json`.

        Only `""` qualifies: a multichoice reads back as a dict (every box
        unchecked is a meaningful selection, see MultiChoiceGroup), and 0 /
        False are values a user deliberately set.

        An argument HIDDEN by its `visible_when` is dropped for the same
        reason, one step further: it is not "left empty", it does not apply at
        all. Sending ASO's 32 `ios_teeth` boxes along with a CBCT run would
        state a selection the user was never shown and never made — and, since
        the server reads what it receives as the selection itself, an argument
        whose default someone changes server-side would still arrive frozen at
        whatever the invisible widget happened to hold.
        """
        values = formgen.collect(self._argWidgets)
        arguments = (self._schema or {}).get("arguments", {})
        collected = {
            name: value
            for name, value in values.items()
            if name not in self._hiddenArgs
            and not (value == "" and not arguments.get(name, {}).get("required"))
        }

        # A file argument satisfied by a hosted MODEL travels as a plain form
        # value - its NAME - not as an upload, so it belongs here rather than
        # in prepareInputFiles. The weights never move, which is the point.
        #
        # A hosted TEST FILE no longer appears here at all. It used to: the
        # name travelled and the server read the file in place. It is
        # downloaded now, so by the time a run starts it is an ordinary local
        # file in `files` - which is what lets the user open the scan beside
        # the panel, the whole reason for fetching it.
        collected.update(self._serverSideSelections())
        return collected

    def _serverSideSelections(self) -> dict:
        """{argument name: hosted name} for every input row whose selection is
        sent as a name rather than uploaded - a model, never a test file (see
        formgen.ServerFileInput.server_name)."""
        chosen = {}
        for arg_name, widget in self._inputWidgets.items():
            if arg_name in self._hiddenArgs:
                continue
            reader = getattr(widget, "server_name", None)
            name = reader() if reader else ""
            if name:
                chosen[arg_name] = name
        return chosen

    def prepareInputFiles(self, workspace: slicer_io.TempWorkspace, batch=None) -> dict:
        """Override for exotic input cases. Default behavior covers every file
        input mode, for each of the tool's file arguments. Returns
        {schema_argument_name: local_file_path}.

        `batch` is `(argument name, [top-level entries])` when this run carries
        one slice of a cohort: that argument is packed from those entries only,
        and every other argument is prepared whole. Sending the rest whole with
        each batch is what keeps a tool matching landmarks or masks to scans by
        patient name working -- it can still find the patient it is looking at.
        """
        axis, entries = batch if batch else (None, None)
        files = {}
        for arg_name, mode in self._inputModes.items():
            path = self._prepareOneInputFile(
                workspace, arg_name, mode,
                entries=entries if arg_name == axis else None,
            )
            if path is not None:
                files[arg_name] = path
        return files

    def _prepareOneInputFile(self, workspace: slicer_io.TempWorkspace, arg_name: str, mode: str,
                             entries=None):
        # Hidden by its `visible_when`: the argument does not apply to this
        # run, so nothing is uploaded for it — same rule as collectArgs.
        if arg_name in self._hiddenArgs:
            return None
        widget = self._inputWidgets.get(arg_name)
        # Satisfied by a volume already open in the scene: export it and send
        # it like any local file. The node is resolved through the same map
        # the dropdown was filled from (_refreshSceneVolumes).
        volume = getattr(widget, "volume_name", None)
        if volume and volume():
            node = self._sceneVolumes.get(volume())
            if node is None:
                return None
            # The format follows the node's own class: a surface written as
            # `.nii.gz` is a file the tool cannot read.
            extension = ".nii.gz"
            kinds = self.SCENE_INPUTS.get(arg_name) or formgen.scene_kinds_for(
                self._schemaArgument(arg_name), arg_name)
            for kind in kinds:
                node_class, candidate = formgen.SCENE_NODE_KINDS[kind]
                if node.IsA(node_class):
                    extension = candidate
                    break
            return slicer_io.export_node(
                node, self._sceneUploadPath(workspace, arg_name, node, extension)
            )
        # Already satisfied by a MODEL the server hosts: nothing to upload,
        # collectArgs sends its name instead (see _serverSideSelections). A
        # hosted test file never reaches this line - it was downloaded, and the
        # row holds its local path.
        reader = getattr(widget, "server_name", None)
        if reader and reader():
            return None
        # Nothing chosen. That is a legitimate state for an OPTIONAL file
        # argument -- Apply no longer waits for one (see _inputReady) -- and the
        # answer is to upload nothing, so the server applies whatever it does
        # when the argument is absent. Returning `widget.currentPath` here sent
        # the empty string on as a path, and the very next thing to touch it
        # failed with "No such file or directory: ''", naming nothing the user
        # could act on. A required argument cannot reach this line: Apply is
        # disabled until it has a path.
        if mode in ("single_file", "folder_zip", "file_or_folder") and not widget.currentPath:
            return None
        if mode == "single_file":
            return widget.currentPath
        if mode == "folder_zip":
            return self._zipFolder(workspace, arg_name, widget.currentPath, entries)
        if mode == "file_or_folder":
            # HTTP carries no folder: a folder selection goes up as a .zip,
            # which the server extracts (stripping a lone root directory).
            # Which one the user gave is read off the path itself — they never
            # had to declare it, so they cannot have declared it wrong.
            if widget.is_folder():
                return self._zipFolder(workspace, arg_name, widget.currentPath, entries)
            return widget.currentPath
        if mode == "volume_node":
            node = widget.currentNode()
            if node is None:
                return None
            return slicer_io.export_volume(
                node, self._sceneUploadPath(workspace, arg_name, node, ".nii.gz"))
        return None

    def _sceneUploadPath(self, workspace, arg_name: str, node, extension: str) -> str:
        """Where a scene node is written before it is sent: under its own name.

        The server reads the patient and the jaw off the file name -- AREG IOS
        to CBCT pairs a surface with its CBCT by the patient's number and tells
        an upper arch from a lower one by `U`/`L`. Written as `AREG_ios.vtk`,
        a node picked from the scene carried neither, and the run stopped with
        "no patient has both an intraoral scan and a CBCT". One folder per
        argument, so two nodes sharing a name cannot overwrite each other.
        """
        name = slicer_io.upload_name(node.GetName(), extension) or f"{self.TOOL_NAME}_{arg_name}"
        folder = workspace.file(arg_name)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, name + extension)

    def _zipFolder(self, workspace: slicer_io.TempWorkspace, arg_name: str, folder: str,
                   entries=None) -> str:
        """Pack a folder argument. `entries` limits it to one batch of a cohort.

        The archive is built straight out of the user's folder either way, so
        splitting a cohort costs no local disk: there is no per-batch staging
        copy to make, and a laptop sending 20 GB in pieces never holds 40.
        """
        # A tool run on this machine reads the folder where it is: packing it
        # only for the run to unpack it again would double the disk and the time.
        if entries is None and getattr(self.client, "is_local", False):
            return folder
        destination = workspace.file(f"{self.TOOL_NAME}_{arg_name}.zip")
        if entries is None:
            return slicer_io.zip_folder(folder, destination)
        return slicer_io.zip_subset(folder, entries, destination)

    def handleResult(self, result) -> None:
        """Override for custom result display."""
        kind = self.resultKind
        if kind == "text":
            self._announce(result.text or "")
        elif kind in ("segmentation", "labelmap", "volume", "model"):
            slicer_io.load_result(result.path, kind)
        elif kind == "save_as":
            self._handleSaveAsResult(result)

    def _handleSaveAsResult(self, result) -> None:
        """A "save_as" tool may return either one file as-is (e.g. SurgMovPred's
        single predictions_outputs.xlsx) or several files bundled into a .zip
        by the server-side wrapper (since one HTTP response can only carry one
        blob). Only unpack a genuine `.zip` — do NOT sniff the file's bytes for
        a zip signature: .xlsx/.docx/.ods are themselves zip containers
        (OOXML), so that would "extract" a result spreadsheet into raw XML
        parts instead of keeping it as the file it is."""
        extracted = getattr(result, "extracted", None)
        if extracted is not None:
            # Unpacked already, on the job's worker thread, where a result
            # archive expanding far beyond its own size (label volumes
            # compress ~100x) freezes nothing (see `_unpackedOnWorker`).
            resultDir = os.path.dirname(result.path)
            self._producedFiles = list(extracted)
            self._producedRoot = resultDir
            self._mergeRunReport(resultDir)
            self._announce(_("Results saved to {path}").format(path=resultDir))
        elif slicer_io.is_extractable_archive(result.path):
            resultDir = os.path.dirname(result.path)
            # Reached when the worker could not unpack it, or for a result
            # that never went through one. Unpacking here runs on the main
            # thread, so say so before starting rather than letting the panel
            # look frozen again. processEvents is what actually paints it:
            # without it the label is only repainted once the (blocking)
            # extraction is already done.
            self._showPhase(_("Extracting results..."))
            slicer.app.processEvents()
            try:
                self._producedFiles = slicer_io.unzip_folder(result.path, resultDir)
                self._producedRoot = resultDir
            finally:
                self._hideProgress()
            os.remove(result.path)
            self._mergeRunReport(resultDir)
            self._announce(_("Results saved to {path}").format(path=resultDir))
        else:
            self._producedFiles = [result.path]
            self._producedRoot = os.path.dirname(result.path)
            self._announce(_("Result saved to {path}").format(path=result.path))

    # ------------------------------------------------------------------
    # Apply / cancel
    # ------------------------------------------------------------------

    # (glob pattern, the slicer_io kind that opens it). Declared per module,
    # because only the module knows what its outputs ARE: AMASSS's `.nii.gz` is
    # a LABELMAP and opens in colour, while the same extension from another tool
    # is a plain volume in greyscale.
    _LOADABLE = ()

    # A cohort run legitimately returns hundreds of files, and a scene holding
    # hundreds of nodes is not a result anyone can read.
    MAX_RESULTS_TO_LOAD = 12

    # {argument: kinds of scene node it can be filled from}, for what the
    # schema cannot say. `describe.py` publishes no extensions for a packaged
    # tool, so `formgen.scene_kinds_for` has nothing to narrow by and answers
    # nothing -- which is right for a spreadsheet argument and wrong for ALI's
    # `input`, which takes a CBCT or an intraoral surface and lets the tool
    # decide from the data. Same shape and same reason as FILE_INPUTS: an
    # override for what the schema is not able to express yet.
    SCENE_INPUTS = {}

    # The volume-rendering preset a module's scan result should be shown with,
    # or "" for none. Declared per module because only it knows what its output
    # IS: a dental CBCT and an MRI want different curves, and the extension has
    # nothing to tell them apart by -- the file extension certainly cannot.
    #
    # Applied to the FIRST volume a run produced, the same one put in the slice
    # views. A cohort gets one rendering, not forty.
    VOLUME_RENDERING = ""

    # The wording on that check box. The default suits any tool; a module
    # overrides it to name what it actually produces, because "the
    # segmentations" or "the registered volumes" is what a clinician
    # recognises on the panel, not "the results".
    LOAD_RESULTS_LABEL = _("Load the results into the scene when done")

    # Set when the box is built, which only happens for a module that declares
    # _LOADABLE. Nothing else may assume it exists.
    _loadResultsCheckBox = None

    def _addLoadResultsCheckBox(self, layout) -> None:
        """Offer to open the results, for a module that can open anything.

        Eight modules built this by hand, identically apart from the wording,
        and each one repeated the same `isChecked()` guard afterwards. What
        decides whether it appears is `_LOADABLE`: a module that declares no
        pattern has nothing to open, so the box would be a control that does
        nothing whichever way it is set.
        """
        if not self._LOADABLE or self._loadResultsCheckBox is not None:
            # Not None means `addExtraWidgets` already built one -- AREG still
            # does. Adding a second would put two check boxes on the panel, the
            # module's wired to its own handler and this one silently winning
            # the attribute.
            return
        self._loadResultsCheckBox = qt.QCheckBox(self.LOAD_RESULTS_LABEL)
        self._loadResultsCheckBox.setChecked(True)
        layout.addWidget(self._loadResultsCheckBox)

    def _maybeLoadResults(self) -> None:
        """Load what the run produced, unless the user unticked the box.

        Safe to call from a module that never built the box: an absent box
        means nothing was offered, and nothing is loaded.
        """
        if self._loadResultsCheckBox and self._loadResultsCheckBox.isChecked():
            self._loadResults()

    # The report a tool writes beside its results. Empty when the module has
    # none. Named rather than derived from TOOL_NAME, because two spellings are
    # in use across the tools -- `<Tool>_report.json` and `run_report.json` --
    # and a module must be able to say which one it gets.
    RUN_REPORT = ""

    @classmethod
    def _readRunReport(cls, outputDir: str):
        """This run's report, or None when there is not a readable one.

        Never fatal: the results themselves are already on disk and are what
        the user asked for. A missing or malformed report costs them the
        summary, not the run -- which is why every failure here returns None
        instead of raising.

        Looked for at the top of `outputDir` first, then anywhere beneath it: a
        tool that mirrors its input tree files the report beside the results
        rather than at the root. Matches are sorted, so a run that somehow
        produced two reports picks the same one every time.
        """
        path = cls._runReportPath(outputDir)
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s: %s", cls.RUN_REPORT, exc)
            return None

    def _mergeRunReport(self, outputDir: str) -> None:
        """Fold this batch's report into the cohort's, on disk.

        Every batch of a cohort writes the SAME file name into the SAME folder,
        so without this the last one to land is the only report that survives:
        a cohort of forty patients would report the four its final batch held,
        and say nothing at all about the thirty-six before it. That is the
        failure this feature could most easily have introduced -- a run that
        succeeded, results all present, and a summary quietly describing a
        tenth of them.

        The merged report is written back where the report was, so
        `_readRunReport` and every module reading it see one report for the
        cohort with nothing to change. It is also complete at every step: a
        cohort abandoned halfway leaves a report of exactly what ran.
        """
        run = getattr(self, "_runInHand", None)
        if run is None or not run.cohort:
            return
        path = self._runReportPath(outputDir)
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                fresh = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s to merge it: %s", self.RUN_REPORT, exc)
            return

        run.cohort.report = (fresh if run.cohort.report is None
                             else _merged_report(run.cohort.report, fresh))
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(run.cohort.report, handle, indent=2)
        except OSError as exc:
            # The batch's own report stays on disk, which is worse than the
            # merge and better than nothing. Never fatal: the results are there.
            logger.warning("Could not write the merged %s: %s", self.RUN_REPORT, exc)

    @classmethod
    def _runReportPath(cls, outputDir: str):
        """Where this run's report landed, or None if it produced none."""
        if not cls.RUN_REPORT:
            return None
        path = os.path.join(outputDir, cls.RUN_REPORT)
        if os.path.exists(path):
            return path
        found = sorted(glob.glob(
            os.path.join(outputDir, "**", cls.RUN_REPORT), recursive=True))
        if not found:
            logger.warning("No %s was produced by this run", cls.RUN_REPORT)
            return None
        return found[0]

    def _loadResults(self) -> None:
        """Open what THIS run produced.

        Reads `_producedFiles` -- the archive's own member list -- and never the
        output folder. That distinction is the whole reason this exists: results
        are unpacked into the folder the user picked, which is the folder their
        EARLIER runs wrote to as well, so a recursive glob there answers "what
        is in this folder" when the question is "what did this run make". A
        single merged segmentation was reported as fourteen files and refused
        for being too many.
        """
        found = [
            (path, kind)
            for path in self._producedFiles
            if not self._isIntermediate(path)
            for pattern, kind in self._LOADABLE
            if fnmatch.fnmatch(os.path.basename(path), pattern)
        ]
        if not found:
            slicer.util.showStatusMessage(
                _("{tool}: no result file found to load.").format(tool=self.TOOL_NAME), 5000)
            return

        # Two patients loaded into one scene put two CBCTs on top of each
        # other. Counted as PATIENTS, not cases: a T1 and a T2 of one patient
        # are two `scene.json` and still one patient to look at.
        patients = _patients_of(self._producedFiles)
        if len(patients) >= 2 or len(found) > self.MAX_RESULTS_TO_LOAD:
            # Several cases, or too many files for the scene at once: that is
            # what VISU is for. It opens one case at a time, each with what its
            # `scene.json` lists.
            # The folder handed over is the narrowest one holding what THIS run
            # produced, so earlier runs beside it in the output folder are not
            # offered as cases unless they share it.
            folder = os.path.commonpath([os.path.dirname(path) for path, _kind in found])
            if self._openReviewer(folder, None):
                slicer.util.showStatusMessage(
                    _("{tool}: {count} result files, opened in VISU one case at a time.").format(
                        tool=self.TOOL_NAME, count=len(found)), 8000)
                return
            slicer.util.infoDisplay(plain_text(
                _("{count} result files were produced - too many to load at once.\n"
                  "They are all saved in {path}.").format(
                      count=len(found), path=folder)
            ))
            return

        self._hideScene()
        failed = []
        opened = []
        for path, kind in found:
            try:
                opened.append((kind, slicer_io.load_result(path, kind)))
            except Exception as exc:  # one bad file must not lose the others
                failed.append("{}: {}".format(os.path.basename(path), exc))
        self._showLoadedResults(opened)
        if failed:
            slicer.util.errorDisplay(plain_text(
                _("Some results could not be loaded:\n{details}").format(
                    details="\n".join(failed))))

    def _showLoadedResults(self, opened: list) -> None:
        """Put what was just loaded in front of the user.

        Loading a node adds it to the scene; it does not decide what the slice
        views show. ASO's oriented CBCT arrived in the scene and the views kept
        whatever was there before -- to a clinician that reads as "only the
        landmarks loaded", with the 102 MB volume sitting in the Data module
        where nobody thought to look.

        Which layer a result belongs in is what `_LOADABLE`'s kind already
        says: a grey scan is a background, labelled voxels are a label layer.
        The FIRST of each is chosen, in the archive's own order, because a
        cohort of forty offers no better answer than a deterministic one.

        Best effort: the results are on disk and in the scene either way, so a
        view that will not cooperate must not turn a finished run into an
        error.
        """
        for kind, node in opened:
            self._makeVisible(kind, node)

        volumes = [node for kind, node in opened
                   if kind == "volume" and node is not None]
        background = volumes[0] if volumes else None
        label = next((node for kind, node in opened
                      if kind == "labelmap" and node is not None), None)
        if background is None and label is None:
            # Models, markups and segmentations carry their own display node
            # and show themselves; there is no slice layer to put them in.
            return
        try:
            slicer.util.setSliceViewerLayers(
                background=background, label=label, fit=True)
        except Exception as exc:  # noqa: BLE001 - a view is never worth a failure
            logger.warning("Could not show the loaded results: %s", exc)

        # ONE scan, or none. A cohort of forty would have one of them rendered
        # and the other thirty-nine not, with nothing on screen saying which --
        # a 3D view that shows a patient the clinician did not choose is worse
        # than a 3D view that shows nothing. They are all in the slice views
        # and on disk either way.
        if len(volumes) == 1:
            self._renderScan(volumes[0])

    def _renderScan(self, node) -> None:
        """Show one scan in 3D, with the preset this module named.

        Used for a RESULT and for an input the clinician just picked: both are
        one scan going into an empty 3D view, and a module that works on CBCTs
        wants the same curve either way.

        Best effort: the scan is in the scene and in the slice views whatever
        happens here.
        """
        if not self.VOLUME_RENDERING or node is None:
            return
        try:
            slicer_io.show_volume_rendering(node, self.VOLUME_RENDERING)
        except Exception as exc:  # noqa: BLE001 - never worth failing a run
            logger.warning("Could not render %s in 3D: %s", self.VOLUME_RENDERING, exc)

    @staticmethod
    def _hideScene() -> None:
        """Hide everything already in the scene before the results go in.

        The results landed on top of the inputs, the pauses' reviews and every
        earlier run, so a finished run read as a cluttered scene where what it
        had just produced was the hard part to find. See `slicer_io.hide_scene`.
        """
        slicer_io.hide_scene()

    @staticmethod
    def _makeVisible(kind: str, node) -> None:
        """Switch a self-displaying result's display node on.

        Markups, models and segmentations are drawn by their own display node
        rather than by a slice layer, and that node can arrive OFF: a markups
        file carries `display.visibility` and both original CLIs wrote `false`,
        so Slicer built the node, listed it, and drew nothing. The tools write
        `true` now -- but every file produced before that still says otherwise,
        and re-running on one is exactly what a clinician does.

        Asked for explicitly here because the user ticked "load the results":
        that is a request to SEE them, and a file's own preference cannot be
        what decides whether a run appears to have produced anything.
        """
        if node is None or kind not in _SELF_DISPLAYING_KINDS:
            return
        try:
            display = node.GetDisplayNode()
            if display is None:
                node.CreateDefaultDisplayNodes()
                display = node.GetDisplayNode()
            if display is not None:
                display.SetVisibility(True)
        except Exception as exc:  # noqa: BLE001 - never worth failing a run
            logger.warning("Could not show a loaded %s: %s", kind, exc)

    def _isIntermediate(self, path: str) -> bool:
        """Whether this file is a CHAIN's output rather than this run's.

        A tool that calls other tools can be asked to return what they produced
        too; the server files those under `intermediate/<NN>_<tool>/` inside the
        archive. They are there to be looked at deliberately -- a prediction
        checked before the thing built on top of it is trusted -- not to be
        poured into the scene beside the results. Loading them doubles what a
        clinician sees and makes the run's own output the harder half to find.

        The directory name is the server's (`runner.INTERMEDIATE_DIRNAME`), and
        this side cannot import it: written down, and pinned by a test naming
        the same string.

        Any FOLDER of the path counts, not only the first: a tool that sorts
        its chain's outputs by patient returns them under
        `<patient>/intermediate/<NN>_<tool>/`. The file's own name never does,
        so a scan merely called `intermediate_scan` is still a result.
        """
        if not self._producedRoot:
            return False
        try:
            # Both resolved: the archive's members come back as real paths
            # (`slicer_io.unzip_folder`), and an output folder reached through
            # a symlink -- ~/Documents moved to another disk -- made every one
            # of them look outside it, so nothing was filtered at all.
            relative = os.path.relpath(os.path.realpath(path),
                                       os.path.realpath(self._producedRoot))
        except ValueError:  # different drive on Windows; not ours then
            return False
        folders = relative.replace("\\", "/").split("/")[:-1]
        if folders[:1] == [".."]:
            return False  # outside the output folder: not the server's layout
        return _INTERMEDIATE_DIRNAME in folders

    def _suggestOutputFolder(self) -> None:
        """Fill the output folder in, so Apply works on a panel nobody set up.

        Only ever over a path this panel itself proposed. Someone who typed a
        folder chose it, and having it replaced between runs -- silently, with
        results already in it -- is worse than any convenience this buys.
        """
        widget = self._outputFolderWidget
        if widget is None:
            return
        current = widget.currentPath
        if current and current != self._suggestedOutput:
            return
        try:
            self._suggestedOutput = slicer_io.default_output_folder()
        except Exception:  # noqa: BLE001 - a convenience must never break a panel
            logger.warning("could not propose an output folder", exc_info=True)
            return
        # Through the generic writer: `currentPath` is read-only on the picker
        # this row is built from, and assigning it raises rather than writing.
        formgen.set_local_path(widget, self._suggestedOutput)

    def _checkCanApply(self, *_args) -> None:
        if not self.applyButton:
            return  # a widget signal fired while the panel is still being built
        arguments = (self._schema or {}).get("arguments", {})
        canApply = self._inputReady() and formgen.all_required_filled(
            self._argWidgets, arguments, hidden=self._hiddenArgs
        )
        if self.resultKind == "save_as":
            canApply = canApply and bool(self._outputFolderWidget and self._outputFolderWidget.currentPath)
        # One run at a time per panel. A second Apply used to queue behind the
        # first, and with a run paused for review that read as the panel
        # having ignored the reader -- or started a run on half-checked data.
        # Disabled rather than hidden, and the tooltip says why: a greyed
        # button with no reason is a question nobody can answer.
        busy = self._busyReason()
        self.applyButton.enabled = canApply and not busy
        self.applyButton.toolTip = busy

    def _pausedRun(self):
        """The first run of this panel waiting at a checkpoint, or None."""
        return next((run for run in self._runs if run.paused is not None), None)

    def _busyReason(self) -> str:
        """Why Apply is refused right now, in a sentence; "" when it is not.

        The batches of one cohort are one Apply, so they never block each
        other: they are already running when this is asked.
        """
        if not getattr(self, "_runs", None):
            return ""
        paused = self._pausedRun()
        if paused is not None:
            step = getattr(paused.paused, "stopped_after", "") or _("a checkpoint")
            return _("Run {number} is paused after {step} and waits for your review "
                     "in VISU. Continue or cancel it before starting another.").format(
                         number=paused.number, step=step)
        return _("Run {number} is still in progress. Wait for it to finish, or "
                 "cancel it, before starting another.").format(number=self._runs[0].number)

    def _goToReview(self) -> None:
        """Back to VISU, where the paused run is waiting."""
        try:
            slicer.util.selectModule(self.REVIEW_MODULE)
        except Exception as exc:  # noqa: BLE001 - a missing module is reported, not raised
            logger.warning("Could not open %s: %s", self.REVIEW_MODULE, exc)

    def _inputReady(self) -> bool:
        arguments = (self._schema or {}).get("arguments", {})
        # What a `required_when` is read against: the same controlling values
        # the visibility pass reads.
        controlling = formgen.controlling_arguments(arguments)
        values = formgen.collect(
            {name: self._argWidgets[name] for name in controlling if name in self._argWidgets}
        )
        for arg_name, mode in self._inputModes.items():
            # A file input hidden by its `visible_when` is not uploaded either
            # (see _prepareOneInputFile), so it cannot be what Apply waits for.
            if arg_name in self._hiddenArgs:
                continue
            # Neither can an OPTIONAL one. `all_required_filled` has always
            # skipped `required: false` scalars; this loop did not, so any
            # optional file argument disabled Apply until something was picked
            # for it -- with no way to tell from the panel that the field was
            # what Apply was waiting for. AREG is the first tool to have one:
            # its `mgl_landmarks` exists only to REUSE landmarks you already
            # have, since the server predicts them otherwise, and requiring it
            # made the ordinary run the one you could not launch.
            spec = arguments.get(arg_name)
            if spec is not None and not formgen.is_required(spec, values):
                continue
            widget = self._inputWidgets.get(arg_name)
            if mode == "volume_node":
                if widget is None or widget.currentNode() is None:
                    return False
                continue
            if widget is None:
                return False
            # A hosted MODEL satisfies the argument just as well as a local
            # file, and leaves currentPath empty on purpose (see
            # ServerFileInput). A hosted TEST FILE needs no clause of its own:
            # it becomes a local path the moment its download lands, and until
            # then Apply stays disabled, which is the honest state.
            reader = getattr(widget, "server_name", None)
            if reader and reader():
                continue
            # So does a volume already open in the scene: it is exported at
            # upload time (_prepareOneInputFile).
            volume = getattr(widget, "volume_name", None)
            if volume and volume():
                continue
            if not widget.currentPath:
                return False
        return True

    def onApplyButton(self) -> None:
        """Queue this cohort: one run, or one run per batch of it.

        The inputs are read HERE, not when the run starts: what the panel says
        now is what the user asked for. A run that starts three minutes later
        because two others were ahead of it must not silently pick up whatever
        the pickers hold by then. A cohort is divided here for the same reason,
        and every batch is packed now, from the folder as it is now.

        A batch is an ORDINARY run. It queues through `_pumpRuns`, reports on
        its own line, cancels on its own button and fails without taking the
        others -- none of which needed a line of code, because a cohort sent in
        pieces is exactly the cohort a clinician could already queue by hand.
        """
        try:
            args = self.collectArgs()
            batches = self._cohortBatches()
        except Exception as exc:
            slicer.util.errorDisplay(plain_text(str(exc)))
            return

        prepared = []
        try:
            for batch in batches:
                workspace = slicer_io.TempWorkspace()
                workspace.__enter__()
                try:
                    # Called with one argument when there is no batch, which is
                    # every run today: an override written against the old
                    # signature is never handed something it cannot take.
                    files = (self.prepareInputFiles(workspace, batch) if batch
                             else self.prepareInputFiles(workspace))
                    prepared.append((workspace, files))
                except Exception:
                    workspace.__exit__(None, None, None)
                    raise
        except Exception as exc:
            # All or nothing: half a cohort queued and half of it reported as an
            # error is the one outcome nobody can act on.
            for workspace, _files in prepared:
                workspace.__exit__(None, None, None)
            slicer.util.errorDisplay(plain_text(str(exc)))
            return

        sizes = [len(batch[1]) for batch in batches] if batches[0] else []
        cohort = _Cohort(len(prepared), sum(sizes)) if len(prepared) > 1 else None
        if cohort:
            # Asked once per cohort, when it is queued: whether this
            # workstation's batches may run side by side is the server
            # operator's decision, and it can change between two cohorts.
            self._batchPolicy = self.client.batch_policy()
        chosen = (self._outputFolderWidget.currentPath
                  if self._outputFolderWidget else None)
        if cohort:
            # Where the whole cohort lands once its last batch has been
            # merged back up. Kept on the cohort rather than reconstructed
            # later with `dirname`: it is the folder the clinician chose.
            cohort.root = chosen
        for index, (workspace, files) in enumerate(prepared, start=1):
            outputDir = chosen if chosen else workspace.path
            if cohort:
                # A batch writes APART while the cohort runs. Together, a
                # finished batch's results sit among a running one's
                # half-written files with nothing saying which is which --
                # and two batches writing one name overwrite in silence.
                # They are folded back into one folder when the last batch
                # lands, which is when the answer is whole.
                outputDir = os.path.join(outputDir, _batch_dirname(index))
                try:
                    os.makedirs(outputDir, exist_ok=True)
                except OSError as exc:
                    slicer.util.errorDisplay(plain_text(
                        _("Could not create {path}: {error}").format(
                            path=outputDir, error=exc)))
                    return
            self._runsStarted += 1
            inputs = self._reviewInputs(files)
            self._runs.append(_Run(
                self._runsStarted,
                self._runLabel(files, index, cohort.total if cohort else None),
                # A copy per run: one dict shared by five runs is one dict any
                # of them could still be reading when another is written to.
                dict(args), files, outputDir, workspace,
                cohort=cohort, cohort_index=index if cohort else None,
                scan_count=sizes[index - 1] if sizes else 0,
            ))
            self._runs[-1].inputs = inputs
        self._pumpRuns()

    def _cohortBatches(self) -> list:
        """How to divide this run's inputs: `[(axis, [entries]), ...]`, or
        `[None]` for a cohort that travels whole.

        The server decides IF and HOW MUCH (its `GET /tools` `batch` field, and
        see its conventions.py for why a tool pairing two folders is never
        offered here). This decides only whether there is anything to divide:
        an axis that is a folder on this machine, holding more than one batch's
        worth. Everything else -- a single file, a volume picked out of the
        scene, a name the server hosts, a server that publishes no plan at all
        -- is one run, byte for byte what it was before this existed.
        """
        # getattr throughout: this runs before anything else reads the panel's
        # state, so it must hold for a panel whose form was never built -- a
        # server that was down at setup(), or a widget under test.
        plan = (getattr(self, "_schema", None) or {}).get("batch")
        axis = (plan or {}).get("axis")
        if not axis:
            return [None]
        widget = (getattr(self, "_inputWidgets", None) or {}).get(axis)
        if not widget or axis in (getattr(self, "_hiddenArgs", None) or ()):
            return [None]
        # A module that builds its own inputs is doing something no rule here
        # anticipated, and dividing what it produces is a guess about work
        # somebody else wrote. It sends its cohort whole, as it always did.
        # Read off the INSTANCE, so an override assigned to one panel is caught
        # as well as one declared on a class.
        prepare = getattr(self, "prepareInputFiles", None)
        if getattr(prepare, "__func__", None) is not ServerToolWidgetBase.prepareInputFiles:
            return [None]
        # Without somewhere for every batch to write, the results of a divided
        # cohort scatter across per-run temporary folders that are removed as
        # each run ends. One output folder is a precondition, not a detail.
        if not getattr(self, "_outputFolderWidget", None):
            return [None]
        # The same order _prepareOneInputFile reads them in: a scene node or a
        # hosted name wins over the path widget, and neither is a folder here.
        for attribute in ("volume_name", "server_name"):
            reader = getattr(widget, attribute, None)
            if reader and reader():
                return [None]

        folder = getattr(widget, "currentPath", "")
        if not folder or not os.path.isdir(folder):
            return [None]
        batches = slicer_io.split_cohort(
            folder, plan.get("max_mb") or 0, plan.get("max_files") or 0)
        if len(batches) < 2:
            return [None]
        logger.info(
            "'%s': %s holds %d entries, sent as %d batches (<= %s MB, <= %s each)",
            self.TOOL_NAME, axis, sum(len(batch) for batch in batches), len(batches),
            plan.get("max_mb"), plan.get("max_files"),
        )
        return [(axis, entries) for entries in batches]

    def _runLabel(self, files: dict, index=None, total=None) -> str:
        """Name a run after what it was given, so several lines of progress read.

        The tool name alone would make every line of a cohort identical, which
        is exactly when a user needs to tell them apart. Batches of one cohort
        are named after the same folder, so they carry their number too.
        """
        name = self.TOOL_NAME
        for path in files.values():
            if isinstance(path, str) and path:
                # Both separators: Qt hands a Windows path over with forward slashes.
                basename = os.path.basename(path.rstrip("/\\"))
                if basename:
                    name = basename
                    break
        if total:
            return _("{name} ({index}/{total})").format(name=name, index=index, total=total)
        return name

    def _concurrentRuns(self) -> int:
        """How many runs may be in flight at once, never below one.

        The configured number, raised to what the server allows this
        workstation when its operator lets this workstation's batches run in
        parallel. The server enforces its rule either way -- a serial
        workstation's batches run one at a time whatever is sent -- so this
        only decides how much is uploaded ahead of time.
        """
        try:
            configured = max(1, int(config.CONCURRENT_RUNS))
        except (AttributeError, TypeError, ValueError):
            configured = 1
        policy = getattr(self, "_batchPolicy", None) or {}
        if policy.get("batches") == "parallel" and any(run.cohort for run in self._runs):
            try:
                return max(configured, int(policy.get("max_parallel") or configured))
            except (TypeError, ValueError):
                return configured
        return configured

    def _pumpRuns(self) -> None:
        """Start queued runs up to the admission limit.

        This one mechanism serves both shapes a clinician might want, and the
        only thing between them is the number: 1 is a queue that works a cohort
        one patient at a time, N lets N transfers overlap one inference.
        """
        limit = self._concurrentRuns()
        for run in self._runs:
            if run.running:
                continue
            if sum(1 for other in self._runs if other.running) >= limit:
                break
            self._startRun(run)
        self._syncRunControls()

    def _startRun(self, run) -> None:
        def task(progress_cb):
            # `event_cb=progress_cb` funnels the server's progress events into
            # the SAME queue the client's own messages already use. The events
            # arrive on a second thread (the client opens one for the run's
            # event stream, since this thread is blocked inside the POST for
            # the whole inference), and a second cross-thread mechanism is
            # exactly what must not be introduced: BackgroundJob's queue plus
            # its main-thread timer is the one place Qt is touched from.
            # _onJobProgress tells the two apart by type.
            result = self.client.run(
                self.TOOL_NAME,
                args=run.args,
                files=run.files,
                output_dir=run.output_dir,
                progress_cb=progress_cb,
                run_id=run.run_id,
                event_cb=progress_cb,
                cancel_event=run.cancel_event,
                batch=({"id": run.cohort.batch_id, "index": run.cohort_index,
                        "total": run.cohort.total} if run.cohort else None),
            )
            return self._unpackedOnWorker(run, result, progress_cb)

        run.phase = _("Sending request...")
        run.started_at = time.monotonic()
        # `run=run` binds the loop variable at definition time: without it every
        # callback would report against whichever run was queued last.
        run.job = BackgroundJob(
            task,
            on_success=lambda result, run=run: self._onJobSuccess(run, result),
            on_error=lambda exc, run=run: self._onJobError(run, exc),
            on_progress=lambda message, run=run: self._onJobProgress(run, message),
            # The run's own event, not the job's: the watcher inside
            # client.run() reads it too, so cancelling closes the progress
            # stream in the same gesture that stops the work.
            cancel_event=run.cancel_event,
        )
        run.job.start()
        self._startElapsedTimer()

    def onCancelButton(self) -> None:
        """Cancel everything in flight, queued runs included.

        The panel-wide button, kept as the lot: a user who wants out usually
        wants out of all of it. One run out of several is the per-run button
        _rebuildRunCancelButtons puts beside each line.
        """
        self._cancelRuns(list(self._runs))
        slicer.util.showStatusMessage(_("Cancelled."), 3000)

    def _cancelRun(self, run) -> None:
        """Cancel exactly one run, leaving the rest of a cohort alone."""
        self._cancelRuns([run])
        slicer.util.showStatusMessage(
            _("Run {number} cancelled.").format(number=run.number), 3000)

    def _cancelRuns(self, runs) -> None:
        """Withdraw these runs, locally and (where there is one) server-side.

        A run that is still QUEUED here has never been sent, so there is
        nothing on the server to withdraw and NO HTTP call is made for it --
        which also means cancelling a queue works with the server unreachable,
        unplugged or gone.
        """
        server_ids = [run.run_id for run in runs if run.started_at is not None]
        reviews = [run.review_folder for run in runs
                   if run.paused is not None and getattr(run, "review_folder", None)]
        for run in runs:
            run.cancel()
            if run in self._runs:
                self._runs.remove(run)
        self._requestServerCancel(server_ids)
        for folder in reviews:
            self._closeReviewer(folder)
        if not self._runs:
            self._stopElapsedTimer()
        # Admission frees up as these leave, exactly as when one finishes: a
        # cancelled run must let the next queued one start rather than leaving
        # the queue stalled behind it.
        self._pumpRuns()
        self._checkCanApply()

    def _requestServerCancel(self, run_ids) -> None:
        """DELETE /runs/{id} for each, from a thread of its own.

        Off the main thread because a Cancel click must be instant. The call is
        a few milliseconds against a healthy server and up to the connect
        timeout against one that is not answering -- and "not answering" is a
        state in which people press Cancel. Multiplied by a cohort, that is a
        frozen Slicer at the exact moment the user asked to be let go.

        A plain daemon thread rather than a BackgroundJob, deliberately: there
        is no outcome to deliver and nothing to render, so there is nothing for
        the queue-and-timer machinery to carry, and this thread touches neither
        Qt nor the scene. Failure is not reported either -- the panel has
        already released the run, the server reaps an abandoned one on its own
        idle timeout, and there is nothing the user could do with the news.
        """
        if not run_ids:
            return
        client = self.client

        def cancel_all():
            for run_id in run_ids:
                try:
                    client.cancel_run(run_id)
                except Exception:
                    logger.debug("Server-side cancel failed", exc_info=True)

        threading.Thread(target=cancel_all, name="sadt-run-cancel", daemon=True).start()

    def _unpackedOnWorker(self, run, result, progress_cb):
        """Do what a delivery costs on disk HERE, on the job's worker thread,
        and hand the result to `on_success` with it done.

        Unpacking the answer -- a checkpoint for review, a "save_as" archive
        -- and digesting what a reader is about to correct used to run in
        `on_success`, which is the Qt main thread: 1.45 s of a frozen Slicer
        for a 500 MB checkpoint, a cohort's worth for every batch. Nothing
        here touches the scene or a widget, which is the one thing a worker
        may not do (see worker.py).

        Best effort: a failure is left for the main thread to meet again in
        `_unpackCheckpoint` / `_handleSaveAsResult`, which already turn a bad
        archive into what the panel says about one.
        """
        checkpoint = getattr(result, "checkpoint", None)
        try:
            if checkpoint is not None:
                if checkpoint.path and run.output_dir:
                    checkpoint.unpacked = self._unpackCheckpointFiles(run, checkpoint)
            elif (self.resultKind == "save_as" and getattr(result, "path", None)
                    and slicer_io.is_extractable_archive(result.path)):
                progress_cb(_("Extracting results..."))
                result.extracted = slicer_io.unzip_folder(
                    result.path, os.path.dirname(result.path))
                os.remove(result.path)
        except Exception as exc:  # noqa: BLE001 - the main thread reports it
            logger.warning("Could not unpack the answer of run %d on its worker: %s",
                           run.number, exc)
        return result

    def _onJobSuccess(self, run, result) -> None:
        if getattr(result, "checkpoint", None) is not None:
            # The run has not finished: it stopped where it was asked to and
            # the server is holding its work. Neither `_finishRun` nor
            # `_countBatch` therefore -- the run is still this panel's, still
            # cancellable, and its batch has not ended.
            run.job = None
            self._reviewCheckpoint(run, result.checkpoint)
            return
        self._narrateEnd(run, _("Finished"), console.OUTPUT)
        self._finishRun(run)
        self._countBatch(run)
        # Which run `handleResult` is handling. It takes only the result -- the
        # signature every module overrides -- so the run it belongs to travels
        # here, the way `_producedRoot` already does. Cleared on every path: a
        # stale one would make the next ordinary run look like a batch.
        self._runInHand = run
        try:
            with slicer.util.tryWithErrorDisplay(_("Failed to handle the tool result."), waitCursor=False):
                self.handleResult(result)
        finally:
            self._runInHand = None
        # After `handleResult`, so the last batch's own results are on disk
        # and its report has been folded into the cohort's.
        self._finishCohort(run)
        # Move the SUGGESTION on, now that this folder holds a result. The next
        # run then lands beside this one instead of into it, which is the whole
        # reason the folders are numbered. A path the user chose is left alone
        # (see _suggestOutputFolder) -- someone who picked a folder meant it.
        #
        # Not between two batches of one cohort: they were all given the folder
        # the user had at Apply, and moving the suggestion under them would put
        # the next Apply somewhere the current cohort is still writing.
        if not run.cohort:
            self._suggestOutputFolder()
        self._showOutcome(run, _("Finished"))

    def _showOutcome(self, run, outcome: str, finished: bool = True) -> None:
        """Leave how the run ended on the panel, once nothing else is running.

        Hiding the progress line at the end left whatever it said last --
        "Extracting results... 2:30 elapsed" -- whenever a late message put it
        back, and said nothing either way about whether the run had finished.
        """
        if self._runs:
            return
        if run.started_at is not None:
            elapsed = int(time.monotonic() - run.started_at)
            template = (_("{outcome} in {minutes}:{seconds:02d}") if finished
                        else _("{outcome} after {minutes}:{seconds:02d}"))
            outcome = template.format(
                outcome=outcome, minutes=elapsed // 60, seconds=elapsed % 60)
        self._showPhase(outcome)

    def _finishCohort(self, run) -> None:
        """Put a divided cohort back in one folder, once its last batch has
        landed.

        Called from the success path AFTER `handleResult`, not from
        `_countBatch` beside it: the count is taken before the archive is
        unpacked, so a merge there would run while the last batch's results
        were still arriving. And from the failure path too -- a cohort whose
        fourth batch failed still has three batches of results that belong
        together, and leaving them in `batch_01/` is telling a clinician to
        do the merge by hand.
        """
        cohort = getattr(run, "cohort", None)
        if cohort is None or not cohort.complete or not cohort.root:
            return
        self._mergeCohortFolders(cohort.root, cohort.report)

    def _mergeCohortFolders(self, root: str, report=None) -> None:
        """Fold every `<root>/batch_NN` back into `<root>`.

        The batch folders exist so that runs in flight do not write over one
        another; once nothing is in flight they are an obstacle -- a clinician
        looking for a patient should not have to know which batch the transfer
        happened to put them in, and no module reading its own results knows
        this feature exists.

        The report is written LAST and from the cohort's merged copy, not
        moved up with the files. Every batch writes the same report name, so
        whichever one happened to be moved first would otherwise survive as
        the cohort's report while describing one batch of it.
        """
        try:
            names = sorted(os.listdir(root))
        except OSError as exc:
            logger.warning("Could not merge the batches in %s: %s", root, exc)
            return
        folded = 0
        for name in names:
            folder = os.path.join(root, name)
            if not _BATCH_DIRNAME.match(name) or not os.path.isdir(folder):
                continue
            try:
                folded += self._hoist(folder, root)
                shutil.rmtree(folder, ignore_errors=True)
            except OSError as exc:
                # Never fatal: the results are on disk either way, and a
                # cohort left in batch folders is readable. Said out loud so
                # it is not discovered as a folder that should not be there.
                logger.warning("Could not merge %s: %s", folder, exc)
        if report is not None and self.RUN_REPORT:
            try:
                with open(os.path.join(root, self.RUN_REPORT), "w",
                          encoding="utf-8") as handle:
                    json.dump(report, handle, indent=2)
            except OSError as exc:
                logger.warning("Could not write the merged %s: %s",
                               self.RUN_REPORT, exc)
        if folded:
            logger.info("'%s': %d file(s) merged back into %s",
                        self.TOOL_NAME, folded, root)

    @staticmethod
    def _hoist(source: str, destination: str) -> int:
        """Move everything under `source` into `destination`, merging trees.

        A file already at the destination is LEFT there and the one below it
        dropped. Two batches hold different patients, so a collision means
        two runs wrote one name -- and the one already there is what the
        merged report and any loaded node already refer to.
        """
        moved = 0
        for entry in sorted(os.listdir(source)):
            origin = os.path.join(source, entry)
            landing = os.path.join(destination, entry)
            if os.path.isdir(origin):
                os.makedirs(landing, exist_ok=True)
                moved += ServerToolWidgetBase._hoist(origin, landing)
                continue
            if os.path.exists(landing):
                logger.warning("Two batches produced %s; kept the first", entry)
                continue
            shutil.move(origin, landing)
            moved += 1
        return moved

    def _countBatch(self, run, succeeded: bool = True) -> None:
        """Record that one batch of a cohort has ended, and how.

        Failures count towards `finished`: it asks whether anything is still
        coming, not whether everything worked. A cohort whose third batch failed
        must still say its last word when the fourth lands.

        The SCANS are counted apart, and done apart from failed, because that
        is the number on the panel: "16 of 20 scans" must never include four a
        batch lost. A batch that failed is reported as failed, not as absent.
        """
        if not run.cohort:
            return
        run.cohort.finished += 1
        if succeeded:
            run.cohort.scans_done += run.scan_count
        else:
            run.cohort.scans_failed += run.scan_count

    def _announce(self, message: str) -> None:
        """Tell the user something, in a dialog they have to dismiss.

        Once per COHORT, not once per batch: five modal dialogs for one Apply
        are four clicks nobody asked for, each interrupting the upload of the
        next batch. The batches before the last say the same thing in the status
        bar, which is where a running commentary belongs.
        """
        run = getattr(self, "_runInHand", None)
        if run is not None and run.cohort and not run.cohort.complete:
            slicer.util.showStatusMessage(message, 3000)
            return
        slicer.util.infoDisplay(plain_text(message))

    def _onJobError(self, run, exc) -> None:
        """One run failing takes only that run: the rest of a cohort goes on."""
        if isinstance(exc, RunCancelled):
            self._narrateEnd(run, _("Cancelled"), console.STEP)
        else:
            self._narrateEnd(run, _("Failed: {reason}").format(
                reason=plain_text(str(exc))), console.ERROR)
        self._finishRun(run)
        self._countBatch(run, succeeded=False)
        # A cohort whose last batch failed still has the earlier ones to put
        # together, and it is over either way.
        self._finishCohort(run)
        if isinstance(exc, RunCancelled):
            # 499: the user asked for this. A cancellation is not a failure and
            # must never open an error dialog -- the panel simply closes the
            # run, which is what the user pressed the button to see. It can
            # arrive without anyone having pressed anything here (someone else
            # holding the run id, an operator stopping a job on the server), so
            # it is answered on its own merits rather than by checking whether
            # we were the ones who asked.
            logger.info("Run %d of '%s' was cancelled", run.number, self.TOOL_NAME)
            slicer.util.showStatusMessage(
                _("Run {number} cancelled.").format(number=run.number), 3000)
            self._showOutcome(run, _("Cancelled"), finished=False)
            return
        self._showOutcome(run, _("Failed"), finished=False)
        slicer.util.errorDisplay(plain_text(str(exc)))

    # ------------------------------------------------------------------
    # Quality control: a run that stopped for somebody to look at it
    # ------------------------------------------------------------------

    # The Slicer module a checkpoint is reviewed in, by NAME and resolved only
    # when one is reached. That module depends on this library and not the
    # other way round, so importing it at the top would be a cycle -- and a
    # deployment that ships without it must still run every tool it does have.
    REVIEW_MODULE = "SADTVisu"

    def _reviewCheckpoint(self, run, checkpoint) -> None:
        """Put what the run produced so far in front of a reader.

        Everything specific to the reviewer is on the other side of
        `_openReviewer`: this method knows a folder and a callback, exactly
        what the viewer's own entry point takes.
        """
        run.paused = checkpoint
        run.clear_server_progress()
        run.phase = _("Stopped after {step} — waiting for your review").format(
            step=checkpoint.stopped_after or _("a checkpoint"))
        self._narrateStep(run)
        self._syncRunControls()

        folder = self._unpackCheckpoint(run, checkpoint)
        run.review_folder = folder
        opened = folder is not None and self._openReviewer(
            folder, lambda reviewed, run=run: self._onReviewed(run, reviewed),
            rewind=self._previousCorrectableStep(run),
            origin={"tool": self.TOOL_NAME,
                    "step": checkpoint.stopped_after or "",
                    "run": run.number},
            inputs=run.inputs,
            focus=self._focusFolder(folder, checkpoint) if folder else None)
        if opened:
            return
        # Nothing to look at, or nowhere to look at it. The run is PAUSED on
        # the server, holding its job directory with a patient's data in it,
        # and carrying it on at once is the only answer that does not leave it
        # there until the reaper. Said out loud: the reader asked for a stop
        # and is not getting one.
        self._announce(_(
            "'{tool}' stopped after {step}, but there is nothing to review "
            "here — carrying on.").format(
                tool=self.TOOL_NAME, step=checkpoint.stopped_after or "?"))
        self._resumeRun(run, {})

    # What a reader may do at a stop, as the server publishes it on the
    # `stop_after` argument. Only these two are somewhere to go BACK to:
    # looking at a result is not a way to change the thing that caused it.
    EDITABLE_KINDS = ("landmarks", "registration")

    def _previousCorrectableStep(self, run):
        """The nearest stop behind this one a reader could actually change.

        Looking at a bad orientation is useless without a way back to the
        landmarks that caused it, so stops that can only be LOOKED at are
        stepped over and the offer lands where something can be done.

        `produced` is the server's own folder names, in the order the steps
        ran -- `("01_ASO", "02_ALI_CBCT")` -- so walking it backwards is
        walking the run backwards. The kind comes from the schema, which the
        server composed off the tool that WROTE each step.

        Returns `{"slot": ..., "tool": ..., "kind": ...}`, or None when there
        is nothing correctable behind the current stop.
        """
        checkpoint = getattr(run, "paused", None)
        if checkpoint is None:
            return None
        kinds = ((getattr(self, "_schema", None) or {}).get("arguments", {})
                 .get("stop_after", {}).get("option_kind") or {})
        behind = list(checkpoint.produced or ())
        # The step the reader is standing on is not somewhere to go BACK to.
        # It is the last one that ran, and offering it would hand them a
        # button that returns to where they already are. A qualified stop
        # (`ASO/ALI_CBCT`) is named by its last segment, as the slots are.
        standing = (checkpoint.stopped_after or "").rsplit("/", 1)[-1]
        for index in range(len(behind) - 1, -1, -1):
            if behind[index].partition("_")[2] == standing:
                del behind[index]
                break
        for slot in reversed(behind):
            # "01_ALI_CBCT" -> "ALI_CBCT". The number is the call's position,
            # which is what keeps two calls to one tool apart; the kind is a
            # property of the tool, not of the position.
            _number, _sep, tool = slot.partition("_")
            kind = kinds.get(tool, "view")
            if kind in self.EDITABLE_KINDS:
                return {"slot": slot, "tool": tool, "kind": kind}
        return None

    def _openReviewer(self, folder: str, on_continue, rewind=None,
                      origin=None, inputs=(), focus=None) -> bool:
        """Hand `folder` to the review module. False when it could not be.

        `rewind` is where flagged patients may be sent BACK to, or None. The
        reviewer decides what to offer from it; this side only knows which
        step it was.

        `origin` says which run is waiting, so the reviewer can tell a reader
        who pressed Apply in a tool panel and found themselves somewhere else
        where they are.

        `inputs` are the run's own files, which the reviewer uses only to put
        landmarks that came back alone on the scan they were placed on.

        `focus` is the folder of the step that stopped, which the reviewer
        shows first (see `_focusFolder`).
        """
        try:
            module = importlib.import_module(self.REVIEW_MODULE)
            return bool(module.open_for_review(folder, on_continue, rewind=rewind,
                                               origin=origin, inputs=tuple(inputs or ()),
                                               focus=focus))
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            logger.warning("Could not open '%s' on %s: %s",
                           self.REVIEW_MODULE, folder, exc)
            return False

    def _closeReviewer(self, folder: str) -> bool:
        """End the review of `folder` in the review module, when it has one.

        A run withdrawn while it waited left the reviewer saying it was
        paused, with a Continue that would resume nothing.
        """
        try:
            module = importlib.import_module(self.REVIEW_MODULE)
            end = getattr(module, "end_review", None)
            return bool(end(folder)) if end is not None else False
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            logger.warning("Could not end the review of %s in '%s': %s",
                           folder, self.REVIEW_MODULE, exc)
            return False

    def _focusFolder(self, folder: str, checkpoint):
        """Where the step this run stopped after wrote its files, or None.

        The last of `produced` is the step that just ran. A stop INSIDE a
        callee (`ASO/ALI_CBCT`) is one level further: the callee's own
        `intermediate/<NN>_ALI_CBCT/` under the outer step's folder.
        """
        produced = list(getattr(checkpoint, "produced", None) or ())
        if not produced:
            return None
        path = self._checkpointSlots(folder, produced).get(produced[-1])
        if not path:
            return None
        stop = getattr(checkpoint, "stopped_after", "") or ""
        if "/" in stop:
            inner = stop.rsplit("/", 1)[-1]
            nested = os.path.join(path, _STEPS_DIRNAME)
            if os.path.isdir(nested):
                for name in sorted(os.listdir(nested)):
                    if name.partition("_")[2] == inner:
                        return os.path.join(nested, name)
        return path

    def _reviewInputs(self, files: dict) -> tuple:
        """Where this run's inputs are on this machine, for a reviewer.

        The folder or file the reader picked, not the archive built from it:
        unpacking a cohort a second time to show one patient's scan is gigabytes
        for nothing. A scene node has no file of its own, so its export -- in
        `files`, alive as long as the run -- stands for it.
        """
        found = []
        widgets = getattr(self, "_inputWidgets", None) or {}
        for arg_name, path in (files or {}).items():
            widget = widgets.get(arg_name)
            chosen = getattr(widget, "currentPath", "") or ""
            location = chosen if chosen and os.path.exists(chosen) else path
            if location and os.path.exists(location) and location not in found:
                found.append(location)
        return tuple(found)

    @staticmethod
    def _checkpointFolderName(run, checkpoint) -> str:
        """The directory ONE stop of ONE run is reviewed in.

        Two levels, and both are load-bearing.

        The RUN comes first because one Apply is several runs: a cohort over
        the server's published batch size is divided, and every batch is an
        ordinary run with its own checkpoint -- into the same output folder.
        Naming a review after the step alone made all of them the same
        directory, so batch 2 landed on batch 1, and batch 2 may well stop
        while a reader is still looking at batch 1. Batches are numbered as
        the progress lines number them, so the folder and the line a reader
        was watching say the same thing.

        The STEP comes second because one run can stop more than once, which
        is what this whole split is for.

        `ASO/ALI_CBCT` is a legal stop name and not a legal directory name,
        so its separator is folded rather than nested: nested, one stop's
        folder would sit inside another's, which is the mixing this prevents.
        """
        index = getattr(run, "cohort_index", None)
        # A run that was not divided is not called a batch: there is no
        # second one to tell it apart from.
        where = (_batch_dirname(index) if index
                 else "run_%02d" % (getattr(run, "number", 0) or 0))
        name = (getattr(checkpoint, "stopped_after", "") or "").strip()
        step = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
        # A stop that named itself nothing still gets its own folder rather
        # than the parent, which `shutil.rmtree` below would then empty.
        return os.path.join(where, step or "checkpoint")

    def _unpackCheckpoint(self, run, checkpoint):
        """Unpack what the stopped run produced, and say where. None if empty.

        Under the run's own output folder rather than a temporary directory:
        the reader is about to CORRECT these files, and a correction that
        disappears with the panel is worse than no correction at all.

        **One folder per stop, per run.** Every checkpoint used to unpack
        into the same directory -- across the stops of one run AND across the
        batches of one cohort, which share an output folder. So a reader was
        handed everything every earlier review had already covered, and the
        baseline below was retaken over the mixture, which made a file from
        an earlier stop, edited during a later review, travel back as a
        correction of a step the run had already left. A review now holds
        exactly what the stop it belongs to produced.

        Emptied before unpacking, for the same reason one directory down: a
        run sent BACK to a step it already stopped at is answered with what
        that step produced this time, and files the previous pass left would
        otherwise read as part of it.

        The digest of everything unpacked is taken as it is unpacked, before
        the reader can touch any of it, and that timing is the whole mechanism:
        it is the only moment at which the folder is known to hold exactly what
        the server produced.
        """
        # Cleared first, so a checkpoint that fails to unpack cannot leave the
        # PREVIOUS one's baseline in place -- against which every file of a
        # second step would read as changed.
        run.checkpoint_digests = {}
        if not checkpoint.path or not run.output_dir:
            return None
        # Done already on the job's worker thread, in the normal case (see
        # `_unpackedOnWorker`); here only for a checkpoint that did not go
        # through one, or that the worker failed to unpack.
        unpacked = getattr(checkpoint, "unpacked", None)
        if unpacked is None:
            try:
                unpacked = self._unpackCheckpointFiles(run, checkpoint)
            except Exception as exc:  # noqa: BLE001 - a bad archive is not a crash
                logger.warning("Could not unpack the checkpoint of run %d: %s",
                               run.number, exc)
                return None
        folder, run.checkpoint_digests = unpacked
        return folder

    def _unpackCheckpointFiles(self, run, checkpoint) -> tuple:
        """`(folder, digests)`: the checkpoint unpacked where its review goes,
        and the digest of every file in it. Disk work only -- no widget, no
        scene, nothing of `run` changed -- so the job's worker thread runs it.
        Raises on an archive that cannot be unpacked."""
        folder = os.path.join(run.output_dir, _CHECKPOINT_DIRNAME,
                              self._checkpointFolderName(run, checkpoint))
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        slicer_io.unzip_folder(checkpoint.path, folder)
        os.remove(checkpoint.path)
        return folder, digest.digest_tree(folder)

    @staticmethod
    def _checkpointSlots(folder: str, produced) -> dict:
        """{step name: the directory holding what that step produced}.

        The archive has two shapes and the server picks between them by
        counting: its zip flattens a SINGLE directory to the archive root, so
        one step's files arrive at the top while two steps' arrive under
        `01_X/` and `02_Y/`. Answered here rather than anywhere the difference
        could be read as a step that went missing.
        """
        # The steps sit under `intermediate/` in what the server sends today
        # (`intermediate/01_ASO/`, `intermediate/02_Crown_Seg/`), and at the
        # top in what it sent before. Looked for in both: found in neither,
        # every correction a reader made was silently dropped, and the run
        # carried on with the landmark they had just moved.
        for root in (os.path.join(folder, _STEPS_DIRNAME), folder):
            slots = {name: os.path.join(root, name) for name in produced
                     if os.path.isdir(os.path.join(root, name))}
            if slots:
                return slots
        if len(produced) == 1 and os.path.isdir(folder):
            return {produced[0]: folder}
        return {}

    def _onReviewed(self, run, reviewed) -> None:
        """The reader pressed Continue, or asked to go back a step.

        Both send the same thing -- whatever they changed here -- and differ
        only in which direction the run then moves. Going back does not throw
        that away: a reader who corrected something on the way to asking for
        an earlier step still corrected it.
        """
        if run.paused is None:
            # Continue on a run that is no longer stopped: it was cancelled,
            # or the panel was torn down while the reader was working. There
            # is nothing left here to carry on.
            logger.info("A review came back for a run that is no longer stopped")
            return
        self._returnFromReview()
        self._resumeRun(run, self._corrections(run, reviewed),
                        rewind_to=(reviewed or {}).get("rewind_to"))

    def _returnFromReview(self) -> None:
        """Bring this panel back up, now that the reader has finished.

        The run carries on HERE -- the progress line, the elapsed time and
        whatever comes back are all on this panel -- and a reader left in the
        reviewer sees none of it. They pressed Continue and then watched a
        viewer do nothing.

        Best effort: failing to switch module must not cost the resume that
        is already under way.
        """
        # `moduleName` when Slicer set it, and the class name otherwise --
        # `ASORemoteWidget` is the `ASORemote` module. NOT `TOOL_NAME`: that is
        # the SERVER's name for the tool, and the two now differ everywhere,
        # since every module carries the `Remote` suffix that keeps it from
        # colliding with AutomatedDentalTools while the tool it calls is still
        # plain `ASO`.
        name = getattr(self, "moduleName", "") or type(self).__name__
        if name.endswith("Widget"):
            name = name[: -len("Widget")]
        try:
            slicer.util.selectModule(name)
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            logger.warning("Could not return to %s: %s", name, exc)

    def _corrections(self, run, reviewed) -> dict:
        """{step name: a zip of the FILES that step's folder changed}, or nothing.

        A step is a cohort's worth of scans; the landmark file a reader moved
        one point in is eight kilobytes. So what goes back is the difference,
        measured against the digest taken when the checkpoint was unpacked --
        changed files and new ones, nothing else. A step nobody touched
        contributes no field at all, and a review that changed nothing
        anywhere sends an empty body, which the server already accepts.

        **This is only safe because the server lays a correction over the
        step's output file by file** (`_Supervisor._substitute`), rather than
        replacing the directory with what arrives. Against the older server
        that swapped the whole directory, a partial set would delete every
        file the reader did not touch.

        The viewer's `written` is still read, and still first: it is the one
        answer available without touching the disk, and it spares the reader
        who only looked a second pass over a cohort to prove that nothing
        moved. It cannot stand in for the digest, though -- it names a
        PATIENT, and what has to be named here is a file.
        """
        if not reviewed.get("written"):
            return {}
        folder = reviewed.get("folder") or ""
        changed = digest.changed_since(run.checkpoint_digests, folder)
        corrections = {}
        for slot, path in self._checkpointSlots(folder, run.paused.produced).items():
            entries = digest.paths_under(changed, folder, path)
            if entries:
                corrections[slot] = self._zipFolder(run.workspace, slot, path, entries)
        return corrections

    def _resumeRun(self, run, corrections, rewind_to=None) -> None:
        """POST the corrections and move the run, on a thread of its own.

        `rewind_to` sends it BACKWARDS to a checkpoint it already cleared,
        instead of onwards. The callbacks are the same either way, and so is
        the answer: a finished run, or another checkpoint.

        The same callbacks as a first attempt, deliberately: the answer of a
        resume is whatever a finished run answers -- or ANOTHER checkpoint,
        when a second one was armed -- so the loop is one recursion rather
        than a second way of handling a result.
        """
        checkpoint, run.paused = run.paused, None
        run.clear_server_progress()
        run.phase = (_("Going back to {step}...").format(step=rewind_to)
                     if rewind_to else _("Carrying the run on..."))

        def task(progress_cb):
            result = self.client.resume_run(
                self.TOOL_NAME,
                checkpoint.run_id,
                corrections=corrections,
                output_dir=run.output_dir,
                progress_cb=progress_cb,
                rewind_to=rewind_to,
                # The same callback a first attempt gets, so the steps after
                # the review are narrated like the ones before it.
                event_cb=progress_cb,
                cancel_event=run.cancel_event,
            )
            return self._unpackedOnWorker(run, result, progress_cb)

        run.job = BackgroundJob(
            task,
            on_success=lambda result, run=run: self._onJobSuccess(run, result),
            on_error=lambda exc, run=run: self._onJobError(run, exc),
            on_progress=lambda message, run=run: self._onJobProgress(run, message),
            cancel_event=run.cancel_event,
        )
        run.job.start()
        # `started_at` is NOT reset: what the panel counts is how long the
        # clinician has been waiting for this run, and the review is part of
        # that wait.
        self._startElapsedTimer()
        self._syncRunControls()

    def _onJobProgress(self, run, payload) -> None:
        """What the run has to say, from either side of the wire.

        A dict is one of the server's progress events, delivered by the
        watcher thread through the job's own queue; anything else is the
        client narrating what IT is doing. One channel for both, because
        BackgroundJob's queue plus its main-thread timer is the only mechanism
        in this file allowed to cross a thread boundary, and a second one would
        be a second way to get Qt wrong.
        """
        if is_log_event(payload):
            # Before the progress branch, and never falling into it: a log
            # line says nothing about where the run is, so it must not touch
            # the phase, the message, the fraction or a cohort's count.
            self._onRunLog(run, payload)
            return
        if isinstance(payload, dict):
            self._onRunEvent(run, payload)
            return
        # Kept as the phase, not printed once and forgotten: the elapsed-time
        # tick below re-renders it every second, so the panel keeps saying what
        # it is doing rather than showing a message frozen minutes ago.
        run.phase = payload
        # The client has moved past whatever the server last said -- "Processing
        # response" comes after "packaging" -- so the older half is dropped
        # rather than left on the panel beside the newer one.
        run.clear_server_progress()
        self._narrateStep(run)
        self._renderProgress()

    def _onRunEvent(self, run, event) -> None:
        """One progress event from the server (see client.normalise_run_event).

        Nothing is logged from it. A progress message is written by a tool and
        may name a file, which on this extension's data means it may name a
        patient; it is rendered on the panel of the person who started the run
        and goes nowhere else.
        """
        run.server_phase = event.get("phase") or ""
        run.server_message = event.get("message") or ""
        run.fraction = event.get("fraction")
        run.depth = event.get("depth") or 0
        call = event.get("call")
        if call and event.get("tool"):
            run.call_tools[call] = event["tool"]
        # Only below the root: the tool the user asked for is already named by
        # the panel it is running in.
        run.nested_tool = run.call_tools.get(call) if run.depth and call else None
        # Not for a call's open or close marker: it names the sub-tool and says
        # nothing it is doing, and the sub-tool's first real step follows it.
        if not event.get("edge"):
            self._narrateStep(run)
        self._renderProgress()

    def _onRunLog(self, run, event) -> None:
        """One of the tool's log lines for the user, printed to the Python console.

        The console and nothing else: no widget of its own, and never Python's
        `logging`, which would also send it to Slicer's log file -- the line
        was written by a tool about a clinician's data, and the console is
        read by the person who started the run, on their own workstation.
        Prefixed with this panel's tool, since every panel prints to the same
        console.
        """
        line = "[{}] {}".format(self.consoleName(run),
                                format_run_log_line(event, self._runPrefix(run)))
        level = event.get("level")
        color = {"warning": console.WARNING, "error": console.ERROR,
                 "metric": console.METRIC}.get(level, console.OUTPUT)
        self._consoleBreak("metric" if level == "metric" else "line",
                           section=event.get("source"))
        # `print` as the fallback only: what it writes is also copied into
        # Slicer's log file, which is exactly where this line must not go.
        if not console.show(line, color):
            print(line)

    def _consoleBreak(self, kind: str, section=None) -> None:
        """A blank line where the console's story changes paragraph.

        Before a run's first line and its last, and around a block of metrics,
        so a run reads as steps, then its numbers, then its outcome, instead of
        one undifferentiated column. `kind` is what is about to be written:
        "start", "line", "metric" or "end". `section` is the sub-tool a line
        comes from, None for the tool itself: entering or leaving a sub-call
        starts a paragraph too.
        """
        previous = self.__dict__.get("_lastConsoleKind")
        if previous is not None and (
                kind in ("start", "end")
                or (kind == "metric") != (previous == "metric")
                or section != self.__dict__.get("_lastConsoleSection")):
            console.show("", console.OUTPUT)
        self._lastConsoleKind = "line" if kind in ("start", "end") else kind
        self._lastConsoleSection = section

    def consoleName(self, run) -> str:
        """The name a run's console lines carry: the tool, by default.

        A module whose one tool routes to several on the server -- AREG sends
        an "IOS to CBCT" run to AREG_IOSCBCT -- overrides this with the one
        actually running, which is the name a reader of the console needs.
        """
        return self.TOOL_NAME

    def _runPrefix(self, run) -> str:
        """"Batch 2" or "Run 7" when several runs share the console, else "".

        Named the way the run's own progress line names it, so a reader can
        match the two.
        """
        if len(self._runs) <= 1:
            return ""
        return (_("Batch {index}").format(index=run.cohort_index)
                if run.cohort_index else
                _("Run {number}").format(number=run.number))

    def _narrate(self, run, text: str, color: str) -> None:
        """One line of the run's story in the console: `[TOOL] HH:MM:SS  text`.

        Through `console.show` and never `print` or `logging`: a step can name
        what the tool is working on, and the console is read by the person who
        started the run, on their own workstation, while Slicer's log file is
        sent around when something goes wrong.
        """
        parts = ["[{}]".format(self.consoleName(run))]
        prefix = self._runPrefix(run)
        if prefix:
            parts.append(prefix + ":")
        parts.append(time.strftime("%H:%M:%S"))
        parts.append(text)
        console.show("  ".join(parts), color)

    def _consoleStepText(self, run) -> str:
        """The step for the console: what is happening, and in which sub-tool.

        Plainer than the panel's line: no phase word when the tool said what it
        is doing, no dash, no arrows. A sub-call is named instead, and set off
        by a blank line (see `_consoleBreak`), which is what makes a chain read.
        """
        if not run.server_phase:
            return (run.phase or "").replace(" — ", ": ")
        text = run.server_message or self._phaseLabel(run.server_phase)
        if run.depth and run.nested_tool:
            text = "{}: {}".format(run.nested_tool, text)
        return text

    def _narrateStep(self, run) -> None:
        """Write the step the run is at, once, when it changes.

        The same words as the panel's progress line, without the percentage: a
        bar moving from 35 % to 36 % is not a new step, and neither is the next
        scan of a cohort (see `step_key`).
        """
        text = self._consoleStepText(run)
        key = step_key(text)
        if not key or key == run.narrated_step:
            return
        self._consoleBreak("start" if run.narrated_step is None else "line",
                           section=run.nested_tool if run.depth else None)
        run.narrated_step = key
        self._narrate(run, text, console.STEP)

    def _narrateEnd(self, run, outcome: str, color: str) -> None:
        """The run's last console line: what happened and how long it took."""
        if run.started_at is not None:
            outcome = "{} ({})".format(
                outcome, format_duration(time.monotonic() - run.started_at))
        self._consoleBreak("end")
        self._narrate(run, outcome, color)

    def _finishRun(self, run) -> None:
        run.job = None
        run.close()
        if run in self._runs:
            self._runs.remove(run)
        if not self._runs:
            self._stopElapsedTimer()
        # Admission frees up as this one leaves, so the next queued run starts
        # here -- that is what makes a cohort walk itself.
        self._pumpRuns()
        self._checkCanApply()

    def _syncRunControls(self) -> None:
        """Apply stays visible whatever is running, disabled while a run is in
        flight (see `_checkCanApply`)."""
        review = getattr(self, "reviewButton", None)
        if review is not None:
            review.setVisible(self._pausedRun() is not None)
        if self.cancelButton is not None:
            self.cancelButton.setVisible(bool(self._runs))
            if self._cohortInFlight() is not None:
                self.cancelButton.setText(_("Cancel the cohort"))
            else:
                self.cancelButton.setText(
                    _("Cancel all") if len(self._runs) > 1 else _("Cancel"))
        if self.applyButton is not None:
            self.applyButton.setVisible(True)
            # A run starting or ending changes whether Apply may be pressed.
            # Only on a built panel: the form it reads is made by setup().
            if hasattr(self, "_argWidgets"):
                self._checkCanApply()
        self._rebuildRunCancelButtons()
        self._renderProgress()

    def _cohortInFlight(self):
        """The one cohort every run in flight belongs to, or None.

        All of them, deliberately. A cohort plus an unrelated run queued behind
        it is not a cohort any more -- it is a queue that happens to contain
        one -- and drawing it as one would put a stranger's progress inside the
        cohort's box and its scans outside the count.
        """
        cohorts = {id(run.cohort): run.cohort for run in self._runs}
        if len(cohorts) != 1:
            return None
        cohort = next(iter(cohorts.values()))
        return cohort if cohort is not None else None

    def _buildCohortView(self, layout, cohort):
        """The cohort's own progress box, built once per change to the run set.

        Built here rather than in the one-second tick for the reason the Cancel
        buttons were: a widget rebuilt under the pointer is a widget the user
        was about to interact with. The tick only writes values into these.
        """
        frame = design.cohort_frame()
        inner = qt.QVBoxLayout(frame)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(design.SPACING_XS)

        total = design.cohort_total_label("")
        bar = design.cohort_bar()
        inner.addWidget(total)
        inner.addWidget(bar)

        # `_runs` is in queue order and a finished run leaves it, so the first
        # entries are the ones in flight and the rest are what comes next. A
        # cohort of a hundred scans is twenty-five batches; listing them all
        # would make the queue the tallest thing on the panel and say nothing
        # the headline count does not.
        shown = self._runs[:design.MAX_BATCH_ROWS]
        rows = {}
        for run in shown:
            label = design.batch_label("")
            batch_bar = design.batch_bar()
            inner.addWidget(label)
            inner.addWidget(batch_bar)
            rows[run.number] = (label, batch_bar)

        remainder = None
        if len(self._runs) > len(shown):
            # Counted, not listed. What a reader needs from the batches beyond
            # the fold is that they exist and how many -- the rest is the
            # headline's job.
            remainder = design.batch_label("")
            inner.addWidget(remainder)

        frame.setVisible(True)
        layout.addWidget(frame)
        return _CohortView(frame, total, bar, rows, cohort, remainder)

    def _rebuildRunCancelButtons(self) -> None:
        """One Cancel per run -- but only once there is more than one run.

        With a single run the panel's own Cancel button already cancels exactly
        that run, and a second button saying the same thing under it is noise;
        the same reasoning that keeps _describeRun from prefixing a lone line
        with "Run 1". With a cohort it is the difference between abandoning the
        patient that is stuck and abandoning the afternoon.

        Rebuilt wholesale into a fresh child rather than by clearing a layout,
        which is the swap _buildForm already makes and for the same reason: no
        widget of the previous set survives, so a button can never outlive the
        run it was bound to. Called only from _syncRunControls, i.e. when the
        set of runs actually changes -- never from the one-second render tick,
        which would destroy a button under the pointer that is about to click
        it.
        """
        if self._runControlsLayout is None:
            # A panel built without setup() (the unit tests do exactly that)
            # has nowhere to put them, and nothing to show them on.
            return

        host = qt.QWidget()
        layout = qt.QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(design.SPACING_XS)
        cohort = self._cohortInFlight()
        self._cohortView = None
        if cohort is not None:
            # A cohort gets NO per-batch Cancel. Abandoning batch 3 of 5 leaves
            # a run whose results cover an arbitrary part of the cohort and
            # whose report says so in a footnote -- an outcome nobody wants and
            # which the panel should not offer. One Apply, one thing to stop.
            self._cohortView = self._buildCohortView(layout, cohort)
        elif len(self._runs) > 1:
            for run in self._runs:
                button = design.compact_danger_button(
                    _("Cancel run {number} ({label})").format(
                        number=run.number, label=run.label)
                )
                # `run=run` for the same reason the job callbacks do it: bound
                # at definition time, or every button cancels the last run.
                # `_checked` swallows the bool a QPushButton's clicked signal
                # sends, which PythonQt passes positionally.
                button.clicked.connect(
                    lambda _checked=False, run=run: self._cancelRun(run))
                layout.addWidget(button)

        self._runControlsLayout.addWidget(host)
        previous = self._runControlsWidget
        self._runControlsWidget = host
        if previous is not None:
            # Reparenting is what takes it out of the layout; the deletion is
            # deferred because this can run from a signal emitted by one of
            # these very buttons -- a button that cancels its own run destroys
            # the set it belongs to, and it has to survive the click it is
            # handling.
            previous.setVisible(False)
            previous.setParent(None)
            previous.deleteLater()

    # ------------------------------------------------------------------
    # "Still working" feedback
    # ------------------------------------------------------------------

    def _startElapsedTimer(self) -> None:
        """Tick once a second for as long as any run is active.

        The worker thread cannot report progress while it is blocked inside a
        single HTTP request, and that request IS the run -- minutes of remote
        inference with no bytes flowing either way. Only a main-thread timer
        can show the panel is alive during it, and without one the run looks
        hung: an AMASSS run was cancelled at three minutes because of this,
        having done nothing wrong and with 40 seconds left to go.

        Idempotent: one timer renders every run, and starting a second run must
        not leave the first one's timer running unowned.
        """
        if self._elapsedTimer is None:
            self._elapsedTimer = qt.QTimer()
            self._elapsedTimer.setInterval(1000)
            self._elapsedTimer.timeout.connect(self._renderProgress)
            self._elapsedTimer.start()
        self._renderProgress()

    def _stopElapsedTimer(self) -> None:
        if self._elapsedTimer:
            self._elapsedTimer.stop()
            self._elapsedTimer = None
        self._hideProgress()

    def _previewPickedFile(self, arg_name: str) -> None:
        """Put what the user just picked in the scene, once."""
        widget = self._inputWidgets.get(arg_name)
        path = getattr(widget, "currentPath", "") if widget is not None else ""
        if self._showInScene(arg_name, path):
            self._showPhase("")

    def _showInScene(self, arg_name: str, path: str, label: str = "") -> bool:
        """Show what this row points at, once. True when something was loaded.

        The ONE answer to "should the user see what they just chose", for both
        halves of the row -- a path picked in the file dialog and a test file
        downloaded from the server. Having had one each is exactly how a
        downloaded scan ended up sitting in the scene unrendered while a picked
        one rendered, and the two kept drifting because nothing said they were
        the same question.

        Three things it deliberately does not do. It never loads a COHORT:
        forty patients would put hundreds of nodes in the scene, which is worse
        than showing nothing -- batches are the run's job, not the picker's. It
        never loads the same pick twice for one argument, or re-choosing would
        stack copies. And it never fails a pick: `slicer_io.load_input` swallows
        a reader Slicer refuses, because the input is already filled in and the
        run works either way.

        The phase message is not decoration. Slicer spends around twenty
        seconds decompressing and building a 94 MB scan, on the main thread;
        without a line saying so, choosing a file looks like a freeze.
        """
        if not path:
            return False
        # A folder is one scan or a cohort, and only its contents say which.
        # `sole_scan_in` answers None for the cohort, which shows nothing --
        # the behaviour every folder used to get unconditionally.
        folder = os.path.isdir(path)
        scan = slicer_io.sole_scan_in(path) if folder else path
        if not scan or slicer_io.scene_kind_for(scan) is None:
            # A .zip, a .csv, a cohort: nothing a scene can hold as one thing,
            # and saying "Loading..." about it would be a lie.
            return False
        # Keyed on what the USER chose, never on the file resolved out of it: a
        # folder is re-picked as the folder, and two of them can hold scans
        # with the same name.
        if self._scenePreviews.get(arg_name) == path:
            return False

        self._scenePreviews[arg_name] = path
        self._showPhase(_("Loading {name} into the scene...").format(
            name=label or os.path.basename(path.rstrip(os.sep))))
        slicer.app.processEvents()
        # The folder's name, when the scan came out of one: a DICOM series is
        # opened through one of its slices and would otherwise arrive called
        # `IMG0001`, which names nothing anyone picked.
        node = slicer_io.load_input(
            scan, os.path.basename(path.rstrip(os.sep)) if folder else "")
        # One scan, whatever the row was pointed at: the cohort was refused
        # above, so rendering this cannot flood anything.
        if slicer_io.scene_kind_for(scan) == "volume":
            self._renderScan(node)
        return True

    def _showPhase(self, message: str) -> None:
        """Put a message on the panel immediately, timer running or not.

        Deliberately independent of the elapsed-time state: _onJobSuccess tears
        the job down BEFORE handleResult, so the work that happens after it
        (unpacking an archive, loading nodes) has no timer left to render with
        and would otherwise report nothing at all.
        """
        if self._progressLabel is None:
            return
        self._progressLabel.setText(message)
        self._progressLabel.setVisible(True)
        slicer.util.showStatusMessage(message)

    def _hideProgress(self) -> None:
        if self._progressLabel:
            self._progressLabel.setVisible(False)
            self._progressLabel.setText("")
        if self._progressBar:
            # Together with the label, always. A bar left at 40% under a blank
            # line is a run that looks stuck at 40% forever.
            self._progressBar.setVisible(False)
            self._progressBar.setValue(0)

    def _renderProgress(self) -> None:
        if not self._runs:
            return
        # getattr: a panel built without __init__ (the unit tests do exactly
        # that) has no view, and no cohort either.
        view = getattr(self, "_cohortView", None)
        if view is not None:
            self._renderCohort(view)
            # The label above the box stays empty: the box says all of it, and
            # the same sentence twice reads as two different runs.
            self._showPhase("")
            return
        self._showPhase("\n".join(self._describeRun(run) for run in self._runs))
        self._renderProgressBar()

    def _renderCohort(self, view) -> None:
        """Write the current state into the cohort box. Values only."""
        cohort = view.cohort
        done, failed = cohort.scans_done, cohort.scans_failed
        text = _("{done} of {total} scans done").format(
            done=done, total=cohort.total_scans)
        if failed:
            # Named, not folded into the count. "16 of 20" with four lost in
            # silence is the report this whole feature exists not to produce.
            text += _("  ·  {failed} failed").format(failed=failed)
        view.total.setText(text)
        view.bar.setValue(int(round(100 * cohort.progress(
            [run for run in self._runs if run.started_at is not None]))))

        if view.remainder is not None:
            waiting = len(self._runs) - len(view.rows)
            view.remainder.setText(_("+ {count} more batches queued").format(
                count=waiting))

        for run in self._runs:
            row = view.rows.get(run.number)
            if row is None:
                continue
            label, bar = row
            label.setText(self._describeBatch(run))
            # Only for a batch actually running: an empty bar under each queued
            # batch is three things that look stuck.
            if run.started_at is None or run.fraction is None:
                bar.setVisible(False)
            else:
                bar.setValue(int(round(100 * run.fraction)))
                bar.setVisible(True)

    def _describeBatch(self, run) -> str:
        """One batch's line inside the cohort box.

        Numbered by its place in the cohort rather than by the panel's run
        counter: "Batch 2 of 5" is where the user is, "Run 7" is bookkeeping
        that means nothing to them.
        """
        where = _("Batch {index} of {total}").format(
            index=run.cohort_index, total=run.cohort.total)
        if run.started_at is None:
            return _("{where}  ·  queued").format(where=where)
        elapsed = int(time.monotonic() - run.started_at)
        return _("{where}  ·  {phase}  ·  {minutes}:{seconds:02d}").format(
            where=where, phase=self._runPhaseText(run) or _("Working..."),
            minutes=elapsed // 60, seconds=elapsed % 60,
        )

    def _renderProgressBar(self) -> None:
        """The determinate bar, shown only when there is a real number behind it.

        One run, one fraction: with several in flight there is no single number
        a bar could honestly show, and each line already carries its own
        percentage. Hidden the rest of the time -- most tools report no
        fraction at all, and a bar that has to invent motion to look busy is
        worse than the elapsed-time line beside it, which never claims to know
        how far along anything is.
        """
        if self._progressBar is None:
            return
        fraction = self._runs[0].fraction if len(self._runs) == 1 else None
        if fraction is None:
            self._progressBar.setVisible(False)
            return
        self._progressBar.setValue(int(round(100 * fraction)))
        self._progressBar.setVisible(True)

    def _describeRun(self, run) -> str:
        """One line per run -- and for a single run, exactly the line it always was.

        The run number and label appear only once there is something to tell
        apart. A panel running one job must look like it always did: that is the
        overwhelmingly common case, and nine modules' worth of habit.
        """
        if run.started_at is None:
            return _("Run {number} ({label})  —  queued").format(
                number=run.number, label=run.label)
        elapsed = int(time.monotonic() - run.started_at)
        line = _("{phase}  —  {minutes}:{seconds:02d} elapsed").format(
            phase=self._runPhaseText(run) or _("Working..."),
            minutes=elapsed // 60, seconds=elapsed % 60,
        )
        if len(self._runs) == 1:
            return line
        return _("Run {number} ({label}): {line}").format(
            number=run.number, label=run.label, line=line)

    def _runPhaseText(self, run, with_fraction: bool = True) -> str:
        """What this run is doing, in one phrase, from whichever side knows.

        The server's word wins while the server is the one working: it is the
        only side that can tell four minutes of queueing for the GPU from four
        minutes of inference, and until it could say so that silence was the
        whole of this panel's problem. The client's own phases (uploading,
        downloading) take over the moment it has something of its own to
        report, since by then the server has finished.
        """
        if not run.server_phase:
            return run.phase
        parts = [self._phaseLabel(run.server_phase)]
        if run.server_message:
            parts.append(run.server_message)
        if with_fraction and run.fraction is not None:
            parts.append("{:.0%}".format(run.fraction))
        text = " — ".join(part for part in parts if part)
        if run.depth:
            # A supervised chain: AREG drives ASO, which drives ALI. Nesting is
            # shown as nesting, and the child is named only when the server
            # named it -- on the marker that opened the call this record came
            # from (see _onRunEvent). An older server sends no name, and
            # guessing which tool of a chain is running would be worse than
            # the arrows alone.
            if run.nested_tool:
                text = "{}: {}".format(run.nested_tool, text)
            text = ("→ " * run.depth) + text
        return text

    @staticmethod
    def _phaseLabel(phase: str) -> str:
        """One phase of the run contract, in words a clinician reads.

        The vocabulary is a small CLOSED set precisely so a panel can translate
        it instead of showing a word written for a server log. Built at call
        time rather than as a module constant: `_` resolves against the
        interface language in force when it runs, and a dict built at import
        time would freeze every label at whatever Slicer started in.

        An unknown phase is shown as-is rather than dropped. This is the seam
        between two repositories: a phase added on the server side must degrade
        to a slightly technical word on the panel, never to a run that looks
        like it stopped saying anything.
        """
        labels = {
            "received": _("Request received"),
            "staging": _("Preparing the input files"),
            "queued_gpu": _("Waiting for the GPU"),
            "running": (_("Running") if _client_is_local()
                        else _("Running on the server")),
            "packaging": _("Packaging the results"),
            "done": _("Finished"),
            "failed": _("Failed"),
            "cancelled": _("Cancelled"),
        }
        return labels.get(phase, phase)

    # ------------------------------------------------------------------
    # Open volumes offered as input sources
    # ------------------------------------------------------------------

    def _onSceneNodesChanged(self, caller=None, event=None) -> None:
        self._refreshSceneVolumes()

    def _refreshSceneVolumes(self) -> None:
        """Re-offer the scene's scalar volumes in every input dropdown that
        can take one (formgen.accepts_volume).

        The display names are disambiguated here and mapped back to nodes at
        upload time through _sceneVolumes: two loaded volumes can share a
        name, and the dropdown must not let one silently shadow the other.
        """
        volumes = {}
        nodes = []
        # Every kind the scene can answer an argument with, not scalar volumes
        # only: ALI's `input` takes an intraoral surface just as readily, and a
        # scene holding nothing but meshes used to offer nothing at all.
        for node_class, _extension in formgen.SCENE_NODE_KINDS.values():
            # The user's own data, not the scene's furniture: every slice view
            # keeps a model node for the plane it draws in 3D, and all three
            # were being offered as surfaces (slicer_io.scene_nodes).
            nodes.extend(slicer_io.scene_nodes(node_class))
        for node in nodes:
            name = node.GetName()
            unique, counter = name, 2
            while unique in volumes:
                unique = f"{name} ({counter})"
                counter += 1
            volumes[unique] = node
        self._sceneVolumes = volumes

        for arg_name, widget in self._inputWidgets.items():
            setter = getattr(widget, "setVolumeChoices", None)
            if setter is None:
                continue
            # Narrowed per ARGUMENT: a scan input is offered scans, a mesh
            # input meshes, and one declaring neither is offered both. A `.csv`
            # argument gets nothing.
            kinds = self.SCENE_INPUTS.get(arg_name) or formgen.scene_kinds_for(
                self._schemaArgument(arg_name), arg_name)
            classes = [formgen.SCENE_NODE_KINDS[kind][0] for kind in kinds]
            label = getattr(widget, "setSceneLabel", None)
            if label is not None:
                label(formgen.scene_label_for(kinds))
            supported = getattr(widget, "setSceneSupported", None)
            if supported is not None:
                supported(bool(classes))
            setter([
                name for name, node in volumes.items()
                if any(node.IsA(node_class) for node_class in classes)
            ] if classes else [])
        self._checkCanApply()

    # ------------------------------------------------------------------
    # The tool's own test files, downloaded from the server on selection
    # ------------------------------------------------------------------

    # Our own key, so the sweep below can tell OUR leftovers from every other
    # module's use of slicer.util.tempDirectory().
    TEST_FILE_DIR_KEY = "ADTRemoteTestFiles"

    # Written inside each directory so the sweep can ask "is the session that
    # made this still running?" instead of guessing from a timestamp. An age
    # threshold was the first attempt and it was wrong in both directions: at
    # twelve hours, thirty-five directories and 2.4 GB piled up in a single
    # afternoon of launching Slicer, while a session left open longer than the
    # threshold could have had its own cohort deleted underneath it.
    OWNER_FILE = ".owner-pid"

    def _testFileDir(self) -> str:
        """A directory for this session's downloaded test files, created once.

        `slicer.util.tempDirectory()` and not ~/Documents: these are whole
        cohorts (648 MB for the semi-automated CBCT set) and a user who clicked
        a test file once to look at it has not asked to keep it. It also hands
        back a NEW directory per call, which is why the first one is kept -- a
        second pick of the same entry has to find the first one's bytes.

        Slicer's own docstring says it plainly: "This directory is not
        automatically cleaned up." The name carries a timestamp, so every
        session makes another one and nothing ever removes them -- 648 MB per
        cohort, per launch, until the operating system gets round to /tmp,
        which on a workstation left running is never. So this sweeps what
        earlier sessions left before making today's.
        """
        if not self._testFileRoot:
            self._sweepLeftoverTestFiles()
            self._testFileRoot = slicer.util.tempDirectory(key=self.TEST_FILE_DIR_KEY)
            try:
                with open(os.path.join(self._testFileRoot, self.OWNER_FILE), "w") as handle:
                    handle.write(str(os.getpid()))
            except OSError:  # the sweep will fall back to leaving it alone
                logger.debug("could not mark the test-file directory", exc_info=True)
        return self._testFileRoot

    @classmethod
    def _sweepLeftoverTestFiles(cls) -> None:
        """Remove test-file directories earlier sessions left behind.

        Ours only, by key, and only ones whose owning process is gone -- so a
        second Slicer running right now keeps its own, however long it has been
        open. A directory with no owner mark is from a build before this and is
        removed. Every failure is ignored: a directory that cannot be removed
        is disk, and disk must never cost a user their run.
        """
        try:
            root = slicer.app.temporaryPath
            for name in os.listdir(root):
                if not name.startswith(cls.TEST_FILE_DIR_KEY):
                    continue
                path = os.path.join(root, name)
                try:
                    if not os.path.isdir(path) or cls._ownerIsAlive(path):
                        continue
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info("removed a leftover test-file directory: %s", path)
                except OSError:
                    continue
        except Exception:  # noqa: BLE001 - housekeeping, never fatal
            logger.debug("could not sweep leftover test files", exc_info=True)

    def _declaredKind(self, arg_name: str, name: str):
        """"file", "folder" or None - what the server said this entry is.

        Read on the MAIN thread and handed to the worker, never read from it:
        it comes off a widget, and a widget belongs to the thread that built
        it. None means the server published no `entries` at all, which is what
        `_hostedKind`'s fallback is for.
        """
        widget = self._inputWidgets.get(arg_name)
        reader = getattr(widget, "hosted_entries", None)
        for entry in (reader() if reader else []):
            if entry.get("name") == name:
                kind = entry.get("kind")
                return kind if kind in ("file", "folder") else None
        return None

    @classmethod
    def _ownerIsAlive(cls, path: str) -> bool:
        """Is the Slicer that created this directory still running?

        An unreadable or nonsensical mark counts as dead: the worst case is
        deleting a cohort someone would have re-downloaded, against a disk that
        fills up for good.
        """
        try:
            with open(os.path.join(path, cls.OWNER_FILE)) as handle:
                pid = int(handle.read().strip())
        except (OSError, ValueError):
            return False
        return _process_is_running(pid)

    def _removeOwnTestFiles(self) -> None:
        """Drop this session's downloaded test files when the panel goes away.

        `cleanup()` runs when the widget is destroyed -- Slicer quitting, or a
        module reload -- so the path an input still holds is about to stop
        mattering. If it never runs (a crash), the next session's sweep finds
        the directory with a dead owner and removes it then. Two layers,
        because Slicer removes nothing on its own and a clinician has no reason
        to know /tmp exists.
        """
        root, self._testFileRoot = self._testFileRoot, None
        self._testFileCache.clear()
        if not root:
            return
        try:
            shutil.rmtree(root, ignore_errors=True)
            logger.info("removed this session's test-file directory: %s", root)
        except Exception:  # noqa: BLE001 - housekeeping, never fatal
            logger.debug("could not remove %s", root, exc_info=True)

    def _onHostedTestFile(self, arg_name: str, name: str) -> None:
        """Download one of the tool's server-hosted test files and use it.

        **On a BackgroundJob, always.** The panel used to reach a hosted test
        file by NAME, which cost nothing because the file never moved; it is
        fetched now, and fetching it on the main thread froze the whole Slicer
        window for as long as it took -- measured at 21 minutes once. The
        elapsed-time label is the same channel a tool run reports on.

        Cached by name for the session: a second pick of the same entry reuses
        what is on disk and issues no request at all, so flipping between two
        test cohorts is free after the first of each.
        """
        widget = self._inputWidgets.get(arg_name)
        if widget is None or not name:
            return

        # Keyed by SCOPE and name, never by the name alone. Two arguments that
        # draw from different subfolders may legitimately offer the same entry
        # name -- `IOSCBCT_TestFile` is the intraoral surfaces under one scope
        # and the CBCT volume under another -- and keyed by name, picking it for
        # the second argument silently handed back the first one's download:
        # meshes in the CBCT field, from a cache that looked like a hit.
        scope = self._selectableScope(arg_name)
        key = (scope, name)

        cached = self._testFileCache.get(key)
        if cached and os.path.exists(cached):
            self._useTestFile(arg_name, name, cached)
            return
        # Tools run on this machine: an installed test file is already on disk.
        # Use it where it is -- no copy, no archive to pack and unpack again,
        # which on a folder of scans held Slicer up for as long as it took.
        local = getattr(self.client, "local_testfile_path", None)
        if local is not None:
            path = local(self.TOOL_NAME, name, scope)
            if path and os.path.exists(path):
                self._useTestFile(arg_name, name, path)
                return

        if self._downloadJob is not None:
            slicer.util.showStatusMessage(
                _("Another test file is still downloading."), 5000
            )
            return

        # The scope is part of the file name on disk for the same reason it is
        # part of the cache key: one staging directory, two entries that share
        # a name and do not share their contents.
        destination = os.path.join(
            self._testFileDir(),
            _safe_name("{}__{}".format(scope, name) if scope else name),
        )
        if os.path.exists(destination):
            # A previous pick in this session that never made it into the
            # cache (a rebuilt panel, the same entry picked twice).
            self._useTestFile(arg_name, name, destination)
            return

        declared = self._declaredKind(arg_name, name)

        def task(progress_cb):
            return self._fetchTestFile(name, destination, declared, progress_cb, scope)

        def finish():
            self._downloadJob = None
            self._hideProgress()

        def on_success(path):
            finish()
            self._useTestFile(arg_name, name, path)

        def on_error(exc):
            finish()
            slicer.util.errorDisplay(plain_text(
                _("Could not download the test file: {error}").format(error=exc)
            ))

        self._downloadJob = BackgroundJob(
            task, on_success=on_success, on_error=on_error, on_progress=self._showPhase
        )
        self._showPhase(_("Downloading {name}...").format(name=name))
        self._downloadJob.start()

    def _selectableScope(self, arg_name: str) -> str:
        """The subfolder this argument's hosted files come from, or "".

        Read in three places -- the cache key, the staged file name, and the
        download itself -- and all three have to agree: two arguments drawing
        from different subfolders may offer the same entry NAME, and treating
        them as one handed the CBCT field a folder of intraoral meshes.
        """
        spec = (self._schema or {}).get("arguments", {}).get(arg_name, {})
        # A facade argument whose engines draw from different subfolders: the
        # scope follows the controlling argument (the mode) as it is now.
        for controller, per_value in (spec.get("selectable_scope_when") or {}).items():
            widget = self._argWidgets.get(controller)
            value = formgen.collect({controller: widget}).get(controller) if widget else None
            if value in per_value:
                return per_value[value] or ""
        return spec.get("selectable_scope") or ""

    def _fetchTestFile(self, name: str, destination: str, declared, progress_cb,
                       scope: str = "") -> str:
        """Worker-thread half: download, unpack a hosted folder, move into place.

        Staged in a sibling directory and renamed at the end, so a failed or
        interrupted download can never leave a half-extracted folder that the
        existence check above would mistake for a completed one.
        """
        staging = destination + ".downloading"
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)
        # Timed per phase and logged. Outside Slicer this whole path measures
        # 0.22 s for a 94 MB file and 0.39 s for a 94 MB folder -- against
        # curl's 0.24 s -- while a user watching the panel reported more than
        # ten seconds. The difference is somewhere in here, and one log line
        # says where instead of inviting a guess.
        timings = []

        def phase(label, work):
            started = time.perf_counter()
            try:
                return work()
            finally:
                timings.append((label, time.perf_counter() - started))

        try:
            payload = os.path.join(staging, _safe_name(name))
            phase("download", lambda: self.client.download_testfile(
                self.TOOL_NAME, name, payload, progress_cb, scope))

            # A hosted FOLDER is zipped by the server on the way out (there
            # being no other way to put a directory on a wire) and is unpacked
            # here, so the input points at a directory the tool can walk. A
            # hosted FILE that happens to be a .zip is left exactly as it is:
            # it is the file, and unpacking it would hand the tool something
            # the server never offered.
            if self._hostedKind(name, declared, payload) == "folder":
                progress_cb(_("Unpacking {name}...").format(name=name))
                unpacked = os.path.join(staging, "unpacked")
                phase("unpack", lambda: slicer_io.unzip_folder(payload, unpacked))
                phase("move", lambda: os.rename(unpacked, destination))
            else:
                phase("move", lambda: os.replace(payload, destination))
        finally:
            phase("clean", lambda: shutil.rmtree(staging, ignore_errors=True))
            # Kept on the instance so the main thread can SHOW it. A
            # `logger.info` alone does not reach Slicer's Python console, so
            # the measurement existed and nobody could read it -- the same
            # mistake as the server's peak VRAM, which was recorded on every
            # run and never read back.
            self._lastTestFileTimings = list(timings)
            logger.info(
                "test file %r: %s", name,
                ", ".join("{} {:.2f}s".format(label, seconds) for label, seconds in timings),
            )
        # Either way the answer is the same path: a file lands as itself and a
        # folder as its unpacked directory, so nothing downstream has to ask
        # which it was.
        return destination

    @staticmethod
    def _hostedKind(name: str, declared, payload: str) -> str:
        """"file" or "folder" for a hosted entry.

        The server says so outright when it publishes `entries`, which is the
        normal case and the only one worth trusting. An older server publishes
        names alone, and then the shape of what arrived has to answer: the
        endpoint zips a folder and streams a file untouched, and a directory
        name carries no extension -- so an extensionless name that really did
        arrive as an archive was a directory. The bytes are sniffed rather than
        the name, here and nowhere else: `is_extractable_archive` refuses to,
        because a RESULT .xlsx is a zip container and must not be unpacked, but
        that reasoning is exactly why the name alone cannot settle this one.
        """
        if declared in ("file", "folder"):
            return declared
        if not os.path.splitext(name)[1] and zipfile.is_zipfile(payload):
            return "folder"
        return "file"

    def _useTestFile(self, arg_name: str, name: str, path: str) -> None:
        """Point the input at the downloaded test file, and show it.

        Order matters: the path is written FIRST and the scene load is a
        courtesy after it. Loading is what the download is for -- a clinician
        asked for this scan so they could look at it beside the panel -- but a
        file Slicer's reader refuses is still a perfectly good input, so a
        failed load is a log line and the run goes ahead (slicer_io.load_input
        never raises).
        """
        load_started = time.perf_counter()
        self._testFileCache[(self._selectableScope(arg_name), name)] = path
        widget = self._inputWidgets.get(arg_name)
        if widget is not None:
            formgen.set_local_path(widget, path)
        self._checkCanApply()

        # `set_local_path` above changed the row, which fires the same preview a
        # hand-picked path gets -- so by the time this line runs the scan may
        # ALREADY be in the scene, and loading it again would put a second copy
        # of the same patient there. `_scenePreviews` is the one record of what
        # has been shown, whichever half of the panel showed it, and
        # `_showInScene` is the one place that reads it. This call is what makes
        # a downloaded scan render exactly as a picked one does; they used to be
        # two code paths, and only one of them rendered.
        #
        # `name` is said out loud because this is the slow half and does not
        # look like it: fetching a 94 MB scan takes 0.3 s over ranged parts,
        # then Slicer spends twenty seconds decompressing it and building the
        # image. A progress line still reading "Downloading..." while that
        # happens makes a fast transfer look like a stalled one.
        loaded_into_scene = self._showInScene(arg_name, path, label=name)
        if not loaded_into_scene and self._scenePreviews.get(arg_name) == path:
            # Already shown by the preview the path change fired, and still
            # worth reporting as scene time -- the seconds were spent.
            loaded_into_scene = True
        # The breakdown goes where a user can actually see it. "It took more
        # than ten seconds" is not a bug report anyone can act on; "download
        # 0.3s, unpack 1.2s, scene 8.4s" is.
        timings = list(getattr(self, "_lastTestFileTimings", []))
        if loaded_into_scene:
            # Only when there WAS one. A cohort is never loaded, and reporting
            # `scene 0.0s` for it would name a phase that did not happen.
            timings.append(("scene", time.perf_counter() - load_started))
        total = sum(seconds for _label, seconds in timings)
        breakdown = ", ".join(
            "{} {:.1f}s".format(label, seconds) for label, seconds in timings
        )
        # LEFT on the panel, not hidden. `_hideProgress` used to run right
        # here, so the one line saying where the seconds went lived only in the
        # status bar for eight seconds -- which is no use to someone trying to
        # find out why a download felt slow. It stays until the next action.
        self._showPhase(
            _("{name} ready in {total:.1f}s -- {breakdown}").format(
                name=name, total=total, breakdown=breakdown
            )
        )
        slicer.util.showStatusMessage(
            _("Test file ready in {total:.1f}s ({breakdown}): {path}").format(
                total=total, breakdown=breakdown, path=path
            ),
            8000,
        )
        # `print` as well as the logger, deliberately. A module logger does not
        # reach Slicer's Python console, so the one measurement that answers
        # "why did that feel slow" was written where nobody could read it --
        # the same mistake as the server's peak VRAM, recorded on every run and
        # never read back. One line per download is not noise; it is the line
        # someone pastes into a bug report.
        summary = "[test file] {} ready in {:.1f}s -- {}".format(name, total, breakdown)
        print(summary)
        logger.info(summary)

    # ------------------------------------------------------------------
    # Server status banner
    # ------------------------------------------------------------------

    def _refreshServerStatus(self) -> None:
        """Keep the job on the instance, never in a local: a BackgroundJob is
        only kept alive by its own reference cycle (job -> QTimer -> bound
        _drain -> job), so a cyclic-GC pass — a module reload triggers one —
        can collect it mid-flight. Its timer dies with it, the callback never
        runs, and the badge stays stuck on "checking...". Owning it also lets
        cleanup() cancel it, so a job started by a widget Qt has since deleted
        can't write into a destroyed badge."""
        if self._statusJob:
            self._statusJob.cancel()

        def task(_progress_cb):
            # Locally the question is whether THIS tool is installed, not
            # whether a server answers.
            if self._isLocal() and hasattr(self.client, "has_tool"):
                return self.client.has_tool(self.TOOL_NAME)
            return self.client.health()

        self._statusJob = BackgroundJob(
            task, on_success=self._onStatusChecked, on_error=lambda _exc: self._onStatusChecked(False)
        )
        self._statusJob.start()

    def _onStatusChecked(self, ok: bool) -> None:
        self._statusJob = None
        if self._statusBadge:
            version, update = "", False
            if self._isLocal() and ok and hasattr(self.client, "tool_version"):
                version = self.client.tool_version(self.TOOL_NAME) or ""
                update = self.client.update_available(self.TOOL_NAME)
            store = getattr(self.client, "store", None)
            target = store.version if store is not None else ""
            design.update_status_badge(self._statusBadge, ok, local=self._isLocal(),
                                       version=version, update=update, target=target,
                                       **self._releaseCheckBadge(store))
            button = getattr(self, "_updateButton", None)
            if button is not None:
                # The version it goes to, on the button itself: "Update now"
                # said nothing of where a clinician's results would come from.
                button.setText(_("Update to {new}").format(new=target) if target else _("Update now"))
                button.setToolTip(_("Install the update: tools {old} → {new}").format(
                    old=version, new=target))
                button.setVisible(bool(update))

        # The panel is built from the schema, once. If the server was down when
        # this module was opened, all it holds is an error label — and the
        # health check coming back green is the one signal that it is worth
        # trying again. Without this the module stays broken for the whole
        # Slicer session, still showing a connection error against a server
        # that is now up.
        if ok and self._schemaError is not None and self.uiWidget:
            logger.info("Server is reachable again, rebuilding the panel for '%s'", self.TOOL_NAME)
            # force_refresh: the cached /tools may be exactly what is wrong
            # (fetched from another server, or before this tool was registered).
            self._buildForm(force_refresh=True)
