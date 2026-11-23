"""A run that stops for somebody to look at it, and is then carried on.

The server can be asked to interrupt a chain after one of its steps: the POST
answers a quality-control record instead of a result, the run stays `paused`,
and `POST /runs/{id}/resume` carries it on with whatever a reader corrected.
What this file pins is the panel's half of that loop, and the two places it
could quietly go wrong:

* a stopped run is NOT a finished one. It keeps its place in the queue, it is
  still cancellable, and nothing downstream of a result may fire for it --
  `handleResult` least of all, since a module's override would open a half-run
  chain's intermediates as if they were the answer;
* and a run that did not stop must go through untouched. That is every run of
  every tool today, so a regression there is not a feature that failed, it is
  the whole extension.

Usage:
    python3 -m unittest test_quality_control
"""

import ast
import contextlib
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest
import zipfile
from unittest import mock

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _CORE)

import test_hosted_test_files as fixtures  # noqa: F401,E402 - importing installs the stubs

# `_onJobSuccess` wraps `handleResult` in it. Same stop-gap as test_runs.py,
# and for the same reason: the shared stub has no need of it.
_slicer_util = sys.modules["slicer.util"]
if not hasattr(_slicer_util, "tryWithErrorDisplay"):
    @contextlib.contextmanager
    def _try_with_error_display(_message, **_kwargs):
        yield

    _slicer_util.tryWithErrorDisplay = _try_with_error_display

from ServerToolsCoreLib import base_widget, transfer  # noqa: E402
from ServerToolsCoreLib.base_widget import ServerToolWidgetBase  # noqa: E402
from ServerToolsCoreLib.client import (  # noqa: E402
    RunCheckpoint,
    ToolResult,
    ToolServerClient,
)

qt = sys.modules["qt"]


def _response(status_code=200, json_data=None, content=b"", headers=None):
    """The same shape test_client.py uses; kept local so the two files can be
    read apart."""
    response = mock.Mock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 300
    response.headers = headers or {"Content-Type": "application/json"}
    response.content = content
    response.iter_content = lambda chunk_size: iter([content] if content else [])
    if json_data is not None:
        response.json.return_value = json_data
    else:
        response.json.side_effect = ValueError("no json body")
    return response


def _zip_bytes(members):
    buffer = tempfile.SpooledTemporaryFile()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    buffer.seek(0)
    return buffer.read()


# ----------------------------------------------------------------------
# The client: recognising the record, and sending the corrections back
# ----------------------------------------------------------------------

class ClientTest(unittest.TestCase):
    """One tool that can be stopped: `stop_after` is a multichoice exactly as
    the server injects it, every step off by default."""

    TOOL = {
        "name": "AREG",
        "output_kind": "files",
        "arguments": {
            "t1": {"type": "path", "required": False},
            "stop_after": {"type": "multichoice", "required": False,
                           "choices": {"ALI_CBCT": False, "ASO": False}},
        },
    }

    QC = {
        "quality_control": True,
        "stopped_after": "ALI_CBCT",
        "produced": ["01_ALI_CBCT"],
        "result_ref": {"result_id": "res123", "filename": "AREG_stopped.zip",
                       "size": 4096, "media_type": "application/zip"},
    }

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)
        self.client = ToolServerClient("https://server", "token", detached_runs=True)

    def _fake_download(self, _session, _url, destination, *_a, **_kw):
        with open(destination, "wb") as handle:
            handle.write(_zip_bytes({"landmarks.mrk.json": "{}"}))
        return destination

    @mock.patch("requests.Session.delete")
    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_stopped_run_answers_a_checkpoint_and_not_a_result(
            self, mock_get, mock_post, mock_delete):
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data=self.QC)

        with mock.patch.object(transfer, "download_ranged",
                               side_effect=self._fake_download):
            result = self.client.run(
                "AREG", args={"stop_after": {"ALI_CBCT": True, "ASO": False}},
                output_dir=self.work, run_id="run-1")

        self.assertEqual(result.kind, "checkpoint")
        self.assertIsNotNone(result.checkpoint)
        self.assertEqual(result.checkpoint.run_id, "run-1")
        self.assertEqual(result.checkpoint.stopped_after, "ALI_CBCT")
        self.assertEqual(result.checkpoint.produced, ("01_ALI_CBCT",))
        self.assertEqual(result.checkpoint.path,
                         os.path.join(self.work, "AREG_stopped.zip"))
        # Fetched and released like any other reference: what happens next is
        # a person reading scans, which is not a wait to hold server-side
        # storage through.
        self.assertEqual(mock_delete.call_args.args[0], "https://server/results/res123")

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_checkpoint_that_collected_nothing_is_still_a_checkpoint(
            self, mock_get, mock_post):
        """A stop declared inside a tool's own work collects no supervised
        call, so the server sends `result_ref: null`. The run is paused all the
        same and something has to carry it on."""
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data=dict(
            self.QC, produced=[], result_ref=None))

        result = self.client.run("AREG", args={"stop_after": {"ALI_CBCT": True}},
                                 output_dir=self.work, run_id="run-1")

        self.assertEqual(result.kind, "checkpoint")
        self.assertIsNone(result.checkpoint.path)
        self.assertEqual(result.checkpoint.produced, ())

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_run_that_asks_to_stop_is_not_detached(self, mock_get, mock_post):
        """The whole feature turns on this. The server builds the record in the
        RESPONSE BODY; its detached path answers 202, writes a non-terminal
        `paused` event and drops the payload -- so a detached run of this kind
        waits on a terminal event that never comes."""
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data=self.QC, headers={
            "Content-Type": "application/json"})

        with mock.patch.object(transfer, "download_ranged",
                               side_effect=self._fake_download):
            self.client.run("AREG", args={"stop_after": {"ALI_CBCT": True}},
                            output_dir=self.work, run_id="run-1")

        self.assertIsNone(
            mock_post.call_args.kwargs["headers"].get("X-Run-Delivery"),
            "a run that will stop must keep the blocking contract")

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_run_that_asks_for_no_stop_is_detached_as_before(
            self, mock_get, mock_post):
        """Every run of every tool today: the opt-out must be exactly as narrow
        as it claims to be."""
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(status_code=202, json_data={})

        with mock.patch.object(ToolServerClient, "_collect_detached",
                               return_value=ToolResult(kind="text", text="done")):
            self.client.run("AREG", args={"stop_after": {"ALI_CBCT": False,
                                                         "ASO": False}},
                            output_dir=self.work, run_id="run-1")

        self.assertEqual(
            mock_post.call_args.kwargs["headers"].get("X-Run-Delivery"), "detached")

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_resume_sends_one_field_per_step_named_after_its_folder(
            self, mock_get, mock_post):
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(
            content=_zip_bytes({"AREG_output.nii.gz": "x"}),
            headers={"Content-Type": "application/zip",
                     "Content-Disposition": 'attachment; filename="AREG_output.zip"'})
        correction = os.path.join(self.work, "01_ALI_CBCT.zip")
        with open(correction, "wb") as handle:
            handle.write(_zip_bytes({"corrected.mrk.json": "{}"}))

        out = os.path.join(self.work, "out")
        result = self.client.resume_run("AREG", "run-1",
                                        corrections={"01_ALI_CBCT": correction},
                                        output_dir=out)

        self.assertEqual(mock_post.call_args.args[0],
                         "https://server/runs/run-1/resume")
        sent = mock_post.call_args.kwargs["files"]
        self.assertEqual(list(sent), ["01_ALI_CBCT"])
        self.assertEqual(sent["01_ALI_CBCT"][0], "01_ALI_CBCT.zip")
        self.assertEqual(result.kind, "file")
        self.assertEqual(result.path, os.path.join(out, "AREG_output.zip"))

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_resume_with_nothing_to_correct_sends_no_files(
            self, mock_get, mock_post):
        """Legal, and the cheap half of the feature: the server carries on with
        what it produced."""
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data={"result": "done"})

        self.client.resume_run("AREG", "run-1", corrections={},
                               output_dir=self.work)

        self.assertIsNone(mock_post.call_args.kwargs["files"])

    @mock.patch("requests.Session.post")
    @mock.patch("requests.Session.get")
    def test_a_resume_that_stops_again_answers_another_checkpoint(
            self, mock_get, mock_post):
        """Two steps can be armed at once, and the second stop arrives on the
        resume's own response. The loop has to be able to go round again."""
        mock_get.return_value = _response(json_data=[self.TOOL])
        mock_post.return_value = _response(json_data=dict(
            self.QC, stopped_after="ASO", produced=["01_ALI_CBCT", "02_ASO"]))

        with mock.patch.object(transfer, "download_ranged",
                               side_effect=self._fake_download), \
                mock.patch("requests.Session.delete"):
            result = self.client.resume_run("AREG", "run-1", corrections={},
                                            output_dir=self.work)

        self.assertEqual(result.kind, "checkpoint")
        self.assertEqual(result.checkpoint.stopped_after, "ASO")

    def test_a_resume_without_a_run_id_is_refused_before_any_request(self):
        with self.assertRaises(Exception):
            self.client.resume_run("AREG", "", corrections={})


# ----------------------------------------------------------------------
# The panel: the loop, driven against a stub job and a stub reviewer
# ----------------------------------------------------------------------

class _Job:
    """Stand-in for BackgroundJob, delivered on command. Same shape as
    test_runs.py's, which is where the reasoning for having no thread is."""

    started = []

    def __init__(self, target, on_success=None, on_error=None, on_progress=None,
                 cancel_event=None):
        self.target = target
        self.on_success = on_success
        self.on_error = on_error
        self.on_progress = on_progress
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()

    def start(self):
        _Job.started.append(self)

    def collect(self):
        """Invoke the target here, the way the real worker thread would.

        Separate from `start` so a case can look at the panel between the two,
        which is the whole reason the stub has no thread of its own.
        """
        return self.target(lambda *_a, **_k: None)

    def cancel(self):
        self.cancel_event.set()

    def succeed(self, result):
        self.on_success(result)


class _Reviewer:
    """The review module, recorded instead of shown.

    Stands in for `VISU.open_for_review`, which is the entire surface the panel
    uses: a folder and a callable in, a bool out.
    """

    def __init__(self):
        self.opened = []
        self.available = True
        self.on_continue = None

    def open_for_review(self, folder, on_continue, rewind=None, origin=None, inputs=(),
                        focus=None):
        self.inputs = tuple(inputs)
        self.focus = focus
        self.rewind = rewind
        self.origin = dict(origin or {})
        self.opened.append(folder)
        self.on_continue = on_continue
        return self.available

    def press_continue(self, flagged=(), written=(), folder=None, rewind_to=None):
        self.on_continue({"folder": folder if folder is not None else self.opened[-1],
                          "flagged": set(flagged), "written": set(written),
                          "rewind_to": rewind_to})


class _RecordingClient:
    def __init__(self):
        self.resumed = []
        self.cancelled = []

    def run(self, tool_name, **kwargs):
        raise AssertionError("the stub job never invokes its target")

    def resume_run(self, tool_name, run_id, corrections=None, output_dir=None,
                   progress_cb=None, rewind_to=None, event_cb=None, cancel_event=None):
        self.resumed.append({"tool": tool_name, "run_id": run_id,
                             "corrections": dict(corrections or {}),
                             "output_dir": output_dir,
                             "rewind_to": rewind_to,
                             "event_cb": event_cb, "cancel_event": cancel_event})
        return ToolResult(kind="text", text="carried on")

    def cancel_run(self, run_id):
        self.cancelled.append(run_id)
        return True


class PanelTest(unittest.TestCase):

    def setUp(self):
        _Job.started = []
        self.addCleanup(setattr, base_widget, "BackgroundJob", base_widget.BackgroundJob)
        base_widget.BackgroundJob = _Job

        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)
        self.reviewer = _Reviewer()
        self.client = _RecordingClient()
        self.announced = []
        self.handled = []
        self.panel = self._panel()

    def _panel(self):
        panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        panel.TOOL_NAME = "AREG"
        panel._runs = []
        panel._runsStarted = 0
        panel._elapsedTimer = None
        panel._outputFolderWidget = None
        panel._statusJob = None
        panel._downloadJob = None
        panel._testFileRoot = None
        panel._progressBar = None
        panel._runControlsLayout = qt.QVBoxLayout()
        panel._runControlsWidget = None
        panel.applyButton = qt.QPushButton("Apply")
        panel.cancelButton = qt.QPushButton("Cancel")
        panel.client = self.client
        panel.prepareInputFiles = lambda workspace: {"t1": "/data/p1.nii.gz"}
        panel.collectArgs = lambda: {"stop_after": {"ALI_CBCT": True}}
        panel.handleResult = self.handled.append
        panel._announce = self.announced.append
        panel._showPhase = lambda _message: None
        panel._hideProgress = lambda: None
        panel._checkCanApply = lambda *_args: None
        panel._suggestOutputFolder = lambda: None
        # The reviewer, in place of the lazy `importlib.import_module` of the
        # real thing. The panel is not allowed to know more about the module
        # than these two lines say it does.
        panel._openReviewer = self.reviewer.open_for_review
        return panel

    # -- fixtures ------------------------------------------------------

    def _checkpoint(self, produced=("01_ALI_CBCT",), members=None, flat=False,
                    steps=None, stopped_after="ALI_CBCT"):
        """A stopped run's answer, with a real archive on disk.

        `flat` is the single-step shape: the server's zip flattens ONE
        directory to the archive root, so a lone step's files arrive at the top
        and its folder name never appears in the archive.

        `steps` lays files out per step for a chain that produced several;
        `members` is the shorthand for the first one, which is every case that
        does not care where the files went.
        """
        members = members or {"p1_lm_Pred.mrk.json": "{}"}
        steps = steps or {produced[0]: members}
        if flat:
            laid_out = dict(members)
        else:
            laid_out = {os.path.join(step, name): data
                        for step, files in steps.items()
                        for name, data in files.items()}
        archive = os.path.join(self.work, "AREG_stopped.zip")
        with open(archive, "wb") as handle:
            handle.write(_zip_bytes(laid_out))
        return ToolResult(kind="checkpoint", checkpoint=RunCheckpoint(
            run_id="run-1", stopped_after=stopped_after,
            produced=tuple(produced), path=archive))

    def _apply(self):
        self.panel.onApplyButton()
        return _Job.started[-1]

    def _stopped(self, **kwargs):
        """Apply, and let the run come back stopped. Returns (run, job)."""
        job = self._apply()
        run = self.panel._runs[0]
        run.output_dir = self.work
        job.succeed(self._checkpoint(**kwargs))
        return run, job

    def _collectResume(self) -> dict:
        """Let the resume job reach the client, and hand back what it sent."""
        _Job.started[-1].collect()
        return self.client.resumed[-1]

    # -- standing in for the reader -------------------------------------

    def _reviewed(self, stop="ALI_CBCT", run="run_01") -> str:
        """Where one STOP of one RUN was unpacked -- what the reader edits.

        One directory per run and per stop: neither two checkpoints of one
        run nor two batches of one cohort ever share a review folder, and
        the batches of one Apply share an output folder.
        """
        return os.path.join(self.work, "quality_control", run, stop)

    def _edit(self, relative, data="moved", stop="ALI_CBCT", run="run_01"):
        """Write into the unpacked checkpoint, as the reviewer's save does.

        Subfolders are created, because a step mirrors its input tree and the
        file a reader corrects can be two directories down.
        """
        path = os.path.join(self._reviewed(stop, run), relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(data)
        return path

    @staticmethod
    def _members(archive_path) -> list:
        with zipfile.ZipFile(archive_path) as archive:
            return sorted(archive.namelist())

    # -- a stopped run is not a finished one ---------------------------

    def test_a_stopped_run_keeps_its_place_and_is_not_handled_as_a_result(self):
        run, _job = self._stopped()

        self.assertIn(run, self.panel._runs, "the run was retired while paused")
        self.assertIsNotNone(run.paused)
        self.assertEqual(self.handled, [], "a half-run chain was opened as the answer")

    def test_a_stopped_run_is_never_started_a_second_time_by_the_pump(self):
        """`running` is what the admission pump reads. A paused run holds no
        thread, so without this it looks startable and the panel sends the same
        request twice."""
        run, _job = self._stopped()
        started = len(_Job.started)

        self.panel._pumpRuns()

        self.assertTrue(run.running)
        self.assertEqual(len(_Job.started), started)

    def test_the_reviewer_is_opened_on_what_the_run_produced(self):
        self._stopped()

        folder = self._reviewed()
        self.assertEqual(self.reviewer.opened, [folder])
        self.assertTrue(os.path.isfile(
            os.path.join(folder, "01_ALI_CBCT", "p1_lm_Pred.mrk.json")))

    def test_the_archive_is_removed_once_it_is_unpacked(self):
        """The reader corrects the unpacked files; the zip beside them is a
        second copy of a cohort nobody will open."""
        self._stopped()
        self.assertFalse(os.path.exists(os.path.join(self.work, "AREG_stopped.zip")))

    # -- and back again ------------------------------------------------

    def test_continue_sends_a_correction_per_step_and_carries_the_run_on(self):
        run, _job = self._stopped()
        self._edit(os.path.join("01_ALI_CBCT", "p1_lm_Pred.mrk.json"))
        self.reviewer.press_continue(flagged={"p1"}, written={"p1"})

        sent = self._collectResume()
        self.assertEqual(sent["run_id"], "run-1")
        self.assertEqual(sorted(sent["corrections"]), ["01_ALI_CBCT"])
        self.assertTrue(zipfile.is_zipfile(sent["corrections"]["01_ALI_CBCT"]))
        self.assertIsNone(run.paused, "the run is going again")

    def test_the_steps_after_the_review_are_still_watched(self):
        """A registration computes its residuals after the last stop. Resumed
        without a watcher, the panel went quiet and they never reached the
        console."""
        run, _job = self._stopped()
        self.reviewer.press_continue(flagged=set(), written=set())

        sent = self._collectResume()
        self.assertIsNotNone(sent["event_cb"], "progress and log lines keep arriving")
        self.assertIs(sent["cancel_event"], run.cancel_event)

    def test_a_reader_who_changed_nothing_sends_nothing_back(self):
        """A step is a cohort's worth of scans. Re-uploading every one of them
        on behalf of somebody who only looked is the expensive half."""
        self._stopped()
        self.reviewer.press_continue(flagged={"p1"}, written=set())

        self.assertEqual(self._collectResume()["corrections"], {})

    def test_the_reviewer_is_told_which_run_is_waiting(self):
        """A reader who pressed Apply in ASO and landed in a viewer needs to
        be told that is where they are. Without it the panel is a folder
        browser that appeared, and nothing says which run wants them."""
        run, _job = self._stopped()

        assert self.reviewer.origin["tool"] == self.panel.TOOL_NAME
        assert self.reviewer.origin["step"] == "ALI_CBCT"
        assert self.reviewer.origin["run"] == run.number

    def test_pressing_continue_brings_the_tool_panel_back(self):
        """The run carries on HERE -- the progress line, the elapsed time and
        whatever comes back. A reader left in the reviewer pressed Continue
        and then watched a viewer do nothing."""
        import slicer

        selected = []
        self.addCleanup(setattr, slicer.util, "selectModule",
                        getattr(slicer.util, "selectModule", None))
        slicer.util.selectModule = selected.append

        self._stopped()
        self.reviewer.press_continue(written=set())

        expected = type(self.panel).__name__
        if expected.endswith("Widget"):
            expected = expected[: -len("Widget")]
        self.assertEqual(selected[-1:], [expected])

    def test_going_back_asks_for_the_step_the_reader_named(self):
        """Continue and Go back travel the same route and differ only in the
        direction the run then moves."""
        self._stopped()
        self.reviewer.press_continue(flagged={"p1"}, rewind_to="01_ALI_CBCT")

        self.assertEqual(self._collectResume()["rewind_to"], "01_ALI_CBCT")

    def test_a_correction_made_on_the_way_back_is_not_thrown_away(self):
        """A reader who fixed something here and THEN asked for an earlier
        step still fixed it. Losing that would make going back cost work."""
        self._stopped()
        self._edit(os.path.join("01_ALI_CBCT", "p1_lm_Pred.mrk.json"))
        self.reviewer.press_continue(written={"p1"}, rewind_to="01_ALI_CBCT")

        sent = self._collectResume()
        self.assertEqual(sent["rewind_to"], "01_ALI_CBCT")
        self.assertEqual(sorted(sent["corrections"]), ["01_ALI_CBCT"])

    def test_an_ordinary_continue_asks_for_no_rewind(self):
        self._stopped()
        self.reviewer.press_continue(written=set())

        self.assertIsNone(self._collectResume()["rewind_to"])

    def test_a_single_step_flattened_into_the_archive_is_still_found(self):
        self._stopped(flat=True)
        self._edit("p1_lm_Pred.mrk.json")
        self.reviewer.press_continue(written={"p1"})

        self.assertEqual(sorted(self._collectResume()["corrections"]),
                         ["01_ALI_CBCT"])

    def test_two_steps_are_two_corrections(self):
        self._stopped(produced=("01_ALI_CBCT", "02_ASO"), steps={
            "01_ALI_CBCT": {"p1_lm_Pred.mrk.json": "{}"},
            "02_ASO": {"p1_Or.nii.gz": "volume"}})
        self._edit(os.path.join("01_ALI_CBCT", "p1_lm_Pred.mrk.json"))
        self._edit(os.path.join("02_ASO", "p1_Or.nii.gz"))

        self.reviewer.press_continue(written={"p1"})

        self.assertEqual(sorted(self._collectResume()["corrections"]),
                         ["01_ALI_CBCT", "02_ASO"])

    # -- only what the reader actually changed -------------------------

    def test_a_checkpoint_nobody_edited_sends_nothing_even_when_written(self):
        """`written` is the viewer's word for "this patient's panel saved
        something", and a save that moved no bytes still sets it. The bytes
        are what decides, so a step nothing changed contributes no field."""
        self._stopped(members={"p1.mrk.json": "{}", "p2.mrk.json": "{}"})

        self.reviewer.press_continue(flagged={"p1"}, written={"p1"})

        self.assertEqual(self._collectResume()["corrections"], {})

    def test_one_changed_file_out_of_several_is_the_only_one_sent(self):
        """The point of the whole feature: a step of AMASSS is hundreds of
        megabytes and the landmark file a reader moved a point in is eight
        kilobytes."""
        self._stopped(flat=True, members={"p1.mrk.json": "{}",
                                          "p2.mrk.json": "{}",
                                          "p3.nii.gz": "volume"})
        self._edit("p2.mrk.json", '{"moved": true}')

        self.reviewer.press_continue(written={"p2"})

        sent = self._collectResume()["corrections"]
        self.assertEqual(self._members(sent["01_ALI_CBCT"]), ["p2.mrk.json"])

    def test_a_file_the_reader_added_is_sent(self):
        """VISU writes an adjustment as its OWN transform beside the scan
        rather than folding it into the tool's. A correction that only ever
        looked at files the server produced would drop it."""
        self._stopped(flat=True, members={"p1.nii.gz": "volume"})
        self._edit("p1_adjust.tfm", "matrix")

        self.reviewer.press_continue(written={"p1"})

        sent = self._collectResume()["corrections"]
        self.assertEqual(self._members(sent["01_ALI_CBCT"]), ["p1_adjust.tfm"])

    def test_a_rewritten_file_whose_bytes_did_not_move_is_not_sent(self):
        """This is why the comparison hashes rather than stats. The reviewer
        saves on the way OUT of a patient, so mtime moves for every file a
        reader merely stepped through -- and trusting it would send the cohort
        back after all."""
        self._stopped(flat=True, members={"p1.mrk.json": "{}"})
        self._edit("p1.mrk.json", "{}")
        os.utime(os.path.join(self._reviewed(), "p1.mrk.json"), (0, 0))

        self.reviewer.press_continue(written={"p1"})

        self.assertEqual(self._collectResume()["corrections"], {})

    def test_a_correction_in_a_subfolder_keeps_its_path_in_the_zip(self):
        """A step mirrors its input tree. Flattening the member name would
        lay the correction over the wrong file, or over nothing at all."""
        self._stopped(flat=True, members={os.path.join("p1", "lm.mrk.json"): "{}",
                                          os.path.join("p2", "lm.mrk.json"): "{}"})
        self._edit(os.path.join("p1", "lm.mrk.json"), "corrected")

        self.reviewer.press_continue(written={"p1"})

        sent = self._collectResume()["corrections"]
        self.assertEqual(self._members(sent["01_ALI_CBCT"]), ["p1/lm.mrk.json"])

    def test_only_the_step_that_changed_contributes_a_field(self):
        """`_stage_corrections` names one field per step, and a step whose
        files are untouched has nothing to say. Sending it empty would cost
        the upload this change exists to avoid."""
        self._stopped(produced=("01_ALI_CBCT", "02_ASO"), steps={
            "01_ALI_CBCT": {"p1_lm_Pred.mrk.json": "{}"},
            "02_ASO": {"p1_Or.nii.gz": "volume"}})
        self._edit(os.path.join("02_ASO", "p1_Or.nii.gz"), "corrected")

        self.reviewer.press_continue(written={"p1"})

        self.assertEqual(sorted(self._collectResume()["corrections"]), ["02_ASO"])

    def test_a_correction_is_sent_from_the_tree_the_server_sends_today(self):
        """The archive puts each step under `intermediate/`. Looked for at the
        top only, no step was found and a moved landmark never went back:
        the run carried on with the point the reader had just corrected."""
        self._stopped(produced=("01_ASO", "04_ALI_CBCT"), steps={
            os.path.join("intermediate", "01_ASO"): {"p1_Or.nii.gz": "volume"},
            os.path.join("intermediate", "04_ALI_CBCT"): {"p1_Or_lm_Pred.mrk.json": "{}"}})
        self._edit(os.path.join("intermediate", "04_ALI_CBCT", "p1_Or_lm_Pred.mrk.json"),
                   "corrected")

        self.reviewer.press_continue(written={"p1"})

        sent = self._collectResume()["corrections"]
        self.assertEqual(sorted(sent), ["04_ALI_CBCT"])
        self.assertEqual(self._members(sent["04_ALI_CBCT"]), ["p1_Or_lm_Pred.mrk.json"])

    def test_a_second_checkpoint_diffs_against_what_the_first_resume_produced(self):
        """One folder is reviewed twice, and the baseline is retaken on every
        unpack. Keeping the first one would resend every correction of round
        one as if the reader had just made it."""
        self._stopped(flat=True, members={"p1.mrk.json": "{}"})
        self._edit("p1.mrk.json", "round one")
        self.reviewer.press_continue(written={"p1"})
        self._collectResume()

        _Job.started[-1].succeed(self._checkpoint(
            produced=("02_ASO",), members={"p2.nii.gz": "volume"}, flat=True))
        self.reviewer.press_continue(written={"p1"})

        self.assertEqual(self._collectResume()["corrections"], {},
                         "round one's correction was sent a second time")

    def test_the_answer_after_a_resume_is_handled_as_any_other_runs(self):
        """The point of reusing the same callbacks: the loop ends in the
        ordinary result path, not in a second one."""
        self._stopped()
        self.reviewer.press_continue(written={"p1"})

        _Job.started[-1].succeed(ToolResult(kind="text", text="carried on"))

        self.assertEqual([result.text for result in self.handled], ["carried on"])
        self.assertEqual(self.panel._runs, [], "the finished run was not retired")

    def test_a_second_checkpoint_goes_round_the_loop_again(self):
        self._stopped()
        self.reviewer.press_continue(written=set())

        _Job.started[-1].succeed(self._checkpoint(produced=("02_ASO",)))

        self.assertEqual(len(self.reviewer.opened), 2)
        self.assertEqual(self.handled, [])
        self.assertIsNotNone(self.panel._runs[0].paused)

    # -- one review holds one stop, and nothing else -------------------
    #
    # Every checkpoint of a run used to unpack into ONE folder, so the second
    # reader was handed everything the first had already reviewed -- and the
    # baseline was retaken over the mixture, which let a file from the first
    # stop, edited during the second review, travel back as a correction of a
    # step the run had already left.

    def test_a_second_stop_is_reviewed_in_a_folder_of_its_own(self):
        self._stopped()
        self.reviewer.press_continue(written=set())

        _Job.started[-1].succeed(self._checkpoint(
            produced=("02_ASO",), stopped_after="ASO",
            steps={"02_ASO": {"p1_Or.nii.gz": "oriented"}}))

        self.assertEqual(self.reviewer.opened,
                         [self._reviewed("ALI_CBCT"), self._reviewed("ASO")])

    def test_the_second_reader_is_not_shown_what_the_first_already_reviewed(self):
        self._stopped()
        self.reviewer.press_continue(written=set())
        _Job.started[-1].succeed(self._checkpoint(
            produced=("02_ASO",), stopped_after="ASO",
            steps={"02_ASO": {"p1_Or.nii.gz": "oriented"}}))

        second = self._reviewed("ASO")
        here = [name for _root, _dirs, files in os.walk(second) for name in files]
        self.assertEqual(sorted(here), ["p1_Or.nii.gz"])
        self.assertFalse(os.path.exists(
            os.path.join(second, "01_ALI_CBCT", "p1_lm_Pred.mrk.json")))

    def test_editing_the_earlier_stop_cannot_correct_the_one_being_reviewed(self):
        """A reader who goes and touches the first stop's files while standing
        at the second changes nothing here. The step is past; a correction
        that reached it would be worse than one that did not."""
        self._stopped()
        self.reviewer.press_continue(written=set())
        _Job.started[-1].succeed(self._checkpoint(
            produced=("02_ASO",), stopped_after="ASO",
            steps={"02_ASO": {"p1_Or.nii.gz": "oriented"}}))

        self._edit("01_ALI_CBCT/p1_lm_Pred.mrk.json", stop="ALI_CBCT")
        self.reviewer.press_continue(written={"p1"})

        sent = self._collectResume()["corrections"]
        self.assertEqual(sent, {}, "a finished step was corrected from behind")

    def _batchRun(self, index):
        """A batch of one cohort, as `_beginRun` builds it: its own run, its
        own checkpoint, and the SAME output folder as its siblings."""
        return types.SimpleNamespace(
            output_dir=self.work, cohort_index=index, number=index,
            checkpoint_digests={})

    def test_two_batches_of_one_cohort_do_not_share_a_review_folder(self):
        """A cohort over the server's batch size is divided, and every batch
        is an ordinary run writing into the one output folder the clinician
        chose. They stop independently, so batch 2 can well stop while a
        reader is still looking at batch 1."""
        first = self.panel._unpackCheckpoint(
            self._batchRun(1),
            self._checkpoint(steps={"01_ALI_CBCT": {"p1_lm_Pred.mrk.json": "{}"}}).checkpoint)
        second = self.panel._unpackCheckpoint(
            self._batchRun(2),
            self._checkpoint(steps={"01_ALI_CBCT": {"p5_lm_Pred.mrk.json": "{}"}}).checkpoint)

        self.assertNotEqual(first, second)
        self.assertEqual(first, self._reviewed(run="batch_01"))
        self.assertEqual(second, self._reviewed(run="batch_02"))
        self.assertTrue(os.path.isfile(
            os.path.join(first, "01_ALI_CBCT", "p1_lm_Pred.mrk.json")),
            "the second batch emptied what the first reader was reading")
        self.assertFalse(os.path.exists(
            os.path.join(second, "01_ALI_CBCT", "p1_lm_Pred.mrk.json")))

    def test_a_run_that_was_not_divided_is_not_called_a_batch(self):
        # There is no second one to tell it apart from, and `batch_01` alone
        # in a folder reads as a cohort whose other batches went missing.
        self.assertTrue(ServerToolWidgetBase._checkpointFolderName(
            types.SimpleNamespace(cohort_index=None, number=3),
            RunCheckpoint(run_id="r", stopped_after="ALI_CBCT",
                          produced=(), path=None)).startswith("run_03"))

    def test_the_reviewer_is_pointed_at_the_step_that_stopped(self):
        """A stop shows what its last step wrote first: Crown_Seg's meshes,
        not the CBCT ASO oriented two steps earlier."""
        self._stopped(produced=("01_ASO", "02_Crown_Seg"), stopped_after="Crown_Seg", steps={
            os.path.join("intermediate", "01_ASO"): {"p1_Or.nii.gz": "volume"},
            os.path.join("intermediate", "02_Crown_Seg"): {"p1_U_Seg.vtk": "mesh"}})
        self.assertTrue(self.reviewer.focus.endswith(
            os.path.join("intermediate", "02_Crown_Seg")))

    def test_a_stop_inside_a_callee_points_at_the_callee_s_folder(self):
        self._stopped(produced=("01_ASO",), stopped_after="ASO/ALI_CBCT", steps={
            os.path.join("intermediate", "01_ASO", "intermediate", "01_ALI_CBCT"):
                {"p1_lm_Pred.mrk.json": "{}"}})
        self.assertTrue(self.reviewer.focus.endswith(
            os.path.join("01_ASO", "intermediate", "01_ALI_CBCT")))

    def test_a_paused_run_refuses_a_second_apply_and_says_why(self):
        """One run at a time: a second Apply on top of a run waiting for review
        started work on data nobody had finished checking."""
        self._stopped()
        reason = self.panel._busyReason()
        self.assertIn("paused after ALI_CBCT", reason)
        self.assertIn("VISU", reason)

    def test_cancelling_a_paused_run_takes_its_review_down(self):
        """The banner kept saying the run was paused after it was withdrawn,
        with a Continue that would have resumed nothing."""
        closed = []
        self.panel._closeReviewer = closed.append
        run, _job = self._stopped()
        self.panel._cancelRun(run)
        self.assertEqual(closed, [self.reviewer.opened[-1]])

    def test_cancelling_a_run_that_never_paused_leaves_the_reviewer_alone(self):
        closed = []
        self.panel._closeReviewer = closed.append
        self._apply()
        self.panel._cancelRun(self.panel._runs[0])
        self.assertEqual(closed, [])

    def test_a_running_run_refuses_a_second_apply_too(self):
        self._apply()
        self.assertIn("still in progress", self.panel._busyReason())

    def test_an_idle_panel_has_nothing_to_refuse(self):
        self.assertEqual(self.panel._busyReason(), "")

    def test_the_way_back_to_the_review_shows_only_while_a_run_is_paused(self):
        self.panel.reviewButton = qt.QPushButton("Review")
        self.panel._syncRunControls()
        self.assertFalse(self.panel.reviewButton.isVisible())
        self._stopped()
        self.panel._syncRunControls()
        self.assertTrue(self.panel.reviewButton.isVisible())

    def test_the_reviewer_is_handed_the_run_s_own_inputs(self):
        """A step that writes only landmarks leaves its scan behind: ALI_IOS
        hands back points on a mesh the caller sent. The panel is the one
        side that has it, so it passes what exists on this machine along."""
        scan = os.path.join(self.work, "p1.nii.gz")
        open(scan, "w").close()
        self.panel.prepareInputFiles = lambda workspace: {
            "t1": scan, "t2": os.path.join(self.work, "gone.nii.gz")}
        self._stopped()
        self.assertEqual(self.reviewer.inputs, (scan,))

    def test_a_nested_stop_name_is_one_folder_rather_than_two(self):
        # `ASO/ALI_CBCT` is a legal stop name. Nested, one stop's folder would
        # sit INSIDE another's, which is the mixing this exists to prevent.
        self._stopped(stopped_after="ASO/ALI_CBCT")
        self.assertEqual(self.reviewer.opened, [self._reviewed("ASO_ALI_CBCT")])

    def test_stopping_at_a_step_again_reviews_what_it_produced_this_time(self):
        """What a run sent BACK to a step is answered with: that step's new
        output, not the union with what the previous pass left."""
        self._stopped(steps={"01_ALI_CBCT": {"p1_lm_Pred.mrk.json": "{}",
                                             "p2_lm_Pred.mrk.json": "{}"}})
        self.reviewer.press_continue(written=set())

        _Job.started[-1].succeed(self._checkpoint(
            steps={"01_ALI_CBCT": {"p1_lm_Pred.mrk.json": "{}"}}))

        folder = os.path.join(self._reviewed(), "01_ALI_CBCT")
        self.assertEqual(sorted(os.listdir(folder)), ["p1_lm_Pred.mrk.json"])

    # -- when there is nothing, or nobody, to review -------------------

    def test_a_checkpoint_with_nothing_to_show_carries_the_run_on_at_once(self):
        """The run is PAUSED on the server holding a patient's data. Leaving it
        there until the reaper because there was nothing to draw is the one
        answer that is worse than not stopping."""
        job = self._apply()
        run = self.panel._runs[0]
        run.output_dir = self.work
        job.succeed(ToolResult(kind="checkpoint", checkpoint=RunCheckpoint(
            run_id="run-1", stopped_after="halfway", produced=(), path=None)))

        self.assertEqual(self.reviewer.opened, [])
        self.assertEqual(self._collectResume()["run_id"], "run-1")
        self.assertTrue(any("carrying on" in said for said in self.announced))

    def test_a_reviewer_that_will_not_open_does_not_strand_the_run(self):
        self.reviewer.available = False
        self._stopped()

        self.assertEqual(self._collectResume()["corrections"], {})

    def test_continue_on_a_run_that_is_no_longer_stopped_does_nothing(self):
        """The reader was working while somebody pressed Cancel."""
        run, _job = self._stopped()
        run.paused = None

        started = len(_Job.started)
        self.reviewer.press_continue(written={"p1"})

        self.assertEqual(len(_Job.started), started, "a dead run was resumed")
        self.assertEqual(self.client.resumed, [])

    # -- the runs that do not stop, which is all of them today ---------

    def test_a_run_that_did_not_stop_is_handled_exactly_as_before(self):
        job = self._apply()
        run = self.panel._runs[0]

        job.succeed(ToolResult(kind="text", text="done"))

        self.assertEqual([result.text for result in self.handled], ["done"])
        self.assertNotIn(run, self.panel._runs)
        self.assertEqual(self.client.resumed, [])

    def test_a_run_that_did_not_stop_never_digests_anything(self):
        """Every run of every tool today. Digesting is two passes over a
        cohort, so a run that stopped at no checkpoint must not pay for one."""
        job = self._apply()
        run = self.panel._runs[0]

        with mock.patch.object(base_widget.digest, "digest_tree") as digesting:
            job.succeed(ToolResult(kind="text", text="done"))

        digesting.assert_not_called()
        self.assertEqual(run.checkpoint_digests, {})

    def test_a_result_that_is_not_a_ToolResult_at_all_still_goes_through(self):
        """`handleResult` takes whatever the client returned, and two of this
        repository's own fixtures hand it a plain string."""
        job = self._apply()

        job.succeed("done")

        self.assertEqual(self.handled, ["done"])

    # -- the disk work of a delivery is the worker's, not Qt's -----------

    def _deliveredOnAWorker(self, answer):
        """Apply, let the job's target run on a thread of its own, the way
        BackgroundJob runs it, then deliver on THIS thread, the main one.
        Returns the unzip and digest calls made here, which is what freezes
        the panel: 1.45 s for a 500 MB checkpoint."""
        main, made = threading.get_ident(), []

        def spy(name, real):
            def recorded(*args, **kwargs):
                if threading.get_ident() == main:
                    made.append(name)
                return real(*args, **kwargs)
            return recorded

        self.client.run = lambda tool_name, **kwargs: answer
        job = self._apply()
        self.panel._runs[0].output_dir = self.work
        with mock.patch.object(base_widget.slicer_io, "unzip_folder",
                               spy("unzip_folder", base_widget.slicer_io.unzip_folder)), \
                mock.patch.object(base_widget.digest, "digest_tree",
                                  spy("digest_tree", base_widget.digest.digest_tree)):
            delivered = []
            worker = threading.Thread(target=lambda: delivered.append(job.collect()))
            worker.start()
            worker.join(30)
            self.assertFalse(worker.is_alive(), "the job's target never returned")
            job.succeed(delivered[0])
        return made

    def test_a_checkpoint_is_unpacked_and_digested_off_the_main_thread(self):
        made = self._deliveredOnAWorker(self._checkpoint())

        self.assertEqual(made, [])
        run = self.panel._runs[0]
        self.assertEqual(self.reviewer.opened, [self._reviewed()])
        self.assertTrue(os.path.isfile(
            os.path.join(self._reviewed(), "01_ALI_CBCT", "p1_lm_Pred.mrk.json")))
        self.assertEqual(sorted(run.checkpoint_digests),
                         [os.path.join("01_ALI_CBCT", "p1_lm_Pred.mrk.json")])

    def test_a_result_archive_is_unpacked_off_the_main_thread(self):
        archive = os.path.join(self.work, "AREG_output.zip")
        with open(archive, "wb") as handle:
            handle.write(_zip_bytes({"p1/p1_lm_Pred.mrk.json": "{}"}))
        del self.panel.handleResult  # the base class's own: "save_as" unpacks
        self.panel.RESULT_KIND = "save_as"
        self.panel._schema = {}

        made = self._deliveredOnAWorker(ToolResult(kind="file", path=archive))

        self.assertEqual(made, [])
        landed = os.path.realpath(os.path.join(self.work, "p1", "p1_lm_Pred.mrk.json"))
        self.assertTrue(os.path.isfile(landed))
        self.assertFalse(os.path.exists(archive))
        self.assertEqual(self.panel._producedFiles, [landed])
        self.assertEqual(self.panel._producedRoot, self.work)


class ReviewModuleTest(unittest.TestCase):
    """The seam between the two halves, which is a NAME and a function.

    `base_widget` reaches the viewer through `importlib` so that a library
    every tool depends on does not import one particular module -- which means
    nothing fails at import time if either side is renamed. This is what fails
    instead.
    """

    def test_the_named_module_exists_and_offers_the_entry_point(self):
        module = os.path.join(_CORE, "..",
                              ServerToolWidgetBase.REVIEW_MODULE,
                              ServerToolWidgetBase.REVIEW_MODULE + ".py")
        if not os.path.isdir(os.path.dirname(module)):
            self.skipTest("the review module is not part of this extension yet")
        self.assertTrue(os.path.isfile(module), module)
        # Read rather than imported: that module needs a running Slicer, and
        # what is being checked is a DECLARATION, which is there in the text.
        with open(module, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        entry = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "open_for_review"]
        self.assertEqual(len(entry), 1,
                         "the review module must offer open_for_review()")
        self.assertEqual([arg.arg for arg in entry[0].args.args],
                         ["folder", "on_continue", "rewind", "origin", "inputs", "focus"],
                         "the seam is a signature, so a change to it is a "
                         "change to what both modules agree on")


class SlotLayoutTest(unittest.TestCase):
    """`_checkpointSlots` alone: the two shapes the archive can have."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_named_folders_are_the_steps(self):
        for name in ("01_ALI_CBCT", "02_ASO"):
            os.makedirs(os.path.join(self.root, name))

        found = ServerToolWidgetBase._checkpointSlots(
            self.root, ("01_ALI_CBCT", "02_ASO"))

        self.assertEqual(sorted(found), ["01_ALI_CBCT", "02_ASO"])
        self.assertEqual(found["02_ASO"], os.path.join(self.root, "02_ASO"))

    def test_one_step_flattened_to_the_root_is_the_root(self):
        with open(os.path.join(self.root, "p1.mrk.json"), "w") as handle:
            handle.write("{}")

        found = ServerToolWidgetBase._checkpointSlots(self.root, ("01_ALI_CBCT",))

        self.assertEqual(found, {"01_ALI_CBCT": self.root})

    def test_two_steps_of_which_one_is_missing_is_the_one_that_is_there(self):
        """A step that produced nothing is not in the archive. Reading the root
        as that step would send one step's files back under another's name."""
        os.makedirs(os.path.join(self.root, "01_ALI_CBCT"))

        found = ServerToolWidgetBase._checkpointSlots(
            self.root, ("01_ALI_CBCT", "02_ASO"))

        self.assertEqual(sorted(found), ["01_ALI_CBCT"])


if __name__ == "__main__":
    unittest.main()


class PreviousCorrectableStepTest(unittest.TestCase):
    """Which stop a reader may send flagged patients back to.

    Looking at a bad orientation is useless without a way back to the
    landmarks that caused it, so a stop that can only be LOOKED at is stepped
    over and the offer lands where something can be done.
    """

    def _panel(self, produced, kinds, stopped_after=""):
        panel = base_widget.ServerToolWidgetBase.__new__(
            base_widget.ServerToolWidgetBase)
        panel._schema = {"arguments": {"stop_after": {"option_kind": kinds}}}
        run = types.SimpleNamespace(
            paused=types.SimpleNamespace(produced=tuple(produced),
                                         stopped_after=stopped_after))
        return panel._previousCorrectableStep(run)

    def test_the_nearest_editable_step_behind_is_offered(self):
        found = self._panel(
            ["01_ALI_CBCT", "02_Crown_Seg"],
            {"ALI_CBCT": "landmarks", "Crown_Seg": "view"})
        self.assertEqual(found["slot"], "01_ALI_CBCT")
        self.assertEqual(found["kind"], "landmarks")

    def test_the_step_the_reader_is_standing_on_is_not_a_way_back(self):
        """ASO offers one checkpoint, so a reader stopped at it has nowhere
        behind them -- and a button that returns to where they already are is
        a button that does nothing twice."""
        self.assertIsNone(self._panel(
            ["01_ALI_CBCT"], {"ALI_CBCT": "landmarks"},
            stopped_after="ALI_CBCT"))

    def test_an_earlier_call_to_the_same_tool_is_still_a_way_back(self):
        """Only the LAST occurrence is the one being stood on. A chain that
        calls one tool twice can still send a reader back to the first."""
        found = self._panel(
            ["01_ALI_CBCT", "02_ALI_CBCT"], {"ALI_CBCT": "landmarks"},
            stopped_after="ALI_CBCT")
        self.assertEqual(found["slot"], "01_ALI_CBCT")

    def test_a_qualified_stop_is_named_by_its_last_segment(self):
        self.assertIsNone(self._panel(
            ["01_ALI_CBCT"], {"ALI_CBCT": "landmarks"},
            stopped_after="ASO/ALI_CBCT"))

    def test_a_step_that_can_only_be_looked_at_is_not_offered(self):
        self.assertIsNone(self._panel(
            ["01_Crown_Seg"], {"Crown_Seg": "view"}))

    def test_a_tool_the_schema_says_nothing_about_is_not_offered(self):
        """The conservative direction: an offer that leads nowhere is worse
        than no offer."""
        self.assertIsNone(self._panel(["01_Mystery"], {}))

    def test_the_slot_number_is_not_part_of_the_kind(self):
        """`02_ALI_CBCT` is the same tool as `01_ALI_CBCT` -- the number is
        the call's position, and what a reader may do there is a property of
        the tool."""
        found = self._panel(["02_ALI_CBCT"], {"ALI_CBCT": "landmarks"})
        self.assertEqual(found["tool"], "ALI_CBCT")

    def test_a_run_that_is_not_paused_offers_nothing(self):
        panel = base_widget.ServerToolWidgetBase.__new__(
            base_widget.ServerToolWidgetBase)
        panel._schema = {}
        self.assertIsNone(
            panel._previousCorrectableStep(types.SimpleNamespace(paused=None)))
