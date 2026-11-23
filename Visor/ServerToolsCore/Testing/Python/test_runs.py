"""Several runs from one panel: the queue, the admission limit, and the cleanup.

A panel used to hold ONE `_job`, and Apply hid itself behind Cancel for the
duration. That is the wrong shape for the wait a clinician actually has, which
is a cohort: the upload of the next patient has no reason to sit behind the
inference of the previous one. `onApplyButton` now queues a `_Run` and
`_pumpRuns` starts as many as `config.CONCURRENT_RUNS` allows.

What these tests pin is the bookkeeping, not the threading -- `test_worker.py`
owns the thread. The failures they exist to catch are the quiet ones: a run
picking up another run's inputs, a scratch directory outliving its run, a
failure taking the whole queue down with it, and the callback closure binding
the wrong run.

The Slicer and Qt stand-ins come from `test_hosted_test_files`, which already
installs exactly the surface `base_widget` touches. Importing the module rather
than copying 120 lines of stubs keeps one definition of what Slicer looks like;
its own test classes stay in its namespace and are not re-run here.
"""

import contextlib
import io
import os
import sys
import threading
import time
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _CORE)

import test_hosted_test_files as fixtures  # noqa: F401,E402 - importing installs the stubs

# `_onJobSuccess` wraps `handleResult` in it, so it has to exist; the panel
# tests never reach that line, which is why the shared stub has no need of it.
_slicer_util = sys.modules["slicer.util"]
if not hasattr(_slicer_util, "tryWithErrorDisplay"):
    import contextlib

    @contextlib.contextmanager
    def _try_with_error_display(_message, **_kwargs):
        """Slicer swallows and reports; the tests want to see what escapes."""
        yield

    _slicer_util.tryWithErrorDisplay = _try_with_error_display

from ServerToolsCoreLib import base_widget, config  # noqa: E402
from ServerToolsCoreLib.base_widget import ServerToolWidgetBase  # noqa: E402
from ServerToolsCoreLib.client import normalise_run_event  # noqa: E402
from ServerToolsCoreLib.errors import RunCancelled, ServerToolError  # noqa: E402

qt = sys.modules["qt"]


class _Job:
    """Stand-in for BackgroundJob: it records itself and delivers on command.

    No thread: what is under test is which run is started, when, and with what.
    A real worker thread would only make the order non-deterministic.
    """

    started = []

    def __init__(self, target, on_success=None, on_error=None, on_progress=None, cancel_event=None):
        self.target = target
        self.on_success = on_success
        self.on_error = on_error
        self.on_progress = on_progress
        # The real BackgroundJob makes one when the caller does not; a panel
        # always does, because the run's watcher reads the same event.
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.cancelled = False

    def start(self):
        _Job.started.append(self)

    def cancel(self):
        self.cancelled = True
        self.cancel_event.set()

    def report(self, message):
        """One of the client's own messages: a plain string."""
        self.on_progress(message)

    def emit(self, **event):
        """One of the SERVER's progress events, which travels down the very
        same channel (see base_widget._startRun: `event_cb=progress_cb`).
        Defaults spell out the contract's shape so a case only states the
        field it is about."""
        payload = {"seq": 0, "at": 0.0, "state": "running", "phase": "running",
                   "fraction": None, "message": "", "depth": 0}
        payload.update(event)
        self.on_progress(payload)

    def log(self, **fields):
        """One of the tool's log lines, normalised by the real client as the
        watcher would, and sent down the same channel as progress."""
        payload = {"seq": 0, "at": 1757400000.0, "kind": "log", "state": "running",
                   "phase": "running", "level": "info", "audience": "user",
                   "message": "", "depth": 0}
        payload.update(fields)
        self.on_progress(normalise_run_event(payload))

    def succeed(self, result="done"):
        self.on_success(result)

    def fail(self, error=None):
        self.on_error(error or RuntimeError("the server said no"))


class _RecordingClient:
    """The panel's client, recording what it was asked for.

    `cancel_run` is the one method a panel calls without a job in between (see
    base_widget._requestServerCancel), so it has to be here or a teardown fails
    on an attribute rather than on its subject. `run` is here for the one case
    that invokes the task closure by hand, to see what the panel puts in it.
    """

    def __init__(self):
        self.cancelled = []
        self.calls = []

    def run(self, tool_name, **kwargs):
        self.calls.append(dict(kwargs, tool_name=tool_name))
        return "result"

    def cancel_run(self, run_id):
        self.cancelled.append(run_id)
        return True


class RunQueueTest(unittest.TestCase):
    def setUp(self):
        _Job.started = []
        self.addCleanup(setattr, base_widget, "BackgroundJob", base_widget.BackgroundJob)
        base_widget.BackgroundJob = _Job
        self.addCleanup(setattr, config, "CONCURRENT_RUNS", config.CONCURRENT_RUNS)
        config.CONCURRENT_RUNS = 1

        self.panel = self._panel()
        self.addCleanup(self.panel.onCancelButton)

    # -- the fixture ---------------------------------------------------

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
        # A real (stub) layout, not None: the per-run Cancel buttons are part
        # of what a cohort's panel offers, so they are built here rather than
        # skipped for want of somewhere to put them.
        panel._runControlsLayout = qt.QVBoxLayout()
        panel._runControlsWidget = None
        panel.applyButton = qt.QPushButton("Apply")
        panel.cancelButton = qt.QPushButton("Cancel")
        panel.client = self
        # Every run id the panel asked the server to cancel, in order. The
        # point of recording them is the runs that must NOT appear: a queued
        # run has never been sent, so cancelling it makes no HTTP call at all.
        self.cancelledRemotely = []

        # What the panel would read off its widgets. Held on the test so a case
        # can change it between two Apply clicks, which is how "the inputs are
        # read at Apply time" is checked at all.
        self.nextInput = "/data/patient_01.nii.gz"
        self.nextArgs = {"suffix": "_reg"}
        panel.prepareInputFiles = lambda workspace: {"t1": self.nextInput}
        panel.collectArgs = lambda: dict(self.nextArgs)

        # Everything downstream of a result, recorded rather than performed.
        self.handled = []
        self.phases = []
        panel.handleResult = self.handled.append
        panel._showPhase = self.phases.append
        panel._hideProgress = lambda: None
        panel._checkCanApply = lambda *args: None
        return panel

    def run_tool(self, *args, **kwargs):
        """The panel's `client.run`; never called, since the job is a stub."""
        raise AssertionError("the stub job never invokes its target")

    def cancel_run(self, run_id):
        """The panel's `client.cancel_run`, called from its own daemon thread."""
        self.cancelledRemotely.append(run_id)
        return True

    def _remoteCancels(self):
        """What reached `cancel_run`, once the thread that calls it has run.

        Joined rather than slept on: _requestServerCancel deliberately does
        this off the main thread, so a test that only looked would sometimes
        look too early.
        """
        for thread in threading.enumerate():
            if thread.name == "sadt-run-cancel":
                thread.join(timeout=5)
        return list(self.cancelledRemotely)

    def _apply(self, path=None, **args):
        if path is not None:
            self.nextInput = path
        if args:
            self.nextArgs = args
        self.panel.onApplyButton()

    def _running(self):
        return [run for run in self.panel._runs if run.running]

    # -- queueing ------------------------------------------------------

    def test_a_second_apply_queues_instead_of_being_refused(self):
        self._apply()
        self._apply()

        self.assertEqual(len(self.panel._runs), 2)
        self.assertEqual(len(_Job.started), 1, "the limit is one at a time")
        self.assertIsNone(self.panel._runs[1].started_at, "the second is queued")

    def test_the_queued_run_starts_by_itself_when_the_first_ends(self):
        self._apply()
        self._apply()

        _Job.started[0].succeed()

        self.assertEqual(len(_Job.started), 2, "the queue must walk itself")
        self.assertEqual(len(self.panel._runs), 1)
        self.assertTrue(self.panel._runs[0].running)

    def test_the_limit_decides_how_many_move_together(self):
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()
        self._apply()

        self.assertEqual(len(_Job.started), 2)
        self.assertEqual(len(self._running()), 2)
        self.assertIsNone(self.panel._runs[2].started_at)

    def test_a_limit_below_one_is_still_one(self):
        """A misconfigured zero must not silently stop every panel."""
        config.CONCURRENT_RUNS = 0
        self._apply()
        self.assertEqual(len(_Job.started), 1)

    def test_a_nonsense_limit_falls_back_to_a_queue(self):
        config.CONCURRENT_RUNS = "beaucoup"
        self._apply()
        self._apply()
        self.assertEqual(len(_Job.started), 1)

    # -- what a run carries --------------------------------------------

    def test_a_queued_run_keeps_the_inputs_it_was_given(self):
        """Read at Apply time, not at start time.

        A run that starts three minutes later because two were ahead of it must
        not pick up whatever the pickers hold by then -- that would silently run
        patient 3's scan under patient 1's request.
        """
        self._apply(path="/data/patient_01.nii.gz", suffix="_one")
        self._apply(path="/data/patient_02.nii.gz", suffix="_two")

        second = self.panel._runs[1]
        self.assertEqual(second.files, {"t1": "/data/patient_02.nii.gz"})
        self.assertEqual(second.args, {"suffix": "_two"})

        self.nextInput = "/data/patient_99.nii.gz"
        _Job.started[0].succeed()

        self.assertEqual(second.files, {"t1": "/data/patient_02.nii.gz"})

    def test_each_run_gets_its_own_scratch_directory(self):
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()

        first, second = self.panel._runs
        self.assertNotEqual(first.workspace.path, second.workspace.path)
        self.assertTrue(os.path.isdir(first.workspace.path))
        self.assertTrue(os.path.isdir(second.workspace.path))

    def test_finishing_one_run_removes_only_its_own_directory(self):
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()
        first, second = self.panel._runs
        gone, kept = first.workspace.path, second.workspace.path

        _Job.started[0].succeed()

        self.assertFalse(os.path.exists(gone))
        self.assertTrue(os.path.isdir(kept), "a sibling's inputs were deleted")

    def test_the_output_folder_defaults_to_the_run_s_own_directory(self):
        self._apply()
        run = self.panel._runs[0]
        self.assertEqual(run.output_dir, run.workspace.path)

    def test_inputs_that_cannot_be_prepared_queue_nothing(self):
        def explode(_workspace):
            raise ValueError("no scan selected")

        self.panel.prepareInputFiles = explode
        self._apply()

        self.assertEqual(self.panel._runs, [])
        self.assertEqual(_Job.started, [])

    # -- callbacks land on the right run -------------------------------

    def test_each_callback_reports_against_its_own_run(self):
        """The closure trap: without `run=run`, every callback would report
        against whichever run was queued last."""
        config.CONCURRENT_RUNS = 2
        self._apply(path="/data/one.nii.gz")
        self._apply(path="/data/two.nii.gz")

        _Job.started[0].report("uploading one")
        _Job.started[1].report("uploading two")

        self.assertEqual(self.panel._runs[0].phase, "uploading one")
        self.assertEqual(self.panel._runs[1].phase, "uploading two")

    def test_a_result_is_handed_over_once_per_run(self):
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()

        _Job.started[1].succeed("second")
        _Job.started[0].succeed("first")

        self.assertEqual(self.handled, ["second", "first"])

    def test_one_run_failing_leaves_the_rest_alone(self):
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()
        self._apply()

        _Job.started[0].fail()

        self.assertEqual(len(self.panel._runs), 2)
        self.assertEqual(len(_Job.started), 3, "the queue advanced past the failure")

    # -- cancelling and closing ----------------------------------------

    def test_cancel_takes_the_queue_with_it(self):
        self._apply()
        self._apply()
        directories = [run.workspace.path for run in self.panel._runs]

        self.panel.onCancelButton()

        self.assertEqual(self.panel._runs, [])
        self.assertTrue(_Job.started[0].cancelled)
        for path in directories:
            self.assertFalse(os.path.exists(path))

    def test_cleanup_cancels_every_run(self):
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()
        directories = [run.workspace.path for run in self.panel._runs]
        self.panel._removeOwnTestFiles = lambda: None

        self.panel.cleanup()

        self.assertEqual(self.panel._runs, [])
        self.assertTrue(all(job.cancelled for job in _Job.started))
        for path in directories:
            self.assertFalse(os.path.exists(path))

    def test_cleanup_leaves_a_paused_run_alone_on_the_server(self):
        """A paused run holds nothing -- not the card, not a worker thread --
        and the server's idle TTL already bounds it, exactly as it bounds an
        abandoned transfer. Cancelling it threw the reader's review away the
        moment the module was reloaded, which is precisely when one reloads."""
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()
        self.panel._removeOwnTestFiles = lambda: None
        asked = []
        self.panel._requestServerCancel = asked.extend
        waiting_on_a_person, still_computing = self.panel._runs
        waiting_on_a_person.paused = object()

        self.panel.cleanup()

        self.assertEqual(asked, [still_computing.run_id])


    # -- what the panel shows ------------------------------------------

    def test_apply_stays_available_so_another_run_can_be_queued(self):
        self._apply()
        self.assertTrue(self.panel.applyButton.isVisible())
        self.assertTrue(self.panel.cancelButton.isVisible())

    def test_the_cancel_button_says_how_much_it_cancels(self):
        self._apply()
        self.assertEqual(self.panel.cancelButton.text, "Cancel")
        self._apply()
        self.assertEqual(self.panel.cancelButton.text, "Cancel all")

    def test_cancel_is_hidden_again_once_nothing_is_left(self):
        self._apply()
        _Job.started[0].succeed()
        self.assertFalse(self.panel.cancelButton.isVisible())

    def test_one_run_reads_exactly_as_it_always_did(self):
        """Nine modules' worth of habit: a single run must gain no prefix."""
        self._apply()
        _Job.started[0].report("Uploading...")

        line = self.phases[-1]
        self.assertIn("Uploading...", line)
        self.assertIn("elapsed", line)
        self.assertNotIn("Run 1", line)

    def test_several_runs_get_one_line_each_naming_what_they_are(self):
        config.CONCURRENT_RUNS = 2
        self._apply(path="/data/patient_01.nii.gz")
        self._apply(path="/data/patient_02.nii.gz")

        lines = self.phases[-1].splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("patient_01.nii.gz", lines[0])
        self.assertIn("patient_02.nii.gz", lines[1])

    def test_a_queued_run_says_it_is_queued(self):
        self._apply(path="/data/patient_01.nii.gz")
        self._apply(path="/data/patient_02.nii.gz")

        lines = self.phases[-1].splitlines()
        self.assertIn("queued", lines[1])
        self.assertNotIn("queued", lines[0])

    def test_run_numbers_keep_counting_across_a_finished_run(self):
        """Two runs must never both be called "Run 1"."""
        self._apply()
        _Job.started[0].succeed()
        self._apply()
        self.assertEqual(self.panel._runs[0].number, 2)

    # -- what the server says about a run ------------------------------

    def test_every_run_carries_an_id_of_its_own(self):
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()

        first, second = self.panel._runs
        self.assertNotEqual(first.run_id, second.run_id)
        for run in (first, second):
            self.assertRegex(run.run_id, r"^[A-Za-z0-9_-]{16,64}$")

    def test_the_id_and_the_cancel_token_are_what_the_client_is_given(self):
        """Both, and the event channel with them: the id makes the run
        readable and cancellable, the token lets the client stop working for a
        run nobody wants, and the callback is how the server's progress reaches
        the panel through the queue the job already owns."""
        recorder = _RecordingClient()
        self.panel.client = recorder
        self._apply()

        _Job.started[0].target(lambda _message: None)

        call = recorder.calls[0]
        run = self.panel._runs[0]
        self.assertEqual(call["run_id"], run.run_id)
        self.assertIs(call["cancel_event"], run.cancel_event)
        self.assertIsNotNone(call["event_cb"])

    def test_a_phase_is_shown_in_the_clinician_s_words_not_the_server_s(self):
        self._apply()
        _Job.started[0].emit(phase="queued_gpu")

        line = self.phases[-1]
        self.assertIn("Waiting for the GPU", line)
        self.assertNotIn("queued_gpu", line)

    def test_the_message_and_the_percentage_ride_with_it(self):
        self._apply()
        _Job.started[0].emit(phase="running", fraction=0.35, message="patient 14 of 40")

        line = self.phases[-1]
        # This extension runs its tools locally: the phase reads "Running".
        self.assertTrue(line.startswith("Running"), line)
        self.assertIn("patient 14 of 40", line)
        self.assertIn("35%", line)

    def test_no_percentage_is_invented_when_the_tool_gave_none(self):
        self._apply()
        _Job.started[0].emit(phase="running", message="segmenting")

        self.assertNotIn("%", self.phases[-1])

    def test_a_phase_this_client_has_never_heard_of_is_still_shown(self):
        """The seam between two repositories: a phase added server-side must
        degrade to a slightly technical word, never to a run that looks like it
        stopped saying anything."""
        self._apply()
        _Job.started[0].emit(phase="uploading_to_pacs")

        self.assertIn("uploading_to_pacs", self.phases[-1])

    def test_a_supervised_chain_reads_as_one(self):
        """AREG drives ASO drives ALI. An older server sends the depth and
        nothing else -- no child's NAME -- so nesting is shown as nesting."""
        self._apply()
        _Job.started[0].emit(phase="running", depth=2, message="orienting")

        self.assertIn("\u2192 \u2192 ", self.phases[-1])

    def test_the_client_s_own_news_replaces_the_server_s_older_word(self):
        """"Downloading results" comes AFTER "packaging"; showing both would
        leave the older of the two on the panel next to the newer."""
        self._apply()
        _Job.started[0].emit(phase="packaging")
        _Job.started[0].report("Downloading results... 8.2 MB")

        line = self.phases[-1]
        self.assertIn("Downloading results", line)
        self.assertNotIn("Packaging", line)

    def test_the_bar_follows_a_real_fraction_and_hides_without_one(self):
        self.panel._progressBar = qt.QProgressBar()
        self._apply()

        _Job.started[0].emit(fraction=0.42)
        self.assertTrue(self.panel._progressBar.isVisible())
        self.assertEqual(self.panel._progressBar.value, 42)

        _Job.started[0].emit(seq=1, fraction=None)
        self.assertFalse(self.panel._progressBar.isVisible())

    # -- a tool's log lines, apart from its progress --------------------

    def test_a_log_line_between_two_progress_events_changes_nothing(self):
        """A log line is never progress. Between two progress events it must
        leave the run's phase, message, fraction and depth exactly where the
        first one put them -- and so the bar, the line and a cohort's count."""
        self.panel._progressBar = qt.QProgressBar()
        self._apply()
        job = _Job.started[0]
        run = self.panel._runs[0]

        job.emit(seq=0, phase="running", fraction=0.4, message="scan 2 of 5")
        before = (run.server_phase, run.server_message, run.fraction, run.depth,
                  self.panel._progressBar.value, self.phases[-1])

        job.log(seq=1, level="warning", message="scan 2 has no mandible", depth=2,
                state="done", phase="done", fraction=0.99)

        self.assertEqual((run.server_phase, run.server_message, run.fraction, run.depth,
                          self.panel._progressBar.value, self.phases[-1]), before)
        self.assertTrue(run.running, "a log line claiming `done` ended nothing")

        job.emit(seq=2, phase="running", fraction=0.6, message="scan 3 of 5")
        self.assertEqual(run.fraction, 0.6)
        self.assertEqual(run.server_message, "scan 3 of 5")

    def test_a_log_line_does_not_move_a_cohort_s_bar(self):
        cohort = base_widget._Cohort(2, total_scans=10)
        self._apply()
        run = self.panel._runs[0]
        run.cohort, run.scan_count = cohort, 5
        _Job.started[0].emit(fraction=0.4)
        progress = cohort.progress([run])

        _Job.started[0].log(seq=1, message="nothing to see")

        self.assertEqual(cohort.progress([run]), progress)

    def _printed(self, act):
        """What `act` printed to the console, line by line."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            act()
        return out.getvalue().splitlines()

    def test_a_log_line_is_printed_to_the_console_with_its_tool(self):
        self._apply()

        (line,) = self._printed(lambda: _Job.started[0].log(
            level="warning", message="scan 4 skipped", source="ALI_CBCT"))

        self.assertTrue(line.startswith("[{}] ".format(self.panel.TOOL_NAME)), line)
        self.assertIn("[ALI_CBCT]", line)
        self.assertIn("WARNING", line)
        self.assertIn("scan 4 skipped", line)
        self.assertNotIn("Run 1", line, "a lone run gains no prefix")

    def test_progress_is_never_printed(self):
        self._apply()
        self.assertEqual(self._printed(
            lambda: _Job.started[0].emit(seq=1, message="scan 5 of 8")), [])

    def test_several_runs_prefix_their_lines_with_the_run_they_came_from(self):
        config.CONCURRENT_RUNS = 2
        self._apply(path="/data/patient_01.nii.gz")
        self._apply(path="/data/patient_02.nii.gz")

        (line,) = self._printed(lambda: _Job.started[1].log(message="from the second"))

        self.assertIn("Run 2:", line)

    def test_nothing_a_tool_says_is_ever_logged(self):
        """The line was written about a clinician's data. It is printed to the
        console and goes nowhere else -- never through `logging`, which would
        also put it in Slicer's log file."""
        self._apply()
        with self.assertNoLogs(level="DEBUG"):
            _Job.started[0].log(level="error", message="scan 4 failed")
            _Job.started[0].emit(seq=1, message="scan 5 of 8")

    def test_a_nested_call_is_named_when_the_server_named_it(self):
        self._apply()
        job = _Job.started[0]
        job.emit(seq=0, depth=1, call="1", tool="ALI_CBCT", edge="open")
        job.emit(seq=1, depth=1, call="1", message="scan 2 of 8")

        line = self.phases[-1]
        self.assertIn("\u2192 ALI_CBCT: ", line)
        self.assertIn("scan 2 of 8", line)

        job.emit(seq=2, depth=0, message="merging")
        self.assertNotIn("ALI_CBCT", self.phases[-1])

    # -- cancelling, and who hears about it ----------------------------

    def test_cancelling_a_queued_run_makes_no_http_call_at_all(self):
        """It was never sent, so there is nothing on the server to withdraw --
        which also means a queue can be emptied with the server unreachable."""
        self._apply()
        self._apply()
        queued = self.panel._runs[1]

        self.panel._cancelRun(queued)

        self.assertNotIn(queued.run_id, self._remoteCancels())
        self.assertNotIn(queued, self.panel._runs)

    def test_cancelling_a_running_run_asks_the_server_to_stop_it(self):
        self._apply()
        run = self.panel._runs[0]

        self.panel._cancelRun(run)

        self.assertEqual(self._remoteCancels(), [run.run_id])
        self.assertTrue(run.cancel_event.is_set())

    def test_cancel_all_withdraws_every_run_that_reached_the_server(self):
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()
        self._apply()
        started = [run.run_id for run in self.panel._runs if run.started_at is not None]

        self.panel.onCancelButton()

        self.assertEqual(sorted(self._remoteCancels()), sorted(started))
        self.assertEqual(len(started), 2, "the third was queued and never sent")

    def test_cancelling_one_of_several_lets_the_queue_move_on(self):
        self._apply()
        self._apply()

        self.panel._cancelRun(self.panel._runs[0])

        self.assertEqual(len(self.panel._runs), 1)
        self.assertTrue(self.panel._runs[0].running, "the queue stalled behind a cancellation")

    def test_a_cancelled_run_closes_quietly(self):
        """A 499 is not a failure. The user asked for this, and an error dialog
        would be the panel arguing with them about it."""
        errors = self._recordErrorDisplays()
        self._apply()

        _Job.started[0].fail(RunCancelled("'AREG' was cancelled.", 499))

        self.assertEqual(errors, [])
        self.assertEqual(self.panel._runs, [])

    def test_a_real_failure_is_still_shown(self):
        """The contrast that makes the case above mean something."""
        errors = self._recordErrorDisplays()
        self._apply()

        _Job.started[0].fail(ServerToolError("The tool failed on the server.", 500))

        self.assertEqual(len(errors), 1)

    def test_a_server_message_is_shown_as_text_never_as_html(self):
        """The server writes the failure's message; in a Qt message box it
        must not become a clickable link."""
        errors = self._recordErrorDisplays()
        self._apply()

        _Job.started[0].fail(ServerToolError(
            "<a href='https://evil.example'>Re-enter your token</a>", 422))

        shown = errors[0][0]
        self.assertNotIn("<a href", shown)
        self.assertIn("&lt;a href=", shown)

    def _recordErrorDisplays(self):
        recorded = []
        self.addCleanup(setattr, _slicer_util, "errorDisplay", _slicer_util.errorDisplay)
        _slicer_util.errorDisplay = lambda *args, **kwargs: recorded.append(args)
        return recorded

    # -- one Cancel per run --------------------------------------------

    def _runButtons(self):
        host = self.panel._runControlsWidget
        return list(host.layout.widgets) if host is not None else []

    def test_a_single_run_gets_no_button_of_its_own(self):
        """The panel's own Cancel already cancels exactly that run; a second
        button saying the same thing under it is noise."""
        self._apply()
        self.assertEqual(self._runButtons(), [])

    def test_several_runs_each_get_one_naming_what_it_would_abandon(self):
        config.CONCURRENT_RUNS = 2
        self._apply(path="/data/patient_01.nii.gz")
        self._apply(path="/data/patient_02.nii.gz")

        labels = [button.text for button in self._runButtons()]
        self.assertEqual(len(labels), 2)
        self.assertIn("patient_01.nii.gz", labels[0])
        self.assertIn("patient_02.nii.gz", labels[1])

    def test_a_run_s_own_button_cancels_that_run_and_no_other(self):
        config.CONCURRENT_RUNS = 2
        self._apply(path="/data/patient_01.nii.gz")
        self._apply(path="/data/patient_02.nii.gz")
        first, second = self.panel._runs

        self._runButtons()[1].clicked.emit()

        self.assertEqual(self.panel._runs, [first])
        self.assertTrue(second.cancel_event.is_set())
        self.assertFalse(first.cancel_event.is_set())

    def test_the_buttons_of_a_finished_set_do_not_outlive_it(self):
        """Rebuilt wholesale rather than edited, so a button can never be left
        bound to a run that is gone."""
        config.CONCURRENT_RUNS = 2
        self._apply()
        self._apply()
        stale = self.panel._runControlsWidget

        _Job.started[0].succeed()

        self.assertIsNot(self.panel._runControlsWidget, stale)
        self.assertTrue(stale.deleted, "the old buttons were left in the layout")


class RunNarrativeTest(RunQueueTest):
    """The run's story in the console, one line per step.

    The panel's progress line is replaced on every event, so by the end of a
    chain -- AREG driving Crown_Seg, ALI_CBCT, ALI_IOS -- nothing on screen
    says what happened. The console keeps one line per step, in blue, and one
    at the end saying how it went and how long it took.

    A RunQueueTest for its panel and fake jobs; the tests it inherits are
    blanked right after the class, so they run once, there.
    """

    def setUp(self):
        super().setUp()
        self.shown = []
        self._show = base_widget.console.show
        base_widget.console.show = lambda text, color: self.shown.append((text, color)) or True

    def tearDown(self):
        base_widget.console.show = self._show
        super().tearDown()

    def _lines(self):
        return [text for text, _color in self.shown]

    def test_each_new_step_is_written_once(self):
        self._apply()
        job = _Job.started[0]
        job.emit(seq=1, message="labelling the crowns", fraction=0.1)
        job.emit(seq=2, message="labelling the crowns", fraction=0.2)
        job.emit(seq=3, message="predicting CBCT landmarks", fraction=0.3)

        steps = [line for line in self._lines() if "labelling" in line or "predicting" in line]
        self.assertEqual(len(steps), 2, steps)
        self.assertNotIn("%", steps[0], "a percentage is the bar's, not a step")
        self.assertTrue(all(color == base_widget.console.STEP for _t, color in self.shown))

    def test_the_next_scan_of_a_cohort_is_not_a_new_step(self):
        self._apply()
        job = _Job.started[0]
        job.emit(seq=1, message="scan 1 of 12")
        job.emit(seq=2, message="scan 2 of 12")

        self.assertEqual(sum("scan" in line for line in self._lines()), 1)

    def test_a_nested_tool_is_named_on_its_step(self):
        self._apply()
        job = _Job.started[0]
        job.emit(seq=0, depth=1, call="1", tool="ALI_CBCT", edge="open")
        job.emit(seq=1, depth=1, call="1", message="placing landmarks")

        self.assertIn("ALI_CBCT: ", self._lines()[-1])

    def test_a_line_names_the_tool_and_the_time(self):
        self._apply()
        _Job.started[0].emit(seq=1, message="merging")

        line = self._lines()[-1]
        self.assertTrue(line.startswith("[{}]".format(self.panel.TOOL_NAME)), line)
        self.assertRegex(line, r"\d\d:\d\d:\d\d")

    def test_a_finished_run_says_so_in_green_with_its_duration(self):
        self._apply()
        _Job.started[0].succeed()

        text, color = self.shown[-1]
        self.assertIn("Finished", text)
        self.assertRegex(text, r"\(\d+ s\)")
        self.assertEqual(color, base_widget.console.OUTPUT)

    def test_a_failed_run_says_why_in_red(self):
        self._apply()
        _Job.started[0].fail(ServerToolError("the GPU ran out of memory"))

        text, color = self.shown[-1]
        self.assertIn("Failed: the GPU ran out of memory", text)
        self.assertEqual(color, base_widget.console.ERROR)

    def test_a_cancelled_run_is_not_a_failure(self):
        self._apply()
        _Job.started[0].fail(RunCancelled("cancelled"))

        text, color = self.shown[-1]
        self.assertIn("Cancelled", text)
        self.assertNotEqual(color, base_widget.console.ERROR)

    def test_a_metric_is_dark_blue_and_says_so(self):
        """A residual or a volume: there to be read when it matters, and set
        apart so it can be skipped when it does not."""
        self._apply()
        _Job.started[0].log(level="metric", message="arch U: landmark fit 1.28 mm")

        text, color = self.shown[-1]
        self.assertIn("METRIC", text)
        self.assertIn("landmark fit 1.28 mm", text)
        self.assertEqual(color, base_widget.console.METRIC)

    def test_a_run_reads_as_steps_then_numbers_then_outcome(self):
        """Blank lines between the paragraphs of a run, so the metrics and the
        outcome are not lost in one column of text."""
        self.panel._lastConsoleKind = "line"  # something came before this run
        self._apply()
        job = _Job.started[0]
        job.emit(seq=1, message="registering")
        job.log(seq=2, level="metric", message="landmark fit: 1.28 mm")
        job.log(seq=3, level="metric", message="final: 93%")
        job.emit(seq=4, message="writing")
        job.succeed()

        shape = ["" if text == "" else ("M" if "METRIC" in text else "L")
                 for text in self._lines()]
        self.assertEqual(shape, ["", "L", "", "M", "M", "", "L", "", "L"])

    def test_a_sub_call_is_its_own_paragraph(self):
        """AREG driving Crown_Seg, then ALI_IOS: each sub-call set off by a
        blank line, entering and leaving it."""
        self._apply()
        job = _Job.started[0]
        job.emit(seq=1, message="labelling the crowns")
        job.emit(seq=2, depth=1, call="1", tool="Crown_Seg", edge="open")
        job.emit(seq=3, depth=1, call="1", message="mesh 1 of 2")
        job.log(seq=4, depth=1, source="Crown_Seg", message="2 meshes labelled")
        job.emit(seq=5, depth=0, message="predicting landmarks")

        shape = ["" if text == "" else "L" for text in self._lines()]
        self.assertEqual(shape, ["L", "", "L", "L", "", "L"])

    def test_the_console_carries_no_dash_and_no_arrow(self):
        self._apply()
        job = _Job.started[0]
        job.emit(seq=1, depth=1, call="1", tool="ALI_CBCT", edge="open")
        job.emit(seq=2, depth=1, call="1", message="placing landmarks", fraction=0.3)

        line = self._lines()[-1]
        self.assertNotIn("\u2014", line)
        self.assertNotIn("\u2192", line)
        self.assertIn("ALI_CBCT: placing landmarks", line)

    def test_the_panel_says_the_run_finished_instead_of_its_last_step(self):
        """It used to keep "Extracting results... 2:30 elapsed" for good."""
        self._apply()
        _Job.started[0].report("Extracting results...")
        _Job.started[0].succeed()
        self.assertRegex(self.phases[-1], r"^Finished in \d+:\d\d$")

    def test_the_panel_says_a_run_failed_or_was_cancelled(self):
        self._apply()
        _Job.started[0].fail(ServerToolError("no"))
        self.assertTrue(self.phases[-1].startswith("Failed after "))
        self._apply()
        _Job.started[-1].fail(RunCancelled("cancelled"))
        self.assertTrue(self.phases[-1].startswith("Cancelled after "))

    def test_nothing_is_written_to_the_log_file(self):
        """A step can name what the tool works on; the log file travels."""
        self._apply()
        with self.assertNoLogs(level="DEBUG"):
            _Job.started[0].emit(seq=1, message="merging")


# Not callable, so the loader skips them: they already run in RunQueueTest.
for _name in dir(RunQueueTest):
    if _name.startswith("test_") and _name not in RunNarrativeTest.__dict__:
        setattr(RunNarrativeTest, _name, None)


class StepKeyTest(unittest.TestCase):
    def test_numbers_do_not_make_a_new_step(self):
        self.assertEqual(base_widget.step_key("scan 3 of 12"), base_widget.step_key("scan 4 of 12"))

    def test_words_do(self):
        self.assertNotEqual(base_widget.step_key("segmenting"), base_widget.step_key("merging"))

    def test_durations_read_like_a_person_would_say_them(self):
        self.assertEqual(base_widget.format_duration(48), "48 s")
        self.assertEqual(base_widget.format_duration(298), "4 min 58 s")
        self.assertEqual(base_widget.format_duration(3720), "1 h 02 min")


class LogLineFormatTest(unittest.TestCase):
    """`HH:MM:SS  [source]  LEVEL  message`, the console's one line shape."""

    def setUp(self):
        self.at = time.mktime((2026, 10, 6, 14, 3, 9, 0, 0, -1))

    def test_time_source_level_and_message_in_that_order(self):
        line = base_widget.format_run_log_line(
            {"at": self.at, "level": "warning", "source": "ALI_CBCT",
             "message": "scan 4 skipped"})
        self.assertEqual(line, "14:03:09  [ALI_CBCT]  WARNING  scan 4 skipped")

    def test_the_tool_that_was_asked_for_is_not_named_again(self):
        line = base_widget.format_run_log_line(
            {"at": self.at, "level": "info", "message": "done"})
        self.assertEqual(line, "14:03:09  INFO     done")

    def test_a_prefix_names_the_run(self):
        line = base_widget.format_run_log_line(
            {"at": self.at, "level": "error", "message": "x"}, prefix="Run 2")
        self.assertTrue(line.startswith("Run 2:  14:03:09  ERROR  "), line)

    def test_without_a_server_time_the_arrival_is_used(self):
        line = base_widget.format_run_log_line(
            {"at": None, "level": "info", "message": "x"}, now=self.at)
        self.assertTrue(line.startswith("14:03:09"), line)


class OneJobPerToolTest(unittest.TestCase):
    """The default shape: one run at a time per tool, several tools at once.

    This is what a clinician expects of a panel, and what the panel did before
    it learned to queue -- minus the part where a second Apply was simply
    refused. Two properties make it up, and they are independent:

    - within ONE tool, a second Apply queues and starts by itself;
    - across tools, nothing is shared, so each panel holds its own run and they
      overlap. The cap that matters there is the server's MAX_CONCURRENT_TOOLS,
      not anything here.
    """

    def setUp(self):
        _Job.started = []
        self.addCleanup(setattr, base_widget, "BackgroundJob", base_widget.BackgroundJob)
        base_widget.BackgroundJob = _Job
        self.addCleanup(setattr, config, "CONCURRENT_RUNS", config.CONCURRENT_RUNS)

    def _panel(self, tool_name):
        panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        panel.TOOL_NAME = tool_name
        panel._runs = []
        panel._runsStarted = 0
        panel._elapsedTimer = None
        panel._outputFolderWidget = None
        panel._progressBar = None
        panel._runControlsLayout = None
        panel._runControlsWidget = None
        panel.client = _RecordingClient()
        panel.applyButton = qt.QPushButton("Apply")
        panel.cancelButton = qt.QPushButton("Cancel")
        panel.prepareInputFiles = lambda workspace: {"t1": "/data/" + tool_name + ".nii.gz"}
        panel.collectArgs = lambda: {}
        panel.handleResult = lambda result: None
        panel._showPhase = lambda text: None
        panel._hideProgress = lambda: None
        panel._checkCanApply = lambda *args: None
        self.addCleanup(panel.onCancelButton)
        return panel

    def test_the_shipped_default_is_one_run_at_a_time(self):
        """A panel running two of the same tool at once is not what a panel
        looks like. The overlap is opt-in, not the resting state."""
        self.assertEqual(config.CONCURRENT_RUNS, 1)

    def test_a_second_apply_on_one_tool_waits_for_the_first(self):
        config.CONCURRENT_RUNS = 1
        panel = self._panel("AMASSS")

        panel.onApplyButton()
        panel.onApplyButton()

        self.assertEqual(len(_Job.started), 1)
        self.assertEqual(len(panel._runs), 2, "the second is kept, not refused")

    def test_two_tools_run_at_the_same_time(self):
        config.CONCURRENT_RUNS = 1
        amasss = self._panel("AMASSS")
        ali = self._panel("ALI")

        amasss.onApplyButton()
        ali.onApplyButton()

        self.assertEqual(len(_Job.started), 2, "one panel held the other back")
        self.assertTrue(amasss._runs[0].running)
        self.assertTrue(ali._runs[0].running)

    def test_one_tool_s_queue_is_not_the_other_s(self):
        config.CONCURRENT_RUNS = 1
        amasss = self._panel("AMASSS")
        ali = self._panel("ALI")

        amasss.onApplyButton()
        amasss.onApplyButton()
        ali.onApplyButton()

        self.assertEqual(len(amasss._runs), 2)
        self.assertEqual(len(ali._runs), 1)
        self.assertEqual(len(_Job.started), 2, "ALI waited behind AMASSS's queue")

    def test_cancelling_one_tool_leaves_the_other_running(self):
        config.CONCURRENT_RUNS = 1
        amasss = self._panel("AMASSS")
        ali = self._panel("ALI")
        amasss.onApplyButton()
        ali.onApplyButton()

        amasss.onCancelButton()

        self.assertEqual(amasss._runs, [])
        self.assertEqual(len(ali._runs), 1)
        self.assertTrue(ali._runs[0].running)

    def test_raising_the_limit_is_all_it_takes_to_overlap(self):
        """The whole difference between the two shapes, in one number."""
        config.CONCURRENT_RUNS = 2
        panel = self._panel("AMASSS")

        panel.onApplyButton()
        panel.onApplyButton()

        self.assertEqual(len(_Job.started), 2)


class RunLabelTest(unittest.TestCase):
    """A run is named after what it was given, so a cohort's lines read."""

    def setUp(self):
        self.panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        self.panel.TOOL_NAME = "AREG"

    def test_it_is_the_input_s_file_name(self):
        self.assertEqual(
            self.panel._runLabel({"t1": "/data/cohort/patient_07.nii.gz"}),
            "patient_07.nii.gz")

    def test_a_folder_keeps_its_own_name_not_its_parent_s(self):
        self.assertEqual(self.panel._runLabel({"t1": "/data/cohort/T1/"}), "T1")

    def test_with_nothing_uploaded_it_falls_back_to_the_tool(self):
        self.assertEqual(self.panel._runLabel({}), "AREG")

    def test_an_empty_path_is_not_a_name(self):
        self.assertEqual(self.panel._runLabel({"t1": ""}), "AREG")


if __name__ == "__main__":
    unittest.main()
