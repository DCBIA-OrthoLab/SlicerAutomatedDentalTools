"""BackgroundJob: the two timers, and why the second one exists.

A worker thread in Slicer is not a worker thread in a plain interpreter.
Slicer's embedded Python keeps the GIL on the main thread while that thread
waits inside Qt's event loop, so a background download barely progresses:
measured in Slicer, a bytecode loop drops from 13,079,723 iterations per second
to 6,632, and a 94 MB test file that a script fetches in under a second took
twenty seconds from a panel. `BackgroundJob` therefore runs a second timer
whose callback does nothing but `time.sleep`, which releases the GIL.

These tests pin the timer's LIFECYCLE -- started with the job, stopped with it,
never left running -- because a yield timer that outlives its job would keep
the main thread napping for nothing, and one that never starts silently
restores the twenty seconds.
"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs  # noqa: E402

qt_stubs.install()

from ServerToolsCoreLib import worker  # noqa: E402


def _drain_until(job, predicate, timeout=5.0):
    """Pump the job's drain timer until `predicate` holds, or give up.

    The stub timer does not fire on its own, so the test plays the event loop:
    that is exactly what the main thread does in Slicer, only deterministically.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        job._drain()
        if predicate():
            return True
        time.sleep(0.005)
    return False


class YieldTimerLifecycle(unittest.TestCase):
    def _job(self, target, **kwargs):
        return worker.BackgroundJob(target=target, **kwargs)

    def test_both_timers_start_with_the_job(self):
        job = self._job(lambda progress: "done")
        self.assertFalse(job._timer.running)
        self.assertFalse(job._yieldTimer.running)
        job.start()
        self.assertTrue(job._timer.running, "the drain timer must run")
        self.assertTrue(job._yieldTimer.running, "the worker gets no GIL without it")

    def test_the_yield_timer_stops_on_success(self):
        outcome = {}
        job = self._job(lambda progress: "done",
                        on_success=lambda value: outcome.setdefault("value", value))
        job.start()
        self.assertTrue(_drain_until(job, lambda: "value" in outcome))
        self.assertEqual(outcome["value"], "done")
        self.assertFalse(job._yieldTimer.running, "napping after the job ended")
        self.assertFalse(job._timer.running)

    def test_the_yield_timer_stops_on_error(self):
        def explode(_progress):
            raise ValueError("no")

        seen = {}
        job = self._job(explode, on_error=lambda exc: seen.setdefault("exc", exc))
        job.start()
        self.assertTrue(_drain_until(job, lambda: "exc" in seen))
        self.assertIsInstance(seen["exc"], ValueError)
        self.assertFalse(job._yieldTimer.running)
        self.assertFalse(job._timer.running)

    def test_cancel_stops_both_timers(self):
        release = threading.Event()
        job = self._job(lambda progress: release.wait(5))
        job.start()
        job.cancel()
        self.assertFalse(job._timer.running)
        self.assertFalse(job._yieldTimer.running)
        release.set()

    def test_progress_leaves_both_timers_running(self):
        """A progress line is not the end of the job, and must not stop either."""
        release = threading.Event()
        seen = []

        def slow(progress):
            progress("halfway")
            release.wait(5)
            return "done"

        job = self._job(slow, on_progress=seen.append)
        job.start()
        self.assertTrue(_drain_until(job, lambda: seen == ["halfway"]))
        self.assertTrue(job._timer.running)
        self.assertTrue(job._yieldTimer.running)
        release.set()


class CancelToken(unittest.TestCase):
    """`cancel()` used to cancel nothing at all.

    It set a flag, stopped the timers, and let the daemon thread run to
    completion inside `requests` with its outcome quietly dropped. The UI half
    of that is right and unchanged. What was missing is a token the TARGET can
    see, so the work itself can stop: a cancelled run must not go on to pull a
    four-hundred-megabyte archive down for a panel that has already closed.
    """

    def test_a_job_always_has_one_even_when_nobody_asked(self):
        job = worker.BackgroundJob(target=lambda progress: None)
        self.assertFalse(job.cancel_event.is_set())

    def test_cancelling_sets_it(self):
        job = worker.BackgroundJob(target=lambda progress: None)
        job.cancel()
        self.assertTrue(job.cancel_event.is_set())

    def test_the_caller_s_own_event_is_used_rather_than_a_second_one(self):
        """The panel hands the run's event in, because the run's progress
        watcher reads the SAME event: cancelling has to close the stream in the
        same gesture that stops the work."""
        shared = threading.Event()
        job = worker.BackgroundJob(target=lambda progress: None, cancel_event=shared)
        self.assertIs(job.cancel_event, shared)
        job.cancel()
        self.assertTrue(shared.is_set())

    def test_the_target_really_sees_it_while_it_is_still_working(self):
        running = threading.Event()
        noticed = threading.Event()
        cancel = threading.Event()

        def work(_progress):
            running.set()
            # What a real target does at its own checkpoints: between two
            # uploads, and before downloading a result.
            while not cancel.is_set():
                time.sleep(0.005)
            noticed.set()
            return "stopped"

        job = worker.BackgroundJob(target=work, cancel_event=cancel)
        job.start()
        self.assertTrue(running.wait(5))
        job.cancel()
        self.assertTrue(noticed.wait(5), "the target never saw the cancellation")

    def test_a_cancelled_job_still_delivers_nothing(self):
        """The UI half, unchanged: whatever the thread ends up doing, its
        outcome must not reach a panel that has moved on."""
        release = threading.Event()
        delivered = []
        job = worker.BackgroundJob(target=lambda progress: release.wait(5) or "done",
                                   on_success=delivered.append)
        job.start()
        job.cancel()
        release.set()
        _drain_until(job, lambda: False, timeout=0.3)
        self.assertEqual(delivered, [])


class YieldCallback(unittest.TestCase):
    def test_the_callback_sleeps_and_does_nothing_else(self):
        job = worker.BackgroundJob(target=lambda progress: None)
        started = time.perf_counter()
        job._yieldGil()
        elapsed = time.perf_counter() - started
        self.assertGreaterEqual(elapsed, worker._YIELD_SECONDS * 0.5)
        # A nap measured in milliseconds: long enough to hand the GIL over, short
        # enough that a UI event never waits a visible amount for it.
        self.assertLess(worker._YIELD_SECONDS, 0.020)

    def test_firing_the_stub_timer_runs_the_nap(self):
        """The callback is really wired to the timer, not merely defined."""
        job = worker.BackgroundJob(target=lambda progress: None)
        started = time.perf_counter()
        job._yieldTimer.fire()
        self.assertGreaterEqual(time.perf_counter() - started,
                                worker._YIELD_SECONDS * 0.5)

    def test_the_yield_timer_is_faster_than_the_drain_timer(self):
        """100 ms of donation is not enough: measured, it leaves the worker at
        453,957 iterations per second against 15,586,975 with this timer."""
        job = worker.BackgroundJob(target=lambda progress: None)
        self.assertLess(job._yieldTimer.interval, job._timer.interval)


class ItReallyUnblocksAWorker(unittest.TestCase):
    """The property the timer exists for, demonstrated without Slicer.

    Outside Slicer the main thread does not hold the GIL while idle, so this
    cannot reproduce the pathology -- it pins the mechanism instead: a sleeping
    main thread lets a worker finish, a spinning one does not.
    """

    def test_a_sleeping_main_thread_lets_the_worker_run(self):
        counted = {}

        def spin():
            loops, end = 0, time.perf_counter() + 0.20
            while time.perf_counter() < end:
                loops += 1
            counted["loops"] = loops

        thread = threading.Thread(target=spin, daemon=True)
        thread.start()
        while thread.is_alive():
            time.sleep(worker._YIELD_SECONDS)
        self.assertGreater(counted["loops"], 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
