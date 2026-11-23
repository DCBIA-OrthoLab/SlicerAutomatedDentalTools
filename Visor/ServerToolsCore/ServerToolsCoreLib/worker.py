"""Off-UI-thread execution so a slow tool call never freezes Slicer.

Hard constraint: never touch the MRML scene from a secondary thread. BackgroundJob
runs `target` on a worker thread; the outcome (or exception) is pushed onto a
queue.Queue and drained by a qt.QTimer on the main thread, which then invokes
on_success/on_error/on_progress. Everything touching slicer.* therefore stays
on the main thread.
"""

import logging
import queue
import threading
import time

import qt

logger = logging.getLogger("ServerToolsCore.worker")

_POLL_INTERVAL_MS = 100

# Slicer's embedded interpreter keeps the GIL on the main thread while that
# thread waits inside Qt's event loop, so a Python worker thread barely runs at
# all. Measured in Slicer: a bytecode loop managing 13,079,723 iterations per
# second on the main thread manages 6,632 on a worker while the main thread is
# idle in Qt -- and a 94 MB test file that downloads in under a second from a
# script took twenty seconds from a panel. The 100 ms drain timer below is not
# enough on its own: it lifts the worker to 453,957, still twenty-nine times
# short.
#
# `time.sleep` releases the GIL for its duration, so a timer whose callback
# does nothing but sleep hands the main thread's idle time to the worker. The
# same loop then measures 15,586,975 iterations per second, i.e. the transfer
# runs at full speed. What it costs is that a UI event can wait up to
# _YIELD_SECONDS for the main thread, which is a fraction of one frame.
_YIELD_INTERVAL_MS = 0
_YIELD_SECONDS = 0.001


class BackgroundJob:
    """Runs `target(progress_cb)` on a worker thread; delivers the outcome on the main thread."""

    def __init__(self, target, on_success=None, on_error=None, on_progress=None, cancel_event=None):
        self._target = target
        self._on_success = on_success
        self._on_error = on_error
        self._on_progress = on_progress
        # A cancel token the TARGET can see, which is the only kind that can
        # actually stop anything. `cancel()` on its own has never interrupted
        # the work -- it drops the outcome and releases the UI, and the thread
        # runs to completion inside `requests` regardless. A target given this
        # event can check it at its own boundaries and stop doing work nobody
        # is waiting for: no further parts uploaded, and above all no result
        # archive pulled down for a panel that has already closed.
        #
        # Supplied by the caller when something ELSE also has to be able to set
        # it (base_widget hands the same event to the run's watcher, so
        # cancelling closes the progress stream too); created here otherwise,
        # so a target that ignores it costs nothing and every job has one.
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self._queue = queue.Queue()
        self._timer = qt.QTimer()
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._drain)
        self._yieldTimer = qt.QTimer()
        self._yieldTimer.setInterval(_YIELD_INTERVAL_MS)
        self._yieldTimer.timeout.connect(self._yieldGil)
        self._thread = None
        self._cancelled = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._timer.start()
        self._yieldTimer.start()

    def cancel(self) -> None:
        """Stop delivering, and tell the target to stop working.

        Two halves, and they are not the same. The UI half is immediate and
        unconditional: nothing further is delivered and the panel is released
        whatever the thread is doing. The work half is cooperative -- setting
        `cancel_event` cannot interrupt a socket read in progress, so a target
        only notices at the checkpoints it chose; what it stops is everything
        after the current one. Stopping the RUN itself is neither of these: it
        is the server's DELETE /runs/{id}, which the panel issues alongside
        this (see base_widget._requestServerCancel).
        """
        self._cancelled = True
        self.cancel_event.set()
        self._stopTimers()

    def _stopTimers(self) -> None:
        """Both timers, always together: the yield timer outliving its job would
        keep the main thread napping for nothing."""
        self._timer.stop()
        self._yieldTimer.stop()

    def _yieldGil(self) -> None:
        """Sleep, and nothing else. The sleep IS the work (see _YIELD_SECONDS)."""
        time.sleep(_YIELD_SECONDS)

    def _run(self) -> None:
        try:
            def progress_cb(message):
                self._queue.put(("progress", message))

            result = self._target(progress_cb)
            self._queue.put(("success", result))
        except Exception as exc:
            logger.exception("Background job failed")
            self._queue.put(("error", exc))

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if self._cancelled:
                    continue
                if kind == "progress" and self._on_progress:
                    self._on_progress(payload)
                elif kind == "success":
                    self._stopTimers()
                    if self._on_success:
                        self._on_success(payload)
                elif kind == "error":
                    self._stopTimers()
                    if self._on_error:
                        self._on_error(payload)
        except queue.Empty:
            pass
