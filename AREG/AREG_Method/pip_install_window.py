"""Install Python requirements with pip without wedging Slicer.

Slicer captures its own stdout and stderr into a pipe it reads back, and
anything written there while the Qt event loop cannot turn risks filling that
pipe with no reader left: the writing thread then blocks in write() for ever.
The extension already had to work around it twice, for a CLI's output and for a
conda tool's, and `slicer.util.pip_install` walks straight into it: its progress
dialog installs a logCallback that calls `print` on every line pip produces, and
the timer draining that queue empties it whole before giving the loop its turn.
A dependency check that downloads torch or numpy prints tens of thousands of
lines, progress bars included, and one of those bursts is what froze a 5.12.4
session on 2026-09-15, panel up, no error, the pip process left as a zombie.

The window below does the same job without ever writing to stdout: pip runs in
Slicer's non-blocking mode, whose reader lives on a worker thread, and the lines
it hands back only ever reach a text widget.

Nothing here is specific to AREG; it is meant to move to a package shared by the
other modules once they are wired to it too.
"""

import logging

import qt
import slicer
from slicer.util import pip_install

logger = logging.getLogger(__name__)

# What the window keeps of a noisy install. Older lines are dropped rather than
# grown into: pip's progress bars alone run to tens of thousands of lines.
PIP_LOG_MAX_LINES = 2000


class PipInstallWindow:
    """A modal window showing one pip install after another.

    Use it as a context manager and call :meth:`install` once per requirement::

        with PipInstallWindow(requester="AREG") as window:
            for requirement in requirements:
                if not window.install(requirement):
                    break
    """

    def __init__(self, requester=None, parent=None):
        self._requester = requester
        title = "Installing Python packages"
        if requester:
            title = f"{requester} - {title}"

        self._dialog = qt.QDialog(parent or slicer.util.mainWindow())
        self._dialog.setModal(True)
        self._dialog.setWindowTitle(title)
        # No close button and no Escape: leaving pip half-way through would
        # strand the environment between two versions of a library.
        self._dialog.setWindowFlags(
            self._dialog.windowFlags() & ~qt.Qt.WindowCloseButtonHint
        )
        self._escapeShortcut = qt.QShortcut(
            qt.QKeySequence(qt.Qt.Key_Escape), self._dialog
        )
        self._escapeShortcut.setContext(qt.Qt.WidgetWithChildrenShortcut)

        layout = qt.QVBoxLayout(self._dialog)

        self._statusLabel = qt.QLabel("Preparing...")
        layout.addWidget(self._statusLabel)

        self._progressBar = qt.QProgressBar()
        self._progressBar.setRange(0, 0)  # indeterminate: pip gives no total
        layout.addWidget(self._progressBar)

        self._logText = qt.QPlainTextEdit()
        self._logText.setReadOnly(True)
        # Bounds the widget's own memory the way the conda queue is bounded.
        self._logText.setMaximumBlockCount(PIP_LOG_MAX_LINES)
        self._logText.setMinimumHeight(180)
        font = qt.QFont("Monospace")
        font.setStyleHint(qt.QFont.TypeWriter)
        self._logText.setFont(font)
        layout.addWidget(self._logText)

        self._dialog.resize(700, 360)

        self._lines = []

    # -- context manager ---------------------------------------------------

    def __enter__(self):
        self.show()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def show(self):
        self._dialog.show()
        slicer.app.processEvents()  # let the window paint before pip starts

    def close(self):
        self._dialog.close()

    # -- content -----------------------------------------------------------

    def setStatus(self, text):
        self._statusLabel.setText(text)

    def appendLine(self, line):
        self._lines.append(line)
        self._logText.appendPlainText(line)
        scroll_bar = self._logText.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum)

    def log(self):
        """Everything pip printed since the window opened."""
        return "\n".join(self._lines)

    # -- the install itself ------------------------------------------------

    def install(self, requirement):
        """Install one requirement. Returns True when pip succeeded.

        Never raises on a pip failure: the caller decides what a failed
        requirement means, and the output stays on screen either way.
        """
        self.setStatus(f"Installing {requirement}...")
        self.appendLine(f"$ pip install {requirement}")

        result = {}

        def onLog(line):
            self.appendLine(line)

        def onCompleted(returnCode):
            result["returnCode"] = returnCode

        try:
            pip_install(
                requirement,
                blocking=False,
                show_progress=True,
                requester=self._requester,
                logCallback=onLog,
                completedCallback=onCompleted,
            )
        except TypeError:
            # Slicer too old for the callback API. Its pip_install prints one
            # line at a time and calls processEvents between them, so the loop
            # gets to drain the pipe: that older path is safe as it stands.
            return self._installBlocking(requirement)
        except Exception as e:
            logger.warning(f"pip could not be started for {requirement}: {e}")
            self.appendLine(f"pip could not be started: {e}")
            return False

        # pip now runs on a worker thread; keep the loop turning so the window
        # repaints and, just as importantly, so Slicer keeps draining the pipe
        # its own captured output goes into.
        while "returnCode" not in result:
            slicer.app.processEvents()
            qt.QThread.msleep(10)

        return_code = result["returnCode"]
        if return_code != 0:
            self.appendLine(f"pip exited with code {return_code}")
        return return_code == 0

    def _installBlocking(self, requirement):
        """Fallback for Slicer versions without the non-blocking pip API."""
        try:
            pip_install(requirement)
            return True
        except Exception as e:
            logger.warning(f"Installing {requirement} failed: {e}")
            self.appendLine(f"Installing {requirement} failed: {e}")
            return False
