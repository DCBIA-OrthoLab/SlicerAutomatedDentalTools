"""What this extension writes to Slicer's Python console, and in which colour.

Slicer's console knows two colours: `print` is green, and everything else --
stderr, `warnings.warn`, VTK, and every log record at WARNING or above, which
Slicer echoes from its application log -- is the same red. A notice that a
file had no coordinate system and a run that crashed look identical, so the
one line that matters drowns among the ones that do not.

`ctkConsole.printMessage(text, QColor)` prints in any colour, and that is the
whole mechanism here:

- `show(text, color)` writes one coloured line, for a module that has
  something to say in a colour of its own.
- `install()` turns this extension's own WARNING records orange. A record is
  "ours" by the file that logged it, not by its logger name: the modules name
  their loggers in a dozen styles (`VISU`, `VFACE_measure`,
  `ServerToolsCore.worker`, `__name__`), and the file is the one thing they
  all share. ERROR and above stay red -- red is kept for what is actually
  wrong -- and so does anything carrying a traceback.

A warning shown in orange is still written to the application log, re-posted
at INFO with a "Warning:" prefix: posted as a warning, Slicer would echo it to
the console a second time, in red. Grepping a log for "Warning" still finds it.

Only the main thread prints. A widget touched from a worker thread can crash
Slicer, so a warning logged off the main thread is left to Slicer's own path,
in red, exactly as before.
"""

import logging
import os
import threading

OUTPUT = "#009600"  # what `print` already looks like
STEP = "#308CC6"  # the console's own prompt blue: a run's steps, as they happen
METRIC = "#1F3F8F"  # dark blue: a number a tool measured, there to be read or skipped
WARNING = "#D97706"  # amber: reads on the light and on the dark console
ERROR = "#FF0000"  # what Slicer already uses for stderr and errors

# The extension's own tree, whatever the layout: in a source checkout this
# file is <repo>/ServerToolsCore/ServerToolsCoreLib/console.py, in an installed
# extension <ext>/lib/Slicer-X.Y/qt-scripted-modules/ServerToolsCoreLib/console.py.
# Three levels up holds every module of the extension in both, and nothing
# else of Slicer's.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _console():
    try:
        import slicer
    except ImportError:
        return None
    getter = getattr(getattr(slicer, "app", None), "pythonConsole", None)
    return getter() if getter is not None else None


def show(text: str, color: str = OUTPUT) -> bool:
    """Write `text` to the Python console in `color`; False if nothing could.

    Never raises: what calls it is a run being narrated or a warning being
    shown, and neither may fail because the console could not print. A line
    ends with a newline whether or not `text` brought one: the console appends
    where the previous message stopped.
    """
    if threading.current_thread() is not threading.main_thread():
        return False
    try:
        console = _console()
        if console is None:
            return False
        import qt

        console.printMessage(text if text.endswith("\n") else text + "\n", qt.QColor(color))
    except Exception:  # noqa: BLE001 - see the docstring
        return False
    return True


def is_ours(pathname: str, root: str = _ROOT) -> bool:
    """Whether the file that logged a record belongs to this extension.

    Compared with a trailing separator, so a sibling checkout named after this
    one (`<repo>-wt-something`) is not mistaken for it.
    """
    if not pathname:
        return False
    prefix = os.path.join(os.path.normcase(os.path.abspath(root)), "")
    return os.path.normcase(os.path.abspath(pathname)).startswith(prefix)


class OrangeWarnings(logging.Filter):
    """Take this extension's WARNING records off Slicer's red path.

    Attached to Slicer's application log handler. For a record it takes, it
    prints the orange line itself, hands the handler an INFO copy for the log
    file, and returns False so the handler does not post the original as a
    warning. Anything it does not take passes through untouched.
    """

    def __init__(self, handler, show=show, root=_ROOT):
        super().__init__()
        self._handler = handler
        self._show = show
        self._root = root

    def filter(self, record) -> bool:
        if record.levelno != logging.WARNING or record.exc_info:
            return True
        if not is_ours(record.pathname, self._root):
            return True
        text = record.getMessage()
        try:
            shown = self._show(f"Warning: {text}", WARNING)
        except Exception:  # noqa: BLE001 - a console that cannot print keeps the old path
            shown = False
        if not shown:
            return True
        quiet = logging.makeLogRecord(record.__dict__)
        quiet.levelno = logging.INFO
        quiet.levelname = logging.getLevelName(logging.INFO)
        quiet.msg = f"Warning: {text}"
        quiet.args = None
        self._handler.emit(quiet)
        return False


def install() -> bool:
    """Attach `OrangeWarnings` to Slicer's log handler, once; False outside Slicer.

    Called on import by the modules that log, so whichever loads first sets it
    up; the second call finds it already there.
    """
    try:
        import slicer
    except ImportError:
        return False
    handler = getattr(slicer, "pythonApplicationLogHandler", None)
    if handler is None:
        return False
    if not any(isinstance(f, OrangeWarnings) for f in handler.filters):
        handler.addFilter(OrangeWarnings(handler))
    return True
