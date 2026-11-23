"""Slicer's console prints every warning and every error in the same red.

A missing coordinate-system header and a crashed run then look alike, and the
line that matters is lost among the ones that do not. `console.OrangeWarnings`
takes this extension's own warnings off that path and prints them orange,
leaving red to errors. What these tests pin is which records it takes, that
it never loses one from the log, and that it steps aside whenever it cannot
print.
"""

import logging
import os
import sys
import threading
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _CORE)

from ServerToolsCoreLib import console  # noqa: E402

_ROOT = os.path.abspath(os.path.join(_CORE, ".."))
OURS = os.path.join(_ROOT, "VISU", "VISU.py")


class _Handler:
    """Slicer's application log handler, reduced to what the filter calls."""

    def __init__(self):
        self.emitted = []

    def emit(self, record):
        self.emitted.append(record)


def _record(level, pathname=OURS, msg="Could not open %s", args=("a.tfm",), exc_info=None):
    return logging.LogRecord("VISU", level, pathname, 248, msg, args, exc_info)


class OrangeWarningsTest(unittest.TestCase):
    def setUp(self):
        self.shown = []
        self.handler = _Handler()
        self.filter = console.OrangeWarnings(
            self.handler, show=lambda text, color: self.shown.append((text, color)) or True,
            root=_ROOT,
        )

    def test_our_warning_is_printed_orange_and_not_posted_red(self):
        self.assertFalse(self.filter.filter(_record(logging.WARNING)),
                         "posted as a warning, Slicer would echo it in red as well")
        self.assertEqual(self.shown, [("Warning: Could not open a.tfm", console.WARNING)])

    def test_our_warning_still_reaches_the_log_file(self):
        """Re-posted at INFO, which Slicer does not echo, with the word that
        lets a grep of the log find it."""
        self.filter.filter(_record(logging.WARNING))

        (quiet,) = self.handler.emitted
        self.assertEqual(quiet.levelno, logging.INFO)
        self.assertEqual(quiet.getMessage(), "Warning: Could not open a.tfm")
        self.assertEqual((quiet.pathname, quiet.lineno), (OURS, 248),
                         "the log keeps pointing at the line that warned")

    def test_an_error_stays_red(self):
        self.assertTrue(self.filter.filter(_record(logging.ERROR)))
        self.assertEqual(self.shown, [])
        self.assertEqual(self.handler.emitted, [])

    def test_a_warning_with_a_traceback_stays_red(self):
        try:
            raise ValueError("boom")
        except ValueError:
            record = _record(logging.WARNING, exc_info=sys.exc_info())

        self.assertTrue(self.filter.filter(record))
        self.assertEqual(self.shown, [])

    def test_info_is_left_alone(self):
        self.assertTrue(self.filter.filter(_record(logging.INFO)))
        self.assertEqual(self.shown, [])

    def test_slicers_own_warnings_are_not_ours_to_recolour(self):
        record = _record(logging.WARNING, pathname="/opt/Slicer/bin/Python/slicer/util.py")
        self.assertTrue(self.filter.filter(record))
        self.assertEqual(self.shown, [])

    def test_a_console_that_cannot_print_keeps_the_old_path(self):
        """No console yet, a worker thread, or a print that raised: the record
        goes out as it always did rather than vanishing."""
        for show in (lambda text, color: False, _raises):
            handler = _Handler()
            keep = console.OrangeWarnings(handler, show=show, root=_ROOT)
            self.assertTrue(keep.filter(_record(logging.WARNING)))
            self.assertEqual(handler.emitted, [])


def _raises(text, color):
    raise RuntimeError("console gone")


class IsOursTest(unittest.TestCase):
    def test_a_file_of_the_extension_is_ours(self):
        self.assertTrue(console.is_ours(OURS, _ROOT))

    def test_a_sibling_checkout_is_not(self):
        """`<repo>-wt-colors` starts with `<repo>`; only a path separator
        tells them apart."""
        sibling = _ROOT + "-wt-colors" + os.sep + "VISU" + os.sep + "VISU.py"
        self.assertFalse(console.is_ours(sibling, _ROOT))

    def test_a_record_without_a_file_is_not(self):
        self.assertFalse(console.is_ours("", _ROOT))


class ShowTest(unittest.TestCase):
    def test_nothing_is_printed_outside_slicer(self):
        self.assertFalse(console.show("hello"))

    def test_nothing_is_printed_off_the_main_thread(self):
        """A widget touched from a worker thread can crash Slicer."""
        result = []
        worker = threading.Thread(target=lambda: result.append(console.show("hello")))
        worker.start()
        worker.join()
        self.assertEqual(result, [False])

    def test_install_outside_slicer_is_a_no_op(self):
        self.assertFalse(console.install())


if __name__ == "__main__":
    unittest.main()
