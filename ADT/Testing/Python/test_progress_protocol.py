# The <filter-progress> channel, and the timer behind the progress labels.
#
# What matters here are the BYTES written: the four CLIs that sent
# `0 -> 2 -> 0` by hand talk to a window that compares the value it receives
# against 200. Changing the emitted line by one character breaks the link, and
# nothing reports it. The cases below therefore compare the output against the
# literal string the original copies wrote.
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

from ADTLib.format import format_timer  # noqa: E402
from ADTLib.progress_protocol import (  # noqa: E402
    PATIENT_DONE,
    STEP_DONE,
    as_fraction,
    emit_event,
    emit_fraction,
    is_event,
)


def captured(fn, *args, **kwargs):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fn(*args, **kwargs)
    return buffer.getvalue()


class EmitEventTest(unittest.TestCase):

    def test_a_patient_event_writes_exactly_what_the_four_clis_wrote(self):
        """AREG_CBCT, PRE_ASO_CBCT, SEMI_ASO_CBCT and Automatrix_CLI wrote
        these three lines by hand. The window compares the value against 200."""
        self.assertEqual(
            captured(emit_event, PATIENT_DONE, pause=0),
            "<filter-progress>0</filter-progress>\n"
            "<filter-progress>2</filter-progress>\n"
            "<filter-progress>0</filter-progress>\n")

    def test_a_step_event_writes_one_instead_of_two(self):
        self.assertEqual(
            captured(emit_event, STEP_DONE, pause=0),
            "<filter-progress>0</filter-progress>\n"
            "<filter-progress>1</filter-progress>\n"
            "<filter-progress>0</filter-progress>\n")

    def test_the_pulse_comes_back_to_zero(self):
        """Without the return to zero, two events in a row would pass for a
        single one: Slicer only wakes the window on a change."""
        lines = captured(emit_event, PATIENT_DONE, pause=0).strip().split("\n")
        self.assertEqual(lines[0], lines[-1])
        self.assertNotEqual(lines[0], lines[1])


class EmitFractionTest(unittest.TestCase):

    def test_a_fraction_is_written_with_four_decimals(self):
        self.assertEqual(captured(emit_fraction, 0.05),
                         "<filter-progress>0.0500</filter-progress>\n")

    def test_what_slicer_would_display_is_unchanged_by_the_new_format(self):
        """The original sites wrote `{x}` in some places and `{x:.2f}` in
        others. What the window takes from it is `int(x * 100)`: that is what
        has to be equal."""
        for raw in (0.0, 0.05, 1 / 3, 0.1, 0.999, 1.0, 2 / 7, 5 / 6):
            written = captured(emit_fraction, raw)
            value = float(written.split(">")[1].split("<")[0])
            self.assertEqual(int(value * 100), int(raw * 100), f"for {raw}")


class DecodeTest(unittest.TestCase):

    def test_the_constants_are_what_the_window_compares_against(self):
        """The tests hard-coded in the Display classes were
        `progress == 100` and `progress == 200`."""
        self.assertEqual(STEP_DONE, 100)
        self.assertEqual(PATIENT_DONE, 200)
        self.assertTrue(is_event(200, PATIENT_DONE))
        self.assertFalse(is_event(200, STEP_DONE))

    def test_a_fraction_passes_through_untouched(self):
        self.assertEqual(as_fraction(0.42), 0.42)
        self.assertEqual(as_fraction(1), 1)

    def test_a_scaled_value_comes_back_between_zero_and_one(self):
        """What AMASSS did by hand: `if progress > 1: progress /= 100`."""
        self.assertEqual(as_fraction(42), 0.42)
        self.assertEqual(as_fraction(200), 2.0)


class FormatTimerTest(unittest.TestCase):

    def reference(self, t):
        """The exact code of the twelve copies, kept here as a witness."""
        if t < 60:
            return f"Time : {int(t)}s"
        elif t < 3600:
            return f"Time : {int(t/60)}min and {int(t%60)}s"
        else:
            return f"Time : {int(t/3600)}h, {int(t%3600/60)}min and {int(t%60)}s"

    def test_identical_to_the_twelve_copies_it_replaces(self):
        for t in (0, 1, 59, 59.9, 60, 61, 192, 3599, 3600, 3661, 3723, 7322.5, 86399):
            self.assertEqual(format_timer(t), self.reference(t), f"for {t}")

    def test_the_three_shapes(self):
        self.assertEqual(format_timer(5), "Time : 5s")
        self.assertEqual(format_timer(192), "Time : 3min and 12s")
        self.assertEqual(format_timer(3723), "Time : 1h, 2min and 3s")


if __name__ == "__main__":
    unittest.main()
