# What ALI CBCT sends on the progress channel, and what its widget makes of it.
#
# The two ends disagreed. The CLI printed percentages -- 5, then 20, then 20 up
# to 100 -- and Slicer multiplies what it reads by a hundred, so the widget
# received 500, 2000, up to 10000. `DisplayALICBCT.isProgress` only ever answers
# to 100 and 200. The bar and the landmark counter therefore never moved, on any
# run, and nothing said so: a progress bar that stays at zero reads like a slow
# run.
#
# The widget's own message says what it wants -- "Landmarks : x / N | Patient :
# y / M" -- so that is what the CLI sends now: one STEP_DONE per landmark, one
# PATIENT_DONE per patient, as the four CLIs whose bar does advance already did.
#
# Two fudges went with the old reading and are gone: `progress += 0.39`, and a
# `pred_step > 3` that kept the bar still until a FOURTH patient was finished --
# on a one-patient run, for ever.
import io
import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
for _path in (os.path.join(_ROOT, "ADT"), os.path.join(_ROOT, "ALI")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from ADTLib.progress_protocol import (  # noqa: E402
    PATIENT_DONE, SCALE, STEP_DONE, emit_event)
from ALI_Method.Progress import DisplayALICBCT, DisplayALIIOS  # noqa: E402

_CHANNEL = re.compile(r"<filter-progress>([\d.]+)</filter-progress>")


def printed_by(emitter):
    """The values a CLI puts on the channel, in order."""
    held, sys.stdout = sys.stdout, io.StringIO()
    try:
        emitter()
        return [float(v) for v in _CHANNEL.findall(sys.stdout.getvalue())]
    finally:
        sys.stdout = held


def as_the_widget_sees(values):
    """Slicer multiplies the printed value by a hundred before the widget."""
    return [int(v * SCALE) for v in values]


class WhatTheCliPrintsTest(unittest.TestCase):

    def test_a_landmark_is_a_pulse_of_one(self):
        """0 -> 1 -> 0: Slicer only notifies the widget when the value changes."""
        self.assertEqual(printed_by(lambda: emit_event(STEP_DONE, pause=0)),
                         [0.0, 1.0, 0.0])

    def test_a_patient_is_a_pulse_of_two(self):
        self.assertEqual(printed_by(lambda: emit_event(PATIENT_DONE, pause=0)),
                         [0.0, 2.0, 0.0])

    def test_the_widget_receives_the_two_values_it_tests(self):
        """The whole point: 100 and 200, not 500 and 2000."""
        self.assertEqual(
            as_the_widget_sees(printed_by(lambda: emit_event(STEP_DONE, pause=0))),
            [0, STEP_DONE, 0])
        self.assertEqual(
            as_the_widget_sees(printed_by(lambda: emit_event(PATIENT_DONE, pause=0))),
            [0, PATIENT_DONE, 0])

    def test_a_percentage_would_arrive_unreadable(self):
        """Why the old emission could not work, kept so the reason survives."""
        self.assertNotIn(as_the_widget_sees([5])[0], (STEP_DONE, PATIENT_DONE))
        self.assertEqual(as_the_widget_sees([5]), [500])
        self.assertEqual(as_the_widget_sees([20]), [2000])


class WhatTheWidgetDoesTest(unittest.TestCase):

    def run_through(self, display, nb_landmark, nb_scan):
        """Drive a display with the events one run of the CLI sends."""
        bar, message = 0.0, ""
        for _ in range(nb_scan):
            for _ in range(nb_landmark):
                if display.isProgress(progress=STEP_DONE, updateProgressBar=False):
                    bar, message = display()
            display.isProgress(progress=PATIENT_DONE, updateProgressBar=False)
        return bar, message

    def test_the_bar_lands_on_a_hundred(self):
        for nb_landmark, nb_scan in ((7, 1), (7, 2), (4, 1), (32, 3)):
            bar, _ = self.run_through(
                DisplayALICBCT(nb_landmark, nb_scan), nb_landmark, nb_scan)
            self.assertAlmostEqual(bar, 100.0, places=6,
                                   msg="%d landmarks x %d scan(s)"
                                       % (nb_landmark, nb_scan))

    def test_the_message_counts_landmarks_and_patients(self):
        _, message = self.run_through(DisplayALICBCT(7, 2), 7, 2)
        self.assertEqual(message, "Landmarks : 14 / 14 | Patient : 2 / 2")

    def test_one_patient_already_moves_the_bar(self):
        """`pred_step > 3` kept it at zero until a fourth patient was done."""
        display = DisplayALICBCT(7, 1)
        self.assertTrue(display.isProgress(progress=STEP_DONE,
                                           updateProgressBar=False))
        bar, message = display()
        self.assertGreater(bar, 0)
        self.assertEqual(message, "Landmarks : 1 / 7 | Patient : 0 / 1")

    def test_it_now_reads_the_channel_as_its_IOS_neighbour_does(self):
        """The two were the same display with two different fudges in CBCT."""
        for progress in (STEP_DONE, PATIENT_DONE, 500, 2000, 0):
            self.assertEqual(
                DisplayALICBCT(7, 1).isProgress(progress=progress,
                                                updateProgressBar=False),
                DisplayALIIOS(7, 1).isProgress(progress=progress,
                                               updateProgressBar=False),
                "the two disagree on %s" % progress)

    def test_a_percentage_moves_nothing(self):
        """What the old CLI sent, against the widget as it stands."""
        display = DisplayALICBCT(7, 1)
        for value in (500, 2000, 10000):
            self.assertFalse(display.isProgress(progress=value,
                                                updateProgressBar=False))


if __name__ == "__main__":
    unittest.main()
