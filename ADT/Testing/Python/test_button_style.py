# The button sheet VFACE carried four times, now generated once.
#
# VFACE spelled out four stylesheets -- standard and cancel, each in dark and
# light. Replacing every colour with a token showed the four to be the same
# 477-character template, and dark to differ from light only in the two
# `:disabled` colours. This pins that: the generated sheet must stay identical
# to what VFACE shipped, because it is what the user sees.
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

from ADTLib.theming import (  # noqa: E402
    BUTTON_ACCENTS, BUTTON_DISABLED, button_stylesheet)

# Copied verbatim from VFACE/VFACE.py before the move. Do not reformat: the
# indentation and the leading and trailing whitespace are part of the value.
PRIMARY_DARK = """
            QPushButton {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4ba3ff, stop:1 #3498db);
              color: white;
              border: none;
              border-radius: 6px;
              font-weight: 600;
              font-size: 10pt;
              padding: 8px;
              margin-top: 4px;
            }
            QPushButton:hover:!pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5cb3ff, stop:1 #2980b9);
            }
            QPushButton:pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1f618d);
            }
            QPushButton:disabled {
              background-color: #555555;
              color: #888888;
            }
            """


def skeleton(sheet):
    """The sheet with every colour replaced, so only its shape is left."""
    return re.sub(r"#[0-9a-fA-F]{6}", "@", sheet)


class ButtonStylesheetTest(unittest.TestCase):

    def test_the_standard_dark_sheet_is_unchanged(self):
        """Character for character, including indentation and whitespace."""
        self.assertEqual(button_stylesheet("primary", dark=True), PRIMARY_DARK)

    def test_the_four_sheets_share_one_shape(self):
        """That is the whole claim: one template, not four stylesheets."""
        shapes = {skeleton(button_stylesheet(accent, dark))
                  for accent in BUTTON_ACCENTS for dark in (True, False)}
        self.assertEqual(len(shapes), 1)

    def test_dark_and_light_differ_only_in_the_disabled_colours(self):
        for accent in BUTTON_ACCENTS:
            dark = button_stylesheet(accent, dark=True)
            light = button_stylesheet(accent, dark=False)
            self.assertNotEqual(dark, light)
            repainted = dark
            for shown, hidden in zip(BUTTON_DISABLED[True], BUTTON_DISABLED[False]):
                repainted = repainted.replace(shown, hidden)
            self.assertEqual(repainted, light, accent)

    def test_each_accent_paints_its_three_gradients(self):
        for accent, gradients in BUTTON_ACCENTS.items():
            sheet = button_stylesheet(accent, dark=True)
            found = re.findall(r"stop:0 (#[0-9a-f]{6}), stop:1 (#[0-9a-f]{6})", sheet)
            self.assertEqual(found, list(gradients), accent)

    def test_an_unknown_accent_is_refused(self):
        with self.assertRaises(KeyError):
            button_stylesheet("chartreuse", dark=True)


if __name__ == "__main__":
    unittest.main()
