# The inputs and the buttons of a panel, in both palettes, from one call.
#
# `apply_theme` is opt-in: a module that keeps its own styling is untouched. The
# string tests below need no application; the widget tests need a running
# QApplication (Slicer) and skip otherwise.
import os
import re
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

import qt  # noqa: E402

from ADTLib import theming  # noqa: E402
from ADTLib.theming import (  # noqa: E402
    BUTTON_ACCENTS, INPUT_COLORS, apply_theme, button_stylesheet, input_stylesheet)


def skeleton(sheet):
    return re.sub(r"#[0-9a-fA-F]{6}", "@", sheet)


class InputStylesheetTest(unittest.TestCase):

    def test_dark_and_light_share_one_shape(self):
        self.assertEqual(skeleton(input_stylesheet(True)), skeleton(input_stylesheet(False)))
        self.assertNotEqual(input_stylesheet(True), input_stylesheet(False))

    def test_every_colour_of_the_palette_is_painted(self):
        for dark, colours in INPUT_COLORS.items():
            sheet = input_stylesheet(dark)
            for colour in colours:
                self.assertIn(colour, sheet)

    def test_the_dark_colours_are_the_ones_the_existing_function_paints(self):
        """A dark panel must not change: the old function and this one agree."""
        source = open(theming.__file__).read()
        old = source[source.index("def update_line_edit_and_combo_box"):source.index("# The button sheet")]
        for colour in INPUT_COLORS[True]:
            self.assertIn(colour, old)

    def test_the_light_text_is_readable_on_its_background(self):
        def luminance(code):
            r, g, b = (int(code[i:i + 2], 16) / 255 for i in (1, 3, 5))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        for dark, (background, _, text, _, _) in INPUT_COLORS.items():
            self.assertGreater(abs(luminance(background) - luminance(text)), 0.5, dark)


    def test_an_arrow_image_adds_the_arrow_rules_and_nothing_else(self):
        with_arrow = input_stylesheet(True, "/tmp/a/down.png")
        self.assertIn("QComboBox::down-arrow", with_arrow)
        self.assertIn("url(/tmp/a/down.png)", with_arrow)
        self.assertNotIn("down-arrow", input_stylesheet(True))
        self.assertEqual(len(with_arrow.split("QComboBox::drop-down")), 2)


class SecondaryAccentTest(unittest.TestCase):

    def test_secondary_has_the_shape_of_the_other_accents(self):
        self.assertIn("secondary", BUTTON_ACCENTS)
        self.assertEqual(skeleton(button_stylesheet("secondary", True)),
                         skeleton(button_stylesheet("primary", True)))

    def test_violet_and_orange_have_the_shape_of_the_others(self):
        for accent in ("violet", "orange"):
            self.assertEqual(skeleton(button_stylesheet(accent, False)),
                             skeleton(button_stylesheet("primary", False)), accent)

    def test_the_existing_accents_are_unchanged(self):
        self.assertEqual(BUTTON_ACCENTS["primary"][0], ("#4ba3ff", "#3498db"))
        self.assertEqual(BUTTON_ACCENTS["danger"][0], ("#e74c3c", "#c0392b"))


def _running_under_slicer():
    application = getattr(qt, "QApplication", None)
    return application is not None and application.instance() is not None


@unittest.skipUnless(_running_under_slicer(), "needs a running QApplication (Slicer)")
class ApplyThemeTest(unittest.TestCase):

    def panel(self):
        root = qt.QWidget()
        ui = types.SimpleNamespace(
            edit=qt.QLineEdit(root), combo=qt.QComboBox(root), label=qt.QLabel(root),
            go=qt.QPushButton(root), stop=qt.QPushButton(root),
            browse=qt.QPushButton(root), download=qt.QPushButton(root))
        return root, ui

    def test_every_widget_gets_its_sheet(self):
        root, ui = self.panel()
        apply_theme(root, ui, primary=("go",), danger=("stop",))
        dark = theming.is_dark_mode()
        self.assertIn(INPUT_COLORS[dark][0], ui.edit.styleSheet)
        self.assertIn(INPUT_COLORS[dark][0], ui.combo.styleSheet)
        self.assertEqual(ui.go.styleSheet, button_stylesheet("primary", dark))
        self.assertEqual(ui.stop.styleSheet, button_stylesheet("danger", dark))

    def test_the_other_buttons_are_secondary_by_default(self):
        root, ui = self.panel()
        apply_theme(root, ui, primary=("go",), danger=("stop",))
        dark = theming.is_dark_mode()
        for name in ("browse", "download"):
            self.assertEqual(getattr(ui, name).styleSheet, button_stylesheet("secondary", dark), name)

    def test_named_accents_take_their_buttons_out_of_the_default_group(self):
        root, ui = self.panel()
        apply_theme(root, ui, primary=("go",), danger=("stop",),
                    accents={"violet": ("download",)})
        dark = theming.is_dark_mode()
        self.assertEqual(ui.download.styleSheet, button_stylesheet("violet", dark))
        self.assertEqual(ui.browse.styleSheet, button_stylesheet("secondary", dark))

    def test_secondary_names_restrict_it(self):
        root, ui = self.panel()
        apply_theme(root, ui, primary=("go",), danger=("stop",), secondary=("browse",))
        self.assertEqual(ui.download.styleSheet, "")

    def test_an_empty_secondary_leaves_the_others_alone(self):
        root, ui = self.panel()
        apply_theme(root, ui, primary=("go",), secondary=())
        self.assertEqual(ui.browse.styleSheet, "")

    def test_the_combo_box_gets_a_drawn_arrow(self):
        root, ui = self.panel()
        theming.apply_input_style(ui, dark=True)
        sheet = ui.combo.styleSheet
        self.assertIn("QComboBox::down-arrow", sheet)
        path = re.search(r"url\(([^)]+)\)", sheet).group(1)
        self.assertTrue(os.path.isfile(path), path)
        self.assertGreater(os.path.getsize(path), 0)

    def test_the_light_input_sheet_reaches_the_widgets(self):
        root, ui = self.panel()
        theming.apply_input_style(ui, dark=False)
        self.assertIn(INPUT_COLORS[False][0], ui.edit.styleSheet)


if __name__ == "__main__":
    unittest.main()
