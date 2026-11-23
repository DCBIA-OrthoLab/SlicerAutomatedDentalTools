"""A choice drawn as segments: every option visible, one pressed, read first.

Usage:
    python3 -m unittest ServerToolsCore/Testing/Python/test_segmented_choice.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import qt_stubs  # noqa: E402

qt, ctk = qt_stubs.install()

from ServerToolsCoreLib import formgen  # noqa: E402

LAYOUT = {
    "type": "choice",
    "ui": formgen.SEGMENTED_UI,
    "choices": {"One folder per type": True, "One folder per patient": False},
    "option_help": {"One folder per patient": "patients/\n    patient001/"},
}


class SegmentedChoiceTest(unittest.TestCase):
    def setUp(self):
        self.layout = qt.QFormLayout()
        self.widgets = formgen.build({"input_layout": dict(LAYOUT)}, self.layout)
        self.widget = self.widgets["input_layout"]

    def test_the_hint_draws_segments_rather_than_a_drop_down(self):
        self.assertIsInstance(self.widget, formgen.SegmentedChoice)
        self.assertEqual(list(self.widget.buttons), list(LAYOUT["choices"]))

    def test_it_opens_on_the_declared_default(self):
        self.assertEqual(formgen._read_widget(self.widget), "One folder per type")
        self.assertTrue(self.widget.buttons["One folder per type"].isChecked())
        self.assertFalse(self.widget.buttons["One folder per patient"].isChecked())

    def test_a_click_selects_and_tells_the_form(self):
        seen = []
        formgen.connect_changed(self.widget, seen.append)

        self.widget.buttons["One folder per patient"].click()

        self.assertEqual(formgen._read_widget(self.widget), "One folder per patient")
        self.assertEqual(seen, ["One folder per patient"])
        self.assertFalse(self.widget.buttons["One folder per type"].isChecked())

    def test_clicking_the_pressed_segment_keeps_it_pressed_and_says_nothing(self):
        """A checkable button un-checks itself on a click; one option is
        always chosen, and nothing changed."""
        seen = []
        formgen.connect_changed(self.widget, seen.append)
        button = self.widget.buttons["One folder per type"]
        button.setChecked(False)  # what Qt does to a checked button on click

        button.click()

        self.assertTrue(button.isChecked())
        self.assertEqual(seen, [])

    def test_each_segment_carries_its_own_help_line_breaks_kept(self):
        """The example tree is the help, and a tree needs its lines."""
        tooltip = self.widget.buttons["One folder per patient"].toolTip()
        self.assertEqual(tooltip, "<qt><pre>patients/\n    patient001/</pre></qt>")
        self.assertFalse(self.widget.buttons["One folder per type"].toolTip())

    def test_help_from_the_server_stays_inert_inside_the_block(self):
        """Only the <pre> is markup: a server's text is escaped like any other."""
        widget = formgen.SegmentedChoice(
            ["a", "b"], "a", help_texts={"a": "<a href='x'>click</a>"})
        self.assertEqual(
            widget.buttons["a"].toolTip(),
            "<qt><pre>&lt;a href=&#x27;x&#x27;&gt;click&lt;/a&gt;</pre></qt>",
        )

    def test_without_the_hint_a_choice_is_still_a_drop_down(self):
        plain = {key: value for key, value in LAYOUT.items() if key != "ui"}
        widgets = formgen.build({"input_layout": plain}, qt.QFormLayout())
        self.assertIsInstance(widgets["input_layout"], qt.QComboBox)


class SegmentedChoicePlacementTest(unittest.TestCase):
    def test_it_heads_its_section_above_rows_already_there(self):
        """The file inputs are laid out before any scalar: the choice that
        decides which of them are asked for has to come above them."""
        layout = qt.QFormLayout()
        layout.addRow(qt.QLabel("T1"), qt.QWidget())
        formgen.build({
            "automation": {"type": "choice", "choices": {"A": True, "B": False}},
            "input_layout": dict(LAYOUT),
        }, layout)

        labels = [label.text for label, _widget in layout.rows]
        self.assertEqual(labels[0], formgen.label_for("input_layout", LAYOUT))
        self.assertEqual(labels[1], "T1")


if __name__ == "__main__":
    unittest.main()
