"""The "Select all" / "Deselect all" pair on a multichoice.

A chain of four steps, or a catalogue of a hundred landmarks, is not something
anyone ticks one box at a time. The pair is opt-in per argument (`select_all`
in the schema) rather than on every multichoice, because turning it on
everywhere would change every panel that has one.

What matters on the wire is that it changes NOTHING: whatever the buttons do,
`value()` reads back the same complete {option: checked} dict it always did.
"""

import os
import sys
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _CORE)

import test_hosted_test_files as fixtures  # noqa: F401,E402 - installs the stubs

from ServerToolsCoreLib import design, formgen  # noqa: E402

STEPS = {"ALI_CBCT": False, "ALI_IOS": False, "ASO": False, "Crown_Seg": False}


def _group(choices=None, **kwargs):
    return formgen.MultiChoiceGroup(dict(choices or STEPS), **kwargs)


class SelectAllTest(unittest.TestCase):
    def test_it_is_absent_unless_the_schema_asks_for_it(self):
        """Every multichoice that exists today renders exactly as it did."""
        group = _group()
        self.assertIsNone(group.selectAllButton)
        self.assertIsNone(group.selectNoneButton)

    def test_asking_for_it_gives_both_buttons(self):
        group = _group(select_all=True)
        self.assertIsNotNone(group.selectAllButton)
        self.assertIsNotNone(group.selectNoneButton)

    def test_one_option_gets_no_buttons(self):
        """Two buttons commanding a single check box read as more of a decision
        than the check box is. ASO's chain is one step long."""
        group = _group({"ALI_CBCT": False}, select_all=True)
        self.assertIsNone(group.selectAllButton)

    def test_select_all_ticks_every_option(self):
        group = _group(select_all=True)
        group.selectAllButton.click()

        self.assertEqual(group.value(), {name: True for name in STEPS})

    def test_select_none_clears_every_option(self):
        group = _group({name: True for name in STEPS}, select_all=True)
        group.selectNoneButton.click()

        self.assertEqual(group.value(), {name: False for name in STEPS})

    def test_the_value_read_back_is_the_same_shape_either_way(self):
        """The invariant the whole widget layer rests on: a presentation hint is
        never visible on the wire."""
        plain, with_buttons = _group(), _group(select_all=True)
        self.assertEqual(plain.value(), with_buttons.value())
        self.assertEqual(list(plain.boxes), list(with_buttons.boxes))

    def test_the_buttons_notify_whoever_was_listening(self):
        """Each box emits its own signal, so Apply re-evaluates exactly as it
        does when a clinician clicks a box."""
        seen = []
        group = _group(select_all=True)
        formgen.connect_changed(group, lambda *a: seen.append(1))

        group.selectAllButton.click()
        self.assertTrue(seen, "ticking four boxes notified nobody")

    def test_a_redraw_does_not_leave_two_of_each_button(self):
        """`rebuild` empties the column; anything added once in __init__ used to
        survive as an orphan. The description had exactly that bug, back when
        the field printed one."""
        group = _group(select_all=True)
        group.rebuild({"ALI_CBCT": False, "ASO": False})

        widgets = getattr(group._column, "widgets", [])
        holders = [w for w in widgets if getattr(w, "layout", None) is not None]
        self.assertEqual(len(holders), 1, "the button row was drawn twice")


if __name__ == "__main__":
    unittest.main()


class TheyLookLikeControlsNotLinksTest(unittest.TestCase):
    """The pair shipped as two underlined captions, and read as one broken
    sentence floating between a grey paragraph and a row of check boxes.

    What it looks like is not decoration here: this is the control that makes a
    four-step chain one click instead of four, and a control nobody recognises
    as one is a control nobody presses.
    """

    def _buttons(self):
        group = _group(select_all=True)
        return group.selectAllButton, group.selectNoneButton

    def test_neither_one_is_underlined(self):
        """An underline is this extension's vocabulary for something that opens
        ELSEWHERE -- Server logs, Check for updates. These act on the list
        directly under them."""
        for button in self._buttons():
            self.assertNotIn("text-decoration: underline", button._stylesheet)

    def test_each_one_is_outlined_so_it_reads_as_a_control(self):
        """The panel's own way of saying "a control": every one of them is a
        surface with a hairline round it, and a bulk command that said it
        differently would read as a different kind of thing."""
        for button in self._buttons():
            self.assertIn("1px solid {}".format(design.tokens()["BORDER"]),
                          button._stylesheet)
            self.assertIn("border-radius", button._stylesheet)
            self.assertIn(design.tokens()["SURFACE"], button._stylesheet)

    def test_neither_one_is_filled_like_apply(self):
        """A filled blue slab a few rows above Apply competes with the one
        button that starts a run. That is why these are not primary buttons,
        and the tabbed layout's full-width pair is."""
        for button in self._buttons():
            self.assertNotIn("qlineargradient", button._stylesheet)
            self.assertNotIn("background-color: " + design.tokens()["PRIMARY"],
                             button._stylesheet)

    def test_the_text_is_the_panels_own_size(self):
        """Shrunk to 8pt they were the smallest thing on the panel and in the
        loudest colour, which is the wrong way round for a secondary control --
        and this panel has already been told twice that its small text cannot
        be read."""
        for button in self._buttons():
            self.assertNotIn("font-size", button._stylesheet)

    def test_the_same_two_words_are_used_wherever_the_pair_appears(self):
        """A tabbed multichoice draws its own per-tab pair. It said "Deselect
        All" while this one said "Select none" -- one action, two names, and a
        reader cannot tell they do the same thing."""
        self.assertEqual(formgen.SELECT_GROUP_LABEL, design.SELECT_ALL_TEXT)
        self.assertEqual(formgen.CLEAR_GROUP_LABEL, design.SELECT_NONE_TEXT)
        for button, text in zip(self._buttons(),
                                (design.SELECT_ALL_TEXT, design.SELECT_NONE_TEXT)):
            self.assertEqual(button.text, text)
