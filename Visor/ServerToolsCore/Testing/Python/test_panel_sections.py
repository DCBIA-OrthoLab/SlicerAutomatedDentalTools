"""Sections that open folded, and the two moments the fold has to be applied.

`_COLLAPSED_SECTIONS` is a one-line intention -- "Advanced" opens shut -- and
it has now failed twice for the same underlying reason: a ctkCollapsibleButton
folds by HIDING the children it has at that instant, so when the fold happens
matters more than the flag does.

The first failure was a row added after the fold, and the fix was to fold at
the end of the build. The second is subtler: `_buildForm` assembles the whole
panel into a DETACHED QWidget and parents it into the module afterwards, so
"the end of the build" is still a tree Qt has never realised, and the hide does
not survive the show that follows. The box reads as folded with its rows drawn
underneath it -- which is exactly what a "crushed" Advanced section looks like.

Neither failure was pinned by a test, which is why the second one shipped.
"""

import os
import sys
import unittest

_HERE = os.path.abspath(os.path.dirname(__file__))
_CORE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _CORE)

import test_hosted_test_files as fixtures  # noqa: E402,F401 - installs the stubs

ctk = fixtures.ctk
from ServerToolsCoreLib.base_widget import ServerToolWidgetBase  # noqa: E402


class FoldedSectionsTest(unittest.TestCase):
    def _panel(self, pending: bool):
        panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        # Falsy, so enter() skips design.apply -- the stylesheet is not the
        # subject and applying it needs a whole widget tree.
        panel.uiWidget = None
        panel._collapsePending = pending
        panel._sectionBoxes = {"Advanced": ctk.ctkCollapsibleButton()}
        # What else enter() does. None of it is the subject, and all of it
        # wants a server.
        panel._sweepLeftoverTestFiles = lambda: None
        panel._refreshSchema = lambda: None
        panel._refreshServerSelectables = lambda: None
        panel._refreshSceneVolumes = lambda: None
        panel._refreshServerStatus = lambda: None
        return panel

    def test_the_first_enter_applies_the_fold_the_build_asked_for(self):
        panel = self._panel(pending=True)
        self.assertFalse(panel._sectionBoxes["Advanced"].collapsed)

        panel.enter()

        self.assertTrue(panel._sectionBoxes["Advanced"].collapsed)
        self.assertFalse(panel._collapsePending, "the request must be consumed")

    def test_a_section_the_user_opened_survives_leaving_and_coming_back(self):
        """Folding on EVERY enter() would be worse than not folding at all: a
        clinician who opened Advanced to reach a field, switched module and came
        back, would find it shut again with no idea why."""
        panel = self._panel(pending=True)
        panel.enter()

        panel._sectionBoxes["Advanced"].collapsed = False  # the user opens it
        panel.enter()

        self.assertFalse(panel._sectionBoxes["Advanced"].collapsed)

    def test_nothing_is_folded_when_no_build_asked_for_one(self):
        panel = self._panel(pending=False)

        panel.enter()

        self.assertFalse(panel._sectionBoxes["Advanced"].collapsed)

    def test_the_fold_records_the_rows_it_hid(self):
        """The stub records how many rows a box held when it folded, because
        folding EMPTY is the first failure this file exists for: a box folded
        before its rows arrive reads as collapsed and draws them anyway."""
        panel = self._panel(pending=True)
        box = panel._sectionBoxes["Advanced"]
        box.layout = fixtures.qt.QFormLayout(box)
        box.layout.addRow("Masks", object())
        box.layout.addRow("Initial transforms", object())

        panel.enter()

        self.assertEqual(box.rowsWhenCollapsed, 2)


if __name__ == "__main__":
    unittest.main()


class TheFoldHasToActuallyRunTest(unittest.TestCase):
    """`collapsed` reading True is not evidence that anything was hidden.

    CTK's setter returns early when handed the value it already holds, so a
    panel that folds during its build and folds again once on screen gets one
    real fold and one silent no-op -- and the one that counted is the one that
    ran on a detached tree. `foldsApplied` counts the passes that did work.
    """

    def _panel(self):
        panel = ServerToolWidgetBase.__new__(ServerToolWidgetBase)
        panel.uiWidget = None
        panel._collapsePending = False
        panel._sectionBoxes = {"Advanced": ctk.ctkCollapsibleButton()}
        panel._sweepLeftoverTestFiles = lambda: None
        panel._refreshSchema = lambda: None
        panel._refreshServerSelectables = lambda: None
        panel._refreshSceneVolumes = lambda: None
        panel._refreshServerStatus = lambda: None
        return panel

    def test_folding_an_already_folded_box_still_hides_its_rows(self):
        panel = self._panel()
        box = panel._sectionBoxes["Advanced"]

        panel._collapseAdvancedSections()          # the build's fold
        self.assertEqual(box.foldsApplied, 1)

        panel._collapsePending = True
        panel.enter()                              # the one that reaches a screen

        self.assertTrue(box.collapsed)
        self.assertEqual(
            box.foldsApplied, 2,
            "the second fold was a no-op, so nothing was hidden on screen",
        )
