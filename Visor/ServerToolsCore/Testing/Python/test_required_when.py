"""An input required in one shape of the request and absent from another.

Usage:
    python3 -m unittest ServerToolsCore/Testing/Python/test_required_when.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import qt_stubs  # noqa: E402

qt, ctk = qt_stubs.install()

from ServerToolsCoreLib import formgen  # noqa: E402

BY_TYPE = {"input_layout": ["One folder per type"]}
T1 = {"type": "path", "required": False, "visible_when": BY_TYPE, "required_when": BY_TYPE}


class RequiredWhenTest(unittest.TestCase):
    def test_it_is_required_while_its_condition_holds(self):
        self.assertTrue(formgen.is_required(T1, {"input_layout": "One folder per type"}))

    def test_it_is_not_required_once_the_condition_fails(self):
        self.assertFalse(formgen.is_required(T1, {"input_layout": "One folder per patient"}))

    def test_required_still_means_always(self):
        self.assertTrue(formgen.is_required({"required": True}, {}))
        self.assertFalse(formgen.is_required({"required": False}, {}))

    def test_its_label_reads_required_not_optional(self):
        """It is shown only where it is needed, so "(optional)" was wrong
        everywhere a reader could see it."""
        self.assertTrue(formgen.may_be_required(T1))
        self.assertFalse(formgen.may_be_required({"required": False}))

    def test_the_panel_re_evaluates_on_the_argument_it_names(self):
        self.assertIn("input_layout", formgen.controlling_arguments({"t1": T1}))


if __name__ == "__main__":
    unittest.main()
