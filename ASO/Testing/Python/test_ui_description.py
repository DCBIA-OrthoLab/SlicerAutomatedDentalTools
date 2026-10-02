# What the interface has to show for each ASO method.
#
# These values used to live in a chain of four `if/elif` branches on dropdown
# indices, and in an `isinstance`. Here they are, frozen as they stood: if a
# page of the `stackedWidget` or a label changes, that is a choice, and this
# test says so.
#
# The classes alone are enough: the description is made of class attributes,
# with no Qt and no instance.
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ADT = os.path.join(_HERE, "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from ASO_Method.CBCT import Auto_CBCT, Semi_CBCT  # noqa: E402
from ASO_Method.IOS import Auto_IOS, Semi_IOS  # noqa: E402

# (input type, mode) -> method, as the `if/elif` chain decided it
COMBOS = {
    (0, 1): Semi_CBCT,
    (0, 0): Auto_CBCT,
    (1, 1): Semi_IOS,
    (1, 0): Auto_IOS,
}

# method -> (page, type, CBCT input shown, label, segmentation model)
EXPECTED = {
    Semi_CBCT: (0, "CBCT", True, None, False),
    Auto_CBCT: (1, "CBCT", True, "Orientation Model Folder", False),
    Semi_IOS: (2, "IOS", False, None, True),
    Auto_IOS: (3, "IOS", False, "Segmentation Model Folder", True),
}


class UiDescriptionTest(unittest.TestCase):

    def test_each_method_describes_what_the_if_elif_chain_did(self):
        for method, expected in EXPECTED.items():
            got = (method.stacked_page, method.scan_type, method.shows_cbct_input,
                   method.model_label, method.uses_segmentation_model)
            self.assertEqual(got, expected, method.__name__)

    def test_the_four_pages_are_distinct(self):
        pages = [m.stacked_page for m in EXPECTED]
        self.assertEqual(sorted(pages), [0, 1, 2, 3])

    def test_the_table_covers_every_combination_of_the_two_dropdowns(self):
        """Two dropdowns of two entries: the table has to be total, otherwise
        one of the user's choices would change nothing -- which is what the
        old chain with no `else` did."""
        self.assertEqual(sorted(COMBOS), [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_the_segmentation_model_flag_says_what_isinstance_said(self):
        """The test was `isinstance(meth, (Auto_IOS, Semi_IOS))`."""
        for method, expected in EXPECTED.items():
            self.assertEqual(method.uses_segmentation_model,
                             issubclass(method, Auto_IOS), method.__name__)

    def test_semi_ios_inherits_from_auto_ios_and_only_changes_two_things(self):
        self.assertTrue(issubclass(Semi_IOS, Auto_IOS))
        self.assertEqual(Semi_IOS.scan_type, Auto_IOS.scan_type)
        self.assertEqual(Semi_IOS.shows_cbct_input, Auto_IOS.shows_cbct_input)
        self.assertNotEqual(Semi_IOS.stacked_page, Auto_IOS.stacked_page)
        self.assertNotEqual(Semi_IOS.model_label, Auto_IOS.model_label)


if __name__ == "__main__":
    unittest.main()
