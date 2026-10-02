# What the interface has to show for each AREG method.
#
# These values used to live in three nested `if/elif` branches on dropdown
# indices. Here they are, frozen as they stood.
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ADT = os.path.join(_HERE, "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from AREG_Method.CBCT import Auto_CBCT, Or_Auto_CBCT, Semi_CBCT  # noqa: E402
from AREG_Method.IOS import Auto_IOS, Semi_IOS  # noqa: E402
from AREG_Method.IOSCBCT import Auto_IOSCBCT, Reg_IOSCBCT, Semi_IOSCBCT  # noqa: E402

SEG = "Segmentation Model Folder"

# method -> (page, scan type, model label)
EXPECTED = {
    Semi_CBCT:    (0, "CBCT", None),
    Auto_CBCT:    (1, "CBCT", SEG),
    Or_Auto_CBCT: (2, "CBCT", SEG),
    Auto_IOS:     (3, "IOS", SEG),
    Semi_IOS:     (3, "IOS", None),
    Auto_IOSCBCT: (4, "IOSCBCT", None),
    Semi_IOSCBCT: (4, "IOSCBCT", None),
    Reg_IOSCBCT:  (4, "IOSCBCT", None),
}

# input type -> how many modes, and the method of each
COMBOS = {
    0: {0: Or_Auto_CBCT, 1: Auto_CBCT, 2: Semi_CBCT},
    1: {0: Auto_IOS, 1: Semi_IOS},
    2: {0: Auto_IOSCBCT, 1: Semi_IOSCBCT, 2: Reg_IOSCBCT},
}


class UiDescriptionTest(unittest.TestCase):

    def test_each_method_describes_what_the_if_elif_chains_did(self):
        for method, expected in EXPECTED.items():
            got = (method.stacked_page, method.scan_type, method.model_label)
            self.assertEqual(got, expected, method.__name__)

    def test_every_mode_of_every_input_type_has_a_method(self):
        self.assertEqual(sorted(COMBOS), [0, 1, 2])
        for input_type, modes in COMBOS.items():
            self.assertEqual(sorted(modes), list(range(len(modes))), input_type)

    def test_the_three_input_types_land_on_three_scan_types(self):
        for input_type, modes in COMBOS.items():
            kinds = {m.scan_type for m in modes.values()}
            self.assertEqual(len(kinds), 1, f"input type {input_type}: {kinds}")

    def test_the_iosbct_family_shares_one_page(self):
        self.assertEqual({m.stacked_page for m in COMBOS[2].values()}, {4})

    def test_semi_ios_inherits_auto_ios_and_only_drops_the_label(self):
        self.assertTrue(issubclass(Semi_IOS, Auto_IOS))
        self.assertEqual(Semi_IOS.stacked_page, Auto_IOS.stacked_page)
        self.assertIsNone(Semi_IOS.model_label)
        self.assertEqual(Auto_IOS.model_label, SEG)


if __name__ == "__main__":
    unittest.main()
