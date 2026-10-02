# ASO IOS keeps the landmarks both files carry, as the CBCT path always did.
#
# `SelectKey` indexed every requested name outright:
#
#     for key in self.list_key:
#         out[key] = input[key]        # KeyError on the first absent one
#
# So one name missing from the scan abandoned the whole pair, however many were
# there. Measured on 2026-09-30: twelve landmarks asked for, four in the
# published Semi-Automated file, and `SEMI_ASO_IOS` ended on "No patient could
# be registered" -- blaming the gold reference, which had all twelve.
#
# The CBCT path has always filtered on what is present and then refused below
# three. Three is not a convention: a 3D rigid registration with two points
# leaves a rotation around the line joining them undetermined.
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
for _path in (os.path.join(_ROOT, "ADT"), os.path.join(_ROOT, "ASO_IOS")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np                                              # noqa: E402

from ASO_IOS_utils.icp import ICP, SelectKey                     # noqa: E402

#: What the Semi-Automated mode asks for, upper arch.
ASKED = ["UR6O", "UR6MB", "UR6DB", "UR4O", "UR4MB", "UR4DB",
         "UL4O", "UL4MB", "UL4DB", "UL6O", "UL6MB", "UL6DB"]

#: What the published test file carries for that arch.
SHIPPED = ["UL1O", "UL6O", "UR1O", "UR6O"]


def landmarks(names):
    return {name: np.array([float(i), float(i * 2), float(i * 3)])
            for i, name in enumerate(names, 1)}


def registration(asked=ASKED):
    """An ICP that only has to reach the key selection, so no icp step."""
    return ICP(list_icp=[], option=SelectKey(asked))


class IntersectionTest(unittest.TestCase):

    def test_the_measured_case_is_refused_and_says_why(self):
        """Two in common: the refusal names them, and is not a KeyError."""
        with self.assertRaises(ValueError) as caught:
            registration().run(landmarks(SHIPPED), landmarks(ASKED))
        said = str(caught.exception)
        self.assertIn("only 2 landmark", said)
        self.assertIn("at least 3", said)
        for common in ("UR6O", "UL6O"):
            self.assertIn(common, said)

    def test_what_used_to_happen_was_a_keyerror(self):
        """Pinned so the difference stays legible: absence is not an error now."""
        with self.assertRaises(KeyError):
            SelectKey(ASKED)(landmarks(SHIPPED))

    def test_five_in_common_registers_on_those_five(self):
        scan = landmarks(ASKED[:5])
        out = registration().run(scan, landmarks(ASKED))
        self.assertEqual(sorted(out["source_int"]), sorted(ASKED[:5]))
        self.assertEqual(sorted(out["target_int"]), sorted(ASKED[:5]))

    def test_exactly_three_is_enough(self):
        scan = landmarks(ASKED[:3])
        out = registration().run(scan, landmarks(ASKED))
        self.assertEqual(len(out["source_int"]), 3)

    def test_all_present_keeps_all(self):
        out = registration().run(landmarks(ASKED), landmarks(ASKED))
        self.assertEqual(sorted(out["source_int"]), sorted(ASKED))

    def test_a_landmark_the_reference_lacks_is_dropped_too(self):
        """Both sides are filtered, not just the scan."""
        gold = landmarks([n for n in ASKED if n != "UR6DB"])
        out = registration().run(landmarks(ASKED), gold)
        self.assertNotIn("UR6DB", out["source_int"])
        self.assertNotIn("UR6DB", out["target_int"])
        self.assertEqual(len(out["source_int"]), len(ASKED) - 1)

    def test_an_option_that_is_not_a_key_selection_is_left_alone(self):
        """The narrowing reads SelectKey; any other callable passes untouched.

        A surface registration uses one, and it must keep being called exactly
        as before -- the filtering has no meaning for a vtkPolyData.
        """
        seen = []

        def other_option(value):
            seen.append(value)
            return value

        icp = ICP(list_icp=[], option=other_option)
        icp.run(landmarks(ASKED), landmarks(ASKED))
        self.assertEqual(len(seen), 2, "called once for source, once for target")
        self.assertIs(icp.option, other_option)


if __name__ == "__main__":
    unittest.main()
