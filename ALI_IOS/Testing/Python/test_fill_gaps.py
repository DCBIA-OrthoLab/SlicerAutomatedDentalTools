# What the rebuilding of a missing mucogingival landmark may and may not span.
#
# The 1.53 mm the module reports was measured by hiding ONE point at a time, so
# it speaks for a hole its neighbours border. Nothing measured a longer one, and
# a run of consecutive holes is exactly where a spline with no support inside it
# goes wild: on a real scan where six landmarks in a row were left out, the
# rebuilt points landed 4 to 32 mm away.
#
# Run with:  python -m unittest discover ALI_IOS/Testing/Python
import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ALI_IOS_utils"))

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

import fill_gaps  # noqa: E402


def Line(spacing=5.0, confidence=0.95):
    """The 13 landmarks on a straight line, every one of them trusted."""
    return {name: {"x": float(i * spacing), "y": 0.0, "z": 0.0,
                   "desc": f"confidence {confidence:.3f}"}
            for i, name in enumerate(fill_gaps.MGL_ORDER)}


class FillGapsTest(unittest.TestCase):
    def test_an_isolated_hole_is_rebuilt_where_it_belongs(self):
        line = Line()
        gone = line.pop("LL3MG")
        filled = fill_gaps.FillGaps(line)
        self.assertEqual(filled, ["LL3MG"])
        self.assertAlmostEqual(line["LL3MG"]["x"], gone["x"], places=6)

    def test_a_rebuilt_point_says_so(self):
        line = Line()
        del line["LL3MG"]
        fill_gaps.FillGaps(line)
        self.assertIn(fill_gaps.REBUILT_NOTE, line["LL3MG"]["desc"])

    def test_a_hole_two_wide_is_still_rebuilt(self):
        line = Line()
        for name in ("LL3MG", "LL2MG"):
            del line[name]
        self.assertEqual(sorted(fill_gaps.FillGaps(line)), ["LL2MG", "LL3MG"])

    def test_a_long_run_of_holes_is_left_alone(self):
        """Six in a row is the case that produced 32 mm errors on a real scan.

        Leaving the hole is the right answer: the curve spans it, and AREG
        builds its band on the points that are real.
        """
        line = Line()
        for name in ("LL5MG", "LL4MG", "LL3MG", "LL2MG", "LL1MG", "L0MG"):
            del line[name]
        self.assertEqual(fill_gaps.FillGaps(line), [])

    def test_the_far_side_of_a_long_run_is_not_rebuilt_either(self):
        """Being bordered on one side is not enough, whichever side it is."""
        line = Line()
        for name in ("LL4MG", "LL3MG", "LL2MG", "LL1MG"):
            del line[name]
        self.assertEqual(fill_gaps.FillGaps(line), [])

    def test_the_ends_of_the_arch_are_never_rebuilt(self):
        line = Line()
        del line["LL6MG"]
        del line["LR6MG"]
        self.assertEqual(fill_gaps.FillGaps(line), [])

    def test_a_doubted_point_counts_as_a_hole(self):
        line = Line()
        line["LL3MG"]["desc"] = "confidence 0.400"
        self.assertEqual(fill_gaps.FillGaps(line), ["LL3MG"])

    def test_a_point_off_the_aim_counts_as_a_hole(self):
        """The note ResolveCollisions writes has to be read here too."""
        line = Line()
        line["LL3MG"]["desc"] = "confidence 0.950; off the aim"
        self.assertEqual(fill_gaps.FillGaps(line), ["LL3MG"])

    def test_too_few_trusted_points_rebuilds_nothing(self):
        line = {name: Line()[name] for name in ("LL6MG", "LL5MG")}
        self.assertEqual(fill_gaps.FillGaps(line), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
