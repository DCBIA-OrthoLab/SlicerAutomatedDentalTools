# Which point array serves as the tooth numbering.
#
# The case that matters is the first one: four of the five copies handed back
# the array AFTER the one they were asked for, because they did `continue`
# instead of `break`. The defect did not show when the preferred array was the
# last one of the mesh -- that is, most of the time.
import os
import sys
import unittest

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

import vtk  # noqa: E402

from ADTLib.labels import array_names, has_label_array, label_array  # noqa: E402


def a_surface_with(*names):
    """A one-point mesh carrying one array per given name, in order."""
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    surf = vtk.vtkPolyData()
    surf.SetPoints(points)
    for name in names:
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.InsertNextValue(1.0)
        surf.GetPointData().AddArray(array)
    return surf


class LabelArrayTest(unittest.TestCase):

    def test_the_preferred_array_wins_even_when_it_is_not_the_last(self):
        """The defect of the four copies: here they handed back "Normals"."""
        surf = a_surface_with("Universal_ID", "Normals")
        self.assertEqual(label_array(surf), "Universal_ID")

    def test_it_also_wins_when_it_is_the_last(self):
        """The only case where the five copies agreed."""
        surf = a_surface_with("Normals", "Universal_ID")
        self.assertEqual(label_array(surf), "Universal_ID")

    def test_the_preference_can_be_asked_for_by_name(self):
        surf = a_surface_with("PredictedID", "Normals")
        self.assertEqual(label_array(surf, "PredictedID"), "PredictedID")

    def test_without_the_preferred_array_the_last_one_is_used(self):
        """Fallback kept as it was: this is what the five copies did."""
        surf = a_surface_with("Normals", "Colors")
        self.assertEqual(label_array(surf), "Colors")

    def test_a_surface_without_any_array_gives_none(self):
        self.assertIsNone(label_array(a_surface_with()))

    def test_array_names_keeps_the_order_of_the_mesh(self):
        self.assertEqual(array_names(a_surface_with("a", "b", "c")), ["a", "b", "c"])


class HasLabelArrayTest(unittest.TestCase):

    def test_true_when_present(self):
        self.assertTrue(has_label_array(a_surface_with("Universal_ID"), "Universal_ID"))

    def test_false_when_absent(self):
        self.assertFalse(has_label_array(a_surface_with("Normals"), "Universal_ID"))

    def test_false_on_a_bare_surface(self):
        self.assertFalse(has_label_array(a_surface_with(), "Universal_ID"))


if __name__ == "__main__":
    unittest.main()
