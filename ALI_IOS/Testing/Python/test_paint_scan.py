# What painting the mucogingival line onto a scan may and may not do to it.
#
# The scan is the user's own file, so the bar is that everything which was in
# it is still in it afterwards, byte for byte where it can be: the geometry,
# the teeth labels, and any array someone else put there.
#
# Run with:  python -m unittest discover ALI_IOS/Testing/Python
import os
import sys
import tempfile
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ALI_IOS_utils"))

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

import paint_scan  # noqa: E402


GRID_STEP = 1.0


def FlatMesh(half_size=20, step=GRID_STEP):
    """A flat triangulated square in z=0, vertices `step` mm apart."""
    coords = np.arange(-half_size, half_size + step, step)
    nx = len(coords)
    xs, ys = np.meshgrid(coords, coords, indexing="ij")
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    cells = vtk.vtkCellArray()
    for i in range(nx - 1):
        for j in range(nx - 1):
            a, b = i * nx + j, i * nx + j + 1
            c, d = (i + 1) * nx + j, (i + 1) * nx + j + 1
            for triangle in ((a, b, d), (a, d, c)):
                cells.InsertNextCell(3)
                for point_id in triangle:
                    cells.InsertCellPoint(point_id)
    polydata.SetPolys(cells)
    return polydata, vertices


def ArchLandmarks(span=12.0):
    xs = np.linspace(-span, span, len(paint_scan.MGL_ORDER))
    return {name: (float(x), 0.0, 0.0)
            for name, x in zip(paint_scan.MGL_ORDER, xs)}


class PaintScanTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="paint_scan_test_")
        self.path = os.path.join(self.folder, "scan.vtk")
        mesh, self.vertices = FlatMesh()
        labels = np.full(len(self.vertices), 33, dtype=np.int64)   # all gingiva
        array = numpy_to_vtk(labels, deep=True)
        array.SetName("Universal_ID")
        mesh.GetPointData().AddArray(array)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(self.path)
        writer.SetInputData(mesh)
        writer.Write()
        self.landmarks = ArchLandmarks()

    def read(self):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.path)
        reader.ReadAllScalarsOn()
        reader.Update()
        return reader.GetOutput()

    def test_both_arrays_land_on_the_scan(self):
        self.assertTrue(paint_scan.PaintScan(self.path, self.landmarks, radius=4.0))
        data = self.read().GetPointData()
        self.assertIsNotNone(data.GetArray(paint_scan.MGL_ARRAY_NAME))
        self.assertIsNotNone(data.GetArray(paint_scan.LANDMARK_ARRAY_NAME))

    def test_one_marked_vertex_per_landmark(self):
        paint_scan.PaintScan(self.path, self.landmarks, radius=4.0)
        marked = vtk_to_numpy(self.read().GetPointData().GetArray(
            paint_scan.LANDMARK_ARRAY_NAME))
        self.assertEqual(int(marked.sum()), len(self.landmarks))

    def test_the_landmarks_are_inside_the_band(self):
        """They are what the band is grown from, so they had better be in it."""
        paint_scan.PaintScan(self.path, self.landmarks, radius=4.0)
        data = self.read().GetPointData()
        band = vtk_to_numpy(data.GetArray(paint_scan.MGL_ARRAY_NAME)) > 0
        marked = vtk_to_numpy(data.GetArray(paint_scan.LANDMARK_ARRAY_NAME)) > 0
        self.assertTrue(band[marked].all())

    def test_the_geometry_is_untouched(self):
        before = vtk_to_numpy(self.read().GetPoints().GetData()).copy()
        paint_scan.PaintScan(self.path, self.landmarks, radius=4.0)
        after = vtk_to_numpy(self.read().GetPoints().GetData())
        np.testing.assert_array_equal(before, after)

    def test_what_was_already_on_the_scan_stays(self):
        paint_scan.PaintScan(self.path, self.landmarks, radius=4.0)
        labels = self.read().GetPointData().GetArray("Universal_ID")
        self.assertIsNotNone(labels)
        self.assertEqual(len(np.unique(vtk_to_numpy(labels))), 1)

    def test_painting_twice_gives_the_same_scan(self):
        paint_scan.PaintScan(self.path, self.landmarks, radius=4.0)
        first = vtk_to_numpy(self.read().GetPointData().GetArray(
            paint_scan.MGL_ARRAY_NAME)).copy()
        paint_scan.PaintScan(self.path, self.landmarks, radius=4.0)
        second = vtk_to_numpy(self.read().GetPointData().GetArray(
            paint_scan.MGL_ARRAY_NAME))
        np.testing.assert_array_equal(first, second)

    def test_a_format_that_cannot_carry_an_array_is_left_alone(self):
        stl = os.path.join(self.folder, "scan.stl")
        with open(stl, "w") as handle:
            handle.write("solid empty\nendsolid empty\n")
        before = open(stl).read()
        self.assertFalse(paint_scan.PaintScan(stl, self.landmarks))
        self.assertEqual(open(stl).read(), before)

    def test_a_failure_never_destroys_the_scan(self):
        """Too few landmarks to build a band: the file must survive intact."""
        before = open(self.path, "rb").read()
        self.assertFalse(paint_scan.PaintScan(self.path, {"LL6MG": (0.0, 0.0, 0.0)}))
        self.assertEqual(open(self.path, "rb").read(), before)

    def Unlabelled(self):
        """The same mesh with no teeth labels, which is what a raw scan is."""
        path = os.path.join(self.folder, "raw.vtk")
        mesh, _ = FlatMesh()
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(mesh)
        writer.Write()
        return path

    def test_a_segmentation_handed_in_keeps_the_band_off_the_crowns(self):
        """ALI segments a copy when the scan it was given carries no labels.

        Those labels are what this argument is for: without them the band has
        nothing to be kept off, and it climbs onto the crowns.
        """
        path = self.Unlabelled()
        labels = np.full(len(self.vertices), 33, dtype=np.int64)
        labels[self.vertices[:, 1] > 1.0] = 19          # a crown above the line
        paint_scan.PaintScan(path, self.landmarks, radius=6.0, labels=labels)

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.ReadAllScalarsOn()
        reader.Update()
        band = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(
            paint_scan.MGL_ARRAY_NAME)) > 0
        self.assertTrue(band.any())
        self.assertFalse((labels[band] == 19).any())

    def test_the_scans_own_labels_are_the_ones_that_count(self):
        """A scan that carries a segmentation is not overruled by a stale one."""
        stale = np.full(len(self.vertices), 19, dtype=np.int64)   # all crown
        paint_scan.PaintScan(self.path, self.landmarks, radius=4.0, labels=stale)
        data = self.read().GetPointData()
        self.assertTrue((vtk_to_numpy(data.GetArray(paint_scan.MGL_ARRAY_NAME)) > 0).any())
        self.assertEqual(int(vtk_to_numpy(data.GetArray("Universal_ID"))[0]), 33)


class TrustedOnlyTest(unittest.TestCase):
    """Which landmarks hold the band up -- the same ones AREG stands on."""

    def setUp(self):
        self.landmarks = ArchLandmarks()

    def test_a_sure_line_is_used_whole(self):
        desc = {name: "confidence 0.950" for name in self.landmarks}
        kept = paint_scan.TrustedOnly(self.landmarks, desc)
        self.assertEqual(set(kept), set(self.landmarks))

    def test_a_forced_point_does_not_hold_the_band_up(self):
        desc = {name: "confidence 0.950" for name in self.landmarks}
        desc["LL6MG"] = "forced (confidence 0.300)"
        self.assertNotIn("LL6MG", paint_scan.TrustedOnly(self.landmarks, desc))

    def test_a_point_off_the_aim_does_not_either(self):
        desc = {name: "confidence 0.950" for name in self.landmarks}
        desc["LL6MG"] = "confidence 0.950; off the aim"
        self.assertNotIn("LL6MG", paint_scan.TrustedOnly(self.landmarks, desc))

    def test_an_unsure_point_does_not_either(self):
        desc = {name: "confidence 0.950" for name in self.landmarks}
        desc["LL6MG"] = "confidence 0.400"
        self.assertNotIn("LL6MG", paint_scan.TrustedOnly(self.landmarks, desc))

    def test_the_threshold_is_the_one_areg_uses(self):
        self.assertAlmostEqual(paint_scan.MIN_CONFIDENCE, 0.785)

    def test_too_few_left_keeps_them_all(self):
        """A doubtful line still beats no line, which is AREG's rule too."""
        desc = {name: "confidence 0.100" for name in self.landmarks}
        self.assertEqual(set(paint_scan.TrustedOnly(self.landmarks, desc)),
                         set(self.landmarks))

    def test_no_descriptions_at_all_changes_nothing(self):
        self.assertEqual(set(paint_scan.TrustedOnly(self.landmarks, None)),
                         set(self.landmarks))

    def test_every_landmark_is_still_marked_even_a_doubtful_one(self):
        """The band leaves it out; the point is still there to be looked at."""
        folder = tempfile.mkdtemp(prefix="paint_scan_marks_")
        path = os.path.join(folder, "scan.vtk")
        mesh, _ = FlatMesh()
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(mesh)
        writer.Write()
        desc = {name: "confidence 0.950" for name in self.landmarks}
        desc["LL6MG"] = "forced (confidence 0.300)"
        paint_scan.PaintScan(path, self.landmarks, radius=4.0, descriptions=desc)

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.ReadAllScalarsOn()
        reader.Update()
        marked = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(
            paint_scan.LANDMARK_ARRAY_NAME))
        self.assertEqual(int(marked.sum()), len(self.landmarks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
