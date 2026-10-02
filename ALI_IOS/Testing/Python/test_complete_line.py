# Placing a mucogingival landmark the prediction could not give.
#
# The two cases are not the same problem and the suite keeps them apart: a hole
# inside the line has points on both sides and the spline is excellent there;
# an end of the arch has not, and what works is the tooth's own gingival collar
# plus putting the answer back on the mesh.
#
# Run with:  python -m unittest discover ALI_IOS/Testing/Python
import os
import sys
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ALI_IOS_utils"))

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

import complete_line  # noqa: E402
import fill_gaps  # noqa: E402


def Arch(radius=22.0, spread=140.0):
    """Thirteen tooth centres on an arc, the shape a mandible actually has.

    A straight row would put the middle tooth on the arch's own centre, where
    "outward" has no direction -- a degenerate case no jaw is in, and testing
    against it would be testing the wrong thing.
    """
    angles = np.radians(np.linspace(-spread / 2, spread / 2, 13)) + np.pi / 2
    return np.column_stack([radius * np.cos(angles),
                            radius * np.sin(angles),
                            np.zeros(13)])


def Line(apical=3.0):
    """The 13 landmarks, each `apical` outside its tooth along the arc.

    Outside, not below: the mucogingival point is on the buccal surface, which
    on a flat test arch is the direction away from the centre.
    """
    centres = Arch()
    middle = centres.mean(axis=0)
    line = {}
    for name, centre in zip(complete_line.MGL_ORDER, centres):
        outward = centre - middle
        outward = outward / np.linalg.norm(outward)
        point = centre + outward * apical
        line[name] = {"x": float(point[0]), "y": float(point[1]),
                      "z": float(point[2]), "desc": "confidence 0.950"}
    return line


def ToothedMesh(inner=14.0, outer=32.0, crown_outer=22.5, step=0.5):
    """A flat annulus about the origin: crowns inside, gingiva outside.

    Arch() puts the tooth centres on radius 22, so a crown reaching to 22.5
    contains its centre and the collar is the circle where it stops. The
    landmarks of Line() sit at radius 25, out on the gingiva, which is where a
    mucogingival point belongs relative to its crown.
    """
    radii = np.arange(inner, outer + step, step)
    angles = np.radians(np.arange(-100.0, 100.0 + 1.0, 1.0)) + np.pi / 2
    grid = np.array([[r * np.cos(a), r * np.sin(a), 0.0]
                     for r in radii for a in angles])

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(grid, deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)

    n_a = len(angles)
    cells = vtk.vtkCellArray()
    for i in range(len(radii) - 1):
        for j in range(n_a - 1):
            a, b = i * n_a + j, i * n_a + j + 1
            c, d = (i + 1) * n_a + j, (i + 1) * n_a + j + 1
            for triangle in ((a, b, d), (a, d, c)):
                cells.InsertNextCell(3)
                for point_id in triangle:
                    cells.InsertCellPoint(point_id)
    mesh.SetPolys(cells)

    centres = Arch()
    tooth_angles = np.arctan2(centres[:, 1], centres[:, 0])
    grid_angles = np.arctan2(grid[:, 1], grid[:, 0])
    grid_radii = np.linalg.norm(grid[:, :2], axis=1)
    half = abs(tooth_angles[1] - tooth_angles[0]) / 2

    labels = np.full(len(grid), 33, dtype=np.int64)          # gingiva
    for i, name in enumerate(complete_line.MGL_ORDER):
        sector = np.abs(grid_angles - tooth_angles[i]) <= half
        labels[sector & (grid_radii <= crown_outer)] = complete_line.TOOTH_OF[name]

    array = numpy_to_vtk(labels, deep=True)
    array.SetName("Universal_ID")
    mesh.GetPointData().AddArray(array)
    return mesh


class CollarFramesTest(unittest.TestCase):
    def test_every_tooth_gets_a_collar(self):
        frames = complete_line.CollarFrames(ToothedMesh())
        self.assertIsNotNone(frames)
        for name in complete_line.MGL_ORDER:
            self.assertIn(complete_line.TOOTH_OF[name], frames)

    def test_no_labels_means_no_frames(self):
        mesh = ToothedMesh()
        mesh.GetPointData().RemoveArray("Universal_ID")
        self.assertIsNone(complete_line.CollarFrames(mesh))

    def test_the_frame_is_a_rotation(self):
        frames = complete_line.CollarFrames(ToothedMesh())
        _, rotation = frames[complete_line.TOOTH_OF["LL3MG"]]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)


class SnapTest(unittest.TestCase):
    def test_a_point_above_the_mesh_comes_back_onto_it(self):
        """The mesh is flat in z, so anything off it has to come back to z = 0."""
        mesh = ToothedMesh()
        on_mesh = np.array(mesh.GetPoint(mesh.GetNumberOfPoints() // 2))
        landed = complete_line.SnapToSurface(on_mesh + [0.0, 0.0, 7.0], mesh)
        self.assertAlmostEqual(landed[2], 0.0, places=5)
        np.testing.assert_allclose(landed[:2], on_mesh[:2], atol=1e-4)

    def test_a_point_already_on_it_does_not_move(self):
        mesh = ToothedMesh()
        on_mesh = np.array(mesh.GetPoint(mesh.GetNumberOfPoints() // 2))
        landed = complete_line.SnapToSurface(on_mesh, mesh)
        np.testing.assert_allclose(landed, on_mesh, atol=1e-5)


class CompleteLineTest(unittest.TestCase):
    """The line always has its 13 points, and says which ones were guessed."""

    def test_a_complete_line_is_left_alone(self):
        line = Line()
        self.assertEqual(complete_line.CompleteLine(line), [])
        self.assertEqual(len(line), 13)

    def test_an_end_of_the_arch_is_filled(self):
        """What FillGaps refuses, since there is nothing to follow on one side."""
        line = Line()
        del line["LL6MG"]
        self.assertEqual(fill_gaps.FillGaps(line), [])        # refuses, rightly
        self.assertEqual(complete_line.CompleteLine(line), ["LL6MG"])
        self.assertEqual(len(line), 13)

    def test_a_long_run_is_filled_too(self):
        line = Line()
        for name in ("LL5MG", "LL4MG", "LL3MG", "LL2MG", "LL1MG", "L0MG"):
            del line[name]
        filled = complete_line.CompleteLine(line)
        self.assertEqual(len(filled), 6)
        self.assertEqual(len(line), 13)

    def test_a_filled_point_is_marked_as_a_guess(self):
        """And not with the note of a point the curve actually bordered."""
        line = Line()
        del line["LL6MG"]
        complete_line.CompleteLine(line)
        self.assertEqual(line["LL6MG"]["desc"], complete_line.EXTRAPOLATED_NOTE)
        self.assertNotIn(fill_gaps.REBUILT_NOTE, line["LL6MG"]["desc"])

    def test_the_note_is_one_areg_leaves_out_of_its_band(self):
        """The word AREG matches on, so a guess cannot drive a registration."""
        self.assertIn("extrapolated", complete_line.EXTRAPOLATED_NOTE)

    def test_a_measured_point_is_never_overwritten(self):
        line = Line()
        del line["LL6MG"]
        before = dict(line["LL5MG"])
        complete_line.CompleteLine(line)
        self.assertEqual(line["LL5MG"], before)

    def test_too_few_points_to_draw_a_line_from(self):
        line = {name: Line()[name] for name in ("LL6MG", "LL5MG")}
        self.assertEqual(complete_line.CompleteLine(line), [])
        self.assertEqual(len(line), 2)

    def test_the_two_stages_together_give_thirteen(self):
        """FillGaps takes what it can stand behind, CompleteLine the rest."""
        line = Line()
        for name in ("LL6MG", "LL3MG", "LR6MG"):
            del line[name]
        fill_gaps.FillGaps(line)
        complete_line.CompleteLine(line)
        self.assertEqual(len(line), 13)
        self.assertIn(fill_gaps.REBUILT_NOTE, line["LL3MG"]["desc"])
        self.assertEqual(line["LL6MG"]["desc"], complete_line.EXTRAPOLATED_NOTE)


class WithTheScanTest(unittest.TestCase):
    """What the scan buys over the spline, on geometry the test controls."""

    def setUp(self):
        self.mesh = ToothedMesh()
        self.truth = Line()

    def error(self, name, surf):
        line = {k: dict(v) for k, v in self.truth.items()}
        expected = np.array([self.truth[name][axis] for axis in "xyz"])
        del line[name]
        complete_line.CompleteLine(line, surf=surf)
        got = np.array([line[name][axis] for axis in "xyz"])
        return float(np.linalg.norm(got - expected))

    def test_an_end_of_the_arch_goes_through_the_collar_when_it_can(self):
        """The mechanism, not its accuracy.

        Which of the two is closer cannot be shown here: this fixture is a
        perfectly regular arc with no noise in its landmarks, and that is
        exactly what a cubic spline extrapolates flawlessly -- it scores
        0.05 mm on it. The collar earns its place on real arches, which are
        irregular and whose landmarks carry the network's error; that is the
        measurement in this module's docstring, on 41 scans held back. What
        this test pins is that the collar path is the one taken for an end
        when the labels are there, and the spline when they are not.
        """
        with_labels = self.error("LL6MG", self.mesh)
        stripped = ToothedMesh()
        stripped.GetPointData().RemoveArray("Universal_ID")
        without_labels = self.error("LL6MG", stripped)
        self.assertNotAlmostEqual(with_labels, without_labels, places=3)

    def test_an_interior_hole_is_placed_the_same_way_labels_or_not(self):
        """Inside the line the spline is used either way, so nothing changes."""
        stripped = ToothedMesh()
        stripped.GetPointData().RemoveArray("Universal_ID")
        self.assertAlmostEqual(self.error("LL3MG", self.mesh),
                               self.error("LL3MG", stripped), places=6)

    def test_a_filled_point_lands_on_the_mesh(self):
        line = {k: dict(v) for k, v in self.truth.items()}
        del line["LL6MG"]
        complete_line.CompleteLine(line, surf=self.mesh)
        got = np.array([line["LL6MG"][axis] for axis in "xyz"])
        self.assertAlmostEqual(got[2], 0.0, places=5)   # the mesh is flat in z

    def test_an_interior_hole_needs_no_scan_to_be_placed_well(self):
        """The spline is already excellent there; the mesh only snaps it."""
        self.assertLess(self.error("LL3MG", None), 1.0)

    def test_a_scan_without_labels_still_completes_the_line(self):
        mesh = ToothedMesh()
        mesh.GetPointData().RemoveArray("Universal_ID")
        line = {k: dict(v) for k, v in self.truth.items()}
        del line["LL6MG"]
        self.assertEqual(complete_line.CompleteLine(line, surf=mesh), ["LL6MG"])


class SnapAllTest(unittest.TestCase):
    """Every landmark on the mesh, whatever put it where it is."""

    def setUp(self):
        self.mesh = ToothedMesh()
        self.line = {k: dict(v) for k, v in Line().items()}

    def lifted(self, height):
        for entry in self.line.values():
            entry["z"] = float(height)
        return self.line

    def heights(self):
        return np.array([entry["z"] for entry in self.line.values()])

    def test_points_hanging_above_the_mesh_come_back_onto_it(self):
        self.lifted(4.0)
        complete_line.SnapAll(self.line, self.mesh)
        np.testing.assert_allclose(self.heights(), 0.0, atol=1e-5)

    def test_what_moved_is_reported_and_by_how_much(self):
        self.lifted(4.0)
        moved = complete_line.SnapAll(self.line, self.mesh)
        self.assertEqual(len(moved), len(self.line))
        for step in moved.values():
            self.assertAlmostEqual(step, 4.0, places=4)

    def test_points_already_on_the_mesh_are_left_exactly_alone(self):
        self.lifted(0.0)
        before = {k: dict(v) for k, v in self.line.items()}
        moved = complete_line.SnapAll(self.line, self.mesh)
        self.assertEqual(moved, {})
        self.assertEqual(self.line, before)

    def test_it_moves_the_point_the_shortest_way(self):
        """Straight down onto the mesh, not sideways along it."""
        self.lifted(4.0)
        before = {k: (v["x"], v["y"]) for k, v in self.line.items()}
        complete_line.SnapAll(self.line, self.mesh)
        for name, (x, y) in before.items():
            self.assertAlmostEqual(self.line[name]["x"], x, places=3)
            self.assertAlmostEqual(self.line[name]["y"], y, places=3)

    def test_no_scan_means_nothing_is_touched(self):
        self.lifted(4.0)
        self.assertEqual(complete_line.SnapAll(self.line, None), {})
        np.testing.assert_allclose(self.heights(), 4.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
