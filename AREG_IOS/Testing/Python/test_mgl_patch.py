# Invariants of the MGL patch AREG registers on, checked on a synthetic mesh so
# the suite needs no patient scan, no model and no GPU.
#
# Both of these guard failures that are invisible on screen -- the patch still
# looks like a patch, and the registration still returns a transform:
#   - a band built at one timepoint from landmarks the other does not have runs
#     further along the arch than its counterpart, and the ICP slides the short
#     one along the long one;
#   - a third molar left inside the band puts an erupting crown in charge of
#     the registration.
#
# Run with:  python -m unittest discover AREG_IOS/Testing/Python
import os
import sys
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "AREG_IOS_utils"))

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

import mgl_patch  # noqa: E402


GRID_STEP = 1.0  # mm between neighbouring vertices of the test plane


def FlatMesh(half_size=20, step=GRID_STEP):
    """A flat triangulated square in z=0, vertices `step` mm apart.

    Geodesic distance across it is Euclidean, so what the band reaches is
    something the test can predict rather than observe.
    """
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


def Label(polydata, labels):
    array = numpy_to_vtk(np.asarray(labels, dtype=np.int64), deep=True)
    array.SetName("Universal_ID")
    polydata.GetPointData().AddArray(array)
    return polydata


def ArchLandmarks(span=12.0):
    """The 13 MG landmarks laid along y = 0, in arch order."""
    xs = np.linspace(-span, span, len(mgl_patch.MGL_ORDER))
    return {name: np.array([x, 0.0, 0.0])
            for name, x in zip(mgl_patch.MGL_ORDER, xs)}


class SharedLandmarksTest(unittest.TestCase):
    def test_both_timepoints_keep_only_what_both_carry(self):
        full = ArchLandmarks()
        t1 = {name: position for name, position in full.items() if name != "LL6MG"}
        t2 = {name: position for name, position in full.items() if name != "LR6MG"}

        trimmed, dropped = mgl_patch.SharedLandmarks({"T1": t1, "T2": t2})

        self.assertEqual(dropped, {"LL6MG", "LR6MG"})
        self.assertEqual(set(trimmed["T1"]), set(trimmed["T2"]))
        self.assertNotIn("LL6MG", trimmed["T2"])
        self.assertNotIn("LR6MG", trimmed["T1"])

    def test_the_terminal_molar_is_what_goes_missing(self):
        """The end of the band is what one timepoint loses, in practice.

        Which is why this matters: the ends are what pin the rotation about
        the vertical axis, so an unmatched end is not a small loss of area.
        """
        full = ArchLandmarks()
        t1 = {name: position for name, position in full.items() if name != "LL6MG"}
        trimmed, _ = mgl_patch.SharedLandmarks({"T1": t1, "T2": full})
        self.assertEqual(sorted(trimmed["T2"]), sorted(trimmed["T1"]))

    def test_matching_sets_are_left_alone(self):
        full = ArchLandmarks()
        trimmed, dropped = mgl_patch.SharedLandmarks({"T1": full, "T2": dict(full)})
        self.assertEqual(dropped, set())
        self.assertEqual(set(trimmed["T1"]), set(full))

    def test_positions_are_never_touched(self):
        full = ArchLandmarks()
        t1 = {name: position for name, position in full.items() if name != "LL6MG"}
        trimmed, _ = mgl_patch.SharedLandmarks({"T1": t1, "T2": full})
        for name, position in trimmed["T2"].items():
            np.testing.assert_allclose(position, full[name])

    def test_too_few_in_common_trims_nothing(self):
        """Two landmarks in common cannot carry a curve, so nothing is trimmed.

        A band on its own beats no registration at all, and the caller learns
        of it through the empty set of dropped names.
        """
        full = ArchLandmarks()
        t1 = {name: full[name] for name in ("LL6MG", "LL5MG", "LL4MG")}
        t2 = {name: full[name] for name in ("LL6MG", "LL5MG", "LR4MG", "LR5MG")}
        trimmed, dropped = mgl_patch.SharedLandmarks({"T1": t1, "T2": t2})
        self.assertEqual(dropped, set())
        self.assertEqual(set(trimmed["T1"]), set(t1))
        self.assertEqual(set(trimmed["T2"]), set(t2))


class AlignOnLandmarksTest(unittest.TestCase):
    """The pose the ICP is handed to start from.

    It is not a cosmetic detail: the band is a strip along a curve, and the
    distance the ICP minimises hardly changes when the strip slides along its
    own length, so a start a few millimetres off stays a few millimetres off
    while the residual reports success.
    """

    def setUp(self):
        self.landmarks = ArchLandmarks()

    @staticmethod
    def Move(landmarks, rotation_deg=0.0, translation=(0.0, 0.0, 0.0)):
        angle = np.radians(rotation_deg)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return {name: rotation @ position + np.asarray(translation)
                for name, position in landmarks.items()}

    def test_a_known_pose_is_recovered(self):
        moved = self.Move(self.landmarks, 17.0, (3.0, -2.0, 1.5))
        matrix = mgl_patch.AlignOnLandmarks(moved, self.landmarks)
        for name, position in moved.items():
            back = matrix[:3, :3] @ position + matrix[:3, 3]
            np.testing.assert_allclose(back, self.landmarks[name], atol=1e-9)

    def test_aligned_landmarks_ask_for_no_move(self):
        matrix = mgl_patch.AlignOnLandmarks(self.landmarks, dict(self.landmarks))
        np.testing.assert_allclose(matrix, np.identity(4), atol=1e-9)

    def test_the_transform_is_a_rotation_never_a_reflection(self):
        """A jaw cannot be mirrored, and svd will hand one over if allowed.

        Landmarks nearly on a line -- which the mucogingival ones are -- make
        that degenerate case reachable rather than theoretical.
        """
        flat = {name: np.array([position[0], 0.0, 0.0])
                for name, position in self.landmarks.items()}
        mirrored = {name: np.array([-position[0], 0.0, 0.0])
                    for name, position in flat.items()}
        matrix = mgl_patch.AlignOnLandmarks(mirrored, flat)
        self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), 1.0, places=9)

    def test_only_the_names_both_carry_are_used(self):
        moved = self.Move(self.landmarks, 10.0, (1.0, 0.0, 0.0))
        moved["STRAY"] = np.array([100.0, 100.0, 100.0])
        matrix = mgl_patch.AlignOnLandmarks(moved, self.landmarks)
        for name, position in self.landmarks.items():
            back = matrix[:3, :3] @ moved[name] + matrix[:3, 3]
            np.testing.assert_allclose(back, position, atol=1e-9)

    def test_too_few_pairs_leaves_the_icp_to_its_own_devices(self):
        two = {name: self.landmarks[name] for name in ("LL6MG", "LL5MG")}
        matrix = mgl_patch.AlignOnLandmarks(two, self.landmarks)
        np.testing.assert_allclose(matrix, np.identity(4))


class ThirdMolarTest(unittest.TestCase):
    """The wisdom teeth belong to the crowns the band is kept off.

    They sit at the very end of the arch, where the band runs out, so a third
    molar inside the band is a crown driving the registration from the one
    place that most constrains it.
    """

    def setUp(self):
        self.mesh, self.vertices = FlatMesh()
        self.landmarks = ArchLandmarks()

    def _patch(self, labels):
        surf = Label(self.mesh, labels)
        out = mgl_patch.MGLPatch(surf, self.landmarks, radius=4.0)
        return vtk_to_numpy(out.GetPointData().GetArray(mgl_patch.MGL_ARRAY_NAME)) > 0

    def test_a_third_molar_is_not_in_the_band(self):
        labels = np.full(len(self.vertices), 33, dtype=np.int64)   # gingiva
        labels[self.vertices[:, 0] < -10] = 17                     # LL8
        labels[self.vertices[:, 0] > 10] = 32                      # LR8
        band = self._patch(labels)
        self.assertTrue(band.any(), "the band should not be empty")
        self.assertFalse(np.isin(labels[band], (17, 32)).any())

    def test_the_same_stretch_is_kept_when_it_is_gingiva(self):
        """Control: the band does reach there, so the exclusion is what removed it."""
        labels = np.full(len(self.vertices), 33, dtype=np.int64)
        band = self._patch(labels)
        reached = np.abs(self.vertices[band][:, 0]) > 10
        self.assertTrue(reached.any())

    def test_every_lower_tooth_label_is_excluded(self):
        self.assertEqual(set(mgl_patch.LOWER_TOOTH_LABELS), set(range(17, 33)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
