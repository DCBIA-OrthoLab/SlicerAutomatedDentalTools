# Invariants of the lower-arch MGL patch engine, checked on a synthetic mesh so
# the suite needs no patient scan and no GPU.
#
# Two of these guard bugs that were live during development and were both
# invisible on screen -- the patch still looked plausible, it was simply the
# wrong size:
#   - an edge graph built from the raw triangle list adds every interior edge
#     twice, doubling its length and halving the reach of the patch;
#   - a height raised on one landmark must not be swallowed where its spline
#     samples land on a vertex a shorter neighbour also claims.
#
# Run with:  python -m unittest discover FlexReg/Testing/Python
import os
import sys
import tempfile
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "FlexReg_utils"))

import mgl_patch  # noqa: E402


GRID_STEP = 1.0  # mm between neighbouring vertices of the test plane


def FlatMesh(half_size=20, step=GRID_STEP):
    """A flat triangulated square in z=0, vertices `step` mm apart.

    Geodesic distance across it is Euclidean, which is what makes the reach of
    the patch something the test can predict in closed form.
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


def ArchLandmarks(span=12.0):
    """The 13 MG landmarks on a shallow in-plane arc.

    Deliberately curved: three collinear points leave the plane the landmarks
    lie in undefined, and with it the apical axis derived from it.
    """
    xs = np.linspace(-span, span, len(mgl_patch.MGL_ORDER))
    ys = xs ** 2 / 60.0 - 2.0
    return {name: np.array([x, y, 0.0])
            for name, x, y in zip(mgl_patch.MGL_ORDER, xs, ys)}


def ReachAround(vertices, labels, builder, index):
    """How far the patch extends from one landmark, over the mesh it owns.

    Measured per landmark rather than across the arch, because a neighbour's
    band can reach the same distance and hide a height that was lost here.
    """
    landmarks = builder._landmarks
    selected = vertices[labels > 0.5]
    if not len(selected):
        return 0.0
    to_each = np.linalg.norm(selected[:, None, :] - landmarks[None, :, :], axis=2)
    owned = selected[to_each.argmin(axis=1) == index]
    if not len(owned):
        return 0.0
    return float(np.linalg.norm(owned - landmarks[index], axis=1).max())


def BandHalfWidth(vertices, labels, x_centre, window=0.6):
    """Extent of the patch across the arch, in the slice of mesh at `x_centre`.

    The band straddles the curve, so this is measured from the curve outwards
    on the wider of the two sides.
    """
    column = np.abs(vertices[:, 0] - x_centre) < window
    selected = column & (labels > 0.5)
    if not selected.any():
        return 0.0
    curve_y = x_centre ** 2 / 60.0 - 2.0
    return float(np.abs(vertices[selected][:, 1] - curve_y).max())


class OrderedLandmarksTest(unittest.TestCase):
    def test_orders_along_the_arch_not_by_json_order(self):
        landmarks = ArchLandmarks()
        shuffled = {name: landmarks[name] for name in reversed(mgl_patch.MGL_ORDER)}
        names, _ = mgl_patch.OrderedLandmarks(shuffled)
        self.assertEqual(names, mgl_patch.MGL_ORDER)

    def test_a_scan_missing_some_landmarks_still_yields_a_curve(self):
        landmarks = ArchLandmarks()
        for name in ("LL6MG", "LR4MG", "LR6MG"):
            del landmarks[name]
        names, positions = mgl_patch.OrderedLandmarks(landmarks)
        self.assertEqual(len(names), len(mgl_patch.MGL_ORDER) - 3)
        self.assertEqual(len(positions), len(names))

    def test_too_few_landmarks_is_an_error_naming_what_was_expected(self):
        with self.assertRaises(ValueError) as caught:
            mgl_patch.OrderedLandmarks({"LL6MG": np.zeros(3), "L0MG": np.ones(3)})
        self.assertIn("LL6MG", str(caught.exception))

    def test_old_training_naming_is_translated_to_canonical(self):
        # An old annotation file : 13 symmetric points whose left names count
        # from the midline point inclusive, so their LL1MG is the midline
        # itself. After translation L0MG must sit exactly there, every point
        # must survive, and the arch order must hold.
        old_names = ['LL7MG', 'LL6MG', 'LL5MG', 'LL4MG', 'LL3MG', 'LL2MG', 'LL1MG',
                     'LR1MG', 'LR2MG', 'LR3MG', 'LR4MG', 'LR5MG', 'LR6MG']
        xs = np.linspace(-12, 12, len(old_names))
        landmarks = {name: np.array([x, x ** 2 / 60.0 - 2.0, 0.0])
                     for name, x in zip(old_names, xs)}
        names, positions = mgl_patch.OrderedLandmarks(landmarks)

        self.assertEqual(names, mgl_patch.MGL_ORDER)  # all 13, L0MG included
        np.testing.assert_allclose(positions[names.index('L0MG')],
                                   landmarks['LL1MG'])
        np.testing.assert_allclose(positions[names.index('LL6MG')],
                                   landmarks['LL7MG'])
        np.testing.assert_allclose(positions[names.index('LR6MG')],
                                   landmarks['LR6MG'])

    def test_the_lr7_variant_of_the_old_naming_is_translated_too(self):
        # Some old files carry the seven names on the RIGHT side : their
        # midline point is LR1MG and LR7MG is their distal-right end.
        old_names = ['LR7MG', 'LR6MG', 'LR5MG', 'LR4MG', 'LR3MG', 'LR2MG', 'LR1MG',
                     'LL1MG', 'LL2MG', 'LL3MG', 'LL4MG', 'LL5MG', 'LL6MG']
        xs = np.linspace(12, -12, len(old_names))
        landmarks = {name: np.array([x, x ** 2 / 60.0 - 2.0, 0.0])
                     for name, x in zip(old_names, xs)}
        names, positions = mgl_patch.OrderedLandmarks(landmarks)

        self.assertEqual(names, mgl_patch.MGL_ORDER)  # all 13, L0MG included
        np.testing.assert_allclose(positions[names.index('L0MG')],
                                   landmarks['LR1MG'])
        np.testing.assert_allclose(positions[names.index('LR6MG')],
                                   landmarks['LR7MG'])
        np.testing.assert_allclose(positions[names.index('LL6MG')],
                                   landmarks['LL6MG'])

    def test_a_missing_midline_point_is_refilled_from_the_right(self):
        # L0 must always exist : when the midline point itself is missing,
        # the counting recentres and the hole surfaces at the LR6 end.
        landmarks = ArchLandmarks()
        del landmarks['L0MG']
        old_lr1 = landmarks['LR1MG'].copy()
        names, positions = mgl_patch.OrderedLandmarks(landmarks)
        self.assertIn('L0MG', names)
        self.assertNotIn('LR6MG', names)
        np.testing.assert_allclose(positions[names.index('L0MG')], old_lr1)

    def test_a_hole_in_the_middle_of_a_side_surfaces_at_its_end(self):
        landmarks = ArchLandmarks()
        del landmarks['LL3MG']
        old_ll4 = landmarks['LL4MG'].copy()
        names, positions = mgl_patch.OrderedLandmarks(landmarks)
        self.assertIn('LL3MG', names)     # refilled by the next point out
        self.assertNotIn('LL6MG', names)  # the hole surfaced at the far end
        self.assertIn('L0MG', names)      # the centre and the right side
        self.assertIn('LR6MG', names)     # are left alone
        np.testing.assert_allclose(positions[names.index('LL3MG')], old_ll4)


class EdgeGraphTest(unittest.TestCase):
    """The reach of the patch is only as trustworthy as the edge lengths."""

    def setUp(self):
        self.polydata, self.vertices = FlatMesh(half_size=4)
        self.builder = mgl_patch.MGLPatchBuilder()
        self.assertTrue(self.builder.prepare(self.polydata, ArchLandmarks(span=3.0)))

    def test_neighbouring_vertices_are_one_step_apart_not_two(self):
        graph = self.builder._graph
        nx = int(round(np.sqrt(len(self.vertices))))
        for a, b in ((0, 1), (0, nx), (5, 6)):
            self.assertAlmostEqual(graph[a, b], GRID_STEP, places=6,
                                   msg="interior edges counted twice")

    def test_graph_is_symmetric(self):
        difference = abs(self.builder._graph - self.builder._graph.T).max()
        self.assertAlmostEqual(difference, 0.0, places=9)


class PatchReachTest(unittest.TestCase):
    def setUp(self):
        self.polydata, self.vertices = FlatMesh()
        self.builder = mgl_patch.MGLPatchBuilder()
        self.assertTrue(self.builder.prepare(self.polydata, ArchLandmarks()))
        self.n = len(self.builder.names())
        self.zero = np.zeros(self.n)

    def patch(self, heights, **kwargs):
        labels, _ = self.builder.compute(self.zero, self.zero, heights, **kwargs)
        return labels

    def test_a_uniform_height_gives_a_band_of_that_height(self):
        for height in (3.0, 6.0):
            labels = self.patch(np.full(self.n, height))
            width = BandHalfWidth(self.vertices, labels, x_centre=0.0)
            # halved edges would put this at height/2, well under the lower bound
            self.assertGreater(width, 0.8 * height)
            self.assertLess(width, 1.25 * height)

    def test_raising_the_height_only_ever_grows_the_patch(self):
        small = self.patch(np.full(self.n, 3.0))
        large = self.patch(np.full(self.n, 6.0))
        self.assertTrue(np.all(large[small > 0.5] > 0.5))
        self.assertGreater(large.sum(), small.sum())

    def test_a_height_set_on_one_end_does_not_reshape_the_other(self):
        heights = np.full(self.n, 2.0)
        heights[:3] = 8.0  # the LL6..LL4 stretch, at negative x
        labels = self.patch(heights)
        tall = BandHalfWidth(self.vertices, labels, x_centre=-11.0)
        short = BandHalfWidth(self.vertices, labels, x_centre=11.0)
        self.assertGreater(tall, 5.0)
        self.assertLess(short, 4.0)
        self.assertGreater(tall, short + 2.0)

    def test_zero_height_keeps_only_the_landmarks(self):
        # Height 0 is the control case : no band, and no curve joining the
        # landmarks either, so the registration runs on the points alone.
        labels = self.patch(np.zeros(self.n))
        selected = self.vertices[labels > 0.5]
        self.assertEqual(len(selected), self.n)  # one vertex per landmark
        to_landmark = np.linalg.norm(
            selected[:, None, :] - self.builder._landmarks[None, :, :],
            axis=2).min(axis=1)
        self.assertLessEqual(to_landmark.max(), GRID_STEP)

    def test_a_stretch_left_at_zero_drops_the_curve_it_carried(self):
        # A height of 0 is local : the flattened stretch keeps its landmarks
        # and nothing else, while the rest of the arch keeps its band.
        heights = np.full(self.n, 4.0)
        heights[:3] = 0.0  # the LL6..LL4 stretch, at negative x
        labels = self.patch(heights)
        between_two_flattened = np.abs(self.vertices[:, 0] + 11.0) < 0.6
        self.assertEqual(labels[between_two_flattened].sum(), 0.0)
        self.assertGreater(BandHalfWidth(self.vertices, labels, x_centre=11.0), 3.0)

    def test_the_landmarks_belong_to_the_patch_whatever_the_height(self):
        ids = self.builder.snappedLandmarkIds(self.zero, self.zero)
        for heights in (np.zeros(self.n), np.full(self.n, 3.0)):
            labels = self.patch(heights)
            self.assertTrue(np.all(labels[ids] > 0.5))

    def test_a_tall_landmark_is_not_swallowed_by_a_short_neighbour(self):
        # Dozens of spline samples land on the same vertex, and the tall
        # landmark is flanked by short ones, so the samples writing that vertex
        # arrive tall then short. Keeping the last one instead of the tallest
        # quietly delivers a fraction of the height that was asked for.
        tall = 12.0
        heights = np.full(self.n, 0.5)
        heights[self.n // 2] = tall
        labels = self.patch(heights)
        reach = ReachAround(self.vertices, labels, self.builder, self.n // 2)
        self.assertGreater(reach, 0.9 * tall)


class ToothExclusionTest(unittest.TestCase):
    """Crowns move between timepoints, so they must stay out of the patch."""

    def setUp(self):
        self.polydata, self.vertices = FlatMesh()
        labels = np.where(self.vertices[:, 1] > 0.0, 20, 0).astype(np.int16)
        array = numpy_to_vtk(labels, deep=True)
        array.SetName("Universal_ID")
        self.polydata.GetPointData().AddArray(array)
        self.crown = self.vertices[:, 1] > 0.0

        self.builder = mgl_patch.MGLPatchBuilder()
        self.assertTrue(self.builder.prepare(self.polydata, ArchLandmarks()))
        self.heights = np.full(len(self.builder.names()), 6.0)
        self.zero = np.zeros(len(self.builder.names()))

    def test_no_crown_vertex_is_labelled(self):
        labels, _ = self.builder.compute(self.zero, self.zero, self.heights)
        self.assertEqual(labels[self.crown].sum(), 0.0)
        self.assertGreater(labels.sum(), 0.0)

    def test_a_landmark_snapped_onto_a_crown_is_dropped_like_the_rest(self):
        # The landmarks survive every height, but not the crown exclusion :
        # a point that moves between the timepoints must never register.
        labels, _ = self.builder.compute(self.zero, self.zero,
                                         np.zeros(len(self.zero)))
        self.assertEqual(labels[self.crown].sum(), 0.0)
        self.assertGreater(labels.sum(), 0.0)  # the landmarks off the crowns stay

    def test_the_crowns_are_within_reach_when_not_excluded(self):
        labels, _ = self.builder.compute(self.zero, self.zero, self.heights,
                                         exclude_teeth=False)
        self.assertGreater(labels[self.crown].sum(), 0.0)


class OffsetTest(unittest.TestCase):
    def setUp(self):
        self.polydata, _ = FlatMesh()
        self.builder = mgl_patch.MGLPatchBuilder()
        self.assertTrue(self.builder.prepare(self.polydata, ArchLandmarks()))
        self.n = len(self.builder.names())
        self.zero = np.zeros(self.n)

    def test_no_offset_leaves_the_landmarks_where_ali_put_them(self):
        moved = self.builder.movedLandmarks(self.zero, self.zero)
        np.testing.assert_allclose(moved, self.builder._landmarks, atol=1e-9)

    def test_a_shared_offset_moves_every_landmark_the_same_distance(self):
        moved = self.builder.movedLandmarks(np.full(self.n, 2.0), self.zero)
        travelled = np.linalg.norm(moved - self.builder._landmarks, axis=1)
        np.testing.assert_allclose(travelled, 2.0, atol=1e-9)

    def test_moving_one_landmark_leaves_the_others_untouched(self):
        offsets = np.zeros(self.n)
        offsets[4] = 3.0
        moved = self.builder.movedLandmarks(offsets, self.zero)
        travelled = np.linalg.norm(moved - self.builder._landmarks, axis=1)
        self.assertAlmostEqual(travelled[4], 3.0, places=9)
        self.assertAlmostEqual(np.delete(travelled, 4).max(), 0.0, places=9)

    def test_a_tangent_offset_slides_along_the_arch(self):
        moved = self.builder.movedLandmarks(self.zero, self.zero,
                                            np.full(self.n, 2.0))
        travelled = np.linalg.norm(moved - self.builder._landmarks, axis=1)
        np.testing.assert_allclose(travelled, 2.0, atol=1e-9)
        # every point moves towards its next neighbour along the arch,
        # i.e. from the LL6 end towards the LR6 end
        for index in range(self.n - 1):
            towards_next = (self.builder._landmarks[index + 1]
                            - self.builder._landmarks[index])
            self.assertGreater(
                np.dot(moved[index] - self.builder._landmarks[index], towards_next), 0)

    def test_snapped_landmarks_stay_on_the_mesh(self):
        # An apical offset leaves the mesh plane entirely; the displayed
        # points must come back onto the surface, near where they started.
        snapped = self.builder.snappedLandmarks(self.zero, np.full(self.n, 3.0))
        self.assertAlmostEqual(np.abs(snapped[:, 2]).max(), 0.0, places=9)
        drift = np.linalg.norm(
            snapped[:, :2] - self.builder._landmarks[:, :2], axis=1)
        self.assertLessEqual(drift.max(), GRID_STEP)


class AsymmetricHeightTest(unittest.TestCase):
    """The band can climb a different distance towards the crowns and towards
    the vestibule.

    The landmark ribbon is tilted out of the mesh plane so its normal -- the
    apical axis -- has an in-plane component, which is what gives 'up' and
    'down' a direction on the flat mesh; tooth labels on the crown side pin
    its sign, exactly as the real segmentation does.
    """

    def setUp(self):
        self.polydata, self.vertices = FlatMesh()
        labels = np.where(self.vertices[:, 1] > 6.0, 20, 0).astype(np.int16)
        array = numpy_to_vtk(labels, deep=True)
        array.SetName("Universal_ID")
        self.polydata.GetPointData().AddArray(array)

        tilted = {name: position + np.array([0.0, 0.0, 0.5 * position[1]])
                  for name, position in ArchLandmarks().items()}
        self.builder = mgl_patch.MGLPatchBuilder()
        self.assertTrue(self.builder.prepare(self.polydata, tilted))
        self.n = len(self.builder.names())
        self.zero = np.zeros(self.n)

    def sideReach(self, labels, x_centre=0.0, window=0.6):
        """Extent of the patch on each side of the curve, in one mesh column.

        Returns (towards the crowns, towards the vestibule): the teeth sit at
        positive y, so positive offsets from the curve are the crown side.
        """
        column = np.abs(self.vertices[:, 0] - x_centre) < window
        selected = column & (labels > 0.5)
        curve_y = x_centre ** 2 / 60.0 - 2.0
        offsets = self.vertices[selected][:, 1] - curve_y
        crown = float(offsets[offsets > 0].max()) if (offsets > 0).any() else 0.0
        vestibule = float(-offsets[offsets < 0].min()) if (offsets < 0).any() else 0.0
        return crown, vestibule

    def test_the_two_sides_of_the_band_are_independent(self):
        labels, _ = self.builder.compute(
            self.zero, self.zero,
            np.full(self.n, 2.0), np.full(self.n, 8.0),
            exclude_teeth=False)
        crown, vestibule = self.sideReach(labels)
        self.assertLess(crown, 4.0)
        self.assertGreater(vestibule, 6.0)

    def test_the_asymmetry_goes_both_ways(self):
        labels, _ = self.builder.compute(
            self.zero, self.zero,
            np.full(self.n, 8.0), np.full(self.n, 2.0),
            exclude_teeth=False)
        crown, vestibule = self.sideReach(labels)
        self.assertGreater(crown, 6.0)
        self.assertLess(vestibule, 4.0)

    def test_no_second_height_keeps_the_symmetric_band(self):
        symmetric, _ = self.builder.compute(
            self.zero, self.zero, np.full(self.n, 5.0), exclude_teeth=False)
        explicit, _ = self.builder.compute(
            self.zero, self.zero,
            np.full(self.n, 5.0), np.full(self.n, 5.0),
            exclude_teeth=False)
        np.testing.assert_array_equal(symmetric, explicit)


class SavedLineTest(unittest.TestCase):
    """What Update records for a future training set."""

    def setUp(self):
        self.polydata, self.vertices = FlatMesh()
        self.builder = mgl_patch.MGLPatchBuilder()
        self.assertTrue(self.builder.prepare(self.polydata, ArchLandmarks()))
        self.n = len(self.builder.names())
        self.zero = np.zeros(self.n)

    def test_the_ids_designate_the_snapped_positions(self):
        apical = np.full(self.n, 2.0)
        ids = self.builder.snappedLandmarkIds(self.zero, apical)
        snapped = self.builder.snappedLandmarks(self.zero, apical)
        np.testing.assert_allclose(self.vertices[ids], snapped)

    def test_landmarks_written_for_training_read_back_identically(self):
        names = mgl_patch.MGL_ORDER[:4]
        positions = np.arange(12, dtype=float).reshape(4, 3)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'line.mrk.json')
            mgl_patch.WriteLandmarks(path, names, positions)
            read = mgl_patch.ReadLandmarks(path)
        self.assertEqual(sorted(read), sorted(names))
        for name, position in zip(names, positions):
            np.testing.assert_allclose(read[name], position)


class ArrayNamingTest(unittest.TestCase):
    def test_the_array_carries_the_name_areg_reads(self):
        builder = mgl_patch.MGLPatchBuilder()
        array = builder.toArray(np.ones(4, dtype=np.float32))
        self.assertEqual(array.GetName(), "Bottom_MGL")
        self.assertEqual(array.GetNumberOfTuples(), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
