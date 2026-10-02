# What the IOS-to-CBCT registration is allowed to do, checked on synthetic
# surfaces so the suite needs no patient scan.
#
# Each case guards a failure that was live and silent -- the run reported
# success and the numbers it printed looked good:
#   - in a closed bite the opposing crowns sit 1 to 3 mm apart, inside the
#     capture radius. An ICP that ignores which way a surface faces locks a
#     maxillary IOS onto the mandible and still reports a perfect fitness;
#   - an ICP that matched nothing returns the identity, and the untouched IOS
#     was written out under the name of a registered one;
#   - three landmarks in a near-straight line fit their targets exactly, so the
#     residual threshold passes them while the rotation about that line is
#     decided by their noise alone.
#
# Run with:  python -m unittest discover AREG_IOSCBCT/Testing/Python
import os
import sys
import unittest

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

import AREG_IOSCBCT as areg  # noqa: E402
import tempfile


def Rigid(axis, degrees, translation):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    matrix[:3, 3] = translation
    return matrix


def Apply(matrix, points):
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (homogeneous @ matrix.T)[:, :3]


def Sheet(z0, phase, facing_down):
    """An undulating occlusal surface, normals on the side it faces."""
    x = np.linspace(-25, 25, 101)
    y = np.linspace(-20, 10, 61)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = z0 + 0.6 * np.sin(X / 4.0 + phase) * np.cos(Y / 5.0 + phase)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    nx, ny = X.shape
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b = i * ny + j, i * ny + j + 1
            c, d = (i + 1) * ny + j, (i + 1) * ny + j + 1
            if facing_down:
                faces += [3, a, c, b, 3, b, c, d]
            else:
                faces += [3, a, b, c, 3, b, d, c]
    mesh = pv.PolyData(points, np.array(faces))
    return mesh.compute_normals(point_normals=True, cell_normals=False,
                                auto_orient_normals=False, inplace=False)


def BiteScene(drop=1.3):
    """A maxillary IOS sitting low over both arches of a closed bite.

    Returns the moving IOS, the CBCT surface holding both arches, and the
    matrix the registration has to recover.
    """
    maxilla = Sheet(0.0, 0.0, facing_down=True)
    mandible = Sheet(-2.2, 1.0, facing_down=False)
    cbct = (maxilla + mandible).compute_normals(
        point_normals=True, cell_normals=False,
        auto_orient_normals=False, inplace=False)

    perturbation = Rigid([1, 0.3, 0.1], 1.2, [0.35, -0.4, -drop])
    return (maxilla.transform(perturbation, inplace=False),
            cbct,
            np.linalg.inv(perturbation))


def ArchLandmarks(xs):
    """Occlusal landmarks along a plausible arch, one per x given."""
    return np.array([[x, 0.02 * x * x - 18.0, 0.0] for x in xs], dtype=float)


class RegistrationTest(unittest.TestCase):

    def PoseError(self, mesh, matrix, truth):
        points = np.asarray(mesh.points)
        return float(np.mean(np.linalg.norm(
            Apply(matrix, points) - Apply(truth, points), axis=1)))

    # ---------------------------------------------------------------- the bite

    def test_the_opposing_arch_is_not_registered_onto(self):
        moving, cbct, truth = BiteScene()
        _, matrix, quality = areg.run_icp_point_to_plane(
            moving, cbct, max_dist=1.0, label="upper")

        self.assertLess(self.PoseError(moving, matrix, truth), 0.2)
        self.assertGreater(quality["rejected_by_normal"], 0,
                           "nothing was dropped as the opposing surface, so the "
                           "normals are not being used to separate the arches")

    def test_without_normals_the_bite_misleads_the_icp(self):
        """The behaviour the normals are there to prevent, pinned down.

        Ignoring which way the surfaces face, the ICP settles millimetres away
        and reports a fitness and an RMSE that both read as a clean result.
        """
        moving, cbct, truth = BiteScene()
        original = areg._point_normals
        areg._point_normals = lambda mesh, name: None
        try:
            _, matrix, quality = areg.run_icp_point_to_plane(
                moving, cbct, max_dist=1.0, label="no normals")
        finally:
            areg._point_normals = original

        self.assertGreater(self.PoseError(moving, matrix, truth), 1.0)
        self.assertGreater(quality["fitness"], 0.9)
        self.assertLess(quality["inlier_rmse"], 0.5)

    # ------------------------------------------------------- ordinary targets

    def test_a_clean_target_is_recovered_exactly(self):
        maxilla = Sheet(0.0, 0.0, facing_down=True)
        perturbation = Rigid([1, 0.3, 0.1], 1.2, [0.35, -0.4, -0.4])
        moving = maxilla.transform(perturbation, inplace=False)

        _, matrix, _ = areg.run_icp_point_to_plane(
            moving, maxilla, max_dist=1.0, label="clean")

        self.assertLess(
            self.PoseError(moving, matrix, np.linalg.inv(perturbation)), 0.05)

    def test_an_inverted_winding_is_found_and_used(self):
        """Which side a normal points to is a property of the file, not the anatomy."""
        maxilla = Sheet(0.0, 0.0, facing_down=True)
        perturbation = Rigid([1, 0.3, 0.1], 1.2, [0.35, -0.4, -0.4])
        moving = maxilla.transform(perturbation, inplace=False).copy()
        moving.point_data["Normals"] = -np.asarray(moving.point_data["Normals"])

        _, matrix, quality = areg.run_icp_point_to_plane(
            moving, maxilla, max_dist=1.0, label="inverted")

        self.assertLess(
            self.PoseError(moving, matrix, np.linalg.inv(perturbation)), 0.05)
        self.assertLess(quality["sign"], 0)

    def test_an_ios_out_of_reach_reports_no_fitness(self):
        """What a skipped pre-alignment leaves behind, and how it is caught."""
        _, cbct, _ = BiteScene()
        maxilla = Sheet(0.0, 0.0, facing_down=True)
        far = maxilla.transform(Rigid([0, 0, 1], 0, [200, 200, 200]), inplace=False)

        _, _, quality = areg.run_icp_point_to_plane(
            far, cbct, max_dist=1.0, label="out of reach")

        self.assertLess(quality["fitness"], areg.MIN_ICP_FITNESS)

    # ------------------------------------------------- stopping, and the crop

    def test_the_icp_stops_once_the_arch_has_settled(self):
        """It used to circle the answer until the iteration cap.

        Converging and stopping are not the same thing: the RMSE and the
        fitness settle within tens of iterations, then flicker by a point or
        two forever, which a threshold on their change never calls done.
        """
        maxilla = Sheet(0.0, 0.0, facing_down=True)
        perturbation = Rigid([1, 0.3, 0.1], 1.2, [0.35, -0.4, -0.4])
        moving = maxilla.transform(perturbation, inplace=False)

        _, _, quality = areg.run_icp_point_to_plane(
            moving, maxilla, max_dist=1.0, label="settling")

        self.assertTrue(quality["settled"])
        self.assertLess(quality["iterations"], areg.ICP_MAX_ITERATIONS)

    def test_the_crop_keeps_what_is_near_the_landmarks(self):
        _, cbct, _ = BiteScene()
        target = areg._Target.FromMesh(cbct, "cbct")
        # A landmark box over the middle of the sheets only.
        anchor = np.array([[-5, -5, 0], [5, 5, 0]], dtype=float)

        cropped = target.Around(anchor, 10.0, "crop")

        self.assertLess(len(cropped), len(target))
        self.assertTrue(np.all(cropped.points[:, 0] >= -15.001))
        self.assertTrue(np.all(cropped.points[:, 0] <= 15.001))
        self.assertEqual(len(cropped.normals), len(cropped.points))

    def test_the_crop_does_not_move_the_answer(self):
        """Cropping bounds the search; it must not change where it lands."""
        moving, cbct, truth = BiteScene()
        target = areg._Target.FromMesh(cbct, "cbct")
        anchor = np.asarray(moving.points)[::500]

        _, whole, _ = areg.run_icp_point_to_plane(
            moving, target, max_dist=1.0, label="whole")
        _, cropped, _ = areg.run_icp_point_to_plane(
            moving, target.Around(anchor, 15.0, "crop"), max_dist=1.0, label="cropped")

        points = np.asarray(moving.points)
        self.assertLess(float(np.max(np.linalg.norm(
            Apply(whole, points) - Apply(cropped, points), axis=1))), 0.01)

    def test_a_crop_with_no_landmark_keeps_the_whole_target(self):
        _, cbct, _ = BiteScene()
        target = areg._Target.FromMesh(cbct, "cbct")

        self.assertIs(target.Around(np.empty((0, 3)), 25.0, "none"), target)
        self.assertIs(target.Around(np.array([[1e6, 1e6, 1e6]]), 1.0, "far"), target)

    # ------------------------------------------------------- reading the names

    def test_a_name_claiming_both_arches_is_refused(self):
        """ALI_IOS names its output after the scan and the model it ran.

        Observed on a real scan: a lower arch whose segmentation held a few
        upper tooth numbers came back as "P09_T1_L_SegOr_Upper_O_Pred.json".
        Read as upper, it takes the place of that patient's real upper
        landmarks.
        """
        folder = tempfile.mkdtemp()
        for name in ("P09_T1_U_SegOr_Upper_O_Pred.json",
                     "P09_T1_L_SegOr_Lower_O_Pred.json",
                     "P09_T1_L_SegOr_Upper_O_Pred.json"):
            with open(os.path.join(folder, name), "w") as handle:
                handle.write("{}")

        found = areg.getPatients(folder, folder, folder, folder)

        upper = found["P9_T1"]["ios_lm_upper"]
        lower = found["P9_T1"]["ios_lm_lower"]
        self.assertEqual(os.path.basename(upper), "P09_T1_U_SegOr_Upper_O_Pred.json")
        self.assertEqual(os.path.basename(lower), "P09_T1_L_SegOr_Lower_O_Pred.json")

    def test_two_markers_that_agree_are_read_normally(self):
        folder = tempfile.mkdtemp()
        for name in ("P09_T1_U_SegOr_Upper_O_Pred.json",
                     "P09_T1_L_SegOr_Lower_O_Pred.json"):
            with open(os.path.join(folder, name), "w") as handle:
                handle.write("{}")

        found = areg.getPatients(folder, folder, folder, folder)

        self.assertIn("ios_lm_upper", found["P9_T1"])
        self.assertIn("ios_lm_lower", found["P9_T1"])

    # --------------------------------------------------- the pre-alignment fit

    def test_a_full_arch_of_landmarks_is_well_spread(self):
        for xs in ([-25, -12, -4, 4, 12, 25], [-25, -12, -4, 4]):
            self.assertGreaterEqual(
                areg._landmark_spread_ratio(ArchLandmarks(xs)),
                areg.MIN_LANDMARK_SPREAD_RATIO)

    def test_anterior_landmarks_alone_are_flagged_as_a_line(self):
        self.assertLess(areg._landmark_spread_ratio(ArchLandmarks([-12, -4, 4])),
                        areg.MIN_LANDMARK_SPREAD_RATIO)

    def test_a_collinear_set_fits_exactly_and_proves_nothing(self):
        """Why the residual threshold cannot stand in for the spread check."""
        collinear = np.array([[-8, -18, 0], [0, -18, 0], [8, -18, 0]], dtype=float)
        truth = Rigid([0.3, 1, 0.2], 12, [15, -8, 4])

        _, matrix, _, _ = areg.align_by_landmarks(
            pv.Sphere(radius=5), collinear, Apply(truth, collinear),
            "Upper", "collinear")

        self.assertLess(
            areg._alignment_residual(collinear, Apply(truth, collinear), matrix),
            0.01)
        self.assertLess(areg._landmark_spread_ratio(collinear),
                        areg.MIN_LANDMARK_SPREAD_RATIO)

    def test_a_flagged_pre_alignment_is_still_applied(self):
        """Under-determined is not wrong: the identity would start further away."""
        source = ArchLandmarks([-12, -4, 4])
        truth = Rigid([0.3, 1, 0.2], 12, [15, -8, 4])
        target = Apply(truth, source)

        _, matrix, _, _ = areg.align_by_landmarks(
            pv.Sphere(radius=5), source, target, "Upper", "flagged")

        self.assertLess(areg._alignment_residual(source, target, matrix), 0.01)


    def test_align_by_landmarks_returns_four_values(self):
        """The shape of the return, pinned so a fifth value is caught here.

        Adding a value to this tuple broke both callers above and neither said
        so: the suite is not run on a change, so `ValueError: too many values
        to unpack` sat in the repository from ad5d96f until an audit ran it by
        hand. One assertion on the contract fails with a readable message at a
        single place, instead of at every call site.
        """
        source = np.array([[-8, -18, 0], [8, -18, 0], [0, -6, 4], [0, -22, -5]],
                          dtype=float)
        result = areg.align_by_landmarks(
            pv.Sphere(radius=5), source,
            Apply(Rigid([0.3, 1, 0.2], 12, [15, -8, 4]), source),
            "Upper", "contract")

        self.assertEqual(
            len(result), 4,
            "align_by_landmarks returns %d values; the callers in this suite "
            "and in AREG_IOSCBCT.py unpack 4 (mesh, matrix, landmarks, kept)."
            % len(result))


if __name__ == "__main__":
    unittest.main()
