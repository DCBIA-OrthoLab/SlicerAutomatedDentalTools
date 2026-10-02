# How the two arches of a patient are told apart, paired, and oriented, checked
# on synthetic segmented surfaces so the suite needs no patient scan.
#
# Each case guards a failure that was live and silent:
#   - a bare letter matched anywhere in a name made "Dupont_03_L.vtk" an upper
#     on the strength of the u in the name, while "P1_T1_U.vtk" had no jaw at
#     all because only "_U_" was looked for, which raised on the whole folder;
#   - a mouth was any two files sharing a name, the first taken as the upper,
#     so a patient with the same arch in two formats was paired as a mouth
#     whose lower arch is a maxilla;
#   - orienting the arches in occlusion fits the upper alone and applies its
#     matrix to the lower unchanged, which leaves the lower nowhere near its own
#     reference whenever the two were not scanned in occlusion.
#
# Run with:  python -m unittest discover ASO_IOS/Testing/Python
import os
import sys
import tempfile
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

ASO_IOS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ASO_IOS)
sys.path.insert(0, os.path.join(ASO_IOS, "PRE_ASO_IOS"))

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

from ASO_IOS_utils.utils import (  # noqa: E402
    JawFromFileName, PatientNumber, StripJawFromFileName, UpperOrLower)
from ASO_IOS_utils.data_file import Files_vtk_link  # noqa: E402
from ASO_IOS_utils.icp import npSameNumberPoint  # noqa: E402
import PRE_ASO_IOS  # noqa: E402


# Universal ids of the teeth each arch is oriented on, and where they sit along
# the arch. The lower set mirrors the upper one tooth for tooth.
UPPER_TEETH = {3: -22, 5: -14, 12: 14, 14: 22}    # UR6 UR4 UL4 UL6
LOWER_TEETH = {19: -21, 21: -13, 28: 13, 30: 21}  # LL6 LL4 LR4 LR6

POINTS_PER_TOOTH = 120


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


def Arch(teeth, z):
    """A blob of points per tooth along a parabolic arch, labelled Universal_ID."""
    generator = np.random.default_rng(0)
    points, ids = [], []
    for tooth, x in teeth.items():
        centre = np.array([x, 0.02 * x * x - 18.0, z])
        points.append(generator.normal(0, 0.8, size=(POINTS_PER_TOOTH, 3)) + centre)
        ids += [tooth] * POINTS_PER_TOOTH
    return np.vstack(points), np.array(ids, dtype=np.int32)


def WriteArch(path, points, ids):
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(*point)
    vertices = vtk.vtkCellArray()
    for index in range(len(points)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(index)

    surface = vtk.vtkPolyData()
    surface.SetPoints(vtk_points)
    surface.SetVerts(vertices)
    labels = numpy_to_vtk(ids, deep=True)
    labels.SetName("Universal_ID")
    surface.GetPointData().AddArray(labels)
    surface.GetPointData().SetActiveScalars("Universal_ID")

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(surface)
    writer.Write()


def MeanPerTooth(path, teeth):
    from vtk.util.numpy_support import vtk_to_numpy
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    surface = reader.GetOutput()
    ids = vtk_to_numpy(surface.GetPointData().GetScalars("Universal_ID"))
    points = vtk_to_numpy(surface.GetPoints().GetData())
    return {tooth: points[ids == tooth].mean(axis=0) for tooth in teeth}


def FolderOf(names):
    folder = tempfile.mkdtemp()
    for name in names:
        with open(os.path.join(folder, name), "w") as handle:
            handle.write("only the name is read")
    return folder


class Arguments:
    """What PRE_ASO_IOS.main reads off argparse: every value in a 1-list."""

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, [value])


class JawFromNameTest(unittest.TestCase):

    def test_the_words_are_read_anywhere(self):
        self.assertEqual(JawFromFileName("GoldUpper.vtk"), "Upper")
        self.assertEqual(JawFromFileName("P1_T1_Lower_Seg.vtk"), "Lower")

    def test_a_letter_only_counts_when_it_stands_alone(self):
        self.assertEqual(JawFromFileName("P1_T1_U.vtk"), "Upper")
        self.assertEqual(JawFromFileName("P1_T1_U_Seg.vtk"), "Upper")
        # The u of Dupont used to carry the vote.
        self.assertEqual(JawFromFileName("Dupont_03_L.vtk"), "Lower")
        self.assertEqual(JawFromFileName("P001_T1_L_Surface.vtk"), "Lower")

    def test_only_the_base_name_is_read(self):
        self.assertEqual(JawFromFileName("/data/lower_arches/P1_T1_U.vtk"), "Upper")

    def test_a_name_that_says_nothing_says_nothing(self):
        self.assertIsNone(JawFromFileName("P1T1U.vtk"))
        # UpperOrLower keeps its documented default for the gold references.
        self.assertEqual(UpperOrLower("P1T1U.vtk"), "Lower")

    def test_both_arches_reduce_to_the_same_pairing_key(self):
        for upper, lower in (("P1_T1_U_Seg", "P1_T1_L_Seg"),
                             ("P1_T1_Upper_Seg", "P1_T1_Lower_Seg"),
                             ("P1_T1_U", "P1_T1_L")):
            self.assertEqual(StripJawFromFileName(upper),
                             StripJawFromFileName(lower))


class PairingTest(unittest.TestCase):

    def test_a_mouth_is_one_upper_and_one_lower(self):
        folder = FolderOf(["P1_T1_U_Seg.vtk", "P1_T1_L_Seg.vtk",
                           "P2_T1_L_Seg.vtk", "P2_T1_U_Seg.vtk"])
        mouths = {mouth.name: (os.path.basename(mouth.Upper),
                               os.path.basename(mouth.Lower))
                  for mouth in Files_vtk_link(folder).list_file}

        self.assertEqual(len(mouths), 2)
        self.assertEqual(mouths["P1_T1_Seg"], ("P1_T1_U_Seg.vtk", "P1_T1_L_Seg.vtk"))
        self.assertEqual(mouths["P2_T1_Seg"], ("P2_T1_U_Seg.vtk", "P2_T1_L_Seg.vtk"))

    def test_the_same_arch_twice_is_not_a_mouth(self):
        """It used to be, with the second copy standing in for the lower."""
        folder = FolderOf(["P3_T1_U_Seg.vtk", "P3_T1_U_Seg.stl"])
        self.assertEqual(Files_vtk_link(folder).list_file, [])

    def test_a_missing_arch_is_not_a_mouth(self):
        folder = FolderOf(["P4_T1_U_Seg.vtk"])
        self.assertEqual(Files_vtk_link(folder).list_file, [])

    def test_names_without_a_trailing_separator_still_pair(self):
        """"P1_T1_U.vtk" used to raise, taking the whole folder down with it."""
        folder = FolderOf(["P5_T1_U.vtk", "P5_T1_L.vtk"])
        mouths = Files_vtk_link(folder).list_file

        self.assertEqual(len(mouths), 1)
        self.assertEqual(os.path.basename(mouths[0].Upper), "P5_T1_U.vtk")
        self.assertEqual(os.path.basename(mouths[0].Lower), "P5_T1_L.vtk")


class OrientationTest(unittest.TestCase):
    """PRE_ASO_IOS end to end, on two arches displaced independently.

    That is the open-bite case: the arches do not hold the relation the gold
    pair holds, so an upper-only fit cannot put the lower where it belongs.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        gold = os.path.join(self.root, "gold")
        self.input = os.path.join(self.root, "in")
        os.makedirs(gold)
        os.makedirs(self.input)

        upper, upper_ids = Arch(UPPER_TEETH, +1.0)
        lower, lower_ids = Arch(LOWER_TEETH, -1.0)
        WriteArch(os.path.join(gold, "Gold_Upper.vtk"), upper, upper_ids)
        WriteArch(os.path.join(gold, "Gold_Lower.vtk"), lower, lower_ids)

        WriteArch(os.path.join(self.input, "P1_T1_U_Seg.vtk"),
                  Apply(Rigid([0.2, 1, 0.3], 25, [40, -15, 12]), upper), upper_ids)
        WriteArch(os.path.join(self.input, "P1_T1_L_Seg.vtk"),
                  Apply(Rigid([1, 0.1, -0.4], -35, [-25, 30, -18]), lower), lower_ids)

        self.gold = gold
        self.output = os.path.join(self.root, "out")

    def Run(self, list_teeth, occlusion, jaw):
        return PRE_ASO_IOS.main(Arguments(
            input=self.input, gold_folder=self.gold, output_folder=self.output,
            add_inname="Or", list_teeth=list_teeth, occlusion=occlusion, jaw=jaw,
            folder_error=os.path.join(self.root, "err"),
            log_path=os.path.join(self.root, "log.txt")))

    def DistanceToGold(self, arch_file, gold_file, teeth):
        placed = MeanPerTooth(os.path.join(self.output, arch_file), teeth)
        reference = MeanPerTooth(os.path.join(self.gold, gold_file), teeth)
        return float(np.sqrt(np.mean([
            np.square(np.linalg.norm(placed[t] - reference[t])) for t in teeth])))

    def test_each_arch_reaches_its_own_reference(self):
        report = self.Run("UR6,UR4,UL4,UL6,LL6,LL4,LR4,LR6", "false", "Upper/Lower")

        self.assertEqual(report["failed"], 0, report["errors"])
        self.assertEqual(report["successful"], 2)
        self.assertLess(
            self.DistanceToGold("P1_T1_U_SegOr.vtk", "Gold_Upper.vtk", UPPER_TEETH), 0.5)
        self.assertLess(
            self.DistanceToGold("P1_T1_L_SegOr.vtk", "Gold_Lower.vtk", LOWER_TEETH), 0.5)

    def test_each_arch_keeps_its_own_matrix(self):
        """One file per arch: cutting the jaw out of the name collided them."""
        self.Run("UR6,UR4,UL4,UL6,LL6,LL4,LR4,LR6", "false", "Upper/Lower")

        self.assertTrue(os.path.exists(os.path.join(self.output, "P1_T1_U_SegOr.tfm")))
        self.assertTrue(os.path.exists(os.path.join(self.output, "P1_T1_L_SegOr.tfm")))

    def test_in_occlusion_the_lower_is_only_carried_along(self):
        """The reason the IOS-to-CBCT pipeline does not orient in occlusion."""
        report = self.Run("UR6,UR4,UL4,UL6", "true", "Upper")

        self.assertEqual(report["successful"], 1, report["errors"])
        # One matrix for the mouth, under the patient's name: AREG_IOS reads it
        # back under that name in Auto_IOS mode.
        self.assertTrue(os.path.exists(os.path.join(self.output, "P1_T1_SegOr.tfm")))
        self.assertLess(
            self.DistanceToGold("P1_T1_U_SegOr.vtk", "Gold_Upper.vtk", UPPER_TEETH), 0.5)
        self.assertGreater(
            self.DistanceToGold("P1_T1_L_SegOr.vtk", "Gold_Lower.vtk", LOWER_TEETH), 10.0)

    def test_teeth_of_one_jaw_only_names_what_is_missing(self):
        """It used to be a bare assertion listing universal ids."""
        report = self.Run("UR6,UR4,UL4,UL6", "false", "Upper/Lower")

        self.assertEqual(report["successful"], 1)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["errors"][0]["stage"], "teeth_for_jaw")
        self.assertIn("Lower teeth", report["errors"][0]["message"])


class PatientNumberTest(unittest.TestCase):
    """What PatientNumber answers, pinned because the name meant two things.

    The module defined it twice. The first returned the first run of digits in
    the name, as an int; the second returned the name with the jaw marker and
    the extension cut off, as a str. Python keeps the last definition, so the
    first was unreachable and the callers had always been getting the string.
    Removing the dead one changes nothing -- these cases say what "nothing" is,
    so the two never silently trade places again.
    """

    def test_returns_the_name_without_jaw_marker_or_extension(self):
        self.assertEqual(PatientNumber("/data/P07_Upper.vtk"), "P07")
        self.assertEqual(PatientNumber("/data/P07_Lower.vtk"), "P07")

    def test_answers_a_string_not_a_number(self):
        """The definition that returned an int is the one that was dead."""
        self.assertIsInstance(PatientNumber("/data/P07_Upper.vtk"), str)

    def test_a_name_holding_no_digit_is_still_answered(self):
        """The int version raised or returned None here; this one does not."""
        self.assertEqual(PatientNumber("/data/Dupont_Upper.vtk"), "Dupont")


class SameNumberOfPointsTest(unittest.TestCase):
    """Both clouds come back the same length, whichever one was larger.

    The branch that trims the target read a name that was never bound, so it
    raised NameError as soon as the source held fewer points than the target --
    half the calls, and the three call sites in icp.py all go through here. The
    branch that trims the source, right above it, shows what was meant.
    """

    def test_a_larger_source_is_trimmed(self):
        source, target = npSameNumberPoint(np.zeros((50, 3)), np.ones((20, 3)))
        self.assertEqual(source.shape, (20, 3))
        self.assertEqual(target.shape, (20, 3))

    def test_a_larger_target_is_trimmed(self):
        """The case that used to raise."""
        source, target = npSameNumberPoint(np.zeros((20, 3)), np.ones((50, 3)))
        self.assertEqual(source.shape, (20, 3))
        self.assertEqual(target.shape, (20, 3))

    def test_the_trimmed_cloud_keeps_its_own_points(self):
        """Trimming picks rows from the right array, not from the other one."""
        _, target = npSameNumberPoint(np.zeros((3, 3)), np.ones((9, 3)))
        self.assertTrue((target == 1).all())

    def test_equal_sizes_are_left_alone(self):
        source, target = npSameNumberPoint(np.zeros((7, 3)), np.ones((7, 3)))
        self.assertEqual(source.shape, (7, 3))
        self.assertEqual(target.shape, (7, 3))


if __name__ == "__main__":
    unittest.main()
