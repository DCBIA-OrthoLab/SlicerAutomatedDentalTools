# The identifier read off a file name, checked against the six chains it
# replaces. Standard library only: no scan, no Slicer, no GPU.
#
# The cases below are not invented. OLD_CHAIN is the chain those six sites
# carried, transcribed marker for marker, and the first test runs both over a
# corpus of names in the shapes the pipelines actually produce. If the two ever
# disagree, the refactor changed what a patient is called -- which silently
# stops a T1 scan from pairing with its T2.
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

from ADTLib.naming import (  # noqa: E402
    patient_id, PATIENT_ID_MARKERS, ASO_CBCT_MARKERS, ASO_CBCT_CLI_MARKERS,
    AREG_IOSCBCT_MARKERS, TMJ_CROP_MARKERS, LANDMARK_SUFFIX_MARKERS)


def OLD_CHAIN(basename):
    """The chain as it stood in the six sites, written out."""
    return (
        basename.split("_Scan")[0]
        .split("_scan")[0]
        .split("_Or")[0]
        .split("_OR")[0]
        .split("_MAND")[0]
        .split("_MD")[0]
        .split("_MAX")[0]
        .split("_MX")[0]
        .split("_CB")[0]
        .split("_lm")[0]
        .split("_T2")[0]
        .split("_T1")[0]
        .split("_Cl")[0]
        .split(".")[0]
    )


CORPUS = [
    "P001_T1.nii.gz", "P001_T2.nii.gz", "P001_Scan_T1.nii.gz",
    "P001_scan.nrrd", "P001_Or.nii.gz", "P001_OR.nii.gz",
    "P001_MAND_T1.nii.gz", "P001_MD.nii.gz", "P001_MAX_T2.nii.gz",
    "P001_MX.nii.gz", "P001_CB_T1.nii.gz", "P001_lm.json",
    "P001_Cl.nii.gz", "P001_Scanreg.nii.gz", "P001_T1_MAND_Or.nii.gz",
    "MG_scan_T1.nii.gz", "Dupont_03_T2_Scan.nii.gz", "P001.vtk",
    "P001_T3.nii.gz", "P001_T4.nii.gz", "sub-01_ses-T1_CBCT.nii.gz",
    "P001_seg_T1.nii.gz", "A_B_C_T1_Scan.nii.gz", "P001_T1_lm_MAX.json",
    "plain", "", "_T1.nii.gz", "P001..nii.gz",
]


class PatientIdTest(unittest.TestCase):

    def test_matches_the_chain_it_replaces_on_every_name(self):
        for name in CORPUS:
            with self.subTest(name=name):
                self.assertEqual(patient_id(name), OLD_CHAIN(name))

    def test_a_longer_marker_is_cut_before_the_shorter_one_inside_it(self):
        """_Scanreg must not lose its tail to _Scan, nor _MAND to _MD."""
        self.assertEqual(patient_id("P001_Scanreg.nii.gz"), "P001")
        self.assertEqual(patient_id("P001_MAND.nii.gz"), "P001")
        order = list(PATIENT_ID_MARKERS)
        for longer, shorter in (("_MAND", "_MD"), ("_MAX", "_MX")):
            self.assertLess(order.index(longer), order.index(shorter))

    def test_the_two_timepoints_of_a_patient_give_the_same_id(self):
        self.assertEqual(patient_id("P001_T1.nii.gz"), patient_id("P001_T2.nii.gz"))

    def test_a_third_timepoint_does_not_pair_yet(self):
        """Documents the known limit rather than pretending it is fixed."""
        self.assertNotEqual(patient_id("P001_T3.nii.gz"), patient_id("P001_T4.nii.gz"))
        self.assertEqual(patient_id("P001_T3.nii.gz"), "P001_T3")


# The four other marker sets of the repository, transcribed from the chains
# they replace. They do NOT give the same identifier as the default set on the
# same name -- which is precisely why they were not merged into it, and these
# cases freeze that.

def ASO_CBCT_CHAIN(name):
    return (name.split("_scan")[0].split("_Scanreg")[0].split("_Scan")[0]
                .split("_Or")[0].split("_OR")[0].split("_lm")[0]
                .split("_T1")[0].split("_T2")[0].split(".")[0])


def ASO_CBCT_CLI_CHAIN(name):
    return (name.split("_Or")[0].split("_OR")[0].split("_scan")[0]
                .split("_Scanreg")[0].split("_Scan")[0].split("_lm")[0].split(".")[0])


def AREG_IOSCBCT_CHAIN(name):
    return name.split("_scan")[0].split("_Scanreg")[0].split("_lm")[0]


def TMJ_CROP_CHAIN(name):
    for marker in ("_Scan", "_scan", "_Or", "_OR", "_MAND", "_MD", "_MAX", "_MX",
                   "_CB", "_lm", "_T2", "_T1", "_Cl", "_seg", "_Seg", "_mask",
                   "_Mask", "_pred", "_Pred", "_crop", "_Crop", "_Left", "_left",
                   "_Right", "_right", "_approximate", "_Approximate", "_CBCT",
                   "_MRI", "_MR", "."):
        name = name.split(marker)[0]
    return name


def LANDMARK_SUFFIX_CHAIN(name):
    return name.split("_lm")[0].split("_Or")[0].split(".")[0]


WIDER_CORPUS = CORPUS + [
    "P001_Scanreg.nii.gz", "P001_seg_T1.nii.gz", "P001_Left_crop.nii.gz",
    "P001_MRI_T2.nii.gz", "P001_Mask.nrrd", "P001_approximate.nii.gz",
    "P001_Or_lm.json", "P001_CBCT_Right.nii.gz", "sub_01_MR.nii.gz",
]


class OtherMarkerSetsTest(unittest.TestCase):
    """Each set gives what the chain it replaces gave, name by name."""

    def _same_as(self, markers, chain):
        for name in WIDER_CORPUS:
            with self.subTest(name=name):
                self.assertEqual(patient_id(name, markers), chain(name))

    def test_aso_cbct(self):
        self._same_as(ASO_CBCT_MARKERS, ASO_CBCT_CHAIN)

    def test_aso_cbct_cli(self):
        self._same_as(ASO_CBCT_CLI_MARKERS, ASO_CBCT_CLI_CHAIN)

    def test_areg_ioscbct(self):
        self._same_as(AREG_IOSCBCT_MARKERS, AREG_IOSCBCT_CHAIN)

    def test_tmj_crop(self):
        self._same_as(TMJ_CROP_MARKERS, TMJ_CROP_CHAIN)

    def test_landmark_suffix(self):
        self._same_as(LANDMARK_SUFFIX_MARKERS, LANDMARK_SUFFIX_CHAIN)

    def test_the_sets_really_do_disagree(self):
        """If two sets always gave the same answer, they should be merged.

        This test would then fail, and that would be the good news: it would
        say that one of the lists has become useless.
        """
        answers = {
            "default": patient_id("P001_MAND_T1.nii.gz"),
            "ASO_CBCT": patient_id("P001_MAND_T1.nii.gz", ASO_CBCT_MARKERS),
            "AREG_IOSCBCT": patient_id("P001_MAND_T1.nii.gz", AREG_IOSCBCT_MARKERS),
        }
        self.assertEqual(answers["default"], "P001")
        self.assertEqual(answers["ASO_CBCT"], "P001_MAND")     # has no _MAND
        self.assertEqual(answers["AREG_IOSCBCT"], "P001_MAND_T1.nii.gz")


if __name__ == "__main__":
    unittest.main()
