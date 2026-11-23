"""The index, against the folder shapes the tools actually produce.

Every tree here is taken from a real tool's naming code rather than invented:
ALI writes landmarks and no scan, ASO writes an oriented scan BESIDE its
landmarks, AMASSS puts the scan's stem in a folder name, Crown_Seg files half
a batch one level deeper than the other half.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from SADTVisuLib import index  # noqa: E402


def tree(root, paths):
    for relative in paths:
        full = os.path.join(root, relative)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as handle:
            handle.write("x")


class PatientStemTest(unittest.TestCase):
    def test_every_tool_s_name_for_one_patient_answers_the_same(self):
        for filename in (
            "P1_scan.nii.gz",            # the acquisition
            "P1_scan_lm_Pred.mrk.json",  # ALI
            "P1_Or.nii.gz",              # ASO's oriented scan
            "P1_lm_Or.mrk.json",         # ASO's landmarks
            "P1_Seg.vtk",                # Crown_Seg
            "P1_MERGED.nii.gz",          # AMASSS merged
        ):
            self.assertEqual(index.patient_stem(filename), "P1", filename)

    def test_a_suffix_inside_a_token_does_not_truncate(self):
        # `P_Seg1` and `P_Seg2` are two subjects. Matching `_Seg` anywhere
        # collapsed them onto one upstream, and lost one of them.
        self.assertEqual(index.patient_stem("P_Seg1_T1.nii.gz"), "P_Seg1_T1")
        self.assertEqual(index.patient_stem("P_Seg2_T1.nii.gz"), "P_Seg2_T1")

    def test_timepoints_are_separate_scans_unless_asked(self):
        self.assertEqual(index.patient_stem("P1_T1_scan.nii.gz"), "P1_T1")
        self.assertEqual(
            index.patient_stem("P1_T1_scan.nii.gz", drop_timepoint=True), "P1"
        )

    def test_a_name_that_is_only_a_suffix_keeps_it(self):
        self.assertEqual(index.patient_stem("_Or.nii.gz"), "_Or")


class KindTest(unittest.TestCase):
    def test_a_mask_token_makes_a_scan_labelled_voxels(self):
        self.assertEqual(index.kind_of("P1_scan.nii.gz"), index.VOLUME)
        self.assertEqual(index.kind_of("P1_Pred_MAND.nii.gz"), index.LABELMAP)
        self.assertEqual(index.kind_of("P1_seg.nii.gz"), index.LABELMAP)

    def test_a_patient_named_after_a_token_is_not_a_mask(self):
        # Whole tokens: `"seg" in name` would make SEGOVIA_01 a segmentation.
        self.assertEqual(index.kind_of("SEGOVIA_01.nii.gz"), index.VOLUME)

    def test_the_other_kinds(self):
        self.assertEqual(index.kind_of("P1_lm_Pred.mrk.json"), index.MARKUPS)
        self.assertEqual(index.kind_of("arch.vtk"), index.MODEL)
        self.assertEqual(index.kind_of("P1_Or_transform.tfm"), index.TRANSFORM)
        self.assertIsNone(index.kind_of("notes.txt"))


class BuildTest(unittest.TestCase):
    def test_ali_landmarks_and_the_scan_they_belong_to_are_one_case(self):
        with tempfile.TemporaryDirectory() as root:
            tree(root, ["scans/P1_scan.nii.gz", "out/P1_scan_lm_Pred.mrk.json"])
            cases = index.build(
                [("Scans", os.path.join(root, "scans")),
                 ("Results", os.path.join(root, "out"))]
            )
            self.assertEqual([case.key for case in cases], ["P1"])
            self.assertEqual(len(cases[0].artifacts), 2)

    def test_reports_and_killed_run_scratch_are_not_patients(self):
        with tempfile.TemporaryDirectory() as root:
            tree(root, [
                "out/P1_scan_lm_Pred.mrk.json",
                "out/run_report.json",
                "out/AMASSS_report.json",
                "out/.amasss_work/p_000_0000.nii.gz",
            ])
            cases = index.build([("Results", os.path.join(root, "out"))])
            self.assertEqual([case.key for case in cases], ["P1"])

    def test_two_patients_of_the_same_name_in_different_folders_stay_apart(self):
        with tempfile.TemporaryDirectory() as root:
            tree(root, ["out/siteA/p01_scan.nii.gz", "out/siteB/p01_scan.nii.gz"])
            cases = index.build([("Results", os.path.join(root, "out"))])
            self.assertEqual(
                [case.key for case in cases],
                [os.path.join("siteA", "p01"), os.path.join("siteB", "p01")],
            )

    def test_amasss_takes_the_patient_from_the_folder_it_invented(self):
        # `<stem>_<prediction_ID>_SegOut/` is the only place the scan's stem
        # survives: the files inside carry a free-text id in the middle.
        with tempfile.TemporaryDirectory() as root:
            tree(root, [
                "scans/sample_scan.nii.gz",
                "out/sample_scan_Pred_SegOut/sample_scan_Pred_MAND.nii.gz",
                "out/sample_scan_Pred_SegOut/sample_scan_Pred_CB.nii.gz",
            ])
            cases = index.build(
                [("Scans", os.path.join(root, "scans")),
                 ("Results", os.path.join(root, "out"))]
            )
            self.assertEqual([case.key for case in cases], ["sample"])
            self.assertEqual(len(cases[0].artifacts), 3)

    def test_crown_seg_s_two_branches_land_in_one_case_each(self):
        with tempfile.TemporaryDirectory() as root:
            tree(root, [
                "out/siteA/passed_Seg.vtk",                      # already labelled
                "out/crownseg_input_Seg/siteA/fresh_Seg.vtk",     # segmented here
            ])
            cases = index.build([("Results", os.path.join(root, "out"))])
            self.assertEqual(
                [case.key for case in cases],
                [os.path.join("siteA", "fresh"), os.path.join("siteA", "passed")],
            )


class RealShapesTest(unittest.TestCase):
    """The three shapes the hosted test data has and a suffix table misses."""

    def test_a_dicom_series_is_one_volume_named_after_its_folder(self):
        # 577 slices called IMG0375.dcm are one scan, and the folder is the
        # only place its patient name appears.
        with tempfile.TemporaryDirectory() as root:
            tree(root, [
                "scans/CBCT_DCM/IC_0005/IMG0001.dcm",
                "scans/CBCT_DCM/IC_0005/IMG0002.dcm",
                "scans/CBCT_DCM/IC_0005_lm_Pred.mrk.json",
            ])
            cases = index.build([("Scans", os.path.join(root, "scans"))])
            self.assertEqual([case.key for case in cases],
                             [os.path.join("CBCT_DCM", "IC_0005")])
            kinds = sorted(artifact.kind for artifact in cases[0].artifacts)
            self.assertEqual(kinds, [index.MARKUPS, index.VOLUME])
            series = cases[0].of_kind(index.VOLUME)[0]
            self.assertTrue(os.path.isdir(series.path))

    def test_a_bare_json_counts_as_markups_only_when_it_is_some(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "out"))
            landmarks = os.path.join(root, "out", "P1_Upper_O_Pred.json")
            with open(landmarks, "w", encoding="utf-8") as handle:
                handle.write('{"@schema": "x", "markups": [{"type": "Fiducial"}]}')
            with open(os.path.join(root, "out", "settings.json"), "w",
                      encoding="utf-8") as handle:
                handle.write('{"device": "cuda"}')
            cases = index.build([("Results", os.path.join(root, "out"))])
            self.assertEqual([case.key for case in cases], ["P1_Upper_O_Pred"])
            self.assertEqual(cases[0].artifacts[0].kind, index.MARKUPS)

    def test_aso_ios_landmarks_reach_the_surface_they_do_not_share_a_stem_with(self):
        # `Upper_new_9.vtk` and `Upper_new_9_Upper_O_Pred.json`, the hosted
        # IOS_SemiAuto pair. No suffix table reaches it; a token-aligned
        # prefix does.
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "ios"))
            with open(os.path.join(root, "ios", "Upper_new_9.vtk"), "w") as handle:
                handle.write("x")
            with open(os.path.join(root, "ios", "Upper_new_9_Upper_O_Pred.json"),
                      "w", encoding="utf-8") as handle:
                handle.write('{"markups": []}')
            cases = index.build([("Scans", os.path.join(root, "ios"))])
            self.assertEqual([case.key for case in cases], ["Upper_new_9"])
            self.assertEqual(len(cases[0].artifacts), 2)

    def test_a_cohort_filed_by_role_is_one_case_per_patient(self):
        # `<root>/CBCT/` beside `<root>/Landmarks/` is how a reader files a
        # cohort. It is one patient, not two.
        with tempfile.TemporaryDirectory() as root:
            tree(root, [
                "data/CBCT/p1_scan.nii.gz",
                "data/CBCT/p2_scan.nii.gz",
                "data/Landmarks/p1_scan_lm_Pred.mrk.json",
                "data/Landmarks/p2_scan_lm_Pred.mrk.json",
            ])
            cases = index.build([("folder", os.path.join(root, "data"))])
            self.assertEqual([case.key for case in cases], ["p1", "p2"])
            self.assertEqual(len(cases[0].artifacts), 2)

    def test_role_folders_do_not_make_two_files_look_co_located(self):
        # Stripping `CBCT/` and `Landmarks/` from the KEY must not also claim
        # the points were written beside the scan. They were not: they are
        # NAMED after it, which is what the view says.
        with tempfile.TemporaryDirectory() as root:
            tree(root, ["d/CBCT/p1_scan.nii.gz", "d/Landmarks/p1_scan_lm_Pred.mrk.json"])
            case = index.build([("folder", os.path.join(root, "d"))])[0]
            views = [v for v in case.views(acquisition="folder") if v.overlays]
            self.assertEqual(len(views), 1)
            self.assertEqual(views[0].anchor.name, "p1_scan.nii.gz")
            self.assertEqual(views[0].basis, index.BASIS_NAMED)

    def test_a_role_folder_of_several_words_is_still_one(self):
        # The hosted AREG fixture files one subject under `CBCT Landmarks/`,
        # `IOS Landmarks/` and `T2/`. Matching exact names left it indexed
        # three times.
        with tempfile.TemporaryDirectory() as root:
            tree(root, [
                "d/T2/P_0001_T2.nii.gz",
                "d/CBCT Landmarks/P_0001_T2_lm_Pred_U.mrk.json",
                "d/IOS Landmarks/P_0001_T2_lm_Pred_L.mrk.json",
            ])
            cases = index.build([("folder", os.path.join(root, "d"))])
            self.assertEqual([case.key for case in cases], ["P_0001_T2"])
            self.assertEqual(len(cases[0].artifacts), 3)

    def test_a_folder_named_after_a_subject_is_never_a_role(self):
        # Every token has to be a role word, or `patient_T1/` would vanish.
        self.assertTrue(index.is_role_level("Landmarks"))
        self.assertTrue(index.is_role_level("CBCT Landmarks"))
        self.assertTrue(index.is_role_level("T2"))
        self.assertFalse(index.is_role_level("patient_T1"))
        self.assertFalse(index.is_role_level("P_0001_T2"))
        self.assertFalse(index.is_role_level("cohort_6"))

    def test_ios_is_a_role_folder_too(self):
        with tempfile.TemporaryDirectory() as root:
            tree(root, ["d/IOS/Upper_new_9.vtk", "d/Landmarks/Upper_new_9_Upper_O.mrk.json"])
            cases = index.build([("folder", os.path.join(root, "d"))])
            self.assertEqual([case.key for case in cases], ["Upper_new_9"])

    def test_loose_masks_join_the_scan_they_were_made_from(self):
        # AMASSS names a mask `<stem>_<prediction_ID>_<CODE>`, and the id is
        # free text no suffix table can strip. Loose in a folder -- no
        # `_SegOut/` level to read it off -- each one used to index as its own
        # patient.
        with tempfile.TemporaryDirectory() as root:
            tree(root, [
                "d/CBCT/p1_scan.nii.gz",
                "d/Masks/p1_scan_Pred_MAND.nii.gz",
                "d/Masks/p1_scan_Pred_MAX.nii.gz",
                "d/Landmarks/p1_scan_lm_Pred.mrk.json",
            ])
            cases = index.build([("folder", os.path.join(root, "d"))])
            self.assertEqual([case.key for case in cases], ["p1"])
            kinds = sorted(a.kind for a in cases[0].artifacts)
            self.assertEqual(kinds, [index.LABELMAP, index.LABELMAP,
                                     index.MARKUPS, index.VOLUME])

    def test_a_mask_with_no_scan_anywhere_is_still_a_case(self):
        # Absorbed only when there is something to absorb it INTO.
        with tempfile.TemporaryDirectory() as root:
            tree(root, ["d/p1_scan_Pred_MAND.nii.gz"])
            cases = index.build([("folder", os.path.join(root, "d"))])
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].artifacts[0].kind, index.LABELMAP)

    def test_a_prefix_that_is_another_patient_does_not_absorb_it(self):
        # `P1` and `P10` are two patients. Matching on substring paired them
        # upstream and padded the list with a sentinel to hide it.
        self.assertTrue(index.is_token_prefix("P1", "P1_lm_Pred"))
        self.assertFalse(index.is_token_prefix("P1", "P10_lm_Pred"))
        self.assertFalse(index.is_token_prefix("P1", "P1"))


class LabelTest(unittest.TestCase):
    def test_a_case_reads_as_the_patient_with_the_folder_behind_it(self):
        case = index.Case(os.path.join("CBCT_SemiAuto", "IC_0005"))
        self.assertEqual(case.patient, "IC_0005")
        self.assertEqual(case.label, "IC_0005  (CBCT_SemiAuto)")

    def test_a_case_at_the_root_is_just_its_name(self):
        self.assertEqual(index.Case("IC_0005").label, "IC_0005")


class ViewTest(unittest.TestCase):
    """Which scan an overlay is drawn on. The one thing that must not lie."""

    def _case(self, root, paths, sources):
        tree(root, paths)
        cases = index.build([(label, os.path.join(root, path)) for label, path in sources])
        self.assertEqual(len(cases), 1, [case.key for case in cases])
        return cases[0]

    def test_aso_landmarks_bind_to_the_oriented_scan_not_the_original(self):
        # ASO's points carry the recentring AND the ICP rotation, so they match
        # the scan ASO WROTE. Drawn on the acquisition they render fine and are
        # wrong by a rotation.
        with tempfile.TemporaryDirectory() as root:
            case = self._case(
                root,
                ["scans/P1_scan.nii.gz",
                 "out/P1_Or.nii.gz",
                 "out/P1_lm_Or.mrk.json",
                 "out/P1_Or_transform.tfm"],
                [("Scans", "scans"), ("Results", "out")],
            )
            views = case.views(acquisition="Scans")
            with_points = [view for view in views if view.overlays]
            self.assertEqual(len(with_points), 1)
            self.assertEqual(with_points[0].anchor.name, "P1_Or.nii.gz")
            self.assertEqual(with_points[0].basis, index.BASIS_COLOCATED)
            # And the acquisition is still offered, carrying nothing.
            self.assertIn("P1_scan.nii.gz", [view.label for view in views])

    def test_ali_landmarks_fall_back_to_the_acquisition_and_say_so(self):
        # ALI writes no scan at all, so its points can only be drawn on the
        # input -- where they are right, measured at 0.000 mm against ITK. The
        # file is named after that input, which is the ground the view gives.
        with tempfile.TemporaryDirectory() as root:
            case = self._case(
                root,
                ["scans/P1_scan.nii.gz", "out/P1_scan_lm_Pred.mrk.json"],
                [("Scans", "scans"), ("Results", "out")],
            )
            views = case.views(acquisition="Scans")
            with_points = [view for view in views if view.overlays]
            self.assertEqual(len(with_points), 1)
            self.assertEqual(with_points[0].anchor.name, "P1_scan.nii.gz")
            self.assertEqual(with_points[0].basis, index.BASIS_NAMED)

    def test_landmarks_named_after_no_scan_still_fall_back_to_the_acquisition(self):
        with tempfile.TemporaryDirectory() as root:
            case = self._case(
                root,
                ["scans/P1_scan.nii.gz", "out/P1_lm_Pred.mrk.json"],
                [("Scans", "scans"), ("Results", "out")],
            )
            with_points = [view for view in case.views(acquisition="Scans") if view.overlays]
            self.assertEqual([view.basis for view in with_points], [index.BASIS_ACQUISITION])

    def test_landmarks_with_no_scan_anywhere_are_offered_without_one(self):
        with tempfile.TemporaryDirectory() as root:
            case = self._case(
                root, ["out/P1_scan_lm_Pred.mrk.json"], [("Results", "out")]
            )
            views = case.views(acquisition="Scans")
            self.assertEqual(len(views), 1)
            self.assertIsNone(views[0].anchor)
            self.assertEqual(views[0].basis, index.BASIS_NONE)



class PauseTest(unittest.TestCase):
    """What a stopped AREG IOS-to-CBCT run hands back, tree for tree: each step
    under `intermediate/<NN>_<Tool>/` holds only what it wrote, so a patient's
    mesh and the landmarks placed on it are two steps apart."""

    ALI_IOS = [
        "intermediate/02_Crown_Seg/ios_case_T2_L_Seg.vtk",
        "intermediate/02_Crown_Seg/ios_case_T2_U_Seg.vtk",
        "intermediate/02_Crown_Seg/run_report.json",
        "intermediate/03_ALI_IOS/ios_case_T2_L_Seg_lm_Pred.mrk.json",
        "intermediate/03_ALI_IOS/ios_case_T2_U_Seg_lm_Pred.mrk.json",
        "intermediate/03_ALI_IOS/run_report.json",
    ]
    ALI_CBCT = [
        "intermediate/01_ASO/cbct_case_T2_Or.nii.gz",
        "intermediate/01_ASO/cbct_case_T2_lm_Or.mrk.json",
        "intermediate/01_ASO/cbct_case_T2_Or_transform.tfm",
        "intermediate/01_ASO/intermediate/01_ALI_CBCT/cbct_case_T2_lm_Pred.mrk.json",
        "intermediate/04_ALI_CBCT/cbct_case_T2_Or_lm_Pred.mrk.json",
    ]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _views(self, case):
        return {(view.anchor.name if view.anchor else None): (
            sorted(a.name for a in view.overlays), view.basis)
            for view in case.views(acquisition="Results") if view.overlays}

    def test_each_arch_is_one_case_with_its_landmarks_on_its_mesh(self):
        tree(self.root, self.ALI_IOS)
        cases = index.build([("Results", self.root)])
        self.assertEqual([case.key for case in cases], ["ios_case_T2_L", "ios_case_T2_U"])
        self.assertEqual(self._views(cases[1]), {
            "ios_case_T2_U_Seg.vtk": (["ios_case_T2_U_Seg_lm_Pred.mrk.json"], index.BASIS_NAMED)})
        # One view, not an empty one with the points behind a second.
        self.assertEqual(len(cases[1].views(acquisition="Results")), 1)

    def test_each_landmark_file_lands_on_the_scan_it_was_placed_on(self):
        """ALI_CBCT inside ASO placed its points on the CBCT the caller sent;
        the later ALI_CBCT on ASO's oriented one. Drawing either on the other
        is wrong by a rotation, and the names are what tell them apart."""
        tree(self.root, self.ALI_CBCT)
        scans = os.path.join(self.root, "inputs")
        tree(scans, ["T2/cbct_case_T2.nii.gz"])
        cases = index.build([("Results", os.path.join(self.root, "intermediate"))],
                            supports=[scans])
        (case,) = cases
        self.assertEqual(self._views(case), {
            "cbct_case_T2_Or.nii.gz": (
                ["cbct_case_T2_Or_lm_Pred.mrk.json", "cbct_case_T2_lm_Or.mrk.json"],
                index.BASIS_COLOCATED_AND_NAMED),
            "cbct_case_T2.nii.gz": (["cbct_case_T2_lm_Pred.mrk.json"], index.BASIS_NAMED),
        })
        # ALI_CBCT's points and ASO's are in ONE view of the oriented scan.
        self.assertEqual(len(case.views(acquisition="Results")), 2)

    def test_the_first_cbct_pause_has_its_points_on_the_input(self):
        """At `ASO/ALI_CBCT` only the landmarks exist: their scan is the input,
        which only the module that sent it has."""
        tree(self.root, ["out/intermediate/01_ASO/intermediate/01_ALI_CBCT/cbct_case_T2_lm_Pred.mrk.json"])
        tree(self.root, ["inputs/cbct_case_T2.nii.gz"])
        (case,) = index.build([("Results", os.path.join(self.root, "out"))],
                              supports=[os.path.join(self.root, "inputs", "cbct_case_T2.nii.gz")])
        self.assertEqual(self._views(case), {
            "cbct_case_T2.nii.gz": (["cbct_case_T2_lm_Pred.mrk.json"], index.BASIS_NAMED)})

    def test_a_segmentation_stop_draws_the_mask_on_the_scan_it_segmented(self):
        """AMASSS writes the labels and nothing else; on their own they were
        a mask on black. They are named after the T1 ASO oriented."""
        tree(self.root, [
            "out/intermediate/01_ASO/cbct_case_T1_Or.nii.gz",
            "out/intermediate/01_ASO/cbct_case_T1_lm_Or.mrk.json",
            "out/intermediate/02_AMASSS/cbct_case_T1_Or_seg_SegOut/cbct_case_T1_Or_seg_CBMASK.nii.gz",
        ])
        (case,) = index.build([("Results", os.path.join(self.root, "out"))],
                              focus=os.path.join(self.root, "out", "intermediate", "02_AMASSS"))
        (view,) = case.views(acquisition="Results")
        self.assertEqual(view.anchor.name, "cbct_case_T1_Or.nii.gz")
        self.assertEqual(view.anchor.origin, os.path.join("intermediate", "01_ASO"))
        self.assertEqual([o.name for o in view.overlays], ["cbct_case_T1_Or_seg_CBMASK.nii.gz"])

    def test_a_step_inside_another_is_never_drawn_on_what_that_one_wrote(self):
        """A cohort sent already named `_Or` meets the `_Or` scan ASO wrote.
        ALI_CBCT ran inside ASO, on the scan sent, before ASO wrote its own:
        its points belong on the input, whatever the names say."""
        tree(self.root, [
            "out/intermediate/01_ASO/cbct_case_T1_Or.nii.gz",
            "out/intermediate/01_ASO/cbct_case_T1_lm_Or.mrk.json",
            "out/intermediate/01_ASO/intermediate/01_ALI_CBCT/cbct_case_T1_Or_lm_Pred.mrk.json",
            "inputs/cbct_case_T1_Or.nii.gz",
        ])
        (case,) = index.build([("Results", os.path.join(self.root, "out"))],
                              supports=[os.path.join(self.root, "inputs")],
                              focus=os.path.join(self.root, "out", "intermediate", "01_ASO"))
        drawn = {(view.anchor.origin, tuple(o.name for o in view.overlays))
                 for view in case.views(acquisition="Results")}
        self.assertEqual(drawn, {
            (os.path.join("intermediate", "01_ASO"), ("cbct_case_T1_lm_Or.mrk.json",)),
            ("", ("cbct_case_T1_Or_lm_Pred.mrk.json",)),
        })

    def test_inputs_never_make_a_case_or_an_empty_view(self):
        tree(self.root, self.ALI_IOS)
        tree(self.root, ["inputs/ios_case_T2_U.vtk", "inputs/P009_T2_U.vtk"])
        cases = index.build([("Results", self.root + "/intermediate")],
                            supports=[os.path.join(self.root, "inputs")])
        self.assertEqual([case.key for case in cases], ["ios_case_T2_L", "ios_case_T2_U"])
        anchors = [view.anchor.name for view in cases[1].views(acquisition="Results")]
        self.assertNotIn("ios_case_T2_U.vtk", anchors)

    def test_a_stop_shows_what_its_last_step_wrote(self):
        """At ALI_CBCT, the oriented CBCT carries ALI_CBCT's points only: ASO's
        own, and the arches of the steps before, are earlier steps' work."""
        tree(self.root, self.ALI_IOS + self.ALI_CBCT)
        (case,) = index.build([("Results", self.root)],
                              focus=os.path.join(self.root, "intermediate", "04_ALI_CBCT"))
        self.assertEqual(self._views(case), {
            "cbct_case_T2_Or.nii.gz": (["cbct_case_T2_Or_lm_Pred.mrk.json"], index.BASIS_NAMED)})

    def test_a_step_that_wrote_meshes_shows_the_meshes_alone(self):
        tree(self.root, self.ALI_IOS + self.ALI_CBCT)
        cases = index.build([("Results", self.root)],
                            focus=os.path.join(self.root, "intermediate", "02_Crown_Seg"))
        self.assertEqual([case.key for case in cases], ["ios_case_T2_L", "ios_case_T2_U"])
        self.assertEqual(self._views(cases[1]), {})

    def test_a_numbered_patient_folder_outside_intermediate_stays_a_patient(self):
        tree(self.root, ["01_P1/scan.nii.gz", "02_P2/scan.nii.gz"])
        keys = [case.key for case in index.build([("Results", self.root)])]
        self.assertEqual(len(keys), 2)

if __name__ == "__main__":
    unittest.main()


class FolderTest(unittest.TestCase):
    """Which level of a folder a case came from."""

    def _cases(self, paths):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        tree(root.name, paths)
        return index.build([("folder", os.path.join(root.name, "d"))])

    def test_a_case_at_the_top_is_named_rather_than_left_blank(self):
        cases = self._cases(["d/p1_scan.nii.gz"])
        self.assertEqual(index.folder_of(cases[0]), index.AT_THE_TOP)

    def test_the_first_level_is_what_a_reader_picks_between(self):
        # Deeper levels are how a tool mirrors an input tree; one chip per
        # patient would be a second case list.
        cases = self._cases(["d/CBCT_SemiAuto/siteA/p1_scan.nii.gz"])
        self.assertEqual(index.folder_of(cases[0]), "CBCT_SemiAuto")

    def test_they_are_listed_in_the_order_the_cases_are(self):
        cases = self._cases(["d/IOS_SemiAuto/u1.vtk",
                             "d/CBCT_SemiAuto/p1_scan.nii.gz",
                             "d/CBCT_SemiAuto/p2_scan.nii.gz"])
        self.assertEqual(index.folders_in(cases), ["CBCT_SemiAuto", "IOS_SemiAuto"])

    def test_a_role_folder_is_never_offered_as_one(self):
        """`CBCT/` beside `Landmarks/` is how ONE cohort is filed, not two to
        pick between -- and stripping them is what made the scan and its
        points one patient in the first place."""
        cases = self._cases(["d/CBCT/p1_scan.nii.gz",
                             "d/Landmarks/p1_scan_lm_Pred.mrk.json"])
        self.assertEqual(len(cases), 1)
        self.assertEqual(index.folders_in(cases), [index.AT_THE_TOP])

    def test_a_folder_with_no_levels_offers_one_entry(self):
        cases = self._cases(["d/p1_scan.nii.gz", "d/p2_scan.nii.gz"])
        self.assertEqual(index.folders_in(cases), [index.AT_THE_TOP])


def write_scene(path, files, space="the CBCT"):
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"version": 1, "space": space, "files": [
            {"path": name, "kind": kind, "role": "r"} for name, kind in files]}, handle)


class SceneTest(unittest.TestCase):
    """A tool that says which files make one scene is believed, not guessed."""

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_an_ios_cbct_patient_is_one_case_and_one_view(self):
        """AREG IOS to CBCT: the meshes, their moved landmarks and the CBCT
        share no stem, and were split across cases by the name rules."""
        tree(self.root, [
            "case_042/scan_T2.nii.gz", "case_042/scan_T2_lm_Pred.mrk.json",
            "case_042/case_042_U_Reg.vtk", "case_042/case_042_U_Reg_lm.mrk.json",
            "case_042/case_042_L_Reg.vtk", "case_042/case_042_U_Reg_transform.tfm",
        ])
        write_scene(os.path.join(self.root, "case_042", "scene.json"), [
            ("scan_T2.nii.gz", "volume"), ("scan_T2_lm_Pred.mrk.json", "markups"),
            ("case_042_U_Reg.vtk", "model"), ("case_042_U_Reg_lm.mrk.json", "markups"),
            ("case_042_L_Reg.vtk", "model"), ("case_042_U_Reg_transform.tfm", "transform"),
        ])
        cases = index.build([("Results", self.root)])
        self.assertEqual([case.key for case in cases], ["case_042"])
        views = cases[0].views()
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].anchor.name, "scan_T2.nii.gz")
        self.assertEqual(views[0].basis, index.BASIS_SCENE)
        self.assertEqual(sorted(a.name for a in views[0].overlays), [
            "case_042_L_Reg.vtk", "case_042_U_Reg.vtk", "case_042_U_Reg_lm.mrk.json",
            "scan_T2_lm_Pred.mrk.json"])

    def test_a_scene_without_a_scan_anchors_on_its_first_surface(self):
        tree(self.root, ["P1_T1_Upper_Reg.vtk", "P1_T2_Upper_Reg.vtk"])
        write_scene(os.path.join(self.root, "P1_Reg_scene.json"), [
            ("P1_T1_Upper_Reg.vtk", "model"), ("P1_T2_Upper_Reg.vtk", "model")])
        cases = index.build([("Results", self.root)])
        self.assertEqual([case.key for case in cases], ["P1_Reg"])
        (view,) = cases[0].views()
        self.assertEqual(view.anchor.name, "P1_T1_Upper_Reg.vtk")
        self.assertEqual([a.name for a in view.overlays], ["P1_T2_Upper_Reg.vtk"])

    def test_two_scenes_in_one_folder_are_two_cases(self):
        tree(self.root, ["CB/P1_CB_Reg.nii.gz", "CB/P2_CB_Reg.nii.gz"])
        for patient in ("P1", "P2"):
            write_scene(os.path.join(self.root, "CB", f"{patient}_CB_Reg_scene.json"),
                        [(f"{patient}_CB_Reg.nii.gz", "volume")])
        cases = index.build([("Results", self.root)])
        self.assertEqual([case.key for case in cases],
                         [os.path.join("CB", "P1_CB_Reg"), os.path.join("CB", "P2_CB_Reg")])

    def test_files_no_scene_names_are_still_indexed_the_old_way(self):
        tree(self.root, ["P1/scan.nii.gz", "P1/other_lm_Pred.mrk.json"])
        write_scene(os.path.join(self.root, "P1", "scene.json"), [("scan.nii.gz", "volume")])
        keys = [case.key for case in index.build([("Results", self.root)])]
        self.assertIn("P1", keys)
        self.assertTrue(any(key.endswith("other") for key in keys), keys)

    def test_an_unreadable_scene_falls_back_to_the_name_rules(self):
        tree(self.root, ["P1_scan.nii.gz", "P1_scan_lm_Pred.mrk.json"])
        with open(os.path.join(self.root, "scene.json"), "w") as handle:
            handle.write("{not json")
        cases = index.build([("Results", self.root)])
        self.assertEqual([case.key for case in cases], ["P1"])
        self.assertFalse(cases[0].scene)

    def test_a_scene_cannot_reach_outside_the_folder_opened(self):
        results = os.path.join(self.root, "results")
        tree(self.root, ["results/inside/scan.nii.gz", "secret.nii.gz"])
        write_scene(os.path.join(results, "inside", "scene.json"), [
            ("scan.nii.gz", "volume"), ("../../secret.nii.gz", "volume")])
        case = [c for c in index.build([("Results", results)]) if c.scene][0]
        self.assertEqual([a.name for a in case.artifacts], ["scan.nii.gz"])

    def test_a_scene_may_point_at_a_file_its_result_shares(self):
        """AREG CBCT lists the oriented T1 every region was registered onto,
        kept once under T1_Oriented/ rather than copied into each region."""
        tree(self.root, ["T1_Oriented/P1_T1_Or.nii.gz", "T1_Oriented/P1_T1_lm_Or.mrk.json",
                         "CB/P1_CB_Reg.nii.gz"])
        write_scene(os.path.join(self.root, "CB", "P1_CB_Reg_scene.json"), [
            ("../T1_Oriented/P1_T1_Or.nii.gz", "volume"),
            ("../T1_Oriented/P1_T1_lm_Or.mrk.json", "markups"),
            ("P1_CB_Reg.nii.gz", "volume")])
        case = [c for c in index.build([("Results", self.root)]) if c.scene][0]
        (view,) = case.views()
        self.assertEqual(view.anchor.name, "P1_T1_Or.nii.gz")
        self.assertEqual(sorted(a.name for a in view.overlays),
                         ["P1_CB_Reg.nii.gz", "P1_T1_lm_Or.mrk.json"])

    def test_a_listed_file_that_is_missing_is_skipped(self):
        tree(self.root, ["P1/scan.nii.gz"])
        write_scene(os.path.join(self.root, "P1", "scene.json"), [
            ("scan.nii.gz", "volume"), ("gone.vtk", "model")])
        (case,) = index.build([("Results", self.root)])
        self.assertEqual([a.name for a in case.artifacts], ["scan.nii.gz"])
