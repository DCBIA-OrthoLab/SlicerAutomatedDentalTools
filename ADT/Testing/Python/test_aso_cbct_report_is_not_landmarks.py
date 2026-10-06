# ALI's not-found report shares the folder with the landmarks, and must not be
# mistaken for them.
#
# When ALI_CBCT cannot place a point it writes `<patient>_lm_NotFound.json`
# beside its predictions, listing which and why. That report carries `patient`
# and `not_found`; it has no `markups`. Everything on the ASO CBCT side globbed
# `*.json` and reached straight for `data["markups"]`:
#
#   - `MergeJson` grouped it with the real landmarks -- the `_lm` infix gives
#     both files the same patient key -- and died on `KeyError: 'markups'`;
#   - `GetPatients` then handed it to ICP as the patient's landmarks, which
#     failed on the same key;
#   - the run ended on "No patient could be registered" while ALI had in fact
#     placed six landmarks out of seven.
#
# So the fix is on the readers, not on the report's name: a folder may hold any
# number of json that are not landmarks, and no naming rule catches them all.
# What is pinned here is that the report is skipped, and survives.
import json
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
for _path in (os.path.join(_ROOT, "ADT"), os.path.join(_ROOT, "ASO_CBCT")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from ADTLib.io.landmarks import IsMarkupsFile                   # noqa: E402
from ASO_CBCT_utils.utils import GetPatients, MergeJson         # noqa: E402


def _markups(*labels):
    return {
        "markups": [{
            "type": "Fiducial",
            "controlPoints": [
                {"label": lm, "position": [0.0, 0.0, 0.0]} for lm in labels
            ],
        }]
    }


def _report(patient, *missing):
    """What ALI_CBCT actually writes, shape for shape."""
    return {
        "patient": patient,
        "not_found": [{"landmark": lm, "reason": "left the readable zone"}
                      for lm in missing],
    }


class IsMarkupsFileTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="aso_cbct_markups_")

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def write(self, name, payload):
        path = os.path.join(self.folder, name)
        with open(path, "w", encoding="utf-8") as handle:
            if isinstance(payload, str):
                handle.write(payload)
            else:
                json.dump(payload, handle)
        return path

    def test_a_landmark_file_is_one(self):
        self.assertTrue(IsMarkupsFile(
            self.write("P01_lm_Pred.mrk.json", _markups("N", "S"))))

    def test_the_not_found_report_is_not(self):
        self.assertFalse(IsMarkupsFile(
            self.write("P01_T1_lm_NotFound.json", _report("P01_T1.nii.gz", "N"))))

    def test_a_json_that_is_not_an_object_is_not(self):
        self.assertFalse(IsMarkupsFile(self.write("list.json", "[1, 2, 3]")))

    def test_markups_must_be_a_list(self):
        """`data["markups"][0]` is what the callers do next."""
        self.assertFalse(IsMarkupsFile(self.write("odd.json", {"markups": 3})))

    def test_malformed_json_answers_no_instead_of_raising(self):
        self.assertFalse(IsMarkupsFile(self.write("broken.json", "{not json")))

    def test_a_missing_file_answers_no(self):
        self.assertFalse(IsMarkupsFile(os.path.join(self.folder, "absent.json")))

    def test_something_that_is_not_json_is_not(self):
        self.assertFalse(IsMarkupsFile(self.write("scan.nii.gz", "")))


class ReportBesideLandmarksTest(unittest.TestCase):
    """The folder ALI leaves behind, read by the step that follows it."""

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="aso_cbct_report_")
        self.landmarks = os.path.join(self.folder, "P01_T1_lm_Pred.mrk.json")
        with open(self.landmarks, "w", encoding="utf-8") as handle:
            json.dump(_markups("S", "Ba", "RPo"), handle)
        # Sorts BEFORE the landmarks, which is how it became `files[0]` and
        # made the merge read ITS "markups" key first.
        self.report = os.path.join(self.folder, "P01_T1_lm_NotFound.json")
        with open(self.report, "w", encoding="utf-8") as handle:
            json.dump(_report("P01_T1.nii.gz", "N"), handle)
        with open(os.path.join(self.folder, "P01_T1.nii.gz"), "w") as handle:
            handle.write("")

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_the_report_sorts_first(self):
        """Without that, the case below would pass for the wrong reason."""
        self.assertLess(os.path.basename(self.report),
                        os.path.basename(self.landmarks))

    def test_merging_does_not_raise_on_the_report(self):
        MergeJson(self.folder)  # raised KeyError: 'markups'

    def test_merging_produces_the_merged_landmarks(self):
        MergeJson(self.folder)
        merged = [f for f in os.listdir(self.folder) if "MERGED" in f]
        self.assertEqual(len(merged), 1, os.listdir(self.folder))
        with open(os.path.join(self.folder, merged[0]), encoding="utf-8") as handle:
            points = json.load(handle)["markups"][0]["controlPoints"]
        self.assertEqual([p["label"] for p in points], ["S", "Ba", "RPo"])

    def test_the_report_survives_the_merge(self):
        """Its deletion pass removes every json it merged; not this one."""
        MergeJson(self.folder)
        self.assertTrue(os.path.isfile(self.report))
        with open(self.report, encoding="utf-8") as handle:
            self.assertEqual(
                [e["landmark"] for e in json.load(handle)["not_found"]], ["N"])

    def test_the_patient_landmarks_are_the_markups_not_the_report(self):
        entry = GetPatients(self.folder)["P01_T1"]
        self.assertTrue(IsMarkupsFile(entry["json"]))
        self.assertNotEqual(os.path.basename(entry["json"]),
                            os.path.basename(self.report))

    def test_a_patient_whose_only_json_is_a_report_has_no_landmarks(self):
        """Better no landmarks than the wrong file: ICP can say so."""
        alone = tempfile.mkdtemp(prefix="aso_cbct_report_only_")
        try:
            with open(os.path.join(alone, "P02_T1_lm_NotFound.json"), "w") as handle:
                json.dump(_report("P02_T1.nii.gz", "N", "S"), handle)
            with open(os.path.join(alone, "P02_T1.nii.gz"), "w") as handle:
                handle.write("")
            entry = GetPatients(alone)["P02_T1"]
            self.assertIn("scan", entry)
            self.assertNotIn("json", entry)
        finally:
            shutil.rmtree(alone, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
