# A review pause on a landmark step must not offer ALI's report as landmarks.
#
# `ReviewSession.build` collects every file whose extension matches the kind of
# pause, and for a LANDMARKS pause that is `.json`. ALI_CBCT writes
# `<patient>_lm_NotFound.json` in that same folder, so the report became one
# more item to review: `loadMarkups` has nothing to make of it, `_load` caught
# the failure and returned None, and the pause opened on an empty view with an
# error line in the log and nothing telling the user what they were looking at.
#
# Not a crash -- `_load` swallows it -- which is exactly why it needed pinning:
# nothing would have failed loudly enough to notice.
import json
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
for _path in (os.path.join(_ROOT, "ADT"), os.path.join(_ROOT, "AREG")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from AREG_Method.Review import LANDMARKS, ReviewSession              # noqa: E402


class LandmarkPauseQueueTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="areg_review_report_")
        self.session = ReviewSession()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def write(self, name, payload):
        path = os.path.join(self.folder, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def landmarks(self, name, *labels):
        return self.write(name, {"markups": [{
            "type": "Fiducial",
            "controlPoints": [{"label": lm, "position": [0.0, 0.0, 0.0]}
                              for lm in labels],
        }]})

    def report(self, name, patient, *missing):
        """The shape ALI_CBCT really writes."""
        return self.write(name, {
            "patient": patient,
            "not_found": [{"landmark": lm, "reason": "left the readable zone"}
                          for lm in missing],
        })

    def build(self):
        return self.session.build(
            {"ReviewFolder": self.folder, "ReviewKind": LANDMARKS},
            expected=["P01_T1"])

    def files_queued(self):
        return sorted(os.path.basename(f)
                      for item in self.build() for f in item["files"])

    def test_the_landmarks_are_queued(self):
        self.landmarks("P01_T1_lm_Pred.mrk.json", "S", "Ba")
        self.assertEqual(self.files_queued(), ["P01_T1_lm_Pred.mrk.json"])

    def test_the_report_is_not_queued_beside_them(self):
        self.landmarks("P01_T1_lm_Pred.mrk.json", "S", "Ba")
        self.report("P01_T1_lm_NotFound.json", "P01_T1.nii.gz", "N")
        self.assertEqual(self.files_queued(), ["P01_T1_lm_Pred.mrk.json"])

    def test_a_patient_whose_only_json_is_a_report_is_not_reviewed(self):
        """Nothing to show is better than an empty view the user cannot read."""
        self.report("P01_T1_lm_NotFound.json", "P01_T1.nii.gz", "N", "S")
        self.assertEqual(self.build(), [])

    def test_the_report_is_left_on_disk(self):
        """Skipped, not consumed: it is the only trace of what ALI missed."""
        self.landmarks("P01_T1_lm_Pred.mrk.json", "S")
        path = self.report("P01_T1_lm_NotFound.json", "P01_T1.nii.gz", "N")
        self.build()
        self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
