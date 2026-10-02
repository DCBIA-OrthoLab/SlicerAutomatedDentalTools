# One landmark whose name has no group must not cost the patient the others.
#
# `SavePredictedLandmarks` looked its group up with `LABEL_GROUPS[landmark]`.
# The published models carry folders named `UL3OI` and `UL3RI`; the table in
# constants.py knows `UL3OIP` and `UL3RIP`. So an agent is built for a name the
# table does not have, it finds its landmark, and the plain lookup raised
# `KeyError: 'UL3OI'` -- before a single file was written. Thirty-odd landmarks
# that had been found were lost with it, and the run still read `ok`, because
# the one line saying so was `Failed to save predictions ...` in the CLI log.
#
# The name mismatch itself is a separate question (is `UL3OI` the same point as
# `UL3OIP`?). This pins the behaviour either way: write what can be written,
# and hand the rest back to be reported.
import os
import shutil
import sys
import tempfile
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
for _path in (os.path.join(_ROOT, "ADT"), os.path.join(_ROOT, "ALI_CBCT")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# See test_ali_cbct_missing: dicom2nifti does not import outside Slicer.
sys.modules.setdefault("dicom2nifti", types.ModuleType("dicom2nifti"))

import numpy as np                                            # noqa: E402

from ALI_CBCT_utils.constants import LABEL_GROUPS              # noqa: E402
from ALI_CBCT_utils.environment import Environment             # noqa: E402


class UnknownGroupTest(unittest.TestCase):

    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="ali_cbct_group_")
        # Built without __init__ on purpose: the constructor reads a real
        # volume off disk, and the only thing under test here is what the
        # save does with the names it is handed.
        self.env = Environment.__new__(Environment)
        self.env.patient_id = "P1.nii.gz"
        self.env.data = {"0-3": {"path": os.path.join(self.out, "P1.nii.gz"),
                                 "origin": np.array([0.0, 0.0, 0.0]),
                                 "spacing": np.array([1.0, 1.0, 1.0])}}

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def written(self):
        return sorted(name for name in os.listdir(self.out)
                      if name.endswith(".mrk.json"))

    def test_a_known_landmark_is_written(self):
        self.assertIn("N", LABEL_GROUPS)
        self.env.predicted_landmarks = {"N": np.array([10.0, 11.0, 12.0])}
        self.assertEqual(self.env.SavePredictedLandmarks("0-3", self.out), [])
        self.assertEqual(self.written(), ["P1_lm_Pred_CB.mrk.json"])

    def test_an_unknown_name_does_not_take_the_others_with_it(self):
        """The case measured on 2026-09-29: UL3OI alongside real landmarks."""
        self.assertNotIn("UL3OI", LABEL_GROUPS)
        self.env.predicted_landmarks = {
            "N": np.array([10.0, 11.0, 12.0]),
            "UL3OI": np.array([20.0, 21.0, 22.0]),
            "Ba": np.array([30.0, 31.0, 32.0]),
        }
        unplaceable = self.env.SavePredictedLandmarks("0-3", self.out)
        self.assertEqual(unplaceable, ["UL3OI"])
        # N and Ba are both in the CB group, so one file, and it exists.
        self.assertEqual(self.written(), ["P1_lm_Pred_CB.mrk.json"])

    def test_the_unknown_one_is_not_in_the_output(self):
        """Silently inventing a group for it would be worse than leaving it out."""
        self.env.predicted_landmarks = {"N": np.array([1.0, 2.0, 3.0]),
                                        "UL3OI": np.array([4.0, 5.0, 6.0])}
        self.env.SavePredictedLandmarks("0-3", self.out)
        with open(os.path.join(self.out, "P1_lm_Pred_CB.mrk.json"),
                  encoding="utf-8") as handle:
            body = handle.read()
        self.assertIn("N", body)
        self.assertNotIn("UL3OI", body)

    def test_only_unknown_names_writes_nothing_and_says_so(self):
        self.env.predicted_landmarks = {"UL3OI": np.array([1.0, 2.0, 3.0]),
                                        "UL3RI": np.array([4.0, 5.0, 6.0])}
        self.assertEqual(self.env.SavePredictedLandmarks("0-3", self.out),
                         ["UL3OI", "UL3RI"])
        self.assertEqual(self.written(), [])


if __name__ == "__main__":
    unittest.main()
