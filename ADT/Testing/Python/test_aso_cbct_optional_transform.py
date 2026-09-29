# The Semi-Automated ASO CBCT mode does not need a prior transform.
#
# `GetPatients` only puts a "tfm" key in when a .tfm sits beside the scan, and
# SEMI_ASO_CBCT read `data["tfm"]` outright. Every patient without one raised
# `KeyError: 'tfm'` before any registration, and the run ended on "No patient
# could be registered". That is every patient of the test set published for this
# very mode: it ships a scan and its landmarks, and the only .tfm in the release
# are OUTPUTS of the Fully-Automated mode.
#
# Two things are pinned here. What GetPatients does and does not put in the
# dict, and the assumption the fix rests on: the prior transform is composed at
# the end of the chain, so its neutral value is the identity.
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

import SimpleITK as sitk                                       # noqa: E402

from ASO_CBCT_utils.utils import GetPatients                    # noqa: E402


class PatientDictTest(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="aso_cbct_patients_")

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def touch(self, name):
        with open(os.path.join(self.folder, name), "w", encoding="utf-8") as handle:
            handle.write("{}")

    def test_a_scan_and_its_landmarks_make_a_patient_without_tfm(self):
        """The shape the published Semi-Automated set actually has."""
        self.touch("IC_0005.nii.gz")
        self.touch("IC_0005_lm_MERGED.mrk.json")
        patients = GetPatients(self.folder)
        self.assertEqual(sorted(patients), ["IC_0005"])
        self.assertIn("scan", patients["IC_0005"])
        self.assertIn("json", patients["IC_0005"])
        self.assertNotIn("tfm", patients["IC_0005"])

    def test_a_tfm_beside_them_is_picked_up(self):
        """And when one IS there, it must still be used."""
        self.touch("Pat_0002_Or.nii.gz")
        self.touch("Pat_0002_lm_Or.mrk.json")
        self.touch("Pat_0002_Or_transform.tfm")
        patients = GetPatients(self.folder)
        entry = patients[sorted(patients)[0]]
        self.assertIn("tfm", entry)
        self.assertTrue(entry["tfm"].endswith(".tfm"))

    def test_reading_the_key_outright_is_what_raised(self):
        """The failure this replaces, kept so the reason stays legible."""
        self.touch("IC_0005.nii.gz")
        self.touch("IC_0005_lm_MERGED.mrk.json")
        entry = GetPatients(self.folder)["IC_0005"]
        with self.assertRaises(KeyError):
            entry["tfm"]
        self.assertIsNone(entry.get("tfm"))


class IdentityIsNeutralTest(unittest.TestCase):
    """ICP composes the prior transform last; absent, it must change nothing."""

    def test_composing_the_identity_leaves_a_point_where_it_was(self):
        chain = sitk.CompositeTransform(3)
        rotation = sitk.Euler3DTransform()
        rotation.SetRotation(0.1, 0.2, 0.3)
        chain.AddTransform(rotation)

        with_identity = sitk.CompositeTransform(chain)
        with_identity.AddTransform(sitk.Euler3DTransform())

        point = (12.5, -3.25, 7.0)
        self.assertEqual(
            tuple(round(c, 9) for c in chain.TransformPoint(point)),
            tuple(round(c, 9) for c in with_identity.TransformPoint(point)))

    def test_a_fresh_euler3d_is_the_identity(self):
        point = (1.0, 2.0, 3.0)
        moved = sitk.Euler3DTransform().TransformPoint(point)
        self.assertEqual(tuple(round(c, 9) for c in moved), point)


if __name__ == "__main__":
    unittest.main()
