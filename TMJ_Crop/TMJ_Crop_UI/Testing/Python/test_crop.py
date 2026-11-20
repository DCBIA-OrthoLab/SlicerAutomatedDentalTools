# The crop on a synthetic head, and the batch around it.
#
# No scan, no weights, no network: a head with two mirror-symmetric condyles is
# built in the oriented frame, then shifted by a known translation into the
# scan's own frame, so the tests can check that the crop recovers that
# translation from the landmarks alone. The geometry cases are SADT-VISOR's
# (tools/TMJ_Crop/tests/test_crop.py, commit 11f3cb8) rewritten for unittest.
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import nibabel as nib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
for _path in (os.path.join(_ROOT, "ADT"), os.path.join(_ROOT, "ASO_CBCT"),
              os.path.join(_ROOT, "TMJ_Crop", "TMJ_Crop_CLI")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# `ASO_CBCT_utils.utils` imports dicom2nifti, which only the DICOM path uses.
sys.modules.setdefault("dicom2nifti", types.ModuleType("dicom2nifti"))

from TMJ_Crop_utils import batch, catalogs, crop  # noqa: E402

REFERENCE_LPS = {
    "Ba": (0.0, 40.0, -25.0), "S": (0.0, -10.0, 10.0), "N": (0.0, -85.0, 20.0),
    "RPo": (-50.0, 0.0, 0.0), "LPo": (50.0, 0.0, 0.0),
    "ROr": (-30.0, -80.0, 20.0), "LOr": (30.0, -80.0, 20.0),
}
SHIFT = np.array([3.0, -2.0, 1.0])
CONDYLES = {"R": np.array([50.0, 8.0, -5.0]), "L": np.array([-50.0, 8.0, -5.0])}
USED = list(REFERENCE_LPS)


def reference_ras():
    return {k: np.array(v) * np.array([-1.0, -1.0, 1.0]) for k, v in REFERENCE_LPS.items()}


def write_markups(path, points):
    """`points` in RAS, written as Slicer does, in LPS."""
    controls = [{"label": label, "position": list(np.array(p) * np.array([-1.0, -1.0, 1.0]))}
                for label, p in points.items()]
    with open(path, "w") as handle:
        json.dump({"markups": [{"coordinateSystem": "LPS", "controlPoints": controls}]}, handle)


def write_scene(folder, name, mask_folder=None):
    """`<name>_Or.nii.gz`, its landmarks and the mandible mask."""
    os.makedirs(folder, exist_ok=True)
    size, origin = 160, -80.0
    axis = origin + np.arange(size, dtype=np.float32)
    world = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
    affine = np.eye(4)
    affine[:3, 3] = origin
    oriented = world + SHIFT
    volume = np.zeros(world.shape[:3], dtype=np.float32)
    mask = np.zeros(world.shape[:3], dtype=np.uint8)
    for centre in CONDYLES.values():
        distance = np.linalg.norm(oriented - centre, axis=-1)
        volume += 1000.0 * np.exp(-(distance / 6.0) ** 2)
        mask |= (distance < 6.0).astype(np.uint8)
    # The body joining the two condyles: the crop keeps the largest connected piece.
    bar = (np.abs(oriented[..., 1] - 8.0) < 3.0) & (np.abs(oriented[..., 2] + 14.0) < 4.0) \
        & (np.abs(oriented[..., 0]) < 50.0)
    mask |= bar.astype(np.uint8)
    scan = os.path.join(folder, f"{name}_Or.nii.gz")
    nib.save(nib.Nifti1Image(volume, affine), scan)
    mask_path = os.path.join(mask_folder or folder, f"{name}_Or_Pred_MAND.nii.gz")
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    markups = os.path.join(folder, f"{name}_lm_Or.mrk.json")
    write_markups(markups, {k: v - SHIFT for k, v in reference_ras().items()})
    return scan, markups, mask_path


class CropGeometryTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def crop(self, settings=None):
        scan, markups, mask = write_scene(os.path.join(self.tmp, "in"), "a")
        out = os.path.join(self.tmp, "out")
        os.makedirs(out)
        result = crop.crop_scan(scan, markups, mask, reference_ras(), USED, {},
                                settings or crop.CropSettings(), "a", out)
        return result, out

    def test_the_translation_is_recovered_from_the_landmarks_alone(self):
        result, _ = self.crop()
        self.assertTrue(np.allclose(result.report["translation"], SHIFT, atol=1e-3))
        self.assertLess(max(result.report["fit_residual_mm"].values()), 0.01)

    def test_both_joints_are_found_and_written_at_the_documented_size(self):
        result, out = self.crop()
        for side in ("R", "L"):
            self.assertEqual(result.report["sides"][side]["status"], "ok")
            self.assertEqual(nib.load(os.path.join(out, f"a_{side}.nii.gz")).shape, (96, 112, 80))
        self.assertEqual(result.report["box_mm"], [28.8, 33.6, 24.0])

    def test_the_left_crop_is_the_mirror_image_of_the_right(self):
        _, out = self.crop()
        right = np.asanyarray(nib.load(os.path.join(out, "a_R.nii.gz")).dataobj)
        left = np.asanyarray(nib.load(os.path.join(out, "a_L.nii.gz")).dataobj)
        self.assertLess(np.abs(right.astype(float) - left.astype(float)).max(), 20)
        self.assertGreater(right.max(), 500)

    def test_a_crop_is_written_in_the_frame_of_the_scan_it_was_cut_from(self):
        from scipy import ndimage

        _, out = self.crop()
        scan = nib.load(os.path.join(self.tmp, "in", "a_Or.nii.gz"))
        cropped = nib.load(os.path.join(out, "a_R.nii.gz"))
        values = np.asanyarray(cropped.dataobj).astype(np.float32)
        voxels = np.indices(values.shape).reshape(3, -1).T
        world = voxels @ cropped.affine[:3, :3].T + cropped.affine[:3, 3]
        inverse = np.linalg.inv(scan.affine)
        sampled = ndimage.map_coordinates(
            np.asanyarray(scan.dataobj).astype(np.float32),
            (world @ inverse[:3, :3].T + inverse[:3, 3]).T, order=1, mode="nearest")
        self.assertGreater(np.corrcoef(sampled, values.ravel())[0, 1], 0.99)

    def test_a_fixed_scale_changes_the_voxel_size_and_not_the_voxel_count(self):
        result, out = self.crop(crop.CropSettings(scale=1.5))
        self.assertAlmostEqual(result.report["spacing_mm"], 0.45)
        self.assertEqual(nib.load(os.path.join(out, "a_R.nii.gz")).shape, (96, 112, 80))

    def test_a_scale_that_means_nothing_is_refused(self):
        with self.assertRaises(ValueError):
            crop.CropSettings(scale=-1.0)
        with self.assertRaises(ValueError):
            crop.CropSettings(scale_by="area")

    def test_no_shared_landmark_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            crop.translation_to_reference({"X": np.zeros(3)}, {"Ba": np.zeros(3)}, ["Ba"])


class BatchTest(unittest.TestCase):
    """The folders the legacy chain leaves, on synthetic scans."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.oriented = os.path.join(self.tmp, "oriented")
        self.masks = os.path.join(self.tmp, "masks")
        self.centred = os.path.join(self.tmp, "centred")
        self.reference = os.path.join(self.tmp, "reference")
        self.output = os.path.join(self.tmp, "out")
        os.makedirs(self.centred)
        os.makedirs(self.reference)
        # The reference folder as the extension publishes it: a scan and its landmarks.
        write_markups(os.path.join(self.reference, "gold.mrk.json"), reference_ras())
        open(os.path.join(self.reference, "gold.nii.gz"), "wb").close()

    def scene(self, name):
        write_scene(self.oriented, name, self.masks)
        # What ALI wrote before SEMI_ASO_CBCT: the landmarks in the scan's own frame.
        write_markups(os.path.join(self.centred, f"{name}_lm_MERGED.mrk.json"),
                      {k: v - SHIFT for k, v in reference_ras().items()})

    def run_batch(self, **options):
        messages = []
        batch.crop_batch(self.oriented, self.centred, self.masks, self.reference, self.output,
                         tell=messages.append, **options)
        return messages

    def case(self, name):
        with open(os.path.join(self.output, f"{name}.json")) as handle:
            return json.load(handle)

    def report(self):
        with open(os.path.join(self.output, batch.REPORT_NAME)) as handle:
            return json.load(handle)

    def test_a_scan_is_cropped_and_reported(self):
        self.scene("a")
        self.run_batch()
        report = self.report()
        self.assertEqual(report["cases"]["a"]["status"], "ok")
        self.assertEqual(report["summary"], {"cases": 1, "cropped": 1, "partial": 0, "failed": 0})
        self.assertEqual(sorted(os.listdir(self.output)),
                         ["a.json", "a_L.nii.gz", "a_R.nii.gz", "a_qc.png", batch.REPORT_NAME])
        case = self.case("a")
        self.assertEqual(case["landmarks_used"], sorted(USED))
        self.assertTrue(np.allclose(case["translation"], SHIFT, atol=1e-3))

    def test_one_side_asked_means_one_side_written(self):
        self.scene("a")
        self.run_batch(sides="Left")
        self.assertTrue(os.path.isfile(os.path.join(self.output, "a_L.nii.gz")))
        self.assertFalse(os.path.exists(os.path.join(self.output, "a_R.nii.gz")))

    def test_extra_outputs_are_written_only_when_asked_for(self):
        self.scene("a")
        self.run_batch(outputs=[catalogs.OUTPUT_ORIENTED_SCAN, catalogs.OUTPUT_MANDIBLE])
        self.assertTrue(os.path.isfile(os.path.join(self.output, "a_oriented.nii.gz")))
        self.assertTrue(os.path.isfile(os.path.join(self.output, "a_mandible.nii.gz")))
        self.assertEqual(self.report()["outputs"], sorted(catalogs.OUTPUT_HELP))

    def test_a_scan_without_a_mask_is_reported_and_the_batch_goes_on(self):
        self.scene("a")
        self.scene("b")
        os.remove(os.path.join(self.masks, "b_Or_Pred_MAND.nii.gz"))
        messages = self.run_batch()
        report = self.report()
        self.assertEqual(report["cases"]["a"]["status"], "ok")
        self.assertEqual(report["cases"]["b"]["status"], "no mandible mask")
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(messages, ["scan 2 of 2: no mandible mask"])

    def test_a_scan_that_was_not_oriented_is_found_from_the_input_folder(self):
        self.scene("a")
        inputs = os.path.join(self.tmp, "in")
        os.makedirs(inputs)
        for name in ("a.nii.gz", "lost.nii.gz"):
            open(os.path.join(inputs, name), "wb").close()
        self.run_batch(input_dir=inputs)
        report = self.report()
        self.assertTrue(report["cases"]["lost"]["status"].startswith("orientation failed"))
        self.assertEqual(report["summary"]["cases"], 2)
        self.assertNotIn(inputs, json.dumps(report))

    def test_a_batch_that_cropped_nothing_raises_and_writes_no_report(self):
        self.scene("a")
        os.remove(os.path.join(self.masks, "a_Or_Pred_MAND.nii.gz"))
        with self.assertRaises(batch.ToolInputError):
            self.run_batch()
        self.assertFalse(os.path.exists(os.path.join(self.output, batch.REPORT_NAME)))

    def test_an_outlier_landmark_is_dropped_the_way_the_registration_dropped_it(self):
        self.scene("a")
        path = os.path.join(self.centred, "a_lm_MERGED.mrk.json")
        points = {k: v - SHIFT for k, v in reference_ras().items()}
        points["Ba"] = points["Ba"] + np.array([0.0, 60.0, 60.0])
        write_markups(path, points)
        self.run_batch()
        case = self.case("a")
        self.assertEqual(case["landmarks_dropped"], {"Ba": "rejected as an outlier"})
        self.assertNotIn("Ba", case["landmarks_used"])

    def test_an_unknown_option_is_refused(self):
        self.scene("a")
        with self.assertRaises(batch.ToolInputError):
            self.run_batch(sides="Both")
        with self.assertRaises(batch.ToolInputError):
            self.run_batch(outputs=["Everything"])

    def test_a_reason_never_carries_a_path(self):
        self.assertEqual(batch.safe_reason("cannot read /data/p/scan one.nii.gz now"),
                         "cannot read <file>")


if __name__ == "__main__":
    unittest.main()
