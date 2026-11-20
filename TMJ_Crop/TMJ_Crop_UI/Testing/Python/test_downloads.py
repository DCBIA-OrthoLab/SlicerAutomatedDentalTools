# The panel fills the model fields from what a Download button already left on disk,
# so a machine that ran ASO, ALI or AMASSS needs no click.
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_UI = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _UI)
sys.path.insert(0, os.path.normpath(os.path.join(_UI, "..", "TMJ_Crop_CLI")))

import TMJ_Crop  # noqa: E402
from ADTLib.testdata import MARKER  # noqa: E402
from TMJ_Crop_utils import catalogs  # noqa: E402


def touch(*parts):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


class FindDownloadsTest(unittest.TestCase):

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = self._temp.name

    def tearDown(self):
        self._temp.cleanup()

    def reference(self):
        return os.path.join(self.root, "ASO", "ASO_CBCT", "Reference", TMJ_Crop.REFERENCE_NAME)

    def make_models(self, landmarks=None):
        for name in landmarks or catalogs.ORIENTATION_LANDMARKS:
            touch(self.root, "ALI", name, "1", "model.pth")

    def test_an_empty_downloads_folder_finds_nothing(self):
        self.assertEqual(TMJ_Crop.find_downloads(self.root), {})

    def test_a_complete_reference_is_found(self):
        touch(self.reference(), MARKER)
        self.assertEqual(TMJ_Crop.find_downloads(self.root), {"reference": self.reference()})

    def test_an_interrupted_reference_is_not_found(self):
        touch(self.reference(), "somefile.json")
        self.assertEqual(TMJ_Crop.find_downloads(self.root), {})

    def test_the_landmark_models_need_every_landmark(self):
        self.make_models(catalogs.ORIENTATION_LANDMARKS[:-1])
        self.assertNotIn("ali", TMJ_Crop.find_downloads(self.root))
        self.make_models()
        self.assertEqual(TMJ_Crop.find_downloads(self.root)["ali"], os.path.join(self.root, "ALI"))

    def test_the_mandible_models_are_found_under_a_nested_bundle(self):
        bundle = os.path.join(self.root, "AMASSS", "Models", "Segmentation")
        touch(bundle, "AMASSS_Models", "MAND", "model.ckpt")
        self.assertEqual(TMJ_Crop.find_downloads(self.root)["amasss"],
                         os.path.join(bundle, "AMASSS_Models"))

    def test_a_bundle_without_mand_is_not_found(self):
        touch(self.root, "AMASSS", "Models", "Segmentation", "MAX", "model.ckpt")
        self.assertNotIn("amasss", TMJ_Crop.find_downloads(self.root))

    def test_amasss_models_folder_returns_the_bundle_when_mand_is_inside_it(self):
        bundle = os.path.join(self.root, "b")
        touch(bundle, "MAND", "x")
        self.assertEqual(TMJ_Crop.amasss_models_folder(bundle), bundle)


if __name__ == "__main__":
    unittest.main()
