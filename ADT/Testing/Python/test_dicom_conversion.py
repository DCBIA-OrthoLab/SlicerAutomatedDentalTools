# A DICOM conversion saves a patient's own volume under that patient's name.
#
# When SimpleITK found no series directly in a patient's folder, the three CBCT
# tools ran dicom2nifti into the folder all patients share and renamed the
# first .nii.gz found there:
#
#   ALI, AREG   that file could be the previous patient's, which then carried
#               this patient's name and was gone under its own.
#   ASO         its `search` takes a `self` the call did not pass, so the same
#               line raised KeyError and stopped the batch.
#
# dicom2nifti also names a file after the series number and description alone,
# so two patients below one folder overwrite each other and leave a single file.
#
# The converter is replaced by one that copies a marker, so that what ends up
# under each name can be read back. The three functions are the real ones.
import contextlib
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", ".."))
if os.path.join(_ROOT, "ADT") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "ADT"))

from ADTLib.io.fs import search                                 # noqa: E402


@contextlib.contextmanager
def _stand_ins(modules):
    """Put `modules` in sys.modules, then put back those names and no other.

    `mock.patch.dict` would also evict numpy and vtk, imported meanwhile, and
    numpy cannot be imported twice in one process.
    """
    absent = object()
    before = {name: sys.modules.get(name, absent) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in before.items():
            if module is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load(relative_path):
    """One converter's module, without what the tools install at first run."""
    package = types.ModuleType("ALI_CBCT_utils")
    package.__path__ = []
    ali_io = types.ModuleType("ALI_CBCT_utils.io")
    ali_io.search = search
    with _stand_ins({"dicom2nifti": types.ModuleType("dicom2nifti"),
                     "itk": types.ModuleType("itk"),
                     "ALI_CBCT_utils": package,
                     "ALI_CBCT_utils.io": ali_io}):
        name = "dicom_conversion_" + relative_path.split("/")[0]
        spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relative_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def copy_markers(folder, output_folder):
    """What convert_directory does: one file per series, named after the series."""
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".marker"):
                shutil.copyfile(os.path.join(root, name),
                                os.path.join(output_folder, name[:-7] + ".nii.gz"))


class ConversionCases:
    """What each of the three converters must do."""

    path = None

    @classmethod
    def setUpClass(cls):
        cls.module = _load(cls.path)

    def setUp(self):
        self.input = tempfile.mkdtemp(prefix="dicom_conversion_")
        self.addCleanup(shutil.rmtree, self.input, True)
        self.output = os.path.join(self.input, "NIFTI")
        os.makedirs(self.output)

    def series(self, patient, content, name="1_cbct"):
        folder = os.path.join(self.input, patient, "series")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, name + ".marker"), "w", encoding="utf-8") as handle:
            handle.write(content)

    def already_converted(self, patient, content):
        # No source folder: os.listdir has no order, and a second pass over
        # this patient could rename a stolen volume back and hide the theft.
        with open(os.path.join(self.output, patient + ".nii.gz"), "w", encoding="utf-8") as handle:
            handle.write(content)

    def convert(self, converter=copy_markers):
        with mock.patch.object(self.module.dicom2nifti, "convert_directory", converter, create=True):
            try:
                self.module.convertdicom2nifti(self.input)
            except Exception as error:
                return error
        return None

    def converted(self):
        found = {}
        for name in os.listdir(self.output):
            with open(os.path.join(self.output, name), encoding="utf-8") as handle:
                found[name] = handle.read()
        return found

    def test_a_nested_series_keeps_its_own_patient(self):
        self.already_converted("0001", "first patient")
        self.series("0002", "second patient")
        self.convert()
        self.assertEqual(self.converted(), {"0001.nii.gz": "first patient",
                                            "0002.nii.gz": "second patient"})

    def test_a_folder_without_dicom_takes_nobody_elses_volume(self):
        self.already_converted("A_patient", "A")
        os.makedirs(os.path.join(self.input, "Z_notes"))
        self.convert()
        self.assertEqual(self.converted(), {"A_patient.nii.gz": "A"})

    def test_a_leftover_in_the_shared_folder_is_never_picked(self):
        with open(os.path.join(self.output, "0_leftover.nii.gz"), "w", encoding="utf-8") as handle:
            handle.write("unknown")
        self.series("P", "P")
        self.convert()
        self.assertEqual(self.converted(), {"0_leftover.nii.gz": "unknown", "P.nii.gz": "P"})

    def test_several_volumes_for_one_patient_are_not_guessed_between(self):
        self.series("P", "one", "1_cbct")
        self.series("P", "two", "2_cbct")
        self.convert()
        self.assertEqual(self.converted(), {})

    def test_a_folder_that_fails_does_not_cost_the_others_theirs(self):
        self.series("good", "good")
        os.makedirs(os.path.join(self.input, "empty"))
        self.assertIsNone(self.convert())
        self.assertEqual(self.converted(), {"good.nii.gz": "good"})

    def test_an_interrupted_copy_is_not_left_to_pass_for_a_conversion(self):
        # An output that exists is skipped on the next run.
        self.series("P", "whole")

        def interrupted(source, destination):
            with open(destination, "w", encoding="utf-8") as handle:
                handle.write("half")
            raise OSError("disk full")

        def converter(folder, output_folder):
            copy_markers(folder, output_folder)
            patch = mock.patch("shutil.copyfile", interrupted)
            patch.start()
            self.addCleanup(patch.stop)

        self.convert(converter)
        self.assertNotIn("half", self.converted().values())

    @unittest.skipUnless(importlib.util.find_spec("pydicom"), "pydicom ships with Slicer")
    def test_two_series_with_the_same_file_name_are_refused(self):
        self.headers("Clinic/P1", slices=4)
        self.headers("Clinic/P2", slices=4)
        self.series("Clinic", "whichever was written last")
        self.convert()
        self.assertEqual(self.converted(), {})

    @unittest.skipUnless(importlib.util.find_spec("pydicom"), "pydicom ships with Slicer")
    def test_a_localizer_beside_the_volume_is_still_accepted(self):
        # dicom2nifti converts no series of three slices or fewer.
        self.headers("P/scout", slices=2)
        self.headers("P/volume", slices=4)
        self.series("P", "volume")
        self.convert()
        self.assertEqual(self.converted(), {"P.nii.gz": "volume"})

    def headers(self, relative_path, slices):
        """DICOM headers without pixels, one folder down so SimpleITK passes them by."""
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

        folder = os.path.join(self.input, relative_path)
        os.makedirs(folder)
        series = generate_uid()
        for index in range(slices):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            path = os.path.join(folder, "%d.dcm" % index)
            header = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
            header.SeriesInstanceUID = series
            header.InstanceNumber = index + 1
            header.ImagePositionPatient = [0, 0, index]
            header.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            header.save_as(path)


class ASOConversionTest(ConversionCases, unittest.TestCase):
    path = "ASO_CBCT/ASO_CBCT_utils/utils.py"

    def test_a_batch_where_nothing_converts_still_fails(self):
        # PRE_ASO would otherwise report success over zero scans.
        os.makedirs(os.path.join(self.input, "empty"))
        self.assertIsInstance(self.convert(), RuntimeError)


class AREGConversionTest(ConversionCases, unittest.TestCase):
    path = "AREG_CBCT/AREG_CBCT_utils/utils.py"


class ALIConversionTest(ConversionCases, unittest.TestCase):
    path = "ALI_CBCT/ALI_CBCT_utils/preprocess.py"


if __name__ == "__main__":
    unittest.main()
