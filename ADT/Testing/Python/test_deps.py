# What `check_lib_installed` answers, and on which spelling of a constraint.
#
# The nine copies this replaces disagreed: two ignored the constraint entirely,
# one compared bare strings, one parsed a single operator by hand, one used
# packaging. The cases below pin the behaviour that was chosen -- constraints
# are applied -- and the two legacy spellings that still have to be understood,
# so that no call site has to be rewritten in the same change.
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ADTLib, which the packages now import: a test suite is an entry point
# like any other, nothing has put it on sys.path before it runs.
_ADT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ADT")
if os.path.isdir(_ADT):
    sys.path.insert(0, _ADT)

from packaging.requirements import Requirement  # noqa: E402

import importlib.metadata  # noqa: E402

from ADTLib.env.deps import (  # noqa: E402
    TORCH_FAMILY, check_lib_installed, normalise_spec, requirement,
    torch_cuda_builds_agree, torch_cuda_conflict, torch_cuda_labels)


class NormaliseSpecTest(unittest.TestCase):

    def test_an_operator_is_kept_as_is(self):
        self.assertEqual(normalise_spec("pydicom", "==3.0.2"), ("pydicom", "==3.0.2"))
        self.assertEqual(normalise_spec("itk", ">=5.4.0"), ("itk", ">=5.4.0"))

    def test_a_bare_version_means_equality(self):
        """ALI wrote ('dicom2nifti', '2.6.2') and meant an exact version."""
        self.assertEqual(normalise_spec("dicom2nifti", "2.6.2"), ("dicom2nifti", "==2.6.2"))

    def test_a_constraint_carried_in_the_name_is_split_off(self):
        """MedX wrote the whole thing as the name: 'numpy<2.0.0'."""
        self.assertEqual(normalise_spec("numpy<2.0.0", None), ("numpy", "<2.0.0"))

    def test_no_constraint_stays_no_constraint(self):
        self.assertEqual(normalise_spec("einops", None), ("einops", None))

    def test_a_compound_constraint_survives(self):
        self.assertEqual(normalise_spec("torch", ">=2.8,<2.13"), ("torch", ">=2.8,<2.13"))


class CheckLibInstalledTest(unittest.TestCase):
    """Checked against a library that is certainly present: packaging itself."""

    def test_an_installed_library_with_no_constraint(self):
        self.assertTrue(check_lib_installed("packaging"))

    def test_a_library_that_is_not_installed(self):
        self.assertFalse(check_lib_installed("a-package-that-does-not-exist-anywhere"))

    def test_a_constraint_that_holds(self):
        self.assertTrue(check_lib_installed("packaging", ">=1.0"))

    def test_a_constraint_that_does_not_hold_is_refused(self):
        """This is the whole point: two of the nine copies answered True here."""
        self.assertFalse(check_lib_installed("packaging", ">=99999"))

    def test_an_exact_version_that_does_not_hold_is_refused(self):
        self.assertFalse(check_lib_installed("packaging", "==0.0.1"))

    def test_a_bare_version_that_does_not_hold_is_refused(self):
        self.assertFalse(check_lib_installed("packaging", "0.0.1"))

    def test_a_missing_library_is_not_an_exception(self):
        """install_function branches on the return value, never on a raise."""
        try:
            result = check_lib_installed("nnunet_version", "==2.8.0")
        except Exception as error:
            self.fail("raised instead of answering: %r" % (error,))
        self.assertFalse(result)


class RequirementTest(unittest.TestCase):
    """What gets handed to pip, for every spelling the call sites use.

    ALI and ASO glued `==` in unconditionally. That produced
    `dicom2nifti==>=2.6.2` as soon as one list entry carried an operator, and
    pip refuses it: `Invalid requirement`. The failure only shows on a machine
    that lacks the library -- a fresh install -- so it is exactly the case the
    developer machine never exercises.
    """

    # What the three modules actually pass, read off the code:
    #   ALI/ALI.py list_libs_cbct / list_libs_ios
    #   ASO/ASO.py libs
    #   FlexReg/FlexReg.py list_libs
    CALL_SITES = [
        ("itk", None, "itk"),
        ("dicom2nifti", ">=2.6.2", "dicom2nifti>=2.6.2"),
        ("pydicom", "3.0.2", "pydicom==3.0.2"),
        ("torch", "2.2.0", "torch==2.2.0"),
        ("pytorch_lightning", None, "pytorch_lightning"),
        ("monai", "1.3.2", "monai==1.3.2"),
        ("monai", "==1.3.2", "monai==1.3.2"),
        ("numpy", "<2.0.0", "numpy<2.0.0"),
        ("numpy<2.0.0", None, "numpy<2.0.0"),
    ]

    def test_every_call_site_spelling(self):
        for lib, version, expected in self.CALL_SITES:
            self.assertEqual(requirement(lib, version), expected,
                             "%r + %r" % (lib, version))

    def test_pip_accepts_every_one_of_them(self):
        """The regression itself: `dicom2nifti==>=2.6.2` does not parse."""
        for lib, version, _ in self.CALL_SITES:
            text = requirement(lib, version)
            try:
                Requirement(text)
            except Exception as error:
                self.fail("pip would refuse %r: %s" % (text, error))

    def test_the_broken_form_is_indeed_broken(self):
        """Guards the test above from passing for the wrong reason."""
        with self.assertRaises(Exception):
            Requirement("dicom2nifti==>=2.6.2")

def _lookup(**versions):
    """A stand-in for importlib.metadata.version, so no wheel has to exist."""
    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]
    return version


ALL_118 = _lookup(torch="2.2.0+cu118", torchvision="0.17.0+cu118",
                  torchaudio="2.2.0+cu118")
# torch and torchvision agree, torchaudio does not: the case the AMASSS copy
# declared "in agreement", because it stopped at the first pair.
LAST_ODD = _lookup(torch="2.2.0+cu118", torchvision="0.17.0+cu118",
                   torchaudio="2.2.0+cu121")
FIRST_ODD = _lookup(torch="2.2.0+cu121", torchvision="0.17.0+cu118",
                    torchaudio="2.2.0+cu118")
# What an ordinary PyPI install lays down: no label at all.
PLAIN = _lookup(torch="2.2.0", torchvision="0.17.0", torchaudio="2.2.0")
MIXED = _lookup(torch="2.2.0+cu118", torchvision="0.17.0", torchaudio="2.2.0")
INCOMPLETE = _lookup(torch="2.2.0+cu118", torchvision="0.17.0+cu118")


class TorchCudaTest(unittest.TestCase):
    """One CUDA build across torch, torchvision and torchaudio, or not.

    A mismatch imports fine and fails much later, inside a model, with an
    `undefined symbol`. AMASSS was the only module looking, on Windows only,
    and its loop compared the first pair and returned on it.
    """

    def test_labels_read_the_local_version(self):
        self.assertEqual(torch_cuda_labels(lookup=ALL_118),
                         {"torch": "118", "torchvision": "118", "torchaudio": "118"})

    def test_a_library_that_is_absent_is_absent_from_the_mapping(self):
        self.assertEqual(sorted(torch_cuda_labels(lookup=INCOMPLETE)),
                         ["torch", "torchvision"])

    def test_present_but_unlabelled_is_None_not_missing(self):
        self.assertEqual(torch_cuda_labels(lookup=PLAIN),
                         {"torch": None, "torchvision": None, "torchaudio": None})

    # --- torch_cuda_conflict: must never cry wolf

    def test_no_conflict_when_they_agree(self):
        self.assertIsNone(torch_cuda_conflict(lookup=ALL_118))

    def test_no_conflict_on_a_plain_pypi_install(self):
        """With no label there is no way to tell: that is not a disagreement."""
        self.assertIsNone(torch_cuda_conflict(lookup=PLAIN))

    def test_no_conflict_when_only_one_declares_a_build(self):
        self.assertIsNone(torch_cuda_conflict(lookup=MIXED))

    def test_a_real_disagreement_is_reported_with_its_members(self):
        self.assertEqual(torch_cuda_conflict(lookup=FIRST_ODD),
                         {"torch": "121", "torchvision": "118", "torchaudio": "118"})

    def test_the_last_one_out_of_step_is_caught_too(self):
        """The defect of the AMASSS copy: it compared the 1st pair only."""
        self.assertIsNotNone(torch_cuda_conflict(lookup=LAST_ODD))

    # --- torch_cuda_builds_agree: stricter, and what AMASSS wants

    def test_agree_when_all_three_come_from_the_same_build(self):
        self.assertTrue(torch_cuda_builds_agree(lookup=ALL_118))

    def test_does_not_agree_when_the_last_is_out_of_step(self):
        self.assertFalse(torch_cuda_builds_agree(lookup=LAST_ODD))

    def test_does_not_agree_on_a_plain_install(self):
        """AMASSS installs from the CUDA index: a PyPI wheel is to be replaced."""
        self.assertFalse(torch_cuda_builds_agree(lookup=PLAIN))

    def test_does_not_agree_when_one_is_missing(self):
        self.assertFalse(torch_cuda_builds_agree(lookup=INCOMPLETE))

    def test_a_plain_trio_is_not_a_conflict(self):
        """The case that blocked AMASSS: three PyPI wheels, exact versions.

        `torch_cuda_builds_agree` answers no -- no label anywhere -- and the
        module then offered to "update torch 2.2.0 to 2.2.0". The decision has
        to rest on a REAL disagreement, not on a missing label.
        """
        self.assertIsNone(torch_cuda_conflict(lookup=PLAIN))
        self.assertFalse(torch_cuda_builds_agree(lookup=PLAIN))

    def test_the_family_is_the_three_amasss_installs_together(self):
        self.assertEqual(TORCH_FAMILY, ("torch", "torchvision", "torchaudio"))

if __name__ == "__main__":
    unittest.main()
