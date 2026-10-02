"""The publication addresses, written once.

This module does not say *which* models a tool offers -- that selection belongs
to the tool and differs from one to the next, which is why the `getModelUrl`
dictionaries stay where they are. It carries only the **release bases**, the
ones that were repeated identically: publishing a new version meant tracking
down 32 literals for the ADT models release alone.

Standard library only.
"""

#: Release carrying the extension's landmark and segmentation models.
#: Changing version happens here, and nowhere else.
ADT_MODELS = (
    "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools"
    "/releases/download/v0.1-v2.0_models"
)

#: The WSL2 installer the modules offer when WSL is missing, or when its system
#: libraries are missing. The link was copied into six modules, and twice in
#: each of them.
WSL2_INSTALLER = (
    "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools"
    "/releases/download/wsl2_windows/installer_WSL2.zip"
)


#: The ASO_CBCT reference-plane models, which four modules offer. Sixteen
#: literals carried this base; the selection itself stays with each module.
ASO_CBCT_GOLD = (
    "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels"
)

#: The ASO_IOS references and models, shared by ALI, ASO and AREG.
ASO_IOS_GOLD = "https://github.com/HUTIN1/ASO/releases/download/v1.0.0"

#: The ASO_IOS test data sets. Two distinct releases, one per mode: the
#: semi-automatic set carries the landmark json files that mode demands, the
#: automatic set carries only the surfaces. Both links were hard-coded in
#: `ASO_Method/IOS.py`, the only test data sets not to go through here.
ASO_IOS_TEST_AUTO = "https://github.com/HUTIN1/ASO/releases/download/v1.0.1"
ASO_IOS_TEST_SEMI = "https://github.com/HUTIN1/ASO/releases/download/v1.0.2"

#: The test data sets published by Slicer, used by `registerSampleData`.
SLICER_TESTING_DATA = (
    "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256"
)


# ---------------------------------------------------------------------------
# The other release bases, each copied two to ten times. Publishing a new
# version of one of these sets happens here, and nowhere else.
#
# This module still does not say *which* models a tool offers: the `getModelUrl`
# dictionaries keep their selection, and that is deliberate -- merging it would
# mean deciding which models each tool offers.
# ---------------------------------------------------------------------------

BASE = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download"

#: The three dental segmentation sets, shared by BATCHDENTALSEG and VFACE.
PEDIATRIC_DENTAL_SEG = f"{BASE}/PEDIATRICDENTALSEG_MODEL"
NASOMAXILLA_DENT_SEG = f"{BASE}/NASOMAXILLADENTSEG_MODEL"
UNIVERSAL_LAB = f"{BASE}/UNIVERSALLAB_MODEL"

#: CBCT segmentation, IOS landmarks, IOS/CBCT registration, TMJ crop, VFACE.
AMASSS_CBCT = f"{BASE}/AMASSS_CBCT"
ALI_IOS_MODELS = f"{BASE}/ALI_IOS_models"
AREG_IOSCBCT_MODELS = f"{BASE}/AREG_IOSCBCT"
TMJ_CROP_MODEL = f"{BASE}/TMJ_CROP_MODEL"
VFACE_MODELS = f"{BASE}/VFACE"

#: Sets published outside the organisation.
ASO_CBCT_PRE = "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_preASOmodels"
ASO_CBCT_TEST_FILES = "https://github.com/lucanchling/ASO_CBCT/releases/download/TestFiles"
AREG_CBCT_TEST_FILES = "https://github.com/lucanchling/Areg_CBCT/releases/download/TestFiles"
AMASSS_CBCT_UPSTREAM = "https://github.com/lucanchling/AMASSS_CBCT/releases/download/v1.0.2"
ALIDDM = "https://github.com/baptistebaquero/ALIDDM/releases/download/v1.0.3"
AREG_IOS_MODELS = "https://github.com/HUTIN1/AREG/releases/download/v1.0.0"
AUTOMATRIX_MIRROR = (
    "https://github.com/GaelleLeroux/DCBIA_Apply_matrix/releases/download/AutoMatrixMirror"
)

#: The test data sets, one base per release. The file itself stays with the
#: tool that offers it: that tool is the one that knows whether it needs one or
#: two. These bases carry /download/, not /tag/ -- /tag/ is the release's web
#: page, which GitHub serves with a 200 and which the copies tried to unzip.
CBCT_TEST_SCAN_RELEASE = "https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1"
ALIDDM_TEST_FILES = "https://github.com/baptistebaquero/ALIDDM/releases/download/v1.0.4"

#: The same scan serves ALI CBCT and AMASSS: a **bare** 99 MB NIfTI, not an
#: archive. It was copied identically into AMASSS and twice into ALI.
AMASSS_TEST_SCAN = f"{CBCT_TEST_SCAN_RELEASE}/MG_test_scan.nii.gz"

# ---------------------------------------------------------------------------
# The test data sets published as a single file. Unlike the bases above, these
# are complete addresses: a test data set is an archive, not a catalogue each
# tool picks from.
# ---------------------------------------------------------------------------

#: The MRI2CBCT test data set. The archive carries `TestFile/`, with the
#: original CBCT and MRI (`CBCT_ori`, `MRI_ori`) and the already preprocessed
#: CBCT/MRI/segmentation triplet (`REG/CBCT`, `REG/MRI`, `REG/Seg`) that feeds
#: the following steps.
MRI2CBCT_TEST_FILES = f"{BASE}/test_files/TestFile.zip"

#: The FlexReg test data set: `TestFiles/T1_test_file.vtk` and
#: `T2_test_file.vtk`.
#:
#: Published on a personal repository, and not under DCBIA-OrthoLab like
#: everything else: the publication should migrate to the organisation, failing
#: which FlexReg's TestFile button depends on an individual account. The address
#: stays the one that answers today -- we cannot republish in their place.
FLEXREG_TEST_FILES = (
    "https://github.com/GaelleLeroux/SlicerAutomatedDentalTools"
    "/releases/download/testfileFlexReg/TestFiles.zip"
)
