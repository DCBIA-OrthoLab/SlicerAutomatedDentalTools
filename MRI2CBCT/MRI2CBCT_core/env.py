"""Where MRI2CBCT keeps its models and its Open3D, on any computer.

Everything lives under Slicer's downloads folder (~/Documents/SlicerDownloads),
next to what ASO, ALI and AMASSS download:

    SlicerDownloads/MRI2CBCT/Models/Dataset516/<trainer folder>/   nnU-Net weights
    SlicerDownloads/MRI2CBCT/py_deps/                              Open3D on Linux

The widget knows that folder (it asks Qt) and passes it to the CLIs through
MRI2CBCT_DOWNLOADS, so the CLIs need no Qt.

Why Open3D can need a folder of its own: on Linux, pip inside Slicer refuses
Open3D's wheel because Slicer caps the manylinux tag at 2_28 while the wheel
is 2_31. The wheel itself runs on any glibc >= 2.31, so it is installed with
that tag accepted, into py_deps, without touching Slicer's own packages.
"""
import os
import sys

DATASET516_TRAINER = "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
#: GitHub release asset. The zip holds DATASET516_TRAINER/ at its root.
DATASET516_URL = os.environ.get(
    "MRI2CBCT_DATASET516_URL",
    "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/"
    "MRI2CBCT_Dataset516/Dataset516_MRICBManual.zip")
OPEN3D = "open3d==0.19.0"
OPEN3D_LINUX_TAG = "manylinux_2_31_x86_64"
PY_ABI = f"cp{sys.version_info.major}{sys.version_info.minor}"


def downloads():
    d = os.environ.get("MRI2CBCT_DOWNLOADS")
    if d:
        return d
    return os.path.join(os.path.expanduser("~"), "Documents", "SlicerDownloads")


def home():
    return os.path.join(downloads(), "MRI2CBCT")


def dataset516_root():
    """Where the downloaded zip is unpacked (ADTLib.testdata marks it complete)."""
    return os.path.join(home(), "Models", "Dataset516")


def dataset516():
    """The nnU-Net trainer folder. MRI2CBCT_DATASET516 overrides it."""
    return os.environ.get("MRI2CBCT_DATASET516",
                          os.path.join(dataset516_root(), DATASET516_TRAINER))


def py_deps():
    return os.environ.get("MRI2CBCT_PY_DEPS", os.path.join(home(), "py_deps"))


def ensure_open3d():
    """Use the interpreter's own open3d, else py_deps."""
    try:
        import open3d  # noqa: F401
        return "interpreter"
    except ImportError:
        pass
    d = py_deps()
    if not os.path.isdir(os.path.join(d, "open3d")):
        raise ImportError(f"Open3D is not installed (looked in Slicer's Python and in {d}).")
    if d not in sys.path:
        sys.path.append(d)   # appended: Slicer's own numpy/scipy must win
    import open3d  # noqa: F401
    return "py_deps"


# Back-compatibility for code written before the move.
DATASET516 = dataset516()
