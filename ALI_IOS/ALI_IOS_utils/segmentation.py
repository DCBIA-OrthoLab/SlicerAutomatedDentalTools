# Give a scan the teeth labels the landmark prediction is built on.
#
# Every ALI_IOS model aims its cameras tooth by tooth, read from a Universal_ID
# array: a scan without one gets no landmark at all. An .stl cannot carry that
# array in the first place, and a .vtk straight off the scanner does not have
# it either, so both used to be turned away.
#
# The crown segmentation writes it, and it lives in the same conda environment
# this CLI is run from, so the scan is segmented on the way in. It writes a
# .vtk whatever it was given, which is the conversion an .stl needs, and it
# leaves the point coordinates untouched (measured: 5e-5 mm over 118k points),
# so the landmarks predicted on the copy are valid in the file the user gave.
import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile

import vtk
from vtk.util.numpy_support import vtk_to_numpy

logger = logging.getLogger("ALI_IOS_segmentation")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Array names a segmentation may carry, in the order they are looked for.
LABEL_ARRAYS = ("Universal_ID", "PredictedID", "UniversalID")

EXECUTABLE = "dentalmodelseg"


def IsSegmented(path):
    """True when the surface already carries teeth labels."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    else:
        # No other surface format this reads can hold a point array.
        return False

    reader.SetFileName(path)
    reader.Update()
    surface = reader.GetOutput()
    if surface is None or surface.GetNumberOfPoints() == 0:
        return False

    point_data = surface.GetPointData()
    for name in LABEL_ARRAYS:
        array = point_data.GetScalars(name) or point_data.GetArray(name)
        if array is not None:
            return len(set(vtk_to_numpy(array).ravel().tolist())) > 1
    return False


def FindExecutable():
    """Path to dentalmodelseg, or None.

    It is on the PATH when this CLI is started through `conda run`, and next to
    the interpreter when the environment's python is called directly.
    """
    found = shutil.which(EXECUTABLE)
    if found:
        return found

    beside = os.path.join(os.path.dirname(sys.executable), EXECUTABLE)
    return beside if os.path.isfile(beside) else None


def SegmentSurface(path, folder=None):
    """Segment `path` and return the labelled .vtk written for it, or None.

    `folder` is where the copy goes; a temporary one is made when it is not
    given, and the caller is then responsible for removing it.
    """
    executable = FindExecutable()
    if executable is None:
        logger.error(
            f"{os.path.basename(path)} carries no teeth segmentation and "
            f"{EXECUTABLE} is not in this environment, so it cannot be given one. "
            "Segment the scans beforehand, or run this from the environment that "
            "provides the crown segmentation")
        return None

    ours = folder is None
    folder = folder or tempfile.mkdtemp(prefix="ALI_IOS_segmented_")

    # The scan is copied first and the copy is what gets segmented. Handed an
    # .stl with --overwrite, shapeaxi deletes it (dental_model_seg.py, "if ext
    # == '.stl': os.remove(args.stl)"), so nothing of the user's is ever passed
    # to it. --overwrite 0 keeps it off that branch as well, and makes it write
    # the labelled copy under --out, which is what is wanted here.
    copy = os.path.join(folder, os.path.basename(path))
    shutil.copy(path, copy)

    # Named through a csv rather than with --surf: given a single file that
    # way, shapeaxi dies writing the result, on an argument its own code never
    # set (AttributeError: Namespace has no attribute 'stl'). The csv is the
    # path the crown segmentation module uses, and it works.
    listing = os.path.join(folder, "surfaces.csv")
    with open(listing, "w") as handle:
        handle.write("surf\n" + copy + "\n")

    command = [executable, "--out", folder, "--overwrite", "0",
               "--crown_segmentation", "0", "--array_name", "Universal_ID",
               "--fdi", "0", "--suffix", "Seg",
               "--csv", listing, "--vtk_folder", folder]

    logger.info(f"{os.path.basename(path)} carries no teeth segmentation: segmenting it "
                "first, which takes about a minute")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")

    # The copy itself is a .vtk when the input was one, so only what the
    # segmentation wrote is looked at.
    produced = sorted(p for p in glob.glob(os.path.join(folder, "**", "*.vtk"), recursive=True)
                      if os.path.abspath(p) != os.path.abspath(copy))
    if not produced:
        tail = "\n".join((result.stdout or "").splitlines()[-8:])
        logger.error(f"The segmentation of {os.path.basename(path)} produced nothing.\n{tail}")
        if ours:
            shutil.rmtree(folder, ignore_errors=True)
        return None

    if len(produced) > 1:
        stem = os.path.splitext(os.path.basename(path))[0]
        named = [p for p in produced if stem in os.path.basename(p)]
        produced = named or produced

    logger.info(f"Segmented into {produced[0]}")
    return produced[0]
