import os
import re
import numpy as np
import json
import SimpleITK as sitk
from ADTLib.io.landmarks import LoadJsonLandmarks  # noqa: F401  (re-exported)
from ADTLib.io.fs import search  # noqa: F401  (re-exported)
from ADTLib.io.surface import ReadSurf, WriteSurf  # noqa: F401  (re-exported)

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("ASO_IOS_utils")


# Which arch a file belongs to is read from its name, and it has to be read the
# same way by everything that reads it. Two rules, because the two spellings do
# not carry the same risk:
#   - the words "upper" and "lower" are looked for anywhere, as before, so names
#     like "GoldUpper.vtk" keep working;
#   - the single letters U and L only count as a jaw when they stand alone
#     between separators ("P1_T1_U.vtk", "U_P1.vtk", "P1_T1_U_Seg.vtk"). A bare
#     letter matched anywhere made "Dupont_03_L.vtk" an upper on the strength of
#     the u in the name, and the old "_U_" wanted a trailing separator the very
#     common "P1_T1_U.vtk" does not have.
_JAW_WORD = {"Upper": "upper", "Lower": "lower"}
_JAW_LETTER = {
    "Upper": re.compile(r"(?:^|[_\-])u(?=[_\-.]|$)", re.IGNORECASE),
    "Lower": re.compile(r"(?:^|[_\-])l(?=[_\-.]|$)", re.IGNORECASE),
}


def JawFromFileName(path_filename, default=None):
    """The arch this file name names: Upper, Lower, or `default` for neither.

    Only the base name is read: a path can hold anything, and an input folder
    called "lower_arches" used to decide the jaw of every file under it.
    """
    filename = os.path.basename(str(path_filename))

    found = {}
    for jaw, word in _JAW_WORD.items():
        index = filename.lower().rfind(word)
        if index != -1:
            found[jaw] = index
    for jaw, pattern in _JAW_LETTER.items():
        if jaw in found:
            continue
        matches = list(pattern.finditer(filename))
        if matches:
            found[jaw] = matches[-1].start()

    if not found:
        return default

    if len(found) == 2:
        # Both arches named in one file name. It happens downstream of ALI_IOS,
        # which appends the jaw of the model it ran to the name of the scan it
        # ran on, so the last marker written is the one that describes the file.
        jaw = max(found, key=found.get)
        logger.warning(
            f"{filename} names both arches; read as {jaw}, its last marker")
        return jaw

    return next(iter(found))


def StripJawFromFileName(name_file):
    """The part of a name that both arches of one patient have in common.

    This is what upper and lower files are paired on, so it has to come out
    identical for the two of them: "P1_T1_U_Seg" and "P1_T1_L_Seg" both reduce
    to "P1_T1_Seg", and so does "P1_T1_Upper_Seg".
    """
    out = re.sub(r"(?:^|[_\-])(?:u|l)(?=[_\-.]|$)", "_", name_file, flags=re.IGNORECASE)
    out = re.sub(r"upper|lower", "_", out, flags=re.IGNORECASE)
    out = re.sub(r"[_\-]{2,}", "_", out)
    return out.strip("_-")


def UpperOrLower(path_filename):
    """tell if the file is for upper jaw of lower

    Args:
        path_filename (str): example /home/..../landmark_upper.json

    Returns:
        str: Upper or Lower, for the example above Upper
    """
    return JawFromFileName(path_filename, default="Lower")


def WriteJsonLandmarks(
    landmarks, output_file, input_file_json, add_innamefile, output_folder
):
    """
    Write the landmarks to a json file

    Parameters
    ----------
    landmarks : dict
        landmarks to write
    output_file : str
        output file name
    """
    # # Load the input image
    dirname, name = os.path.split(output_file)
    name, extension = os.path.splitext(name)
    output_file = os.path.join(output_folder, name + add_innamefile + extension)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(input_file_json, "r") as outfile:
        temp_data = json.load(outfile)
    for i in range(len(landmarks)):
        pos = landmarks[temp_data["markups"][0]["controlPoints"][i]["label"]]
        temp_data["markups"][0]["controlPoints"][i]["position"] = [
            pos[0],
            pos[1],
            pos[2],
        ]
    with open(output_file, "w") as outfile:

        json.dump(temp_data, outfile, indent=4)


def listlandmark2diclandmark(list_landmark):
    upper = []
    lower = []
    list_landmark = list_landmark.split(",")
    for landmark in list_landmark:
        if "U" == landmark[0]:
            upper.append(landmark)
        else:
            lower.append(landmark)

    out = {"Upper": upper, "Lower": lower}

    return out


def WritefileError(file, folder_error, message):
    if not os.path.exists(folder_error):
        os.mkdir(folder_error)
    name = os.path.basename(file)
    name, _ = os.path.splitext(name)
    with open(os.path.join(folder_error, f"{name}Error.txt"), "w") as f:
        f.write(message)

def PatientNumber(path):
    matrix_pat = os.path.basename(path).split('_U')[0].split('_L')[0].split('.')[0].replace('.','')
    return matrix_pat


def saveMatrixAsTfm(matrix, output_path):
    assert matrix.shape == (4, 4), "Expected a 4x4 matrix."
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    inverted_matrix = np.linalg.inv(matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(inverted_matrix[:3, :3].flatten())
    transform.SetTranslation(inverted_matrix[:3, 3])
    sitk.WriteTransform(transform, output_path)