from torch.utils.data import Dataset
import torch
import os
from AREG_IOS_utils.utils import ReadSurf, ComputeNormals, GetColorArray
from vtk.util.numpy_support import vtk_to_numpy
from AREG_IOS_utils.orientation import orientation
from AREG_IOS_utils.transformation import ScaleSurf
from AREG_IOS_utils.vtkSegTeeth import ToothNoExist, NoSegmentationSurf
import glob

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("AREG_IOS_dataset")


class DatasetPatch(Dataset):
    """
    DatasetPatch class allow to normalise the meshes and creating a bacth before the prediction of the patch
    """

    def __init__(self, T1, T2, surf_property):
        self.list_upper, self.list_lower = Sort(T1, T2)
        self.surf_property = surf_property

    def __len__(self):

        return len(self.list_upper)

    def __getitem__(self, args):
        index, time = args

        surf = ReadSurf(self.list_upper[index][time])
        name = os.path.basename(self.list_upper[index][time])

        bool_error = False
        try:
            surf, matrix = orientation(
                surf,
                [[-0.5, -0.5, 0], [0, 0, 0], [0.5, -0.5, 0]],
                ["3", "8", "9", "14"],
            )
        except NoSegmentationSurf as error:
            logger.error(f"The surf {name} cant be oriented because {error}")
            bool_error = True
        except ToothNoExist:
            logger.error(
                f"The surf {name} was not oriented because the one of UR6, UR1, UL1 or UR6 is missing"
            )
            bool_error = True
        if bool_error:
            logger.warning(
                f"The prediction of the patch can be wrong if the scan has not been oriented beforehand"
            )

        surf = ScaleSurf(surf)

        surf = ComputeNormals(surf)

        V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        F = torch.tensor(
            vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
        ).to(torch.int64)
        CN = torch.tensor(
            vtk_to_numpy(GetColorArray(surf, "Normals")) / 255.0, dtype=torch.float32
        )

        return V, F, CN

    def isLower(self):
        out = True
        if self.list_lower == None:
            out = False
        return out

    def getLowerSurf(self, idx, time):
        return ReadSurf(self.list_lower[idx][time])

    def getUpperSurf(self, idx, time):
        return ReadSurf(self.list_upper[idx][time])

    def getUpperPath(self, idx, time):
        return self.list_upper[idx][time]

    def getLowerPath(self, idx, time):
        return self.list_lower[idx][time]


def Sort(T1: str, T2: str) -> tuple[list, list]:
    """
    Return two list of dictionnary, one with only Upper and the other with Lower.
    The index is linked to the same patient.
    If there are not lower scan in folders, the Lower list return None
    Args:
        T1 (str): T1 folder path
        T2 (str): T12 folder path

    Returns:
        tuple[list,list]:
            list_reg_Upper = [{'T1':'path/P1_UpperT1.vtk', 'T2':'path/P1_UpperT2.vtk'},
            ...,
            {'T1':'path/PX_UpperT1.vtk', 'T2':'path/PX_UpperT2.vtk'}]

            list_reg_Lower = [{'T1':'path/P1_LowerT1.vtk', 'T2':'path/P1_LowerT2.vtk'},
            ...,
            {'T1':'path/PX_LowerT1.vtk', 'T2':'path/PX_LowerT2.vtk'}]
    """
    # Get all files and filter to only surface files (exclude .tfm transformation files)
    all_t1_files = glob.glob(os.path.join(T1, "*"))
    all_t2_files = glob.glob(os.path.join(T2, "*"))
    
    # Filter to only include surface files (.vtk, .vtp, .stl, .obj), exclude .tfm
    surface_extensions = ('.vtk', '.vtp', '.stl', '.obj')
    t1_files = [f for f in all_t1_files if f.lower().endswith(surface_extensions)]
    t2_files = [f for f in all_t2_files if f.lower().endswith(surface_extensions)]

    if not insideLower(t1_files):  # check if there are Lower arches in list of file
        # if there are not lower arches
        list_reg_upper = sort(t1_files, t2_files)
        list_reg_lower = None

    else:  # if there are lower arches

        t1_uppers = []
        t1_lowers = []
        t2_uppers = []
        t2_lowers = []

        for file in t1_files:
            if isLowerUpper(file, choice="Upper"):
                t1_uppers.append(file)
            else:
                t1_lowers.append(file)

        for file in t2_files:
            if isLowerUpper(file, choice="Upper"):
                t2_uppers.append(file)
            else:
                t2_lowers.append(file)

        list_reg_upper_tmp = sort(t1_uppers, t2_uppers)
        list_reg_lower_tmp = sort(t1_lowers, t2_lowers)

        # organize Lower and Upper list, to have the order of file
        list_reg_upper = []
        list_reg_lower = []

        for upper in list_reg_upper_tmp:
            upper_name = os.path.basename(upper["T1"]).replace("T1", "")
            upper_name = removeLowerUpper(upper_name, choice="Upper")

            for lower in list_reg_lower_tmp:
                lower_name = os.path.basename(lower["T1"]).replace("T1", "")
                lower_name = removeLowerUpper(lower_name, choice="Lower")

                if upper_name == lower_name:
                    list_reg_upper.append(upper)
                    list_reg_lower.append(lower)
                    continue

    return list_reg_upper, list_reg_lower


def SortLower(T1: str, T2: str) -> list:
    """Pair the lower scans of two folders, whether or not upper scans are there.

    Sort() only keeps a lower pair when the matching upper pair exists, since
    the palatal registration always starts from the maxilla. The MGL
    registration works on the mandible alone, so it pairs the lower files on
    their own.

    The search goes through subfolders: the crown segmentation writes its
    output into a folder of its own under the one it was given, so scans that
    were segmented on the way here sit one level down, and a flat listing
    would pair nothing at all.

    Returns:
        list[dict]: [{'T1': 'path/P1_LowerT1.vtk', 'T2': 'path/P1_LowerT2.vtk'}, ...]
    """
    surface_extensions = ('.vtk', '.vtp', '.stl', '.obj')

    def lower_files(folder):
        files = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                 if f.lower().endswith(surface_extensions)]
        return [f for f in files if not isLowerUpper(f, choice="Upper")]

    return sort(lower_files(T1), lower_files(T2))


def insideLower(list_files: list) -> bool:
    """Check if there are lower archer in list of file

    Args:
        list_files (list): contain list of file path

    Returns:
        bool: return if there are lower archer in list of file
    """
    out = False

    for file in list_files:
        if isLowerUpper(file, choice="Lower"):
            out = True
            continue

    return out


def removeLowerUpper(file_name: str, choice: str = "Upper") -> str:
    """remove the appelation in the file name of upper or lower depend of the argument choice

    Args:
        file_name (str): file path
        choice (str, optional): _description_. Defaults to 'Upper'.

    Returns:
        str:  return file name without the appelation in the file name of upper or lower depend of the argument choice
    """
    list_word = []

    if choice == "Lower":
        list_word = ["Lower", "_L", "L_", "Mandibule", "Md"]
    elif choice == "Upper":
        list_word = ["Upper", "_U", "U_", "Maxilla", "Mx"]

    for word in list_word:
        file_name = file_name.replace(word, "")

    return file_name


def isLowerUpper(file_name: str, choice: str = "Upper") -> bool:
    """Check if the file name is for Lower of Upper depend of the choice

    Args:
        file_name (str): _description_
        choice (str, optional): _description_. Defaults to 'Upper'.

    Returns:
        bool: _description_
    """
    out = False
    list_word = []

    if choice == "Lower":
        list_word = ["Lower", "_L", "L_", "Mandibule", "Md"]
    elif choice == "Upper":
        list_word = ["Upper", "_U", "U_", "Maxilla", "Mx"]

    for lower_word in list_word:
        if lower_word in file_name:
            out = True
            continue

    return out


def sort(T1_files: list, T2_files: list) -> list[dict]:
    """Link T1 file and T2 file

    Args:
        T1_files (list): contain list of T1 files
        T2_files (list): contain list of T2 files

    Returns:
        list[dict]: example: [{'T1':'path/patient5T1.vtk','T2':'path/patient5T2},...,{'T1':'path/patient90T1.vtk','T2':'path/patient90T2}]
    """
    list_reg = []

    for t1_file in T1_files:
        t1_name = os.path.basename(t1_file)
        t1_name = t1_name.replace("T1", "")

        for t2_file in T2_files:
            t2_name = os.path.basename(t2_file)
            t2_name = t2_name.replace("T2", "")

            if t2_name == t1_name:
                list_reg.append({"T1": t1_file, "T2": t2_file})
                continue

    return list_reg
