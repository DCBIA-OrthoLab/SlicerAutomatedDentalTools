import os
from ADTLib.naming import patient_id as read_patient_id

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.io.fs import search as search_files

logger = get_logger("MRI2CBCT_utils")

def GetListFiles(folder_path, file_extension):
    """Return a list of files in folder_path finishing by file_extension"""
    file_list = []
    for extension_type in file_extension:
        file_list += search(folder_path, file_extension)[extension_type]
    return file_list

def GetListNamesSegType(segmentationType):
    dic = {
        "CB": ["cb"],
        "MAND": ["mand", "md"],
        "MAX": ["max", "mx"],
    }
    return dic[segmentationType]

def GetPatients(folder_path, time_point="T1", segmentationType=None):
    """Return a dictionary with patient id as key"""
    file_extension = [".nii.gz", ".nii", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz"]
    json_extension = [".json"]
    file_list = GetListFiles(folder_path, file_extension + json_extension)

    patients = {}

    # TIMEPOINT-SUFFIX: the chain that builds this id now lives in
    # ADTLib.naming, together with the note on what it would take.
    for file in file_list:
        basename = os.path.basename(file)
        patient = read_patient_id(basename)

        if patient not in patients:
            patients[patient] = {}

        if True in [i in basename for i in file_extension]:
            if True in [i in basename.lower() for i in ["mask", "seg", "pred"]]:
                if segmentationType is None:
                    patients[patient]["seg" + time_point] = file
                else:
                    if True in [
                        i in basename.lower()
                        for i in GetListNamesSegType(segmentationType)
                    ]:
                        patients[patient]["seg" + time_point] = file

            else:
                patients[patient]["scan" + time_point] = file

        if True in [i in basename for i in json_extension]:
            if time_point == "T2":
                patients[patient]["lm" + time_point] = file

    return patients


def GetMatrixPatients(folder_path):
    """Return a dictionary with patient id as key and matrix path as data"""
    file_extension = [".tfm"]
    file_list = GetListFiles(folder_path, file_extension)

    patients = {}
    for file in file_list:
        basename = os.path.basename(file)
        patient = basename.split("reg_")[1].split("_Cl")[0]
        if patient not in patients and True in [i in basename for i in file_extension]:
            patients[patient] = {}
            patients[patient]["mat"] = file

    return patients


def GetDictPatients(
    folder_t1_path,
    folder_t2_path,
    segmentationType=None,
    todo_str="",
    matrix_folder=None,
):
    """Return a dictionary with patients for both time points"""
    patients_t1 = GetPatients(
        folder_t1_path, time_point="T1", segmentationType=segmentationType
    )
    patients_t2 = GetPatients(folder_t2_path, time_point="T2", segmentationType=None)
    patients = MergeDicts(patients_t1, patients_t2)

    if matrix_folder is not None:
        patient_matrix = GetMatrixPatients(matrix_folder)
        patients = MergeDicts(patients, patient_matrix)
    patients = ModifiedDictPatients(patients, todo_str)
    return patients


def MergeDicts(dict1, dict2):
    """Merge t1 and t2 dictionaries for each patient"""
    patients = {}
    for patient in dict1:
        patients[patient] = dict1[patient]
        try:
            patients[patient].update(dict2[patient])
        except KeyError:
            continue
    return patients


def ModifiedDictPatients(patients, todo_str):
    """Modify the dictionary of patients to only keep the ones in the todo_str"""

    if todo_str != "":
        liste_todo = todo_str.split(",")
        todo_patients = {}
        for i in liste_todo:
            patient = list(patients.keys())[int(i) - 1]
            todo_patients[patient] = patients[patient]
        patients = todo_patients

    return patients


def search(path, *args):
    """Delegated to ADTLib. This call site sorts: the patient order depends on it."""
    return search_files(path, *args, sort=True)