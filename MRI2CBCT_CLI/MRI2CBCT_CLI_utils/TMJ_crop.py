import os
from pathlib import Path

def GetListFiles(folder_path, extensions):
    return [str(p) for ext in extensions for p in Path(folder_path).rglob(f"*{ext}")]

def extract_patient_id(filename: str) -> str:
    # Remove suffixes like _Scan, _CB, _T1, _seg, _mask, etc.
    return (
        Path(filename).stem
        .split("_Scan")[0]
        .split("_scan")[0]
        .split("_Or")[0]
        .split("_OR")[0]
        .split("_MAND")[0]
        .split("_MD")[0]
        .split("_MAX")[0]
        .split("_MX")[0]
        .split("_CB")[0]
        .split("_lm")[0]
        .split("_T2")[0]
        .split("_T1")[0]
        .split("_Cl")[0]
        .split("_seg")[0]
        .split("_mask")[0]
        .split("_pred")[0]
        .split("_CBCT")[0]
        .split("_MRI")[0]
        .split("_MR")[0]
        .split("_Seg")[0]
        .split(".")[0]
    )

def GetPatients(cbct_folder, mri_folder, seg_folder):
    extensions = [".nii.gz", ".nii", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz"]
    patients = {}

    for file in GetListFiles(cbct_folder, extensions):
        pid = extract_patient_id(file)
        patients.setdefault(pid, {})["cbct"] = file

    for file in GetListFiles(mri_folder, extensions):
        pid = extract_patient_id(file)
        patients.setdefault(pid, {})["mri"] = file

    for file in GetListFiles(seg_folder, extensions):
        pid = extract_patient_id(file)
        patients.setdefault(pid, {})["seg"] = file

    return patients
