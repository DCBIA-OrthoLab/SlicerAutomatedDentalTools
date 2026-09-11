import os
from pathlib import Path
import sys
import logging

# ===== Logging Configuration =====
logger = logging.getLogger("MRI2CBCT_CLI_utils_TMJ_Crop")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def GetListFiles(folder_path, extensions):
    return [str(p) for ext in extensions for p in Path(folder_path).rglob(f"*{ext}")]

# TIMEPOINT-SUFFIX: only _T1/_T2 are stripped here, so _T3/_T4 inputs break
# patient pairing. See the full note above GetPatients in
# AREG_CBCT/AREG_CBCT_utils/utils.py before changing this.
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
        .split("_Seg")[0]
        .split("_mask")[0]
        .split("_Mask")[0]
        .split("_pred")[0]
        .split("_Pred")[0]
        .split("_crop")[0]
        .split("_Crop")[0]
        .split("_Left")[0]
        .split("_left")[0]
        .split("_Right")[0]
        .split("_right")[0]
        .split("_approximate")[0]
        .split("_Approximate")[0]
        .split("_CBCT")[0]
        .split("_MRI")[0]
        .split("_MR")[0]
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

    # Segmentation files must not create new patient entries on their own -
    # mis-tagged or stray seg files (e.g. "B002_Pred_CB.nii.gz") would
    # otherwise show up as spurious patients with no CBCT/MRI. Only attach a
    # seg to a patient that already exists from the CBCT/MRI folders.
    seg_files_by_id = {}
    for file in GetListFiles(seg_folder, extensions):
        pid = extract_patient_id(file)
        seg_files_by_id[pid] = file

    for pid, files in patients.items():
        if pid in seg_files_by_id:
            files["seg"] = seg_files_by_id[pid]

    return patients
