from pathlib import Path
from ADTLib.naming import patient_id as read_patient_id, TMJ_CROP_MARKERS

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("MRI2CBCT_CLI_utils_TMJ_Crop")

def GetListFiles(folder_path, extensions):
    return [str(p) for ext in extensions for p in Path(folder_path).rglob(f"*{ext}")]

# TIMEPOINT-SUFFIX: the chain lives in ADTLib.naming, with the note. This site
# uses the widest marker set of the repository -- the default plus seventeen
# markers of its own -- declared there as TMJ_CROP_MARKERS.
def extract_patient_id(filename: str) -> str:
    return read_patient_id(Path(filename).stem, TMJ_CROP_MARKERS)

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
