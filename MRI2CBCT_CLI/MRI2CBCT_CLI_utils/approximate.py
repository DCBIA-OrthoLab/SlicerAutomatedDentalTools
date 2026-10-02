import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

from MRI2CBCT_CLI_utils.approx_utils import get_corresponding_file, compute_rotation_correction, world_center_of_mass
from MRI2CBCT_CLI_utils.condyle_segmentation import segment_condyle


# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.progress_protocol import emit_fraction

logger = get_logger("MRI2CBCT_CLI_utils_approximate")


def _patientIdFromCbctFilename(cbct_file):
    # Prefer splitting on "CBCT" when present (e.g. "B010_CBCT_01.nii.gz"
    # -> "B010"); fall back to the leading token for filenames that don't
    # contain it (e.g. "B010.nii.gz" -> "B010"), instead of skipping them.
    if "CBCT" in cbct_file:
        return cbct_file.split("CBCT")[0].rstrip("_")
    return cbct_file.split(".")[0].split("_")[0]


def _findMriForPatient(mri_folder, patient_id):
    # Try the stricter "filename also carries a modality token" match first
    # (matches the convention used elsewhere in this codebase, e.g. when CBCT
    # and MRI files might share a folder). Since mri_folder is normally its
    # own dedicated folder, the modality token isn't actually needed to be
    # unambiguous, so fall back to a bare patient-ID match (e.g. "B003.nii"
    # in both the cbct and mri folders, with no "MRI"/"MR" in the filename).
    for modality in ("MRI", "MR"):
        mri_path = get_corresponding_file(mri_folder, patient_id, modality)
        if mri_path:
            return mri_path

    for root, _, files in os.walk(mri_folder):
        for file in files:
            if file.startswith(patient_id) and (file.endswith(".nii.gz") or file.endswith(".nii")):
                return os.path.join(root, file)
    return None


def _findPairs(cbct_folder, mri_folder):
    """Yields (patient_id, cbct_path, mri_path) for every CBCT in cbct_folder
    that has a matching MRI in mri_folder."""
    pairs = []
    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if not (cbct_file.endswith(".nii") or cbct_file.endswith(".nii.gz")):
                logger.warning(f"Skipping non-NIfTI file: {cbct_file}")
                continue
            patient_id = _patientIdFromCbctFilename(cbct_file)
            cbct_path = os.path.join(root, cbct_file)
            mri_path = _findMriForPatient(mri_folder, patient_id)
            if not mri_path:
                logger.warning(f"No corresponding MRIs file found for: {cbct_file}")
                continue
            pairs.append((patient_id, cbct_path, mri_path))
    return pairs


def approximation(cbct_folder, mri_folder, output_folder, model_folder, tmp_folder):
    """
    For each CBCT/MRI pair, locate the condyle in the CBCT with the same nnUNet
    segmentation used by TMJ-crop, mark a corresponding point at the center of
    the MRI, and write both points (plus the header-based rotation correction)
    to a per-patient JSON file. The actual point-matching registration is done
    afterwards by the calling widget, since it requires Slicer's own fiducial
    registration machinery which isn't available in this CLI subprocess.

    cbct_folder/mri_folder can each be a folder (batch mode: every CBCT in
    cbct_folder is paired by patient ID with a matching MRI in mri_folder) or
    a single NIfTI file (single-case mode: cbct_folder and mri_folder are then
    treated as exactly one CBCT/MRI pair) - mirroring how AMASSS accepts
    either a folder or a single file for its scan input.

    Args:
        cbct_folder (str): CBCT folder, or path to a single CBCT file.
        mri_folder (str): MRI folder, or path to a single MRI file.
        output_folder (str): Path to save the per-patient point JSON files into
            (a "points" subfolder is created under it).
        model_folder (str): Path to the nnUNet condyle segmentation model.
        tmp_folder (str): Scratch folder for nnUNet's per-case inputs/outputs.
    """

    cbct_is_file = os.path.isfile(cbct_folder)
    mri_is_file = os.path.isfile(mri_folder)
    if cbct_is_file != mri_is_file:
        raise ValueError(
            f"cbct_folder and mri_folder must either both be single files or both be folders: "
            f"{cbct_folder}, {mri_folder}")

    if cbct_is_file:
        patient_id = _patientIdFromCbctFilename(os.path.basename(cbct_folder))
        pairs = [(patient_id, cbct_folder, mri_folder)]
    else:
        if not os.path.isdir(cbct_folder): raise ValueError(f"CBCT folder does not exist: {cbct_folder}")
        if not os.path.isdir(mri_folder): raise ValueError(f"MRI folder does not exist: {mri_folder}")
        pairs = _findPairs(cbct_folder, mri_folder)

    os.makedirs(output_folder, exist_ok=True)
    points_folder = os.path.join(output_folder, "points")
    os.makedirs(points_folder, exist_ok=True)

    tmp_dir = Path(tmp_folder)
    model_dir = Path(model_folder)

    total_patients = len(pairs)
    patient_count = 0

    for patient_id, cbct_path, mri_path in pairs:
        logger.info(f"Treating patient: {patient_id}")

        case_dir = tmp_dir / patient_id
        try:
            mask, cbct_half, side = segment_condyle(cbct_path, mri_path, case_dir, model_dir)
        except Exception as e:
            logger.error(f"Condyle segmentation failed for {patient_id}: {e}")
            shutil.rmtree(case_dir, ignore_errors=True)
            continue

        if mask.sum() == 0:
            logger.warning(f"No condyle found for {patient_id} (side {side}); skipping approximation for this patient")
            shutil.rmtree(case_dir, ignore_errors=True)
            continue

        nz = np.array(np.nonzero(mask))
        centroid_ijk = nz.mean(axis=1)
        cbct_point = cbct_half.affine[:3, :3] @ centroid_ijk + cbct_half.affine[:3, 3]

        mri_point = world_center_of_mass(nib.load(mri_path))
        rotation = compute_rotation_correction(mri_path, cbct_path)
        mri_point_rotated = rotation @ mri_point

        points = {
            "patient_id": patient_id,
            "side": side,
            "cbct_point_ras": cbct_point.tolist(),
            "mri_point_rotated_ras": mri_point_rotated.tolist(),
            "rotation_ras": rotation.tolist(),
            "mri_path": mri_path,
            "cbct_path": cbct_path,
        }
        points_path = os.path.join(points_folder, f"{patient_id}_approx_points.json")
        with open(points_path, "w") as f:
            json.dump(points, f, indent=2)
        logger.info(f"Saved approximation points for {patient_id} to {points_path}")

        shutil.rmtree(case_dir, ignore_errors=True)

        patient_count += 1
        if total_patients > 0:
            progress = patient_count / total_patients
            emit_fraction(progress)
            sys.stdout.flush()
            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Locate matching CBCT/MRI condyle points for approximation.')
    parser.add_argument('--cbct_folder', type=str, help='Path to a folder of CBCT images, or a single CBCT file')
    parser.add_argument('--mri_folder', type=str, help='Path to a folder of MRI images, or a single MRI file')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output point files will be saved')
    parser.add_argument('--model_folder', type=str, help='Path to the nnUNet condyle segmentation model')
    parser.add_argument('--tmp_folder', type=str, help='Scratch folder for nnUNet per-case files')
    args = parser.parse_args()

    approximation(args.cbct_folder, args.mri_folder, args.output_folder, args.model_folder, args.tmp_folder)
