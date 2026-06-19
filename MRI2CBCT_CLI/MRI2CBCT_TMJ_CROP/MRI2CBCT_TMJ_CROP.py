#!/usr/bin/env python-real

import os
import argparse, subprocess, shutil, itertools
from nnunetv2.inference.predict_from_raw_data import predict_entry_point
from typing import Optional
from pathlib import Path
import numpy as np
import nibabel as nib
from  scipy.ndimage import label

import sys
import logging

# ===== Logging Configuration =====
logger = logging.getLogger("MRI2CBCT_CLI_TMJ_Crop")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from MRI2CBCT_CLI_utils import GetPatients

# ── CONFIG ──────────────────────────────────────────────────────────
DATASET      = "Dataset001_myseg"
CONFIG       = "3d_fullres"
PLAN         = "nnUNetResEncUNetXLPlans"
MARGIN       = 3                          # voxels besides B-box
PROBA_THR    = 0.02
FIXED_BBOX_VOXELS = [400, 400, 400]       # set to None to disable fixed box; otherwise [sx,sy,sz] in voxels

# ── Tools ──────────────────────────────────────────────────────────
def crop_with_affine(img: nib.Nifti1Image, start: np.ndarray, end: np.ndarray) -> nib.Nifti1Image:
    """Beneath-volume [start,end[ and keep world geometry."""
    data   = img.get_fdata()
    affine = img.affine.copy()
    sub    = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    affine[:3, 3] += affine[:3, :3] @ start          # move origin
    return nib.Nifti1Image(sub, affine)

def biggest_cc(mask: np.ndarray) -> np.ndarray:
    lbl, n = label(mask)
    if n <= 1: return mask
    vols = [(lbl == i).sum() for i in range(1, n+1)]
    return (lbl == 1+np.argmax(vols))

def nnunet_predict(case_dir: Path, out_dir: Path, model_folder: Path) -> None:
    os.environ['nnUNet_results'] = str(model_folder.parent.parent)

    subprocess.check_call([
        "nnUNetv2_predict",
        "-i", str(case_dir),
        "-o", str(out_dir),
        "-d", DATASET,
        "-c", CONFIG,
        "-p", PLAN,
        "--disable_tta",
        "--save_probabilities",
        "-f", "0"
    ])

    
def process_patient(cbct_path: Path, mri_path: Path, seg_path: Optional[Path], tmp_dir: Path, out_dir: Path, model_folder: Path) -> None:
    def crop_by_world_corners(img: nib.Nifti1Image, corners_world: np.ndarray) -> tuple[Optional[nib.Nifti1Image], Optional[tuple[np.ndarray, np.ndarray]]]:
        """
        Crop `img` with the B-box determined by 8 edges in world coordinates
        Return (volume_crop, (imin, imax)) ou (None, None) if no intersection.
        """
        inv   = np.linalg.inv(img.affine)
        ijk   = (inv @ np.c_[corners_world, np.ones(8)].T)[:3].T
        imin  = np.floor(ijk.min(0)).astype(int)
        imax  = np.ceil (ijk.max(0)).astype(int) + 1
        imin  = np.maximum(imin, 0)
        imax  = np.minimum(imax, img.shape)
        if np.any(imax <= imin):
            return None, None
        return crop_with_affine(img, imin, imax), (imin, imax)

    def _save(vol: Optional[nib.Nifti1Image], file_name: str, folder_name: str):
            if vol is None:
                logger.info(f"{file_name}: No intersection (not save)")
            else:
                if not os.path.exists(os.path.join(out_dir,folder_name)):
                    os.makedirs(os.path.join(out_dir,folder_name))
                nib.save(vol, os.path.join(os.path.join(out_dir,folder_name),file_name))

    
    ### ------------------------------------------------------------------ ###
    
    
    name = cbct_path.stem.split(".")[0]
    name = name.split("_")[0]
    logger.info(f"\n{name}")

    cbct = nib.load(cbct_path)
    mri  = nib.load(mri_path)

    
    ### ------------------------------------------------------------------ ###
    
    
    mid      = cbct.shape[0] // 2
    cog      = np.array(np.nonzero(mri.get_fdata() > 0)).mean(axis=1)
    cog_w    = mri.affine[:3, :3] @ cog + mri.affine[:3, 3]
    cog_cbct = np.linalg.inv(cbct.affine)[:3, :3] @ cog_w + \
               np.linalg.inv(cbct.affine)[:3, 3]
    side = "Left" if cog_cbct[0] < mid else "Right"
    logger.info(f"MRI side: {side}")

    
    ### ------------------------------------------------------------------ ###


    if side == "Left":
        start_half = np.array([0, 0, 0])
        end_half   = np.array([mid, *cbct.shape[1:]])
    else:
        start_half = np.array([mid, 0, 0])
        end_half   = np.array(cbct.shape)
    cbct_half = crop_with_affine(cbct, start_half, end_half)

    case_dir = tmp_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    for f in case_dir.glob("*.nii.gz"):
        f.unlink()
    nib.save(cbct_half, case_dir / f"{name}_0000.nii.gz")


    ### ------------------------------------------------------------------ ###
    
    
    pred_dir = case_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    nnunet_predict(case_dir, pred_dir, model_folder)
    logger.info("Prediction done")

    pred = nib.load(pred_dir / f"{name}.nii.gz").get_fdata()
    if pred.ndim == 4:
        pred = np.argmax(pred, 0)

    mask = pred if pred.max() > 1 else (pred > PROBA_THR)
    mask = biggest_cc(mask)
    # save mask for inspection (in cbct_half voxel space)
    try:
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), cbct_half.affine)
        _save(mask_img, f"{name}_Mask_TMJ_{side}.nii.gz", "Mask")
    except Exception:
        # best-effort save, do not fail the pipeline
        pass
    if mask.sum() == 0:
        logger.info("No voxel ignored")
        return

    nz   = np.array(np.nonzero(mask))
    pmin = np.maximum(nz.min(axis=1) - MARGIN, 0)
    pmax = np.minimum(nz.max(axis=1) + 1 + MARGIN, mask.shape)

    # If a fixed box size is requested, center that box on the mask centroid
    if FIXED_BBOX_VOXELS is not None:
        center = np.round(nz.mean(axis=1)).astype(int)
        size = np.array(FIXED_BBOX_VOXELS)
        half = size // 2
        pmin = center - half
        pmin = np.maximum(pmin, 0)
        pmax = pmin + size
        # clamp to image and if clamped, re-adjust pmin to keep requested size when possible
        pmax = np.minimum(pmax, mask.shape)
        pmin = np.maximum(pmax - size, 0)


    ### ------------------------------------------------------------------ ###
    
    
    ijk_corners   = np.array(list(itertools.product(
                       [pmin[0], pmax[0] - 1],
                       [pmin[1], pmax[1] - 1],
                       [pmin[2], pmax[2] - 1])))
    world_corners = (cbct_half.affine
                     @ np.c_[ijk_corners, np.ones(8)].T)[:3].T


    ### ------------------------------------------------------------------ ###
    
    
    out_dir.mkdir(parents=True, exist_ok=True)

    cbct_crop, _      = crop_by_world_corners(cbct, world_corners)
    mri_crop,  bbox_m = crop_by_world_corners(mri,  world_corners)
    
    _save(mri_crop,  f"{name}_MRI_TMJ_crop{side}.nii.gz","MRI")


    ### ------------------------------------------------------------------ ###
    
    
    if mri_crop is not None:
        from nibabel.processing import resample_from_to
        # cible = (shape, affine) crop MRI
        cbct_on_mri = resample_from_to(cbct, (mri_crop.shape, mri_crop.affine), order=1)
        _save(cbct_on_mri, f"{name}_CBCT_TMJ_crop{side}.nii.gz","CBCT")
        pred_on_mri = resample_from_to(nib.load(seg_path),
                                       (mri_crop.shape, mri_crop.affine),
                                       order=0)
        _save(pred_on_mri, f"{name}_Seg_TMJ_crop{side}.nii.gz","CBCT seg")
        # also resample and save the binary mask as a label-map on the MRI grid (nearest)
        try:
            mask_img = nib.Nifti1Image(mask.astype(np.uint8), cbct_half.affine)
            mask_on_mri = resample_from_to(mask_img, (mri_crop.shape, mri_crop.affine), order=0)
            _save(mask_on_mri, f"{name}_Mask_TMJ_crop{side}.nii.gz","CBCT seg")
        except Exception:
            pass


# ── MAIN ────────────────────────────────────────────────────────────
def main(args):
    output_dir = Path(args.output_folder)
    tmp_folder = Path(args.tmp_folder)
    model_folder = Path(args.model_folder)
    
    patients = GetPatients(args.cbct_folder, args.mri_folder, args.seg_folder)

    for pid, files in sorted(patients.items()):
        cbct_path = Path(files.get("cbct", ""))
        mri_path  = Path(files.get("mri", ""))
        seg_path  = Path(files.get("seg", ""))

        if not cbct_path.exists() or not mri_path.exists() or not seg_path.exists():
            logger.info(f"Skipping {pid}: missing CBCT, MRI, or SEG")
            continue

        logger.info(f"\nPatient: {pid}")
        logger.info(f"   CBCT: {cbct_path}")
        logger.info(f"   MRI:  {mri_path}")
        logger.info(f"   SEG:  {seg_path}")

        process_patient(cbct_path, mri_path, seg_path, tmp_folder, output_dir, model_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cbct_folder", type=str, help="Input path CBCT folder")
    parser.add_argument("mri_folder", type=str, help="Input path MRI folder")
    parser.add_argument("seg_folder", type=str, help="Input path segmentation folder")
    parser.add_argument("output_folder", type=str, help="Output path folder")
    parser.add_argument("model_folder", type=str, help="Path to nnUNet model folder")
    parser.add_argument("tmp_folder", type=str, help="Temporary folder for nnUNet processing")
    args = parser.parse_args()

    main(args)
