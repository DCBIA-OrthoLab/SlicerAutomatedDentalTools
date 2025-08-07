#!/usr/bin/env python-real

import os
import argparse, subprocess, shutil, sys, itertools
from nnunetv2.inference.predict_from_raw_data import predict_entry_point
from typing import Optional
from pathlib import Path
import numpy as np
import nibabel as nib
from  scipy.ndimage import label         # garder la plus grosse CC
fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from MRI2CBCT_CLI_utils import GetPatients

# ── CONFIG ──────────────────────────────────────────────────────────
DATASET      = "Dataset001_myseg"
CONFIG       = "3d_fullres"
PLAN         = "nnUNetResEncUNetXLPlans"
MARGIN       = 3                          # voxels autours B-box
PROBA_THR    = .5

# ── OUTILS ──────────────────────────────────────────────────────────
def crop_with_affine(img: nib.Nifti1Image, start: np.ndarray, end: np.ndarray) -> nib.Nifti1Image:
    """Sous-volume [start,end[ en conservant la géométrie monde."""
    data   = img.get_fdata()
    affine = img.affine.copy()
    sub    = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    affine[:3, 3] += affine[:3, :3] @ start          # décale l’origine
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
        Coupe `img` avec la B-box définie par 8 coins en coordonnées monde.
        Retourne (volume_coupé, (imin, imax)) ou (None, None) si intersection nulle.
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

    def _save(vol: Optional[nib.Nifti1Image], fname: str):
        if vol is None:
            print(f"  ↳ {fname}: intersection nulle (non sauvegardé)")
        else:
            nib.save(vol, out_dir / fname)

    
    ### ------------------------------------------------------------------ ###
    
    
    name = cbct_path.stem.split(".")[0]
    print(f"\n▶ {name}")

    cbct = nib.load(cbct_path)
    mri  = nib.load(mri_path)

    
    ### ------------------------------------------------------------------ ###
    
    
    mid      = cbct.shape[0] // 2
    cog      = np.array(np.nonzero(mri.get_fdata() > 0)).mean(axis=1)
    cog_w    = mri.affine[:3, :3] @ cog + mri.affine[:3, 3]
    cog_cbct = np.linalg.inv(cbct.affine)[:3, :3] @ cog_w + \
               np.linalg.inv(cbct.affine)[:3, 3]
    side = "Left" if cog_cbct[0] < mid else "Right"
    print("MRI side:", side)

    
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
    print("✅ prediction done")

    pred = nib.load(pred_dir / f"{name}.nii.gz").get_fdata()
    if pred.ndim == 4:
        pred = np.argmax(pred, 0)

    mask = pred if pred.max() > 1 else (pred > PROBA_THR)
    mask = biggest_cc(mask)
    if mask.sum() == 0:
        print("⚠️  no voxel — ignored")
        return

    nz   = np.array(np.nonzero(mask))
    pmin = np.maximum(nz.min(axis=1) - MARGIN, 0)
    pmax = np.minimum(nz.max(axis=1) + 1 + MARGIN, mask.shape)


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
    
    # _save(cbct_crop, f"{name}_CBCT_cropped.nii.gz")
    _save(mri_crop,  f"{name}_MRI_crop{side}.nii.gz")

    # if seg_path and seg_path.exists():
    #     seg_crop, _ = crop_by_world_corners(nib.load(seg_path), world_corners)
    #     _save(seg_crop, f"{name}_CBCT_Seg_cropped.nii.gz")


    ### ------------------------------------------------------------------ ###
    
    
    if mri_crop is not None:
        from nibabel.processing import resample_from_to
        # cible = (shape, affine) du MRI croppé
        cbct_on_mri = resample_from_to(cbct, (mri_crop.shape, mri_crop.affine), order=1)
        _save(cbct_on_mri, f"{name}_CBCT_crop{side}.nii.gz")

        pred_on_mri = resample_from_to(nib.load(seg_path),
                                       (mri_crop.shape, mri_crop.affine),
                                       order=0)
        _save(pred_on_mri, f"{name}_Seg_crop{side}.nii.gz")

    print("── finished.")


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
            print(f"❌ Skipping {pid}: missing CBCT, MRI, or SEG")
            continue

        print(f"\n▶ Patient: {pid}")
        print(f"   CBCT: {cbct_path}")
        print(f"   MRI:  {mri_path}")
        print(f"   SEG:  {seg_path}")

        process_patient(cbct_path, mri_path, seg_path, tmp_folder, output_dir, model_folder)

    # Optionally clean temp dir
    # shutil.rmtree(tmp_folder)

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
