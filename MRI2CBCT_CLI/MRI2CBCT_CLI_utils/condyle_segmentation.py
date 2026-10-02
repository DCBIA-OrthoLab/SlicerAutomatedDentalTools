import os
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import label


# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("MRI2CBCT_CLI_utils_condyle_segmentation")

# nnUNet condyle segmentation model config, shared by TMJ-crop and Approximate.
DATASET   = "Dataset001_myseg"
CONFIG    = "3d_fullres"
PLAN      = "nnUNetResEncUNetXLPlans"
PROBA_THR = 0.02


def crop_with_affine(img: nib.Nifti1Image, start: np.ndarray, end: np.ndarray) -> nib.Nifti1Image:
    """Crop a volume to [start, end) while keeping its world geometry intact."""
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

    # nnUNetv2_predict defaults to "-device cuda"; on a machine with no usable
    # GPU it still tries to pin_memory() for a CUDA transfer and crashes with
    # "Cannot access accelerator device when none is available" instead of
    # falling back to CPU on its own.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = subprocess.run(
        [
            "nnUNetv2_predict",
            "-i", str(case_dir),
            "-o", str(out_dir),
            "-d", DATASET,
            "-c", CONFIG,
            "-p", PLAN,
            "--disable_tta",
            "--save_probabilities",
            "-f", "0",
            "-device", device,
        ],
        capture_output=True, text=True,
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        # subprocess.check_call's CalledProcessError only reports the exit
        # code, hiding the actual reason nnUNet failed - log+raise the real
        # stderr so it's visible in the caller's error message.
        logger.error(result.stderr or "nnUNetv2_predict failed with no stderr output")
        raise RuntimeError(
            f"nnUNetv2_predict failed (exit code {result.returncode}): "
            f"{(result.stderr or '').strip()[-2000:]}")


def detect_side_and_half(cbct: nib.Nifti1Image, mri: nib.Nifti1Image):
    """Decide whether the MRI sits on the Left or Right of the CBCT, in world
    space (so it doesn't depend on either scan's own voxel-axis direction),
    then crop the CBCT down to that half so it matches the field of view the
    condyle segmentation model was trained on.
    """
    mid = cbct.shape[0] // 2

    cog          = np.array(np.nonzero(mri.get_fdata() > 0)).mean(axis=1)
    cog_w        = mri.affine[:3, :3] @ cog + mri.affine[:3, 3]
    cbct_mid_ijk = np.array(cbct.shape) / 2.0
    cbct_mid_w   = cbct.affine[:3, :3] @ cbct_mid_ijk + cbct.affine[:3, 3]
    side = "Right" if cog_w[0] > cbct_mid_w[0] else "Left"

    axis0_step_world = cbct.affine[:3, 0]
    increasing_index_is_right = axis0_step_world[0] > 0
    if (side == "Right") == increasing_index_is_right:
        start_half = np.array([mid, 0, 0])
        end_half   = np.array(cbct.shape)
    else:
        start_half = np.array([0, 0, 0])
        end_half   = np.array([mid, *cbct.shape[1:]])
    cbct_half = crop_with_affine(cbct, start_half, end_half)
    return side, cbct_half


def segment_condyle(cbct_path, mri_path, case_dir: Path, model_folder: Path):
    """Run the same nnUNet condyle segmentation used by TMJ-crop on the half of
    the CBCT that contains the MRI's side.

    Returns (mask, cbct_half, side). `mask` may be all-zero (caller decides
    whether that counts as "not found"); `cbct_half` is needed by the caller to
    map mask voxel indices back to world coordinates.
    """
    cbct = nib.load(cbct_path)
    mri  = nib.load(mri_path)

    side, cbct_half = detect_side_and_half(cbct, mri)

    name = Path(cbct_path).stem.split(".")[0].split("_")[0]
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    for f in case_dir.glob("*.nii.gz"):
        f.unlink()
    nib.save(cbct_half, case_dir / f"{name}_0000.nii.gz")

    pred_dir = case_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    nnunet_predict(case_dir, pred_dir, model_folder)

    pred = nib.load(pred_dir / f"{name}.nii.gz").get_fdata()
    if pred.ndim == 4:
        pred = np.argmax(pred, 0)

    mask = pred if pred.max() > 1 else (pred > PROBA_THR)
    mask = biggest_cc(mask)

    return mask, cbct_half, side
