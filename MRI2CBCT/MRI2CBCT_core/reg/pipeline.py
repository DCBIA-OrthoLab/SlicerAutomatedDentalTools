"""One joint in, one transform + QC verdict out.

Reproduces run_02_register_w48.py / register_leakfree_noCC.py exactly: largest
connected component on the MRI mask, CBCT target cut to the scanned side with a
5 mm margin, NDRAW independent RANSAC draws keeping the highest inlier fitness.
"""
import numpy as np

from .surface import surface_points_normals, largest_cc
from .global_reg import register

# Published settings. Changing any of these makes the result not comparable to
# the Dataset516 numbers.
NPTS = 20000      # surface points sampled per mask
NDRAW = 50        # RANSAC restarts, best inlier fitness kept
MARGIN = 5.0      # mm kept past the midline when cutting the CBCT to one side
VOXEL = 1.5       # mm downsample; FPFH radius = 5*VOXEL, correspondence = 1.5*VOXEL

# Reference-free QC thresholds, as used for the published verdict tags.
GATE = 0.75       # fitness below this -> FLAG
MAXROT = 90.0     # pose rotation above this -> FLIP (impossible for one head)
SMALL = 11.21     # mask cm^3 below this -> SMALL (every flip came from this group,
                  # 5/46 vs 0/172, Fisher p=3.5e-04) -- a PRE-registration warning


def qc_tag(T, fitness, mask_cm3):
    """FLIP / FLAG / SMALL / OK. All three criteria need no reference transform.

    FLIP is the only one that caught the laterality failures: fitness did not.
    """
    T = np.asarray(T, float)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    if ang > MAXROT:
        tag = "FLIP"
    elif fitness < GATE:
        tag = "FLAG"
    elif mask_cm3 is not None and mask_cm3 < SMALL:
        tag = "SMALL"
    else:
        tag = "OK"
    return tag, round(ang, 2)


def cbct_surface(cbct_mask_path, npts=NPTS):
    """CBCT cranial-base surface points + normals in RAS. Cache this per patient:
    it is reused by both joints and is the slow part."""
    import nibabel as nib
    im = nib.load(str(cbct_mask_path))
    a = np.asarray(im.dataobj) > 0.5
    return surface_points_normals(a, im.affine, npts)


def register_joint(mri_mask_path, cbct_surf, side=None, ndraw=NDRAW,
                   npts=NPTS, margin=MARGIN, progress=None):
    """Register one MRI cranial-base mask onto a CBCT cranial-base surface.

    mri_mask_path : native-space MRI cranial-base mask (nnU-Net Dataset516 output)
    cbct_surf     : (points, normals) from cbct_surface(), or a path to the CBCT mask
    side          : 'R', 'L' or None. Cuts the CBCT to the scanned hemisphere, which
                    is what makes a single-joint MRI crop registrable against a
                    full-head CBCT. None uses the whole CBCT surface.
    progress      : optional callable(i, ndraw) for a Slicer progress bar.

    Returns a dict: status, T (4x4 RAS, MRI->CBCT), fitness, fitness_median,
    mask_cm3, qc, rotation_deg. status != 'ok' means no transform was produced.
    """
    import nibabel as nib

    if not isinstance(cbct_surf, (tuple, list)):
        cbct_surf = cbct_surface(cbct_surf, npts)
    Cp, Cn = (np.asarray(x, dtype=np.float64) for x in cbct_surf)

    r = {"mri_mask": str(mri_mask_path), "side": side}
    im = nib.load(str(mri_mask_path))
    arr = np.asarray(im.dataobj) > 0.5
    if arr.sum() < 50:
        r["status"] = "empty_mask"
        return r

    arr, ncc = largest_cc(arr)
    r["n_components"] = int(ncc)
    vox = float(np.prod(im.header.get_zooms()[:3]))
    r["mask_cm3"] = round(float(arr.sum()) * vox / 1000.0, 2)

    Mp, Mn = surface_points_normals(arr, im.affine, npts)
    if Mp is None:
        r["status"] = "mask_too_small"
        return r

    if side in ("R", "L"):
        mid = float(np.median(Cp[:, 0]))
        sgn = 1.0 if side == "R" else -1.0
        keep = (Cp[:, 0] - mid) * sgn >= -margin
        if keep.sum() < 500:
            r["status"] = "side_crop_too_small"
            return r
        Cp, Cn = Cp[keep], Cn[keep]
    r["cbct_points"] = int(len(Cp))

    best = (-1.0, None)
    fits = []
    for i in range(ndraw):
        T, fit, rmse = register(Mp, Mn, Cp, Cn, voxel=VOXEL)
        fits.append(float(fit))
        if fit > best[0]:
            best = (float(fit), np.asarray(T, dtype=float), float(rmse))
        if progress:
            progress(i + 1, ndraw)
    if best[1] is None:
        r["status"] = "no_solution"
        return r

    tag, ang = qc_tag(best[1], best[0], r["mask_cm3"])
    r.update(status="ok", T=best[1].tolist(), fitness=round(best[0], 4),
             inlier_rmse=round(best[2], 4),
             fitness_median=round(float(np.median(fits)), 4),
             qc=tag, rotation_deg=ang)
    return r
