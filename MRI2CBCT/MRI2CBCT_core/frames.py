"""Transforms between the frames a run produces.

Conventions (one place, so the rest of the code never guesses):
  * RAS 4x4 "point maps": T maps a point of frame A to the same anatomy in B.
  * ITK / SimpleITK .tfm files are LPS and store the RESAMPLING direction,
    i.e. the point map from the fixed (output) frame back to the moving one.

ASO's <PID>_Or_transform.tfm maps oriented-frame points back to the ORIGINAL
CBCT (resampling convention: applied in Slicer to the original scan, it lands
in the oriented pose).
"""
import numpy as np
import SimpleITK as sitk

L = np.diag([-1.0, -1.0, 1.0, 1.0])


def affine_of(transform):
    """4x4 LPS point map of any linear SimpleITK transform (composites included),
    read off by transforming points rather than by trusting parameter layouts."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * 100.0
    out = np.array([transform.TransformPoint(p.tolist()) for p in pts])
    A = np.eye(4)
    A[:3, 3] = out[0]
    A[:3, :3] = ((out[1:] - out[0]) / 100.0).T
    return A


def read_tfm_lps(path):
    return affine_of(sitk.ReadTransform(str(path)))


def aso_original_to_oriented_ras(aso_tfm_path):
    """RAS point map: original CBCT -> ASO-oriented CBCT."""
    oriented_to_original_lps = read_tfm_lps(aso_tfm_path)
    return L @ np.linalg.inv(oriented_to_original_lps) @ L


def to_itk_resampling(T_ras):
    """RAS point map moving->fixed  ->  SimpleITK AffineTransform for resampling
    the moving image onto the fixed grid (and for writing the .tfm Slicer reads)."""
    Ti = L @ np.linalg.inv(np.asarray(T_ras, float)) @ L
    a = sitk.AffineTransform(3)
    a.SetMatrix(Ti[:3, :3].ravel().tolist())
    a.SetTranslation(Ti[:3, 3].tolist())
    return a


def write_tfm(T_ras, path):
    sitk.WriteTransform(to_itk_resampling(T_ras), str(path))


def move_header(in_path, T_ras, out_path, label=False):
    """Put an image where T_ras sends it by rewriting its origin and direction.

    No voxel is touched: the MRI keeps its own resolution and opens on top of
    the CBCT in any viewer that reads the header. Works for reflections too
    (a mirrored result has a left-handed direction matrix, which NIfTI stores).
    """
    img = sitk.ReadImage(str(in_path))
    P = L @ np.asarray(T_ras, float) @ L                  # LPS point map
    D = np.array(img.GetDirection()).reshape(3, 3)
    img.SetDirection((P[:3, :3] @ D).ravel().tolist())
    img.SetOrigin((P[:3, :3] @ np.array(img.GetOrigin()) + P[:3, 3]).tolist())
    if label:
        img = sitk.Cast(img > 0, sitk.sitkUInt8)
    sitk.WriteImage(img, str(out_path), True)
    return str(out_path)


def resample(moving_path, reference_path, T_ras, out_path, label=False):
    """Resample `moving` onto `reference`'s grid; T_ras maps moving -> reference."""
    ref = sitk.ReadImage(str(reference_path))
    mov = sitk.ReadImage(str(moving_path))
    interp = sitk.sitkNearestNeighbor if label else sitk.sitkLinear
    out = sitk.Resample(mov, ref, to_itk_resampling(T_ras), interp, 0.0,
                        sitk.sitkUInt8 if label else mov.GetPixelID())
    sitk.WriteImage(out, str(out_path), True)
    return str(out_path)
