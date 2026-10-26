"""RAS <-> ITK/LPS conversion and .tfm writing.

global_reg.register() returns a 4x4 that maps MRI world points to CBCT world
points in nibabel's RAS convention. Slicer transform nodes and ITK .tfm files
are LPS, and an ITK transform stores the RESAMPLING direction (fixed -> moving),
i.e. the inverse. Both corrections are applied here:

    A_itk = L @ inv(T_ras) @ L,   L = diag(-1, -1, 1)

This is the same conversion used by run_03_export.py for the published
Dataset516 transforms, so a .tfm written here is interchangeable with those.
"""
import numpy as np

FLIP = np.diag([-1.0, -1.0, 1.0])


def ras_to_itk(T):
    """4x4 RAS MRI->CBCT transform -> SimpleITK AffineTransform (LPS, resampling)."""
    import SimpleITK as sitk
    Ti = np.linalg.inv(np.asarray(T, float))
    a = sitk.AffineTransform(3)
    a.SetMatrix((FLIP @ Ti[:3, :3] @ FLIP).ravel().tolist())
    a.SetTranslation((FLIP @ Ti[:3, 3]).tolist())
    return a


def write_tfm(T, path):
    """Write the RAS transform as an ITK .tfm. Load the NATIVE MRI in Slicer and
    set this as its transform node -- full native resolution, nothing
    interpolated, reversible."""
    import SimpleITK as sitk
    sitk.WriteTransform(ras_to_itk(T), str(path))
    return str(path)


def resample_onto(moving_path, fixed_path, T, out_path, label=False):
    """Resample a volume onto the fixed image's grid using the RAS transform.

    DERIVED data -- putting a 2.75 mm-slice MRI on a 0.3-0.5 mm CBCT grid
    interpolates and cannot be undone. Prefer the .tfm for anything quantitative.
    """
    import SimpleITK as sitk
    tf = ras_to_itk(T)
    fix = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    mov = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    interp = sitk.sitkNearestNeighbor if label else sitk.sitkLinear
    out = sitk.Resample(mov, fix, tf, interp, 0.0)
    out = sitk.Cast(out > 0.5, sitk.sitkUInt8) if label else sitk.Cast(out, sitk.sitkInt16)
    sitk.WriteImage(out, str(out_path), True)
    return str(out_path)


def slicer_matrix(T):
    """The same transform as a 4x4 list for vtkMRMLLinearTransformNode.SetMatrixTransformToParent.

    Slicer's transform node also stores fixed->moving in LPS, so this is the
    identical matrix the .tfm holds.
    """
    Ti = np.linalg.inv(np.asarray(T, float))
    M = np.eye(4)
    M[:3, :3] = FLIP @ Ti[:3, :3] @ FLIP
    M[:3, 3] = FLIP @ Ti[:3, 3]
    return M.tolist()
