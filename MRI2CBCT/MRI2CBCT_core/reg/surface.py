"""Surface points + outward normals from a binary mask, in world (RAS) space.

Lifted verbatim (except for the dropped `torch` import, which this function never
used) from PIPELINE/registration/snap_icp.py, the code that produced the published
Dataset516 registrations. numpy + scipy only, so it runs inside PythonSlicer.
"""
import numpy as np


def surface_points_normals(mask, affine, max_pts=8000, seed=0):
    """Surface voxels of a binary mask, with outward normals, in world space."""
    from scipy import ndimage
    m = mask > 0.5
    if m.sum() < 50:
        return None, None
    surf = m ^ ndimage.binary_erosion(m)
    # normal = gradient of a smoothed occupancy field
    g = ndimage.gaussian_filter(m.astype(np.float32), 1.5)
    gz, gy, gx = np.gradient(g)
    idx = np.argwhere(surf)
    if len(idx) > max_pts:
        idx = idx[np.random.RandomState(seed).choice(len(idx), max_pts, replace=False)]
    n_vox = np.stack([gz[tuple(idx.T)], gy[tuple(idx.T)], gx[tuple(idx.T)]], axis=1)
    R = affine[:3, :3]
    pts = (R @ idx.T.astype(float)).T + affine[:3, 3]
    # normals transform by the inverse transpose, then renormalise
    nrm = (np.linalg.inv(R).T @ n_vox.T).T
    L = np.linalg.norm(nrm, axis=1, keepdims=True)
    keep = L[:, 0] > 1e-6
    return pts[keep], (nrm[keep] / L[keep])


def largest_cc(arr):
    """Largest connected component of a boolean array (6-connectivity).

    MANDATORY before registration: global_reg.to_cloud() calls
    orient_normals_consistent_tangent_plane(20), which propagates normal
    orientation across a connectivity graph. Detached islands in the mask get
    their normals flipped against the main surface and the FPFH descriptors
    that follow are then computed on inconsistent normals.
    """
    from scipy import ndimage
    lab, n = ndimage.label(arr, ndimage.generate_binary_structure(3, 1))
    if n <= 1:
        return arr, n
    s = np.bincount(lab.ravel())
    s[0] = 0
    return lab == s.argmax(), n
