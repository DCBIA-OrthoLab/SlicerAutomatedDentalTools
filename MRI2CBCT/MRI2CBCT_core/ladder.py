"""One joint: MRI cranial-base mask -> CBCT cranial-base surface, with the FLIP ladder.

The first rung is the published Dataset516 registration, unchanged: largest
connected component of the MRI mask, CBCT surface cut to the scanned side with
a 5 mm margin past the midline, 50 FPFH+RANSAC+ICP draws, best inlier fitness
kept. Only the draws run in parallel.

If the result is a FLIP (pose rotation > 90 deg), the ladder climbs:
    1. more RANSAC draws, keeping the best result that does not flip
    2. the flipped pose turned 180 deg about each axis, refined by ICP
    3. the opposite side of the CBCT (the MRI side label may be wrong)
    4. the MRI mask mirrored, registered again (fixes a left-right handedness
       error in the file itself; a wrong mirror fits visibly badly)
and if nothing passes, the best result is written with a _FLAGGED suffix.
A rung is accepted when its result does not flip and its fitness reaches the
QC gate.
"""
import os
import sys
import multiprocessing as mp

import numpy as np

from .env import ensure_open3d
ensure_open3d()              # the reg package imports open3d at load time

from .reg.pipeline import NPTS, NDRAW, MARGIN, VOXEL, GATE, qc_tag
from .reg.surface import surface_points_normals, largest_cc

EXTRA_DRAWS = 100            # rung 1
THREADS_PER_WORKER = 4       # Open3D oversubscribes badly when unconstrained

# ----------------------------------------------------------------- workers
# Set by _init in each worker process, so the clouds are sent once, not per draw.
_W = {}


def _init(paths, Mp, Mn, Cp, Cn):
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    from MRI2CBCT_core.env import ensure_open3d
    ensure_open3d()
    _W.update(Mp=Mp, Mn=Mn, Cp=Cp, Cn=Cn)


def _draw(_):
    from MRI2CBCT_core.reg.global_reg import register
    T, fit, rmse = register(_W["Mp"], _W["Mn"], _W["Cp"], _W["Cn"], voxel=VOXEL)
    return np.asarray(T, float), float(fit), float(rmse)


def _icp(T0):
    import open3d as o3d
    from MRI2CBCT_core.reg.global_reg import to_cloud
    src = to_cloud(_W["Mp"], _W["Mn"], VOXEL)
    tgt = to_cloud(_W["Cp"], _W["Cn"], VOXEL)
    fine = o3d.pipelines.registration.registration_icp(
        src, tgt, VOXEL * 1.5, np.asarray(T0, float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return np.asarray(fine.transformation), float(fine.fitness), float(fine.inlier_rmse)


class _Pool:
    """Worker pool bound to one (MRI cloud, CBCT cloud) pair."""

    def __init__(self, Mp, Mn, Cp, Cn, workers):
        os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
        paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(workers, initializer=_init, initargs=(paths, Mp, Mn, Cp, Cn))

    def draws(self, n, progress=None):
        out = []
        for i, r in enumerate(self.pool.imap_unordered(_draw, range(n))):
            out.append(r)
            if progress:
                progress(i + 1, n)
        return out

    def icp(self, starts):
        return self.pool.map(_icp, starts)

    def close(self):
        self.pool.terminate()
        self.pool.join()


def default_workers():
    return max(1, min(NDRAW, (os.cpu_count() or 4) // THREADS_PER_WORKER))


# ----------------------------------------------------------------- geometry
def side_cut(Cp, Cn, side, margin=MARGIN):
    """The published hemisphere cut: keep CBCT points up to `margin` mm past the
    median x (RAS: +x is the patient's RIGHT)."""
    if side not in ("L", "R"):
        return Cp, Cn
    mid = float(np.median(Cp[:, 0]))
    sgn = 1.0 if side == "R" else -1.0
    keep = (Cp[:, 0] - mid) * sgn >= -margin
    return Cp[keep], Cn[keep]


def rot180(axis, centre):
    R = np.diag([1.0, 1.0, 1.0])
    for i in range(3):
        if i != axis:
            R[i, i] = -1.0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centre - R @ centre
    return T


def mirror_matrix(direction, centre):
    """Reflection across the plane through `centre` normal to `direction`."""
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    H = np.eye(3) - 2.0 * np.outer(n, n)
    T = np.eye(4)
    T[:3, :3] = H
    T[:3, 3] = centre - H @ centre
    return T


def apply(T, pts, nrm):
    return pts @ T[:3, :3].T + T[:3, 3], nrm @ T[:3, :3].T


def mri_surface(mask_path, npts=NPTS):
    """MRI cranial-base surface, published preprocessing. Returns dict or raises."""
    import nibabel as nib
    im = nib.load(str(mask_path))
    arr = np.asarray(im.dataobj) > 0.5
    if arr.sum() < 50:
        raise ValueError("the MRI cranial-base segmentation is empty")
    arr, ncc = largest_cc(arr)
    vox = float(np.prod(im.header.get_zooms()[:3]))
    Mp, Mn = surface_points_normals(arr, im.affine, npts)
    if Mp is None:
        raise ValueError("the MRI cranial-base segmentation is too small to register")
    # world direction of the index axis closest to the patient's left-right
    A = im.affine[:3, :3]
    lr = A[:, int(np.argmax(np.abs(A[0, :])))]
    return dict(Mp=Mp, Mn=Mn, n_components=int(ncc), lr_dir=lr,
                mask_cm3=round(float(arr.sum()) * vox / 1000.0, 2))


def cbct_surface(mask_path, npts=NPTS):
    import nibabel as nib
    im = nib.load(str(mask_path))
    a = np.asarray(im.dataobj) > 0.5
    Cp, Cn = surface_points_normals(a, im.affine, npts)
    if Cp is None:
        raise ValueError("the CBCT cranial-base segmentation is empty")
    return np.asarray(Cp, float), np.asarray(Cn, float)


# ----------------------------------------------------------------- ladder
def _verdict(T, fit, mask_cm3):
    tag, ang = qc_tag(T, fit, mask_cm3)
    return {"T": np.asarray(T, float), "fitness": float(fit), "qc": tag, "rotation_deg": ang}


def _passes(c):
    return c["qc"] != "FLIP" and c["fitness"] >= GATE


def register_joint(mri_mask_path, cbct_surf, side, workers=None, log=print, progress=None):
    """Returns a result dict. T is 4x4 RAS, MRI -> CBCT, and may be a reflection
    when result["mirrored"] is True."""
    workers = workers or default_workers()
    m = mri_surface(mri_mask_path)
    Cp_all, Cn_all = cbct_surf
    res = {"side_in": side, "side": side, "mask_cm3": m["mask_cm3"],
           "n_components": m["n_components"], "mirrored": False, "rungs": []}

    def run(Mp, Mn, s, n, label, mask_cm3=m["mask_cm3"], pre=None):
        Cp, Cn = side_cut(Cp_all, Cn_all, s)
        if len(Cp) < 500:
            log(f"  {label}: CBCT side cut left too few points")
            return []
        pool = _Pool(Mp, Mn, Cp, Cn, min(workers, n))
        try:
            draws = pool.draws(n, progress)
        finally:
            pool.close()
        out = []
        for T, fit, rmse in draws:
            c = _verdict(T, fit, mask_cm3)
            c["rmse"] = rmse
            if pre is not None:                   # mirrored: report the composed transform
                c["T"] = c["T"] @ pre
            out.append(c)
        return out

    def best(cands):
        return max(cands, key=lambda c: c["fitness"]) if cands else None

    def best_passing(cands):
        ok = [c for c in cands if _passes(c)]
        return best(ok)

    # rung 0: the published registration
    first = run(m["Mp"], m["Mn"], side, NDRAW, "registration")
    if not first:
        res.update(status="failed", reason="no registration could be computed")
        return res
    top = best(first)
    res["fitness_median"] = float(np.median([c["fitness"] for c in first]))
    res["rungs"].append({"rung": "published", "fitness": top["fitness"], "qc": top["qc"],
                         "rotation_deg": top["rotation_deg"]})
    log(f"  registration: fitness {top['fitness']:.3f}, rotation {top['rotation_deg']:.1f} deg, {top['qc']}")
    if top["qc"] != "FLIP":
        return _finish(res, top, "published")

    flipped = top
    pool_all = list(first)

    # rung 1: more draws
    log(f"  FLIP detected: {EXTRA_DRAWS} more RANSAC draws")
    pool_all += run(m["Mp"], m["Mn"], side, EXTRA_DRAWS, "more draws")
    c = best_passing(pool_all)
    res["rungs"].append(_rung("more_draws", c))
    if c:
        return _finish(res, c, "more_draws")

    # rung 2: turn the flipped pose 180 deg about each axis, refine with ICP
    log("  still flipped: trying the flipped pose turned 180 deg, refined by ICP")
    centre = m["Mp"].mean(axis=0)
    starts = [flipped["T"] @ rot180(a, centre) for a in range(3)]
    Cp, Cn = side_cut(Cp_all, Cn_all, side)
    pool = _Pool(m["Mp"], m["Mn"], Cp, Cn, 3)
    try:
        refined = [_verdict(T, fit, m["mask_cm3"]) for T, fit, _ in pool.icp(starts)]
    finally:
        pool.close()
    c = best_passing(refined)
    res["rungs"].append(_rung("rotate_180", c))
    if c:
        return _finish(res, c, "rotate_180")

    # rung 3: the other side of the CBCT
    if side in ("L", "R"):
        other = "R" if side == "L" else "L"
        log(f"  still flipped: trying the other side of the CBCT ({other})")
        cands = run(m["Mp"], m["Mn"], other, NDRAW, "other side")
        c = best_passing(cands)
        res["rungs"].append(_rung("other_side", c))
        if c:
            res["side"] = other
            log(f"  the joint registers on side {other}: side relabelled {side} -> {other}")
            return _finish(res, c, "other_side")

    # rung 4: mirror the MRI mask and register again
    log("  still flipped: trying the MRI mirrored left-right")
    M = mirror_matrix(m["lr_dir"], centre)
    Mp_m, Mn_m = apply(M, m["Mp"], m["Mn"])
    # run() scores QC on the rigid registration of the mirrored points, then
    # composes the reflection in: qc_tag cannot read a rotation off a reflection
    cands = run(Mp_m, Mn_m, side, NDRAW, "mirrored", pre=M)
    c = best_passing(cands)
    res["rungs"].append(_rung("mirrored", c))
    if c:
        res["mirrored"] = True
        return _finish(res, c, "mirrored")

    log("  unresolved: writing the best registration with _FLAGGED")
    res["flagged"] = True
    return _finish(res, flipped, "unresolved")


def _rung(name, c):
    if c is None:
        return {"rung": name, "passed": False}
    return {"rung": name, "passed": True, "fitness": c["fitness"], "qc": c["qc"],
            "rotation_deg": c["rotation_deg"]}


def _finish(res, c, how):
    res.update(status="ok", T=c["T"].tolist(), fitness=round(c["fitness"], 4),
               qc=c["qc"], rotation_deg=c["rotation_deg"], resolved_by=how)
    res.setdefault("flagged", False)
    return res
