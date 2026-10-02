#!/usr/bin/env python-real

import os
import sys
import argparse

import pyvista as pv
import SimpleITK as sitk
import numpy as np
import json
import vtk

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

# --- LOGGING CONFIGURATION ---
from ADTLib.logging_setup import get_logger
import re

logger = get_logger("AREG_IOSCBCT")

# vtkLandmarkTransform needs as many source points as target points: given
# anything else it logs an error on stderr and returns the identity, so the
# mesh is never pre-aligned and nothing says so in the log. ALI_CBCT regularly
# finds only part of the landmarks, so the IOS and CBCT lists match neither in
# size nor in content, and pairing them by position would align landmarks that
# have nothing to do with each other. Only keep what exists on both sides.
MIN_LANDMARK_PAIRS = 3
MAX_LANDMARK_RESIDUAL_MM = float(os.environ.get("AREG_MAX_LANDMARK_RESIDUAL", 10.0))

# How flat a landmark set is allowed to be, as the ratio of its second spread to
# its first. Landmarks strung out along a line leave the rotation about that
# line to be decided by nothing but their noise, and the residual does not show
# it: three points always fit three points exactly, whichever way the arch ends
# up facing. A full arch measures about 0.32 here, half an arch 0.15, and three
# anterior landmarks 0.05 to 0.09.
MIN_LANDMARK_SPREAD_RATIO = 0.10

# An ICP correspondence is kept when the two surfaces face the same way there.
# In a closed bite the opposing crowns sit 1 to 3 mm apart, well inside the
# capture radius, but they face each other: the maxillary occlusal surface
# points down and the mandibular one points up. Without this test a maxillary
# IOS that starts a little low locks onto the mandible and reports a perfect
# fitness while sitting millimetres from the truth.
MIN_NORMAL_AGREEMENT = float(os.environ.get("AREG_MIN_NORMAL_AGREEMENT", 0.5))

# A registration that found next to nothing to match is not a registration. The
# floor is deliberately low: an IOS carries gingiva that no CBCT surface answers
# to, so a healthy run does not match all of its points either.
MIN_ICP_FITNESS = float(os.environ.get("AREG_MIN_ICP_FITNESS", 0.05))

# How far around an arch's own CBCT landmarks the CBCT surface is kept as a
# registration target, in mm. The isosurface of a head at 0.3 mm spacing is
# millions of points, nearly all of them cranial base, vertebrae and jaw body
# that no intraoral scan can ever match; every ICP iteration was querying the
# lot. The margin has to clear the crowns, the gingiva the IOS carries, and
# whatever the landmark pre-alignment left on the table -- its residual runs to
# a few mm on real patients -- so it is set well past all three.
CBCT_CROP_MARGIN_MM = float(os.environ.get("AREG_CBCT_CROP_MARGIN", 25.0))

# The isosurface the IOS is registered onto used to be taken at a fixed 400,
# which is not the enamel: at that level the contour runs through the
# partial-volume halo around the crowns and on into the alveolar bone, so the
# ICP matched the IOS against a surface some way outside the teeth and slid the
# arch along it. Measured on a patient whose landmarks agree to 1.4 mm, a
# threshold of 400 moved the arch 2.2 mm away from them while improving the
# median surface distance by 0.09 mm; nearer the enamel edge it moves it 0.7 mm
# and keeps the landmark agreement it started with.
#
# So the level is read off the scan instead, by the half-maximum rule: the
# boundary of a structure lies halfway between its own plateau and that of what
# surrounds it, which is where a contour localises the edge best. The plateau is
# measured at the CBCT landmarks, which sit on the crowns by construction, and
# the surround from the soft tissue around the arch.
#
# Measuring both ends on the image is what makes this work on a scan that is not
# calibrated: plenty of CBCTs come in arbitrary grey values rather than HU, and
# any fixed number -- 400 or 1200 -- is meaningless on those. A ratio of two
# levels taken from the same image is not.
SURFACE_THRESHOLD = os.environ.get("AREG_CBCT_SURFACE_THRESHOLD")

# The window, in voxels either side, over which the enamel plateau is read at
# each landmark. Wide enough to survive a landmark a voxel or two off the crown,
# narrow enough not to reach the next tooth.
ENAMEL_PROBE_RADIUS_VOXELS = 2

# What counts as soft tissue when the surround is measured. Water sits at 0 and
# cancellous bone starts well above 300, so this band holds mucosa, gingiva and
# muscle without letting bone into the median.
SOFT_TISSUE_BAND = (-300.0, 300.0)

# The Universal_ID value CrownSegmentation gives a point that belongs to no
# tooth. Teeth are 1 to 32, so 33 is the gingiva and 0 is unlabelled.
GINGIVA_LABELS = (0, 33)

# Below this share of the arch left as crowns, the labels are not to be trusted
# and the whole surface is registered instead. A healthy segmentation leaves
# about 40% of an arch as crowns.
MIN_CROWN_SHARE = 0.10

# How much further from its CBCT landmarks the ICP may leave an arch than the
# pre-alignment did, in mm of RMS, before the run says so. The landmarks are the
# only check on the ICP that the ICP does not grade itself: its own fitness and
# RMSE are computed over the correspondences it chose to keep, so a fit that has
# slid along a smooth occlusal surface reports the same healthy numbers as one
# that has not. Drifting away from twelve independently placed points while
# claiming to improve is the signature of that slide, and it is what a 400
# threshold did on every patient without anything noticing.
MAX_ICP_LANDMARK_DRIFT_MM = float(os.environ.get("AREG_MAX_ICP_LANDMARK_DRIFT", 1.0))

# A rigid fit spreads one bad landmark over all the others: least squares cannot
# tell "this point is wrong" from "everything is a little off", so it splits the
# difference and tilts the whole arch. On a real patient ALI_CBCT put UR3O about
# 10 mm from where the IOS has it, and that single point took the upper arch
# from 2.8 mm to 4.8 mm -- the ICP then started from 2 mm further out than it
# needed to.
#
# So a pair the fit is measurably better without is dropped, and the fit redone.
#
# The test is what removing the pair does to the fit, not how far out the pair
# looks: the least squares that produced the residuals has already spread the
# bad point over the good ones, so on this patient UR3O came out at 8.3 mm
# against a median of 3.5 -- under any sane multiple of it, while being the
# whole problem. Its removal, on the other hand, takes the arch from 4.8 mm to
# 2.8 mm, which nothing else comes near.
#
# Both conditions are needed. A pair must be far enough out to be wrong rather
# than noisy, and its removal must actually buy something; either alone strips
# points off a fit that was never in trouble.
LANDMARK_OUTLIER_IMPROVEMENT = float(os.environ.get("AREG_LANDMARK_OUTLIER_IMPROVEMENT", 0.75))
LANDMARK_OUTLIER_FLOOR_MM = float(os.environ.get("AREG_LANDMARK_OUTLIER_FLOOR", 3.0))

# Never trimmed below this many pairs. Three is what a rigid fit needs, so
# stopping at four leaves it one pair of redundancy rather than none: at three
# the fit is exact whatever the points are, and a bad one can no longer show up
# as a residual at all.
MIN_LANDMARK_PAIRS_AFTER_TRIM = 4

# A CBCT landmark is meant to sit on an occlusal surface, so the scan around it
# has to hold a tooth. Where the neighbourhood never reaches the level the scan
# is contoured at, there is no enamel there at all and the point is not on a
# tooth -- the ICP has no surface to match there either, so the pre-alignment is
# fitted without it.
#
# This catches a failure the residual cannot. ALI_CBCT places occlusal points,
# and in a closed bite the two occlusal surfaces sit 1 to 3 mm apart and look
# alike; nothing in it tells them apart, so one arch's points can land on the
# other jaw. Measured on one patient: UR3O came back in soft tissue at 316 and
# UL3O on bone at 1050, against a contour level of 1072, both of them below the
# whole of the opposing arch. Dropping the two took the upper from 4.8 mm to
# 2.7 mm, while the lower -- whose six points were all on enamel -- was left
# exactly as it was.
#
# Read from the image rather than from the landmark set, so a set that is
# largely wrong cannot vouch for itself the way a median-based test lets it.
CHECK_LANDMARKS_ON_ENAMEL = os.environ.get("AREG_CHECK_LANDMARKS_ON_ENAMEL", "1") != "0"

# When the arch has stopped moving, in mm of the furthest-travelling point over
# one iteration, and for how many iterations in a row.
#
# The ICP used to stop on the RMSE and the fitness both changing by less than
# 1e-8, which on a real patient never happens: it settles within about sixty
# iterations and then circles the answer forever, each turn moving the scan by
# some 3e-5 to 1e-4 mm. It spent the remaining 1900-odd iterations of its cap
# doing that, around a hundred seconds per arch per reading of the normals. A
# micron is three orders of magnitude below the 0.3 mm voxels this is
# registered against, so stopping there costs nothing anyone could measure.
ICP_SETTLED_SHIFT_MM = float(os.environ.get("AREG_ICP_SETTLED_SHIFT", 1e-3))
ICP_SETTLED_ITERATIONS = 3

# The cap, for the case where it never settles. Real arches settle in twenty to
# sixty iterations from a pre-alignment several mm out; what is still moving
# after three hundred is circling the answer, not approaching it, and stopping
# it there costs nothing but the circling. The reading of the normals that
# loses used to spend the whole of a 2000 cap doing exactly that.
ICP_MAX_ITERATIONS = int(os.environ.get("AREG_ICP_MAX_ITERATIONS", 300))


def _labeled_landmarks(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    out = {}
    if data.get("markups"):
        for cp in data["markups"][0].get("controlPoints", []):
            label = cp.get("label")
            if label:
                out[label] = list(cp["position"])
    return out


def _pair_landmarks(cbct_json, ios_json, jaw, patient_id):
    cbct = _labeled_landmarks(cbct_json)
    ios = _labeled_landmarks(ios_json)
    common = [l for l in ios if l in cbct]
    missing = [l for l in ios if l not in cbct]
    logger.info("%s / %s: %d common landmark pair(s) %s%s" % (
        patient_id, jaw, len(common), common,
        "  |  missing from the CBCT: %s" % missing if missing else ""))
    return (common,
            np.array([cbct[l] for l in common], dtype=float).reshape(-1, 3),
            np.array([ios[l] for l in common], dtype=float).reshape(-1, 3))


def _landmark_spread_ratio(landmarks):
    """Second spread over first: 1 is a disc, 0 is a straight line."""
    if len(landmarks) < 3:
        return 0.0
    centred = np.asarray(landmarks, dtype=float)
    centred = centred - centred.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return float(singular[1] / singular[0]) if singular[0] > 0 else 0.0


def _alignment_residual(moving_lms, fixed_lms, matrix):
    """How far each moved landmark still sits from its counterpart, as an RMS.

    A rigid transform preserves distances, so a large residual proves the two
    sets do not describe the same anatomy and the transform fitted to them is a
    meaningless compromise.
    """
    res = []
    for src, dst in zip(moving_lms, fixed_lms):
        moved = matrix.MultiplyPoint([src[0], src[1], src[2], 1])[:3]
        res.append(np.linalg.norm(np.array(moved) - np.array(dst)))
    return float(np.sqrt(np.mean(np.square(res)))) if res else float("inf")


def _report_landmark_drift(aligned_lms, fixed_lms, matrix, jaw, patient_id, kept=None):
    """Say whether the ICP moved the arch towards its landmarks or away.

    Measured over the pairs the pre-alignment was fitted on, so the two numbers
    are the same measurement taken twice. Including a pair the pre-alignment
    rejected would compare the ICP against a residual nothing tried to minimise,
    and report a drift on an arch that had not moved.

    Returns the two residuals so a caller can act on them; the judgement itself
    is only logged, because a registration can legitimately trade a little
    landmark agreement for a much better surface fit. What it must not do is
    trade it silently.
    """
    if aligned_lms is None or len(aligned_lms) == 0:
        return None, None

    if kept is not None and len(kept):
        aligned_lms = np.asarray(aligned_lms)[kept]
        fixed_lms = np.asarray(fixed_lms)[kept]

    before = float(np.sqrt(np.mean(np.square(
        np.linalg.norm(np.asarray(aligned_lms) - np.asarray(fixed_lms), axis=1)))))
    moved = (np.hstack([aligned_lms, np.ones((len(aligned_lms), 1))]) @ np.asarray(matrix).T)[:, :3]
    after = float(np.sqrt(np.mean(np.square(
        np.linalg.norm(moved - np.asarray(fixed_lms), axis=1)))))

    drift = after - before
    if drift > MAX_ICP_LANDMARK_DRIFT_MM:
        logger.warning(
            "%s / %s: the ICP left the arch %.1f mm from its landmarks where the "
            "pre-alignment had it at %.1f mm, a drift of %.1f mm. The surface it "
            "settled on is not where the landmarks say the teeth are; on a smooth "
            "occlusal surface that is the fit sliding along the arch, and its own "
            "fitness cannot see it. Check the registration before using it."
            % (patient_id, jaw, after, before, drift))
    else:
        logger.info("%s / %s: landmarks %.1f mm from their CBCT counterparts after "
                    "the ICP, against %.1f mm before it"
                    % (patient_id, jaw, after, before))
    return before, after


def _write_positions(cbct_json, positions, labels):
    """Store the registered positions in the CBCT json, matched by label.

    This used to walk the CBCT control points and index the IOS array by
    position, so a CBCT holding one landmark against six on the IOS side saved
    the IOS UL1O coordinates under the UR6O label.
    """
    if not cbct_json.get("markups"):
        return
    control_points = cbct_json["markups"][0].get("controlPoints", [])
    if labels is None:
        for i, cp in enumerate(control_points):
            if i < len(positions):
                cp["position"] = list(positions[i])
        return
    by_label = dict(zip(labels, positions))
    kept = []
    for cp in control_points:
        label = cp.get("label")
        if label in by_label:
            cp["position"] = list(by_label[label])
            kept.append(cp)
        else:
            logger.warning("CBCT landmark %s has no IOS counterpart: dropped from the output" % label)
    cbct_json["markups"][0]["controlPoints"] = kept


def _fit_rigid(moving_lms, fixed_lms):
    """The rigid transform taking the moving landmarks onto the fixed ones."""
    landmark_transform = vtk.vtkLandmarkTransform()

    points_moving = vtk.vtkPoints()
    points_fixed = vtk.vtkPoints()

    for p in moving_lms: points_moving.InsertNextPoint(p)
    for p in fixed_lms: points_fixed.InsertNextPoint(p)

    landmark_transform.SetSourceLandmarks(points_moving)
    landmark_transform.SetTargetLandmarks(points_fixed)
    landmark_transform.SetModeToRigidBody()  # preserve shape (no deformation)
    landmark_transform.Update()

    # The filter owns the matrix it returns and overwrites it on the next
    # Update(); a copy outlives this call.
    matrix = vtk.vtkMatrix4x4()
    matrix.DeepCopy(landmark_transform.GetMatrix())
    return matrix


def _fit_rigid_without_outliers(moving_lms, fixed_lms, labels, jaw, patient_id,
                                on_enamel=None):
    """Fit, then drop any pair the fit cannot account for, and fit again.

    Two different tests, in the order their evidence is worth. A landmark the
    scan says is not on a tooth goes first and unconditionally: that verdict
    comes from the image, so it holds however the other landmarks look. What is
    left is then trimmed on the fit, one pair per round, and only while enough
    remain to hold a rigid transform down: each drop changes every other
    residual, so deciding them all from one fit would throw away points that
    were never the problem.

    Returns the matrix and the labels it was fitted on.
    """
    keep = list(range(len(moving_lms)))

    if on_enamel and labels:
        off = [i for i in keep if on_enamel.get(labels[i]) is False]
        if off and len(keep) - len(off) >= MIN_LANDMARK_PAIRS:
            logger.warning(
                "%s / %s: %s %s not on a tooth in the CBCT, so the pre-alignment "
                "is fitted on the remaining %d. Worth correcting %s at the CBCT "
                "landmark pause."
                % (patient_id, jaw, ", ".join(labels[i] for i in off),
                   "is" if len(off) == 1 else "are", len(keep) - len(off),
                   "it" if len(off) == 1 else "them"))
            keep = [i for i in keep if i not in off]
        elif off:
            logger.warning(
                "%s / %s: %d of %d CBCT landmarks are not on a tooth, which "
                "leaves too few to fit with. They are all kept, but this arch's "
                "landmarks should be corrected before its registration is used."
                % (patient_id, jaw, len(off), len(keep)))
    while True:
        matrix = _fit_rigid(moving_lms[keep], fixed_lms[keep])
        rms = _alignment_residual(moving_lms[keep], fixed_lms[keep], matrix)
        residuals = np.array([
            np.linalg.norm(np.array(matrix.MultiplyPoint([p[0], p[1], p[2], 1])[:3]) - q)
            for p, q in zip(moving_lms[keep], fixed_lms[keep])])

        if len(keep) <= MIN_LANDMARK_PAIRS_AFTER_TRIM:
            break

        # What the fit would be without each pair in turn.
        without = []
        for position in range(len(keep)):
            subset = [k for j, k in enumerate(keep) if j != position]
            trial = _fit_rigid(moving_lms[subset], fixed_lms[subset])
            without.append((_alignment_residual(
                moving_lms[subset], fixed_lms[subset], trial), position))

        best, position = min(without)
        if not (residuals[position] > LANDMARK_OUTLIER_FLOOR_MM
                and best <= LANDMARK_OUTLIER_IMPROVEMENT * rms):
            break

        dropped = labels[keep[position]] if labels else "#%d" % keep[position]
        logger.warning(
            "%s / %s: %s sits %.1f mm from where the rest of the arch puts it, and "
            "the fit over the other %d landmarks is %.1f mm against %.1f mm with "
            "it. It is not the same point on both scans, so the pre-alignment is "
            "fitted without it -- worth correcting that landmark on the CBCT."
            % (patient_id, jaw, dropped, residuals[position], len(keep) - 1, best, rms))
        keep.pop(position)

    return matrix, [labels[i] for i in keep] if labels else None, keep


def align_by_landmarks(moving_mesh, moving_lms, fixed_lms, jaw="", patient_id="",
                       labels=None, on_enamel=None):
    moving_lms = np.asarray(moving_lms, dtype=float)
    fixed_lms = np.asarray(fixed_lms, dtype=float)

    matrix, kept_labels, kept = _fit_rigid_without_outliers(
        moving_lms, fixed_lms, labels, jaw, patient_id, on_enamel)

    # Falling back to the identity leaves the ICP to start from the raw pose,
    # which is what already happened whenever the two lists had different
    # sizes -- only now it is a decision, and it is written down.
    # Everything below judges the fit on the pairs it was actually fitted to: a
    # pair dropped as an outlier would otherwise be counted twice, once in
    # having it removed and again in the residual that decides whether to keep
    # the result at all.
    n = len(kept)
    if len(moving_lms) < MIN_LANDMARK_PAIRS:
        logger.warning(
            "%s / %s: %d landmark pair(s), %d needed. A rigid fit is "
            "underdetermined below that (one point is a plain translation, two "
            "leave a free rotation about the axis). Skipping the pre-alignment."
            % (patient_id, jaw, len(moving_lms), MIN_LANDMARK_PAIRS))
        matrix = vtk.vtkMatrix4x4()
    else:
        rms = _alignment_residual(moving_lms[kept], fixed_lms[kept], matrix)
        if rms > MAX_LANDMARK_RESIDUAL_MM:
            logger.warning(
                "%s / %s: %.1f mm residual over %d pairs (threshold %.1f). The "
                "IOS and CBCT landmarks do not describe the same points. "
                "Skipping the pre-alignment."
                % (patient_id, jaw, rms, n, MAX_LANDMARK_RESIDUAL_MM))
            matrix = vtk.vtkMatrix4x4()
        else:
            logger.info("%s / %s: pre-alignment accepted, %.1f mm residual over %d "
                        "pair(s)%s"
                        % (patient_id, jaw, rms, n,
                           " (%s)" % ", ".join(kept_labels) if kept_labels else ""))
            # Kept rather than refused: being under-determined is not being
            # wrong, and the identity would start the ICP from the raw pose,
            # which is further still. If the free rotation has in fact turned
            # the arch away, the ICP finds nothing to match and the run says so.
            spread = _landmark_spread_ratio(moving_lms[kept])
            if spread < MIN_LANDMARK_SPREAD_RATIO:
                logger.warning(
                    "%s / %s: the %d landmarks are nearly in a straight line "
                    "(spread %.2f, under the %.2f floor; a full arch measures "
                    "about 0.32). The rotation about that line rests on their "
                    "noise alone, and the %.1f mm residual cannot show it. "
                    "Landmarks further apart on both sides of the arch would "
                    "pin it down."
                    % (patient_id, jaw, n, spread, MIN_LANDMARK_SPREAD_RATIO, rms))
    # Apply it with PyVista
    aligned_mesh = moving_mesh.transform(matrix,inplace=False)
    
    aligned_lms = []
    for p in moving_lms:
        p_transformed = matrix.MultiplyPoint([p[0], p[1], p[2], 1])
        aligned_lms.append(p_transformed[:3])
    
    aligned_lms = np.array(aligned_lms)
    
    return aligned_mesh, matrix, aligned_lms, kept

class _Target:
    """The CBCT surface as the ICP actually consumes it: points and normals.

    Cropping is then index selection on two arrays. Cutting the mesh itself
    instead, with extract_points, costs more than the ICP saves: on one patient
    it removed three quarters of the points and still added nearly two minutes,
    because rebuilding a 2.8 million point surface is dearer than querying it.

    Normals are taken from the whole surface before any crop, so a point keeps
    the normal its neighbourhood gives it rather than one bent by the cut.
    """

    def __init__(self, points, normals):
        self.points = points
        self.normals = normals

    @classmethod
    def FromMesh(cls, mesh, name):
        return cls(np.asarray(mesh.points), _point_normals(mesh, name))

    def __len__(self):
        return len(self.points)

    def Around(self, anchor_points, margin, label):
        """The part of the surface an arch can plausibly be registered to.

        Anchored on that arch's own CBCT landmarks rather than on the
        pre-aligned IOS: the landmarks are in CBCT coordinates whatever the
        pre-alignment did, while an IOS whose pre-alignment was skipped is
        still in the frame ASO left it in and would drag the box across the
        whole head.

        The opposing arch stays in the box -- at this margin it cannot be
        excluded by position, and it does not need to be, the normals tell it
        apart. What goes is everything that was never a candidate: cranial
        base, vertebrae, the far side of the jaw.
        """
        if anchor_points is None or len(anchor_points) == 0:
            logger.warning("%s: no CBCT landmark to crop around, the whole "
                           "surface is kept as the target" % label)
            return self

        anchor_points = np.asarray(anchor_points, dtype=float)
        low = anchor_points.min(axis=0) - margin
        high = anchor_points.max(axis=0) + margin
        inside = np.all((self.points >= low) & (self.points <= high), axis=1)

        kept = int(np.sum(inside))
        if kept < 3:
            logger.warning("%s: nothing of the CBCT surface lies within %.0f mm "
                           "of its landmarks, the whole surface is kept as the "
                           "target" % (label, margin))
            return self

        logger.info("%s: CBCT target cropped to %d of %d points (%.1f%%) within "
                    "%.0f mm of the arch's landmarks"
                    % (label, kept, len(self), 100.0 * kept / max(len(self), 1),
                       margin))
        return _Target(self.points[inside],
                       self.normals[inside] if self.normals is not None else None)


def _point_normals(mesh, name):
    """Per-point outward normals, in the order of `mesh.points`.

    Cell normals were taken first here and never used, which was as well: they
    are one per triangle, so indexing them with a point index read the normal of
    an unrelated part of the surface.
    """
    if "Normals" in mesh.point_data:
        return np.asarray(mesh.point_data["Normals"], dtype=float)
    try:
        with_normals = mesh.compute_normals(
            point_normals=True, cell_normals=False,
            auto_orient_normals=False, inplace=False)
        return np.asarray(with_normals.point_data["Normals"], dtype=float)
    except Exception as e:
        logger.warning("%s: no surface normals could be computed (%s). The "
                       "opposing arch cannot be told apart by orientation."
                       % (name, e))
        return None


def _crown_points(mesh, label):
    """The part of an IOS the CBCT has anything to answer with.

    An intraoral scan is about 60% gingiva, and gingiva has no counterpart in a
    CBCT surface: the nearest thing under it is the alveolar bone, a millimetre
    or two further in. Those points are not matched so much as dragged, and they
    outnumber the crowns. Registering on the crowns alone does not move the
    answer much -- it is the same anatomy either way -- but it roughly doubles
    the share of the moving surface that finds a real match, which is what the
    run is judged on.

    Returns None when the mesh carries no usable labels, and the caller then
    registers the whole thing, as it always did.
    """
    if "Universal_ID" not in mesh.point_data:
        logger.info("%s: no Universal_ID on the IOS, the whole arch is registered "
                    "including its gingiva" % label)
        return None

    labels = np.asarray(mesh.point_data["Universal_ID"])
    crowns = ~np.isin(labels, GINGIVA_LABELS)
    share = float(np.mean(crowns)) if crowns.size else 0.0
    if share < MIN_CROWN_SHARE:
        logger.warning("%s: only %.0f%% of the IOS is labelled as crowns, which is "
                       "too little to trust. The whole arch is registered instead."
                       % (label, 100 * share))
        return None

    logger.info("%s: registering on the %d crown point(s) of %d (%.0f%%); the "
                "gingiva has no counterpart in the CBCT"
                % (label, int(np.sum(crowns)), len(crowns), 100 * share))
    return crowns


def _point_to_plane_step(source, target, normals):
    """Rigid step minimising the distance to the target's tangent plane.

    Point-to-point pulls a surface towards particular neighbours, which on the
    smooth, near-flat occlusal surfaces here means it slides along them and
    stalls. Measuring along the target normal lets the surface slide freely and
    only resists what actually separates it from the other one.

    Linearised in the rotation: for a correspondence (p, q, n) the residual is
    (p - q).n + w.(p x n) + t.n, which is linear in the six unknowns [w, t].
    """
    A = np.hstack([np.cross(source, normals), normals])
    b = np.einsum("ij,ij->i", target - source, normals)

    solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 6 or not np.all(np.isfinite(solution)):
        return None

    omega, translation = solution[:3], solution[3:]
    # A linearised step is only meaningful while it stays small; a large one
    # means the system is being driven by outliers rather than by the surface.
    if np.linalg.norm(omega) > 0.5:
        return None

    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(omega).as_matrix()
    delta[:3, 3] = translation
    return delta


def _point_to_point_step(source, target):
    """Procrustes fit, the fallback when the point-to-plane system is degenerate."""
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)

    H = (source - source_center).T @ (target - target_center)
    U, _, vt = np.linalg.svd(H)
    R = vt.T @ U.T
    if np.linalg.det(R) < 0:
        vt[-1, :] *= -1
        R = vt.T @ U.T

    delta = np.eye(4)
    delta[:3, :3] = R
    delta[:3, 3] = target_center - R @ source_center
    return delta


def _icp_run(moving_pts, moving_normals, fixed_pts, fixed_normals, kdtree,
             sign, max_dist, max_iterations, label):
    """One ICP, under one reading of which way the two meshes wind their faces.

    `sign` is +1 when a normal means the same thing on both meshes and -1 when
    one of them is wound the other way; it multiplies the CBCT normals before
    they are compared and before they are used as tangent planes.
    """
    use_normals = moving_normals is not None and fixed_normals is not None

    transformation = np.eye(4)
    moving_pts_transformed = moving_pts.copy()
    moving_normals_transformed = moving_normals.copy() if use_normals else None

    inlier_rmse = float("inf")
    fitness = 0.0
    n_pairs = 0
    rejected_by_normal = 0
    settled = 0
    iteration = 0

    for iteration in range(max_iterations):
        # workers=-1 spreads the query over every core: it is the dominant cost
        # of the loop (one lookup per moving point, per iteration) and the
        # default of a single worker left the other cores idle.
        distances, indices = kdtree.query(moving_pts_transformed, k=1, workers=-1)

        near = distances < max_dist
        valid_mask = near
        if use_normals:
            agreement = np.einsum("ij,ij->i", moving_normals_transformed,
                                  sign * fixed_normals[indices])
            valid_mask = near & (agreement > MIN_NORMAL_AGREEMENT)
            # The peak, not the last iteration: once the arch has settled on
            # its own side nothing nearby faces the wrong way any more, so the
            # final count says nothing about how ambiguous the start was.
            rejected_by_normal = max(
                rejected_by_normal, int(np.sum(near & ~valid_mask)))

        valid_indices = indices[valid_mask]
        valid_moving = moving_pts_transformed[valid_mask]
        valid_distances = distances[valid_mask]

        inlier_rmse = (float(np.sqrt(np.mean(valid_distances ** 2)))
                       if len(valid_distances) > 0 else float("inf"))
        fitness = float(np.sum(valid_mask)) / len(moving_pts)
        n_pairs = int(np.sum(valid_mask))

        # Checked here rather than after the update, so the numbers reported
        # are the ones that describe the transform actually returned.
        if settled >= ICP_SETTLED_ITERATIONS:
            logger.debug("%s: settled at iteration %d" % (label, iteration))
            break

        if n_pairs < 3:
            logger.debug("%s: iteration %d has %d usable correspondence(s), "
                         "the ICP stops here" % (label, iteration, n_pairs))
            break

        target = fixed_pts[valid_indices]
        delta = None
        if use_normals:
            delta = _point_to_plane_step(
                valid_moving, target, sign * fixed_normals[valid_indices])
        if delta is None:
            delta = _point_to_point_step(valid_moving, target)

        transformation = delta @ transformation

        moving_pts_homogeneous = np.hstack(
            [moving_pts, np.ones((moving_pts.shape[0], 1))])
        previous_pts = moving_pts_transformed
        moving_pts_transformed = (moving_pts_homogeneous @ transformation.T)[:, :3]
        if use_normals:
            moving_normals_transformed = moving_normals @ transformation[:3, :3].T

        shift = float(np.max(np.linalg.norm(
            moving_pts_transformed - previous_pts, axis=1)))
        settled = settled + 1 if shift < ICP_SETTLED_SHIFT_MM else 0

    return transformation, {
        "iterations": iteration,
        "settled": settled >= ICP_SETTLED_ITERATIONS,
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
        "pairs": n_pairs,
        "rejected_by_normal": rejected_by_normal,
        "used_normals": use_normals,
        "sign": sign,
    }


def run_icp_point_to_plane(moving_mesh, fixed_mesh, max_dist=1.5, label=""):
    """Register `moving_mesh` onto `fixed_mesh`.

    Returns the registered mesh, the 4x4 matrix, and what the run is worth:
    fitness (the share of moving points that found a match), the inlier RMSE,
    and how many correspondences the answer stands on.

    Whether a normal points out of the tooth or into it is a property of how
    each file was written, and the two modalities do not have to agree. It
    cannot be read off the starting pose either: that is precisely where an
    IOS sitting between the two arches has most of its nearest neighbours on
    the wrong one, and averaging over them reads the bite as an inversion and
    then locks the registration onto the opposing arch. So both readings are
    registered and the one that actually fits the CBCT better is kept -- the
    IOS crowns are the same anatomy as their own arch in the CBCT and nothing
    else, so the right reading wins on the merits.
    """
    label = label or "IOS"
    target = (fixed_mesh if isinstance(fixed_mesh, _Target)
              else _Target.FromMesh(fixed_mesh, "CBCT surface"))

    moving_pts = np.asarray(moving_mesh.points)
    fixed_pts = target.points
    fixed_normals = target.normals
    moving_normals = _point_normals(moving_mesh, label)
    use_normals = fixed_normals is not None and moving_normals is not None

    # Only the crowns drive the fit; the whole arch still moves by the matrix
    # they settle on, gingiva included, so nothing is lost from the output.
    # Taken here rather than by cutting the mesh, which would have to be rebuilt
    # and would leave the normals to be recomputed at the cut.
    crowns = _crown_points(moving_mesh, label)
    if crowns is not None:
        moving_pts = moving_pts[crowns]
        if moving_normals is not None:
            moving_normals = moving_normals[crowns]

    max_iterations = ICP_MAX_ITERATIONS
    # Only the moving points change from one iteration to the next, so the tree
    # over the fixed points is built once instead of being rebuilt up to
    # max_iterations times over the very same coordinates.
    kdtree = cKDTree(fixed_pts)

    attempts = []
    for sign in ((1.0, -1.0) if use_normals else (1.0,)):
        transformation, quality = _icp_run(
            moving_pts, moving_normals, fixed_pts, fixed_normals, kdtree,
            sign, max_dist, max_iterations, label)
        attempts.append((transformation, quality))
        if use_normals:
            logger.debug("%s: normals read as %s gives %.1f%% matched at %.3f mm"
                         % (label, "aligned" if sign > 0 else "opposed",
                            100 * quality["fitness"], quality["inlier_rmse"]))

    # More of the IOS matched is the first thing that matters; a tie on that is
    # broken by how closely it matched.
    transformation, quality = max(
        attempts, key=lambda a: (round(a[1]["fitness"], 3), -a[1]["inlier_rmse"]))

    # Only of the attempt that was kept: the reading of the normals that loses
    # is expected to wander, and saying so about it reads as a doubt over the
    # answer that was actually returned.
    if not quality["settled"]:
        logger.warning(
            "%s: the ICP used all %d iterations without settling to within "
            "%g mm. Its answer is wherever it had got to."
            % (label, max_iterations, ICP_SETTLED_SHIFT_MM))

    if use_normals and quality["sign"] < 0:
        logger.info("%s: the IOS and the CBCT surface wind their faces the "
                    "opposite way; the IOS normals were flipped to compare them"
                    % label)

    final_mesh = moving_mesh.transform(transformation, inplace=False)

    logger.info(
        "%s: ICP done in %d iterations, %.1f%% of the IOS matched (%d points) "
        "at %.2f mm RMSE%s"
        % (label, quality["iterations"], 100 * quality["fitness"],
           quality["pairs"], quality["inlier_rmse"],
           ", up to %d nearby points dropped as the opposing surface"
           % quality["rejected_by_normal"] if quality["rejected_by_normal"] else ""))

    return final_mesh, transformation, quality

def save_registered_ios(registered_vtk_upper,registered_vtk_lower,output_path,num_patient):
    file_path_u = os.path.join(output_path,f"{num_patient}_Reg_U.vtk")
    registered_vtk_upper.save(file_path_u)
    file_path_l = os.path.join(output_path,f"{num_patient}_Reg_L.vtk")
    registered_vtk_lower.save(file_path_l)

def apply_matrix_and_save_landmarks(aligned_upper_lm,aligned_lower_lm,mat_u,mat_l,json_output_path,num_patient,landmarks_json_cbct_U,landmarks_json_cbct_L,labels_u=None,labels_l=None):
    # Apply transformations using NumPy (no Open3D dependency)
    # Convert landmarks to homogeneous coordinates, apply transformation, convert back
    
    # Upper landmarks transformation
    aligned_icp_lm_upper_homo = np.hstack([aligned_upper_lm, np.ones((aligned_upper_lm.shape[0], 1))])
    aligned_icp_lm_upper = (aligned_icp_lm_upper_homo @ mat_u.T)[:, :3]
    
    # Lower landmarks transformation
    aligned_icp_lm_lower_homo = np.hstack([aligned_lower_lm, np.ones((aligned_lower_lm.shape[0], 1))])
    aligned_icp_lm_lower = (aligned_icp_lm_lower_homo @ mat_l.T)[:, :3]

    json_output_path_ios_u = os.path.join(json_output_path,f"{num_patient}_lm_Reg_U.mrk.json")
    json_output_path_ios_l = os.path.join(json_output_path,f"{num_patient}_lm_Reg_L.mrk.json")

    _write_positions(landmarks_json_cbct_U, aligned_icp_lm_upper, labels_u)

    with open(json_output_path_ios_u, "w") as file:
        json.dump(landmarks_json_cbct_U, file,indent=4, ensure_ascii=False)

    _write_positions(landmarks_json_cbct_L, aligned_icp_lm_lower, labels_l)

    with open(json_output_path_ios_l, "w") as file:
        json.dump(landmarks_json_cbct_L, file,indent=4, ensure_ascii=False)

def get_landmarks (json_path):
    with open(json_path, 'r') as f:
        landmarks_json = json.load(f)

    landmarks = []
    if 'markups' in landmarks_json:
        for markup in landmarks_json['markups'][0]['controlPoints']:
            x, y, z = markup['position']
            landmarks.append([x, y, z])
    
    return np.array(landmarks)

def _enamel_and_soft_levels(image_array, ijk_to_lps, landmarks, patient_id):
    """The two intensities the crown boundary lies between.

    The enamel plateau is read at the landmarks, which sit on the occlusal
    surfaces, as the brightest voxel in a small window around each -- brightest
    rather than nearest, so a landmark a voxel off the crown still reads the
    tooth and not the gap beside it. The median over the landmarks then carries
    no single bad one.

    The surround is the median of everything in the soft-tissue band across the
    arch, which is the mucosa and gingiva the crowns actually emerge through.

    Returns (None, None) when the scan cannot be probed, and the caller falls
    back rather than guessing.
    """
    if landmarks is None or len(landmarks) == 0:
        return None, None

    depth, height, width = image_array.shape
    lps_to_ijk = np.linalg.inv(ijk_to_lps)
    homogeneous = np.hstack([landmarks, np.ones((len(landmarks), 1))])
    ijk = np.rint((homogeneous @ lps_to_ijk.T)[:, :3]).astype(int)

    radius = ENAMEL_PROBE_RADIUS_VOXELS
    peaks = []
    for i, j, k in ijk:
        if not (0 <= i < width and 0 <= j < height and 0 <= k < depth):
            continue
        window = image_array[max(k - radius, 0):k + radius + 1,
                             max(j - radius, 0):j + radius + 1,
                             max(i - radius, 0):i + radius + 1]
        if window.size:
            peaks.append(float(window.max()))

    if not peaks:
        logger.warning("%s: no CBCT landmark falls inside the scan, the surface "
                       "level cannot be read from the crowns" % patient_id)
        return None, None

    low = np.clip(ijk.min(axis=0) - radius, 0, None)
    high = ijk.max(axis=0) + radius + 1
    region = image_array[low[2]:high[2], low[1]:high[1], low[0]:high[0]]
    in_band = region[(region > SOFT_TISSUE_BAND[0]) & (region < SOFT_TISSUE_BAND[1])]

    enamel = float(np.median(peaks))
    soft = float(np.median(in_band)) if in_band.size else float(region.min())
    return enamel, soft


def _landmarks_on_enamel(image_array, ijk_to_lps, json_path, level, jaw, patient_id):
    """Which of a CBCT landmark file's points actually sit on a tooth.

    Returns {label: bool}, empty when the check is switched off or the file
    cannot be read -- an empty verdict lets every landmark through.
    """
    if not CHECK_LANDMARKS_ON_ENAMEL:
        return {}
    try:
        labelled = _labeled_landmarks(json_path)
    except Exception as e:
        logger.warning("%s / %s: could not read %s to check its landmarks sit on "
                       "teeth (%s)" % (patient_id, jaw, os.path.basename(json_path), e))
        return {}
    if not labelled:
        return {}

    depth, height, width = image_array.shape
    lps_to_ijk = np.linalg.inv(ijk_to_lps)
    radius = ENAMEL_PROBE_RADIUS_VOXELS

    verdict = {}
    for label, position in labelled.items():
        i, j, k = np.rint((np.append(np.asarray(position, dtype=float), 1.0)
                           @ lps_to_ijk.T)[:3]).astype(int)
        if not (0 <= i < width and 0 <= j < height and 0 <= k < depth):
            logger.warning("%s / %s: %s falls outside the scan entirely"
                           % (patient_id, jaw, label))
            verdict[label] = False
            continue
        window = image_array[max(k - radius, 0):k + radius + 1,
                             max(j - radius, 0):j + radius + 1,
                             max(i - radius, 0):i + radius + 1]
        peak = float(window.max()) if window.size else float("-inf")
        verdict[label] = peak >= level
        if not verdict[label]:
            logger.warning(
                "%s / %s: %s reads %.0f at its brightest, under the %.0f the scan "
                "is contoured at, so it is not on a tooth. In a closed bite that "
                "is usually a point that landed on the opposing arch."
                % (patient_id, jaw, label, peak, level))
    return verdict


def surface_threshold(image_array, ijk_to_lps, landmarks, patient_id=""):
    """The level to contour the CBCT at, halfway between enamel and its surround.

    An explicit AREG_CBCT_SURFACE_THRESHOLD wins, for the scan this rule cannot
    read.
    """
    if SURFACE_THRESHOLD is not None:
        level = float(SURFACE_THRESHOLD)
        logger.info("%s: CBCT contoured at %.0f, set by AREG_CBCT_SURFACE_THRESHOLD"
                    % (patient_id, level))
        return level

    enamel, soft = _enamel_and_soft_levels(image_array, ijk_to_lps, landmarks, patient_id)
    if enamel is None or not (enamel > soft):
        logger.warning(
            "%s: the crowns are no brighter than what surrounds them in this "
            "scan, so the surface level cannot be read from it. Falling back to "
            "400, which is what every run used before this was measured; if the "
            "registration comes out poor, set AREG_CBCT_SURFACE_THRESHOLD."
            % patient_id)
        return 400.0

    level = 0.5 * (enamel + soft)
    logger.info("%s: CBCT contoured at %.0f -- enamel reads %.0f at the landmarks, "
                "the soft tissue around the arch %.0f"
                % (patient_id, level, enamel, soft))
    return level


def load_data(scan_path,json_path_cbct_u,json_path_cbct_l,json_path_ios_u,json_path_ios_l,patient_id=""):
    
    lm_cbct_u = get_landmarks(json_path_cbct_u)
    lm_cbct_l = get_landmarks(json_path_cbct_l)
    lm_ios_u = get_landmarks(json_path_ios_u)
    lm_ios_l = get_landmarks(json_path_ios_l)

    image = sitk.ReadImage(scan_path)
    image_array = sitk.GetArrayFromImage(image)
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    ijk_to_lps = np.eye(4)
    ijk_to_lps[:3, :3] = direction @ np.diag(spacing)
    ijk_to_lps[:3, 3] = origin

    # Both arches: the level describes the scan, not one jaw, and reading it off
    # twelve landmarks rather than six makes the median that much steadier.
    arch_landmarks = [a for a in (lm_cbct_u, lm_cbct_l) if a is not None and len(a)]
    all_landmarks = np.vstack(arch_landmarks) if arch_landmarks else np.empty((0, 3))
    level = surface_threshold(image_array, ijk_to_lps, all_landmarks, patient_id)

    # Judged here, where the voxels are already in hand: the array is gigabytes
    # and is dropped on the way out of this function.
    on_enamel = {
        "Upper": _landmarks_on_enamel(image_array, ijk_to_lps, json_path_cbct_u,
                                      level, "Upper", patient_id),
        "Lower": _landmarks_on_enamel(image_array, ijk_to_lps, json_path_cbct_l,
                                      level, "Lower", patient_id),
    }

    vol = pv.wrap(image_array.transpose(2, 1, 0))
    cbct_raw_mesh = vol.contour(isosurfaces=[level])

    cbct_surface = cbct_raw_mesh.transform(ijk_to_lps, inplace=False)

    return lm_cbct_u,lm_cbct_l,lm_ios_u,lm_ios_l,cbct_surface,on_enamel

def getPatients(ios_folder, cbct_folder, ios_lm_folder, cbct_lm_folder):
    """
    Scans the 4 folders and generates a dictionary with patient IDs as keys
    and paths to IOS scan, CBCT scan, and landmarks files as values.
    
    Uses pattern matching to identify timepoints (T0, T1, T2) and jaws (U/u, L/l)
    """
    
    patients = {}
    
    def extract_timepoint(filename):
        """Extract timepoint from filename (T0, T1, T2, t0, t1, t2)"""
        match = re.search(r'[Tt]([0-2])', filename)
        return match.group(0) if match else None
    
    def extract_jaw(filename):
        """Extract jaw from filename (_u, _U, u_, _l, _L, l_, _upper, _lower)

        The letter has to be a token of its own, delimited by an underscore,
        the start of the name or a dot. Matching a bare "u" anywhere used to
        read "Dupont_003_T1_L.vtk" or "P001_T1_L_Surface.vtk" as upper, which
        registers the lower arch against the upper CBCT landmarks without any
        error being raised.

        A name carrying both jaws is refused rather than read as the first one
        found. ALI_IOS runs the model of each jaw over every scan and names what
        it writes after both the scan and the model, so a lower arch whose
        segmentation holds a few upper tooth numbers comes back as
        "P09_T1_L_SegOr_Upper_O_Pred.json": upper landmarks sitting on a lower
        arch, which belongs to neither and would have taken the place of that
        patient's real upper landmarks. Two markers that agree are the ordinary
        case ("..._U_SegOr_Upper_...") and are read normally.
        """
        upper = re.search(r'(?:^|_)(?:u|upper)(?=_|\.|$)', filename, re.IGNORECASE)
        lower = re.search(r'(?:^|_)(?:l|lower)(?=_|\.|$)', filename, re.IGNORECASE)
        if upper and lower:
            logger.warning(
                "%s names both an upper and a lower arch, so which one it "
                "describes cannot be told from it: ignored" % filename)
            return None
        if upper:
            return 'upper'
        if lower:
            return 'lower'
        return None
    
    def extract_patient_id(filename):
        """Extract patient ID from filename (letter + digits before timepoint)"""
        # TIMEPOINT-SUFFIX: [Tt][0-2] accepts T0..T2 only, so a _T3 file matches
        # nothing at all and this returns None - the patient is skipped rather than
        # mispaired. Accepting more timepoints means \d+ here. See the full note
        # above GetPatients in AREG_CBCT/AREG_CBCT_utils/utils.py.
        match = re.search(r'([A-Za-z]+)[_]?([0-9]+)[_]?[Tt][0-2]', filename)
        if match:
            return match.group(1) + match.group(2)  # Combine letter and digits
        return None
    
    def normalize_patient_id(patient_id):
        """Normalize patient ID by:
        1. Removing underscores (P_0001 -> P0001)
        2. Removing leading zeros from digits (P0001 -> P1, P00001 -> P1)
        E.g., P001, P_0001, and P00001 all become P1"""
        if patient_id:
            # Remove underscores
            patient_id = patient_id.replace('_', '')
            
            # Separate letters and digits
            match = re.match(r'([A-Za-z]*)([0-9]*)', patient_id)
            if match:
                letters = match.group(1)
                digits = match.group(2)
                if digits:
                    # Remove leading zeros from digits
                    digits = str(int(digits))
                return letters + digits
            return patient_id
        return None
    
    # Parse IOS VTK files
    ios_files = os.listdir(ios_folder)
    for filename in ios_files:
        if filename.endswith('.vtk'):
            timepoint = extract_timepoint(filename)
            jaw = extract_jaw(filename)
            patient_id = normalize_patient_id(extract_patient_id(filename))
            
            if timepoint and jaw and patient_id:
                key = f"{patient_id}_{timepoint}"
                
                if key not in patients:
                    patients[key] = {}
                
                jaw_key = f"ios_{'upper' if jaw == 'upper' else 'lower'}"
                patients[key][jaw_key] = os.path.join(ios_folder, filename)
    
    # Parse CBCT files
    cbct_files = os.listdir(cbct_folder)
    for filename in cbct_files:
        if filename.endswith('.nii.gz'):
            timepoint = extract_timepoint(filename)
            patient_id = normalize_patient_id(extract_patient_id(filename))
            
            if timepoint and patient_id:
                key = f"{patient_id}_{timepoint}"
                
                if key not in patients:
                    patients[key] = {}
                
                patients[key]["cbct"] = os.path.join(cbct_folder, filename)
    
    # Parse IOS JSON landmarks
    ios_lm_files = os.listdir(ios_lm_folder)
    for filename in ios_lm_files:
        if filename.endswith('.json'):
            timepoint = extract_timepoint(filename)
            jaw = extract_jaw(filename)
            patient_id = normalize_patient_id(extract_patient_id(filename))
            
            # Check if it's an IOS landmark file
            if timepoint and jaw and patient_id:
                key = f"{patient_id}_{timepoint}"
                
                if key not in patients:
                    patients[key] = {}
                
                jaw_key = f"ios_lm_{'upper' if jaw == 'upper' else 'lower'}"
                patients[key][jaw_key] = os.path.join(ios_lm_folder, filename)
    
    # Parse CBCT JSON landmarks
    cbct_lm_files = os.listdir(cbct_lm_folder)
    for filename in cbct_lm_files:
        if filename.endswith('.json'):
            timepoint = extract_timepoint(filename)
            jaw = extract_jaw(filename)
            patient_id = normalize_patient_id(extract_patient_id(filename))
            
            # Check if it's a CBCT landmark file
            if timepoint and jaw and patient_id:
                key = f"{patient_id}_{timepoint}"
                
                if key not in patients:
                    patients[key] = {}
                
                jaw_key = f"cbct_lm_{'upper' if jaw == 'upper' else 'lower'}"
                patients[key][jaw_key] = os.path.join(cbct_lm_folder, filename)
    
    # Log found patients
    logger.info(f"Found {len(patients)} patients")
    for patient_key in sorted(patients.keys()):
        logger.debug(f"Patient {patient_key}: {patients[patient_key]}")
    
    return patients

def main(args):
    patients = getPatients(args.IOS_folder, args.CBCT_folder, args.IOS_lm_folder, args.CBCT_lm_folder)
    logger.info("Running AREG_IOSCBCT for all patients")

    if not patients:
        logger.warning("No files to process has been found. Please check the input folders and folder_name")
    
    # Create output directory if it doesn't exist
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Process each patient
    registered = 0
    for patient_id, patient_data in sorted(patients.items()):
        logger.info(f"Processing patient {patient_id}...")
        
        try:
            # 1. LOAD DATA
            logger.debug(f"Loading data for patient {patient_id}")
            lm_cbct_u, lm_cbct_l, lm_ios_u, lm_ios_l, cbct_surface, on_enamel = load_data(
                patient_data["cbct"],
                patient_data["cbct_lm_upper"],
                patient_data["cbct_lm_lower"],
                patient_data["ios_lm_upper"],
                patient_data["ios_lm_lower"],
                patient_id
            )
            
            # Keep only the landmarks present on both sides, in the same order
            labels_U, lm_cbct_u, lm_ios_u = _pair_landmarks(
                patient_data["cbct_lm_upper"], patient_data["ios_lm_upper"], "Upper", patient_id)
            labels_L, lm_cbct_l, lm_ios_l = _pair_landmarks(
                patient_data["cbct_lm_lower"], patient_data["ios_lm_lower"], "Lower", patient_id)

            # Load IOS meshes (VTK files)
            ios_upper_mesh = pv.read(patient_data["ios_upper"])
            ios_lower_mesh = pv.read(patient_data["ios_lower"])
            logger.debug(f"Loaded IOS upper and lower meshes")
            logger.debug(f"IOS Upper mesh: n_points={ios_upper_mesh.n_points}, n_cells={ios_upper_mesh.n_cells}, bounds={ios_upper_mesh.bounds}")
            logger.debug(f"IOS Lower mesh: n_points={ios_lower_mesh.n_points}, n_cells={ios_lower_mesh.n_cells}, bounds={ios_lower_mesh.bounds}")
            
            # 2. ALIGN BY LANDMARKS
            logger.debug(f"Aligning IOS upper jaw by landmarks")
            aligned_ios_upper, mat_ios_upper, aligned_lms_ios_upper, kept_u = align_by_landmarks(
                ios_upper_mesh, lm_ios_u, lm_cbct_u, "Upper", patient_id, labels_U,
                on_enamel.get("Upper")
            )
            logger.info(f"IOS Upper landmarks after alignment:\n{aligned_lms_ios_upper}")
            
            logger.debug(f"Aligning IOS lower jaw by landmarks")
            aligned_ios_lower, mat_ios_lower, aligned_lms_ios_lower, kept_l = align_by_landmarks(
                ios_lower_mesh, lm_ios_l, lm_cbct_l, "Lower", patient_id, labels_L,
                on_enamel.get("Lower")
            )
            logger.debug(f"IOS Lower landmarks after alignment shape: {aligned_lms_ios_lower.shape}")
            logger.debug(f"IOS Lower landmarks after alignment:\n{aligned_lms_ios_lower}")
            
            # 3. RUN ICP REGISTRATION
            # Each arch gets its own slice of the CBCT surface. The two ICPs
            # each built a tree over the whole head before, and then queried it
            # once per moving point per iteration, for structures an intraoral
            # scan has no counterpart to.
            # Normals over the whole surface once, rather than once per arch
            # inside each ICP, and the crop is then a slice of those arrays.
            cbct_target = _Target.FromMesh(cbct_surface, f"{patient_id} / CBCT")
            cbct_upper = cbct_target.Around(lm_cbct_u, CBCT_CROP_MARGIN_MM,
                                            f"{patient_id} / Upper")
            cbct_lower = cbct_target.Around(lm_cbct_l, CBCT_CROP_MARGIN_MM,
                                            f"{patient_id} / Lower")

            logger.debug(f"Running ICP for upper jaw")
            registered_ios_upper, mat_icp_upper, quality_upper = run_icp_point_to_plane(
                aligned_ios_upper, cbct_upper, max_dist=1.0,
                label=f"{patient_id} / Upper"
            )
            
            logger.debug(f"Running ICP for lower jaw")
            registered_ios_lower, mat_icp_lower, quality_lower = run_icp_point_to_plane(
                aligned_ios_lower, cbct_lower, max_dist=1.0,
                label=f"{patient_id} / Lower"
            )
            logger.info(f"ICP registration completed for patient {patient_id}")

            # The one check on the ICP that the ICP does not grade itself.
            _report_landmark_drift(aligned_lms_ios_upper, lm_cbct_u,
                                   mat_icp_upper, "Upper", patient_id, kept_u)
            _report_landmark_drift(aligned_lms_ios_lower, lm_cbct_l,
                                   mat_icp_lower, "Lower", patient_id, kept_l)

            # An ICP that matched nothing still returns a matrix, and writing it
            # out put an untouched IOS in the results folder under the name of a
            # registered one. It happens whenever the pre-alignment is skipped
            # and the raw pose is nowhere near the CBCT.
            starved = [jaw for jaw, quality in (("Upper", quality_upper),
                                                ("Lower", quality_lower))
                       if quality["fitness"] < MIN_ICP_FITNESS]
            if starved:
                raise RuntimeError(
                    "the %s arch matched under %.0f%% of its points to the CBCT "
                    "surface (upper %.1f%%, lower %.1f%%). Nothing was written "
                    "for this patient: check its landmarks, the pre-alignment "
                    "above says whether it was accepted."
                    % (" and ".join(starved), 100 * MIN_ICP_FITNESS,
                       100 * quality_upper["fitness"], 100 * quality_lower["fitness"]))

            # 4. SAVE RESULTS
            logger.info(f"Saving registered meshes and landmarks")
            
            # Save registered meshes
            save_registered_ios(registered_ios_upper, registered_ios_lower,output_dir,patient_id)
            
            # Load landmark JSON files to update them
            with open(patient_data["cbct_lm_upper"], 'r') as f:
                landmarks_json_cbct_u = json.load(f)
            with open(patient_data["cbct_lm_lower"], 'r') as f:
                landmarks_json_cbct_l = json.load(f)
            
            # Save registered landmarks
            apply_matrix_and_save_landmarks(
                aligned_lms_ios_upper, aligned_lms_ios_lower,
                mat_icp_upper, mat_icp_lower,
                output_dir, patient_id,
                landmarks_json_cbct_u, landmarks_json_cbct_l,
                labels_U, labels_L
            )
            registered += 1
            logger.info(f"Patient {patient_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}", exc_info=True)
            continue
    
    logger.info("AREG_IOSCBCT processing completed")
    # Every patient can fail on a missing landmark file and the loop still ends
    # normally, which used to exit 0 and let Slicer report the whole pipeline as
    # a success while the output folder stayed empty.
    if not registered:
        raise RuntimeError(
            "No patient could be registered: check that every patient has an "
            "IOS surface, a CBCT, and the landmark files for both arches.")
    logger.info(f"{registered}/{len(patients)} patient(s) registered")


if __name__ == "__main__":
    try:
        logger.info("AREG_IOSCBCT entry point initiated")
        
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("IOS_folder", type=str)
            parser.add_argument("CBCT_folder", type=str)
            parser.add_argument("IOS_lm_folder", type=str)
            parser.add_argument("CBCT_lm_folder", type=str)
            parser.add_argument("output", type=str)

            args = parser.parse_args()
            logger.debug(f"Arguments parsed successfully")
        except Exception as e:
            logger.error(f"Error parsing command line arguments: {e}")
            raise

        try:
            logger.info("Calling main() function")
            main(args)
            logger.info("AREG_IOSCBCT completed successfully")
        except Exception as e:
            logger.error(f"Error in main() execution: {e}")
            raise

    except SystemExit as e:
        logger.info(f"Script exited with code: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.critical(f"Fatal error in entry point: {e}")
        sys.exit(f"Fatal error: {e}")
