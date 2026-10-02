# Place a mucogingival landmark the prediction could not give, as well as it
# can be placed without the network.
#
# Two holes need two answers, and the difference is measured:
#
#   inside the line, a cubic spline through the points around it is excellent --
#   0.52 mm from where the network puts such a point, 99% of them within 2 mm;
#
#   at an end of the arch there is nothing to follow on one side, and the same
#   spline runs away: 5.1 mm, 6% within 2 mm, and on real scans it has landed
#   27 mm out, off the mesh. What works there is the tooth's own gingival
#   collar -- the ring where its crown meets the gingiva, read off the mesh --
#   with the crown-to-landmark offset copied from the single nearest landmark
#   the scan does have. That is 1.44 mm and 70% within 2 mm.
#
# And whatever placed it, the answer is put back on the mesh. The mucogingival
# point is on the mucosa by definition, and snapping is what turned 2.6 mm into
# 1.4 mm at the ends -- the single biggest gain of the lot, bigger than the
# choice of method.
#
# Chosen on 163 scans and quoted on the 41 held back, split once by scan. The
# two agree to the decimal (1.44 both sides), so none of this is fitted to the
# scans it was picked on. A straight line from the last two points was tried
# and is worse everywhere once snapping is on; so is reading the point off its
# mirror image across the arch, by a long way.

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from ADTLib.logging_setup import get_logger

logger = get_logger("ALI_IOS_complete")

MGL_ORDER = ['LL6MG', 'LL5MG', 'LL4MG', 'LL3MG', 'LL2MG', 'LL1MG', 'L0MG',
             'LR1MG', 'LR2MG', 'LR3MG', 'LR4MG', 'LR5MG', 'LR6MG']
INDEX = {name: i for i, name in enumerate(MGL_ORDER)}
TOOTH_OF = {name: 19 + i for i, name in enumerate(MGL_ORDER)}

# The two ends. They are the only places the spline has nothing to interpolate
# between, and between them they are most of what goes missing on a real run.
ARCH_ENDS = ("LL6MG", "LR6MG")

# What a point placed here says about itself. Kept apart from FillGaps's
# "rebuilt from its neighbours": that one is bordered by trusted points, this
# one is what is left when nothing measured anything nearby. AREG and FlexReg
# match on the word "extrapolated" and leave such a point out of the band, so
# a placement is never mistaken for a measurement.
EXTRAPOLATED_NOTE = "extrapolated, no measurement near it"

LOWER_TOOTH_LABELS = range(17, 33)
MIN_SUPPORT = 4


def _labels(surf):
    for name in ("Universal_ID", "PredictedID", "UniversalID"):
        array = surf.GetPointData().GetScalars(name) or surf.GetPointData().GetArray(name)
        if array is not None:
            return vtk_to_numpy(array).astype(int)
    return None


def CollarFrames(surf):
    """Per tooth, its buccal gingival collar and the frame it sits in.

    The collar is the ring of crown vertices that touch something else -- the
    gingiva, or the neighbouring tooth -- kept to the half facing the cheek.
    It beats the crown's centroid as an anchor because a terminal molar is a
    big tooth, and because the landmark is a fixed step below the collar rather
    than a fixed step from the middle of the crown.
    """
    labels = _labels(surf)
    if labels is None:
        return None

    points = vtk_to_numpy(surf.GetPoints().GetData())
    teeth = [int(t) for t in np.unique(labels) if int(t) in LOWER_TOOTH_LABELS]
    if len(teeth) < 3:
        return None

    centres = np.array([points[labels == t].mean(axis=0) for t in teeth])
    middle = centres.mean(axis=0)
    # the arch spreads in two directions and is thin in the third: that one is
    # the vertical, whatever pose the scan happens to be in
    _, _, axes = np.linalg.svd(centres - middle)
    vertical = axes[2]

    surf.BuildLinks()
    ids = vtk.vtkIdList()
    border = {t: set() for t in teeth}
    for cell in range(surf.GetNumberOfCells()):
        surf.GetCellPoints(cell, ids)
        corners = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        seen = {int(labels[i]) for i in corners}
        if len(seen) == 1:
            continue
        for i in corners:
            tooth = int(labels[i])
            if tooth in border:
                border[tooth].add(i)

    frames = {}
    for tooth, centre in zip(teeth, centres):
        ring = sorted(border[tooth])
        if len(ring) < 6:
            continue
        ring_points = points[ring]
        outward = centre - middle
        outward = outward - np.dot(outward, vertical) * vertical
        length = np.linalg.norm(outward)
        if length < 1e-6:
            continue
        outward = outward / length
        side = (ring_points - centre) @ outward
        buccal = ring_points[side > np.median(side)]
        if len(buccal) < 3:
            buccal = ring_points
        frames[tooth] = (buccal.mean(axis=0),
                         np.column_stack([outward, np.cross(vertical, outward), vertical]))
    return frames or None


def _positions(group_data):
    return {name: np.array([group_data[name][axis] for axis in "xyz"], dtype=float)
            for name in group_data}


def _from_collar(present, name, frames):
    """The point its own tooth's collar puts it at, offset copied from a neighbour."""
    tooth = TOOTH_OF.get(name)
    if tooth not in frames:
        return None
    usable = [other for other in present if TOOTH_OF.get(other) in frames]
    if not usable:
        return None
    # the single nearest along the arch: a neighbour two teeth away describes a
    # different part of the vestibule, and averaging several measured worse
    nearest = min(usable, key=lambda other: abs(INDEX[other] - INDEX[name]))
    base, rotation = frames[TOOTH_OF[nearest]]
    offset = rotation.T @ (present[nearest] - base)
    base, rotation = frames[tooth]
    return base + rotation @ offset


def _from_spline(present, name):
    from scipy.interpolate import CubicSpline
    names = [other for other in MGL_ORDER if other in present]
    if len(names) < MIN_SUPPORT:
        return None
    x = np.array([INDEX[other] for other in names], dtype=float)
    y = np.array([present[other] for other in names])
    return CubicSpline(x, y, axis=0, extrapolate=True)(float(INDEX[name]))


def SnapToSurface(point, surf):
    """The closest point of the mesh. The landmark is on the mucosa, not above it."""
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(surf)
    locator.BuildLocator()
    return _snap_with(point, locator)


def _snap_with(point, locator):
    closest = [0.0, 0.0, 0.0]
    cell, sub, squared = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    locator.FindClosestPoint([float(v) for v in point], closest,
                             vtk.vtkGenericCell(), cell, sub, squared)
    return np.array(closest)


def CompleteLine(group_data, surf=None, note=None):
    """Give the line its 13 points. Returns the names filled.

    With the scan in hand an end of the arch is placed off its own tooth and
    every answer is put back on the mesh; without it, only the spline is left
    and an end of the arch is the 5 mm guess it always was.
    """
    note = note or EXTRAPOLATED_NOTE

    missing = [name for name in MGL_ORDER if name not in group_data]
    if not missing:
        return []

    present = _positions(group_data)
    if len(present) < MIN_SUPPORT:
        logger.info(f"Only {len(present)} landmark(s) on the line: too few to "
                    "place the rest from")
        return []

    frames = CollarFrames(surf) if surf is not None else None
    if surf is not None and frames is None:
        logger.info("No teeth labels on the scan: the ends of the arch fall back "
                    "on the spline, which is a 5 mm guess there")
    locator = None
    if surf is not None:
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(surf)
        locator.BuildLocator()

    filled, how = [], []
    for name in missing:
        answer, way = None, None
        if name in ARCH_ENDS and frames is not None:
            answer = _from_collar(present, name, frames)
            way = "collar"
        if answer is None:
            answer = _from_spline(present, name)
            way = "spline"
        if answer is None:
            continue
        if locator is not None:
            answer = _snap_with(answer, locator)
        group_data[name] = {"x": float(answer[0]), "y": float(answer[1]),
                            "z": float(answer[2]), "desc": note}
        filled.append(name)
        how.append(f"{name} ({way})")

    if filled:
        logger.info(f"Completing the line with {len(filled)} point(s) nothing "
                    f"measured, marked as such: {', '.join(how)}")
    return filled


def SnapAll(group_data, surf):
    """Put every landmark back on the mesh. Returns how far each one moved.

    A mucogingival point is on the mucosa: that is what the word means. Three
    steps of this pipeline can leave it hanging in the air, and none of them
    means to -- the spline through the neighbours does not follow the surface,
    and neither does the pull toward the line that smoothing applies. Measured
    on one timepoint of 14 patients: the points the network places itself sit
    exactly on the mesh, the ones rebuilt from their neighbours a median
    0.60 mm above it and as much as 4.89 mm.

    Done once, at the end, after everything has had its say -- snapping before
    the smoothing would simply be undone by it.
    """
    if surf is None or not group_data:
        return {}

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(surf)
    locator.BuildLocator()

    moved = {}
    for name, entry in group_data.items():
        before = np.array([entry[axis] for axis in "xyz"], dtype=float)
        after = _snap_with(before, locator)
        step = float(np.linalg.norm(after - before))
        if step > 1e-9:
            entry["x"], entry["y"], entry["z"] = (float(after[0]), float(after[1]),
                                                  float(after[2]))
            moved[name] = step

    if moved:
        steps = np.array(list(moved.values()))
        logger.info(f"Put {len(moved)} landmark(s) back on the mesh, by a median "
                    f"{np.median(steps):.2f} mm (largest {steps.max():.2f} mm)")
    return moved
