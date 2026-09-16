"""Repair of a scan the segmentation numbered in both jaws at once.

The crown segmentation names every point on its own, and nothing in a local
neighbourhood says which jaw the scan is: a maxilla and a mirrored mandible have
the same shape, only the palate tells them apart. On an arch it cannot place, it
splits each tooth between its own number and the same rank in the other arch.

That costs this module its orientation. PRE_ASO_IOS fits an arch on three or
four named teeth, and a tooth whose points all went to the other jaw's number is
simply gone: the ICP raises ToothNoExist and the whole arch is dropped, before
ALI_IOS -- which carries the same repair for the scans it is handed directly --
ever sees it.

Kept in step with ALI_IOS_utils.surface.UnifyArchLabels on purpose. The two CLIs
are separate Slicer modules with no shared package, so the logic is duplicated
rather than imported; change one and change the other.
"""
import logging

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

logger = logging.getLogger(__name__)

# Universal numbering runs 1-16 over the upper arch from the patient's right,
# then 17-32 over the lower from the left. Tooth t and tooth t+16 hold the same
# rank in their own arch, which puts them on opposite sides of the mouth.
ARCH_OFFSET = 16
# Two labels whose points share a centre this closely are one tooth, not two.
# Measured on the scan this was written for, the two families sit 2.8 to 8.0 mm
# apart while every other pairing of the same labels starts at 18.9 mm, so the
# threshold sits in the middle of that gap. A scan genuinely holding both arches
# puts t and t+16 on opposite sides AND opposite jaws, far past this.
SAME_TOOTH_MM = 12.0
# One coincidence is a stray patch. A numbering that has split shows on the arch.
MIN_SPLIT_TEETH = 3


def ArchLabelSplit(labels, points, max_distance=SAME_TOOTH_MM):
    """Teeth carrying both an upper and a lower number, as [(upper, mm), ...]."""
    labels = np.asarray(labels).ravel()
    points = np.asarray(points)
    split = []
    for upper in range(1, ARCH_OFFSET + 1):
        lower = upper + ARCH_OFFSET
        here, there = labels == upper, labels == lower
        if not here.any() or not there.any():
            continue
        gap = float(np.linalg.norm(points[here].mean(axis=0) - points[there].mean(axis=0)))
        if gap <= max_distance:
            split.append((upper, gap))
    return split


def UnifyArchLabels(surf, jaw, property_name="Universal_ID"):
    """Renumber a doubly-numbered arch into `jaw`'s numbering, in place.

    Returns the number of points moved, 0 when there was nothing to repair.
    `jaw` is the arbiter and has to come from outside the geometry: an isolated
    arch does not determine its own left and right until the jaw is known.

    Every point of the losing family is moved, not only those on the teeth
    caught carrying both numbers. Where a tooth was named in the wrong
    numbering alone it is missing from the split entirely, and those are exactly
    the ones whose absence stops this module.
    """
    if jaw not in ("Upper", "Lower"):
        return 0
    array = surf.GetPointData().GetScalars(property_name)
    if array is None:
        array = surf.GetPointData().GetArray(property_name)
    if array is None:
        return 0

    labels = vtk_to_numpy(array)
    points = vtk_to_numpy(surf.GetPoints().GetData())
    split = ArchLabelSplit(labels, points)
    if len(split) < MIN_SPLIT_TEETH:
        return 0

    if jaw == "Lower":
        wrong = (labels >= 1) & (labels <= ARCH_OFFSET)
        shift = ARCH_OFFSET
    else:
        wrong = (labels > ARCH_OFFSET) & (labels <= 2 * ARCH_OFFSET)
        shift = -ARCH_OFFSET

    moved = int(wrong.sum())
    if not moved:
        return 0

    labels[wrong] += shift
    array.Modified()
    logger.warning(
        "%d teeth are numbered twice, once in each arch (%s), so the segmentation "
        "could not tell which jaw this scan is. Reading it as %s from its name and "
        "moving %d point(s) onto that numbering."
        % (len(split), ", ".join("%d/%d at %.1f mm" % (t, t + ARCH_OFFSET, d)
                                 for t, d in split[:4]), jaw.lower(), moved))
    return moved
