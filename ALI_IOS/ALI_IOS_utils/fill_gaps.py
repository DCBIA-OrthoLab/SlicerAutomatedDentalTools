# Put back a mucogingival landmark the prediction could not give.
#
# A tooth absent from the scan gets no camera, and a point the network was
# unsure of is worth little: either way the line comes out with a hole, which
# is awkward to work with clinically. The line is smooth, so a hole between two
# points that are trusted can be filled by following the curve through them.
#
# Measured by hiding, one at a time, points whose tooth was really there and
# rebuilding them from the rest: 1.53 mm from the hand annotation, 68% within
# 2 mm, against 1.20 mm and 67% for the network's own prediction of the same
# points. As good as predicting, in other words.
#
# Not at the ends of the arch: there is nothing on one side to follow, and
# extrapolating there landed 9.7 mm away, none of it within 2 mm.
import re

import numpy as np

from ADTLib.logging_setup import get_logger

logger = get_logger("ALI_IOS_gaps")

# The 13 MG points, in arch order: the position in this list is the parameter
# the curve is followed along.
MGL_ORDER = ['LL6MG', 'LL5MG', 'LL4MG', 'LL3MG', 'LL2MG', 'LL1MG', 'L0MG',
             'LR1MG', 'LR2MG', 'LR3MG', 'LR4MG', 'LR5MG', 'LR6MG']

# Under this the network was, on the corpus it was trained on, wrong by 3.8 mm
# where a confident point is wrong by 1.0 mm: too little to build a curve on.
MIN_CONFIDENCE = 0.785

# Fewest trusted points a curve may be drawn through.
MIN_SUPPORT = 4

# How far along the arch a trusted point may be and still be an anchor for a
# hole. The 1.53 mm above was measured by hiding ONE point at a time, so it
# says what the curve is worth across a hole its neighbours border -- nothing
# about a longer one. Left unbounded, a run of consecutive holes is rebuilt
# from a spline with no support inside it: measured on a scan where six
# landmarks in a row went missing, the rebuilt points landed 4 to 32 mm away,
# which is far worse than the hole they filled.
MAX_SUPPORT_DISTANCE = 2

REBUILT_NOTE = "rebuilt from its neighbours"



def _confidence(description):
    found = re.search(r"confidence ([0-9.]+)", (description or "").lower())
    return float(found.group(1)) if found else 1.0


def _trusted(name, entry):
    """A point worth drawing the curve through: predicted outright, and sure."""
    description = (entry.get("desc") or "").lower()
    if any(mark in description for mark in ("forced", "fallback", "arch fit",
                                            "off the aim", REBUILT_NOTE)):
        return False
    return _confidence(description) >= MIN_CONFIDENCE


def FillGaps(group_data, minimum_confidence=MIN_CONFIDENCE):
    """Fill the holes of an MG landmark set, in place. Returns the names filled.

    A hole is a point missing altogether or one the network was unsure of. It
    is rebuilt only when trusted points lie on both sides of it, and it is
    marked as rebuilt so nothing downstream mistakes it for a measurement.
    """
    from scipy.interpolate import CubicSpline

    support = [(index, name) for index, name in enumerate(MGL_ORDER)
               if name in group_data and _trusted(name, group_data[name])]
    if len(support) < MIN_SUPPORT:
        logger.info(f"Only {len(support)} trusted landmark(s): the line is left as it is")
        return []

    indices = np.array([index for index, _ in support], dtype=float)
    points = np.array([[group_data[name]["x"], group_data[name]["y"], group_data[name]["z"]]
                       for _, name in support])
    curve = CubicSpline(indices, points, axis=0)

    filled = []
    for index, name in enumerate(MGL_ORDER):
        if name in group_data and _trusted(name, group_data[name]):
            continue
        if not (indices.min() < index < indices.max()):
            # an end of the arch: nothing to follow on one side
            continue
        before = index - indices[indices < index].max()
        after = indices[indices > index].min() - index
        if max(before, after) > MAX_SUPPORT_DISTANCE:
            # the nearest trusted point is too far along the arch for the
            # curve between them to mean anything here
            logger.info(f"{name}: the nearest trusted landmark is "
                        f"{int(max(before, after))} places away, too far to "
                        "rebuild from; the hole is left")
            continue
        position = curve(float(index))
        group_data[name] = {"x": float(position[0]), "y": float(position[1]),
                            "z": float(position[2]), "desc": REBUILT_NOTE}
        filled.append(name)

    if filled:
        logger.info(f"Rebuilt {len(filled)} landmark(s) from the curve through the "
                    f"{len(support)} trusted ones: {', '.join(filled)}")
    return filled

