# Settle the mucogingival landmarks onto the line they belong to.
#
# The network places each point on its own, from three images of one tooth, and
# knows nothing of the twelve others. The mucogingival line, however, is one
# continuous curve: a point that sits off the curve its neighbours draw is
# wrong far more often than they all are. Pulling each point part of the way
# back onto that curve trades a little of its own evidence for the agreement of
# the rest.
#
# Only part of the way. Moving a point all the way onto its neighbours' curve
# throws its own measurement away and lands 1.60 mm from the annotation with
# 65% within 2 mm; leaving it alone gives 0.70 mm and 88%. A fifth to a third
# of the way is where the two work together.
#
# Measured on the 19 held-out scans, after the second look at each tooth:
#            strength   median   within 2 mm   90th pct
#                 0%    0.69 mm      89%        2.1 mm
#                20%    0.66 mm      90%        1.9 mm
#                30%    0.68 mm      91%        1.8 mm
# The tail is what moves: the worst points come back, the good ones barely
# stir. Beware that the strength was chosen on those same scans, so the gain it
# reports for itself is the optimistic end of what to expect.
#
# Not at the ends of the arch: the curve through the others does not reach
# them, and following it out that far is extrapolation.

import numpy as np

from ADTLib.logging_setup import get_logger

logger = get_logger("ALI_IOS_smooth")

# The 13 MG points, in arch order.
MGL_ORDER = ['LL6MG', 'LL5MG', 'LL4MG', 'LL3MG', 'LL2MG', 'LL1MG', 'L0MG',
             'LR1MG', 'LR2MG', 'LR3MG', 'LR4MG', 'LR5MG', 'LR6MG']

DEFAULT_STRENGTH = 0.3      # how far along, from the point to its neighbours' curve
MIN_SUPPORT = 5             # fewest points for the curve through the others to mean anything
SAMPLES = 400               # how finely that curve is walked to find the nearest spot

# Left in the description so a point that was nudged says so. Deliberately
# worded clear of the marks AREG and FlexReg read as doubtful, since a smoothed
# point is still a predicted one.
SMOOTHED_NOTE = "settled onto the line"


def _curve(points, n_samples=SAMPLES):
    """Walk a cubic spline through `points`, returned as (n_samples, 3)."""
    from scipy.interpolate import CubicSpline

    t = np.arange(len(points), dtype=float)
    fine = np.linspace(0, len(points) - 1, n_samples)
    return CubicSpline(t, points, axis=0)(fine)


def SmoothAlongArch(group_data, strength=DEFAULT_STRENGTH):
    """Pull each interior MG landmark toward its neighbours' curve, in place.

    Every point is moved from where the network left it, so the shifts are all
    computed before any of them is applied. Returns the names that moved.
    """
    if strength <= 0:
        return []

    names = [name for name in MGL_ORDER if name in group_data]
    if len(names) < MIN_SUPPORT:
        logger.info(f"Only {len(names)} MG landmark(s): the line is left as it is")
        return []

    positions = np.array([[group_data[name]["x"], group_data[name]["y"], group_data[name]["z"]]
                          for name in names])

    moved, shifts = [], []
    for index in range(1, len(names) - 1):
        others = np.delete(positions, index, axis=0)
        curve = _curve(others)
        nearest = curve[np.linalg.norm(curve - positions[index], axis=1).argmin()]
        settled = (1 - strength) * positions[index] + strength * nearest

        entry = group_data[names[index]]
        shifts.append(float(np.linalg.norm(settled - positions[index])))
        entry["x"], entry["y"], entry["z"] = (float(settled[0]), float(settled[1]),
                                              float(settled[2]))
        description = entry.get("desc") or ""
        entry["desc"] = f"{description}, {SMOOTHED_NOTE}" if description else SMOOTHED_NOTE
        moved.append(names[index])

    if moved:
        logger.info(f"Settled {len(moved)} landmark(s) {int(strength * 100)}% of the way onto "
                    f"the line their neighbours draw, by a median {np.median(shifts):.2f} mm "
                    f"(largest {max(shifts):.2f} mm)")
    return moved
