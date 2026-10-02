# Keep the marked faces that can belong to the tooth the cameras were aimed at,
# and drop the ones that belong to its neighbour.
#
# The MG network segments its images into "mucogingival point" and background.
# It is never told which tooth it is looking at, so it answers "where is a
# mucogingival point in this picture" -- and the picture of one tooth also
# shows its neighbours' gingiva. When two of those points win, every face they
# cover is averaged into one position, which lands between them, on a spot that
# belongs to neither. The average is then snapped to the nearest vertex, so two
# neighbouring landmarks can come out bit-for-bit identical.
#
# Measured on 14 scans: 8 pairs of neighbouring landmarks closer than 1 mm, two
# of them at exactly the same position. Reproduced with a second, unrelated
# segmentation of the same scans, and present on the corpus the model was
# trained on, so it is the prediction and not the input.
#
# Narrowing the render removes the neighbours from the picture and does fix it,
# and costs far too much: at a 0.70 framing the collapses go to zero while the
# error against hand annotations goes from 0.76 mm to 1.70 mm and the share
# under 2 mm from 84% to 59%. The network needs the wide view to be accurate,
# so the ambiguity has to be resolved after the render, not before.
#
# The rule: a face is kept when it lies closer to the aim than half the way to
# the neighbouring tooth. Half the tooth pitch is measured on the scan itself,
# from the segmentation, so nothing here carries a tuned distance -- a crowded
# arch and a spaced one get their own bound. It is deliberately generous: the
# aim is an anatomical prior, not a measurement, and the point of this is to
# exclude the neighbour's answer, not to second-guess the network's.

import numpy as np

# The 13 MG landmarks in arch order: neighbours in this list are the pairs
# that can end up on one spot.
MGL_ORDER = ['LL6MG', 'LL5MG', 'LL4MG', 'LL3MG', 'LL2MG', 'LL1MG', 'L0MG',
             'LR1MG', 'LR2MG', 'LR3MG', 'LR4MG', 'LR5MG', 'LR6MG']

from ADTLib.logging_setup import get_logger

logger = get_logger("ALI_IOS_patch")


def ToothPitch(region_ids, vertices, minimum_teeth=4):
    """Median distance between neighbouring tooth centres, on this scan.

    The teeth are ordered around the arch rather than by label, so a gap in the
    numbering -- a tooth the segmentation does not have -- does not turn into a
    double-width pitch. None when there are too few teeth to measure one.
    """
    region_ids = np.asarray(region_ids).ravel()
    labels = [int(v) for v in np.unique(region_ids) if 17 <= int(v) <= 32]
    if len(labels) < minimum_teeth:
        return None

    centres = np.array([vertices[region_ids == label].mean(axis=0) for label in labels])
    middle = centres.mean(axis=0)
    # the arch is a horseshoe: going round it by angle is going along it
    angles = np.arctan2(centres[:, 1] - middle[1], centres[:, 0] - middle[0])
    ordered = centres[np.argsort(angles)]
    steps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    return float(np.median(steps)) if len(steps) else None


def PickNearAim(face_ids, face_vertices, vertex_positions, aim, radius, name=""):
    """(faces within `radius` of `aim`, whether none of them was).

    Two things happen when a neighbour's mucogingival point is marked in this
    tooth's picture, and they need different answers:

    - both points are marked. The neighbour's faces are dropped here and this
      tooth keeps its own answer, which is what the bound is for.
    - only the neighbour's is marked. Nothing can be recovered -- this tooth's
      point was never found -- so the faces are kept as they are and the caller
      is told, in order to record the landmark as one not to be trusted.
      Measured, that is the commoner of the two: filtering the faces alone took
      8 collapsed pairs to 7, while what makes the difference is that the point
      stops being passed off as a measurement.
    """
    if radius is None or not len(face_ids):
        return face_ids, False

    faces = np.array([int(f) for f in face_ids])
    aim = np.asarray(aim, dtype=float).reshape(3)
    centres = np.asarray(vertex_positions, dtype=float)[face_vertices[faces]].mean(axis=1)
    near = np.linalg.norm(centres - aim, axis=1) <= radius

    if near.all():
        return face_ids, False
    if not near.any():
        logger.info(f"{name or 'landmark'}: every marked face is further than "
                    f"{radius:.3f} from the aim -- the point found is not this "
                    "tooth's, recording it as doubtful")
        return face_ids, True

    logger.info(f"{name or 'landmark'}: dropping {int((~near).sum())} of "
                f"{len(faces)} marked face(s), further from the aim than half "
                f"the way to the next tooth ({radius:.3f})")
    return [face for face, keep in zip(face_ids, near) if keep], False


# What the description says when the network marked nothing near the aim. The
# tools downstream read it: AREG leaves such a point out of the band it builds,
# and ALI rebuilds it from its neighbours rather than keeping it.
OFF_AIM_NOTE = "off the aim"


def ResolveCollisions(group_data, aims, pitch, note=None):
    """Mark, of two landmarks sitting on one spot, the one that is not its own.

    Two mucogingival landmarks cannot share a position: they belong to
    neighbouring teeth, one tooth apart. When they do, the network found the
    same point twice -- it is never told which tooth it is looking at, and one
    picture shows its neighbours' gingiva as well.

    Filtering the marked faces catches this only when both points were marked
    in the same picture; more often only the neighbour's is, and then there is
    nothing to filter. What can still be decided is which of the two claims the
    spot: the one whose own aim it is nearer. The other is marked, so that
    FillGaps rebuilds it from the curve its neighbours draw rather than leaving
    a duplicate, and so that AREG leaves it out of the band.

    `aims` holds the aim of each landmark, in the same frame as the positions.
    Only pairs closer than half the tooth pitch are considered: that is one
    tooth's width measured on this scan, so it says "these two are on the same
    spot" without a tuned distance.

    Returns the names marked.
    """
    if pitch is None or len(group_data) < 2:
        return []

    names = [name for name in MGL_ORDER if name in group_data]
    marked = []
    for first, second in zip(names, names[1:]):
        if first in marked or second in marked:
            continue
        a = np.array([group_data[first][axis] for axis in "xyz"], dtype=float)
        b = np.array([group_data[second][axis] for axis in "xyz"], dtype=float)
        if float(np.linalg.norm(a - b)) > pitch / 2:
            continue
        if first not in aims or second not in aims:
            continue

        # whichever of the two the spot is really the landmark of keeps it
        loser = (second if np.linalg.norm(a - np.asarray(aims[first], dtype=float))
                 <= np.linalg.norm(b - np.asarray(aims[second], dtype=float)) else first)
        entry = group_data[loser]
        existing = entry.get("desc")
        entry["desc"] = f"{existing}; {note}" if existing else note
        marked.append(loser)
        logger.info(
            f"{first} and {second} are {np.linalg.norm(a - b):.2f} mm apart, closer "
            f"than two teeth can be: {loser} is the further from its own aim and is "
            "recorded as doubtful")

    return marked
