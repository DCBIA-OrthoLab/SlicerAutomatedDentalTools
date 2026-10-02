# Bring a lower arch into the frame the mucogingival model was trained in.
#
# The MG cameras are built around a vertical axis taken to be Z: the arch
# tangent and the buccal normal are flattened onto the horizontal plane and the
# aim point is lowered toward the gum along Z (see agent.py). A scan that comes
# off the scanner in another pose therefore gets its cameras aimed at the wrong
# part of the arch, and the landmarks land on the crowns instead of the
# gingival margin -- on one dataset, nine of the thirteen.
#
# Rather than change camera geometry the model was trained with, the scan is
# rotated into that frame before the prediction and the landmarks are brought
# back afterwards. The rotation is read off four teeth, the way ASO and FlexReg
# already orient an arch.

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ADTLib.logging_setup import get_logger

logger = get_logger("ALI_IOS_orientation")

# Universal ids of the lower teeth. The occlusal plane is fitted through the
# centroids of every one the segmentation knows, rather than four named ones:
# a single missing tooth is common in these arches and would otherwise leave
# the scan uncorrected, which is what happened to one of twenty-eight.
LOWER_TEETH = tuple(range(18, 32))

# Fewest teeth, and shortest stretch of arch, the plane may be fitted through.
MIN_TEETH = 4
MIN_SPAN = 4

# Only the vertical axis is corrected, by the shortest rotation that brings the
# occlusal plane level. Sending the arch to a fixed frame would also turn it
# within that plane, and a scan that arrives properly laid out comes out of the
# prediction worse for it: measured on one, a landmark on a crown, two swapped
# along the arch and the band twice as loose.
UPRIGHT = np.array([0.0, 0.0, 1.0])

# How far the occlusal plane may lean before it is worth correcting. Under
# this, the scan is already in the frame the model was trained on and is left
# strictly untouched.
MAX_TILT_DEGREES = 12.0

# Array names a segmentation may carry, in the order they are looked for.
LABEL_ARRAYS = ("Universal_ID", "PredictedID", "UniversalID")


def _labels(surf):
    point_data = surf.GetPointData()
    for name in LABEL_ARRAYS:
        array = point_data.GetScalars(name) or point_data.GetArray(name)
        if array is not None:
            return vtk_to_numpy(array).ravel()
    return None


def _rotation_between(source, target):
    """Rotation bringing the unit vector `source` onto `target`.

    The dot product is clamped: rounding can push it past 1, where arccos
    returns NaN. Aligned and opposed vectors have a null cross product, which
    would normalise into NaN, so both are answered explicitly.
    """
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    axis = np.cross(source, target)

    if np.linalg.norm(axis) < 1e-8:
        if dot > 0:
            return np.identity(3)
        fallback = np.array([0.0, 1.0, 0.0]) if abs(source[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        axis = np.cross(source, fallback)

    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.identity(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _plane_normal(centroids):
    """Normal of the plane the tooth centroids lie closest to."""
    centred = centroids - centroids.mean(axis=0)
    normal = np.linalg.svd(centred)[2][2]
    return normal / np.linalg.norm(normal)


def LowerArchMatrix(surf, max_tilt=MAX_TILT_DEGREES):
    """4x4 rotation standing a segmented lower arch upright, or None.

    None when there is nothing to do -- the arch is already level, the scan
    carries no segmentation, or the teeth the plane is read from are missing --
    and the caller then works on the scan exactly as it came.
    """
    labels = _labels(surf)
    if labels is None:
        logger.warning("No teeth segmentation on the scan, it cannot be oriented")
        return None

    points = vtk_to_numpy(surf.GetPoints().GetData()).astype(float)
    present, centroids = [], []
    for tooth in LOWER_TEETH:
        selection = labels == tooth
        if selection.any():
            present.append(tooth)
            centroids.append(points[selection].mean(axis=0))

    if len(present) < MIN_TEETH or (max(present) - min(present)) < MIN_SPAN:
        logger.warning(f"Only {len(present)} lower teeth are segmented: not enough to "
                       "read the occlusal plane, the landmarks will be predicted on "
                       "the scan as it is")
        return None

    normal = _plane_normal(np.array(centroids))

    # The crowns must end up on top: the aim point of every camera is lowered
    # from the tooth toward the gum along the vertical axis.
    crowns = np.isin(labels, list(LOWER_TEETH))
    if crowns.any() and not crowns.all():
        upward = points[crowns].mean(axis=0) - points[~crowns].mean(axis=0)
        if np.dot(normal, upward) < 0:
            normal = -normal

    tilt = np.degrees(np.arccos(float(np.clip(np.dot(normal, UPRIGHT), -1.0, 1.0))))
    if tilt <= max_tilt:
        logger.info(f"The arch leans {tilt:.0f}deg off upright, close enough to "
                    "the frame of the model: the scan is left as it is")
        return None

    logger.info(f"The arch leans {tilt:.0f}deg off upright: standing it up for the prediction")
    matrix = np.identity(4)
    matrix[:3, :3] = _rotation_between(normal, UPRIGHT)
    return matrix


def TransformSurf(surf, matrix):
    """A copy of `surf` with `matrix` applied to its points."""
    points = vtk_to_numpy(surf.GetPoints().GetData()).astype(float)
    moved = points @ matrix[:3, :3].T + matrix[:3, 3]

    output = vtk.vtkPolyData()
    output.DeepCopy(surf)
    output.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(moved), deep=True))
    return output


def TransformPoint(position, matrix):
    """A single point through `matrix`, as a plain list."""
    moved = matrix[:3, :3] @ np.asarray(position, dtype=float) + matrix[:3, 3]
    return moved.tolist()


# How wide the framing is made: the scan extent the normalisation is told to
# see, as a multiple of the distance between the first molars.
#
# Not the training corpus average (1.08) but the value the model actually works
# best at, found by sweeping it: widening the framing costs accuracy, gently at
# first and then sharply -- at 1.18 the share of landmarks within 2 mm fell from
# 80% to 69%. Held at 0.98 the corpus improves (1.05 -> 0.95 mm) and a dataset
# that arrives at 1.24, framed far wider, improves more.
TRAINED_SCAN_TO_ARCH = 0.98

# The two molars the arch is measured across: the ends of the MG line.
ARCH_TEETH = (19, 30)


def ArchScale(surf, ratio=TRAINED_SCAN_TO_ARCH):
    """Unit-sphere scale factor read off the arch instead of the scan extent.

    ScaleSurf divides by the bounding box half-diagonal, which measures how
    much was scanned rather than the patient: a scan that captured more
    vestibule comes out smaller, and the cameras -- placed at a fixed distance
    in that normalised space -- then sit further away and aim lower. Between
    the training corpus and another dataset that difference measured 13%, which
    put the aim 1.1 mm below where it was trained to be, on a band 5 mm tall.

    Scaling on the distance between the first molars instead makes the framing
    depend on the jaw and not on the operator. The ratio is the one the
    training corpus had, so a scan like those is scaled as it always was.

    None when the two molars are not both segmented, leaving ScaleSurf to its
    own measure.
    """
    labels = _labels(surf)
    if labels is None:
        return None

    points = vtk_to_numpy(surf.GetPoints().GetData()).astype(float)
    centres = []
    for tooth in ARCH_TEETH:
        selection = labels == tooth
        if not selection.any():
            logger.info(f"Tooth {tooth} is not segmented, the scan keeps the scale "
                        "read from its own extent")
            return None
        centres.append(points[selection].mean(axis=0))

    width = float(np.linalg.norm(centres[0] - centres[1]))
    if width < 1e-6:
        return None
    return 1.0 / (ratio * width)
