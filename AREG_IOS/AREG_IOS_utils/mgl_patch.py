# Build the registration patch of the lower arch from the mucogingival (MGL)
# landmarks predicted by ALI_IOS.
#
# The upper arch uses a patch painted by a neural network on the palate. The
# mandible has no such stable plateau, but it has the mucogingival line: the
# 13 MG landmarks run along the arch and can be joined into a smooth curve. The
# band of surface around that curve plays the same role as the palatal patch,
# and is written as a 0/1 point array of the same shape, under its own name so
# a mandible is never labelled after the palate.
#
# Two properties matter for the result to be usable:
#   - every sample of the curve is snapped onto the mesh, because a curve
#     interpolated between landmarks floats off the surface in the concavities
#     between teeth;
#   - the band grows along the surface (geodesic), never through it, so a
#     buccal patch cannot leak onto the lingual side where the ridge is thin.
import heapq
import json
import logging
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("AREG_IOS_MGL")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Landmark names of the MG model, in arch order. L0MG is the midline (tooth 25),
# so the right side is shifted by one against the tooth numbers.
MGL_ORDER = ['LL6MG', 'LL5MG', 'LL4MG', 'LL3MG', 'LL2MG', 'LL1MG', 'L0MG',
             'LR1MG', 'LR2MG', 'LR3MG', 'LR4MG', 'LR5MG', 'LR6MG']

# Names written by predictions made before the MG suffix was added.
MGL_ORDER_LEGACY = [name[:-2] for name in MGL_ORDER]

# Name of the point array the patch is written to. The palatal patch is called
# "Butterfly" after its shape; the band along the mucogingival line is neither
# butterfly-shaped nor on the same arch, so it carries its own name.
MGL_ARRAY_NAME = "Bottom_MGL"

DEFAULT_RADIUS = 5.0        # mm, half-height of the band around the curve
                            # 0 leaves no band at all: the landmarks alone
DEFAULT_SAMPLES = 300       # samples along the spline

# Universal_ID labels of the lower teeth. The gingiva carries its own label, so
# the patch can be kept off the crowns, which are the structures that move
# between the two timepoints and must not drive the registration.
LOWER_TOOTH_LABELS = range(18, 32)


def DropDoubtfulLandmarks(landmarks, path):
    """Landmarks minus the ones ALI itself was not sure of.

    ALI records how it came by each point in the description of the markup:
    a point it forced out of the top pixels, one it fell back on the tooth for,
    one whose tooth was absent and whose cameras were aimed at an estimated
    position. Measured over 364 predictions, those sit a median 4.2 mm off the
    curve their neighbours draw, against 1.2 mm for the rest, so they are the
    ones that pull the mucogingival line out of shape.

    The curve is built from whatever is left, and a hole is nothing new: the
    spline spans it. All of them are kept when too few would remain, since a
    doubtful line still beats no registration.
    """
    try:
        with open(path) as f:
            markups = json.load(f)["markups"][0]["controlPoints"]
    except Exception as error:
        logger.warning(f"Could not read the landmark descriptions from {path}: {error}")
        return landmarks

    doubtful = {point["label"]: point["description"] for point in markups
                if point.get("description")}
    if not doubtful:
        return landmarks

    kept = {name: position for name, position in landmarks.items() if name not in doubtful}
    if len(kept) < 3:
        logger.warning(
            f"{len(doubtful)} of the {len(landmarks)} landmarks are flagged by ALI, "
            "too many to leave out: the line is built on all of them")
        return landmarks

    logger.info(f"Leaving out {len(doubtful)} landmark(s) ALI was unsure of: "
                + ", ".join(f"{name} ({reason})" for name, reason in sorted(doubtful.items())))
    return kept


def OrderedMGLandmarks(landmarks):
    """Return the MG landmark positions in arch order, as an (N, 3) array.

    Accepts both the current names (LL6MG...) and the older suffix-less ones,
    and tolerates missing teeth: a scan where ALI could not place every point
    still yields a usable curve, as long as three points remain.
    """
    for order in (MGL_ORDER, MGL_ORDER_LEGACY):
        points = [np.asarray(landmarks[name], dtype=float)
                  for name in order if name in landmarks]
        if len(points) >= 3:
            missing = [name for name in order if name not in landmarks]
            if missing:
                logger.warning(f"MG landmarks missing from the prediction: {missing}")
            return np.array(points)

    raise ValueError(
        "Fewer than 3 MG landmarks found. Expected names such as "
        f"{MGL_ORDER[:3]}, got {sorted(landmarks)}"
    )


def SplineThroughLandmarks(points, n_samples=DEFAULT_SAMPLES):
    """Sample a B-spline passing through `points`, as an (n_samples, 3) array.

    The landmarks are sparse (one per tooth), so the curve between them is an
    interpolation, not a measurement: it is only used to place the band.
    """
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(*point)

    spline = vtk.vtkParametricSpline()
    spline.SetPoints(vtk_points)
    spline.ClosedOff()

    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(spline)
    source.SetUResolution(n_samples)
    source.Update()

    return vtk_to_numpy(source.GetOutput().GetPoints().GetData())


def SnapToSurface(surf, samples):
    """Return, for each sample, the id of the closest vertex of `surf`.

    A spline drawn through landmarks that sit on the surface still leaves it
    between them, so the samples are snapped back before growing the band.
    Duplicate ids are removed: consecutive samples often land on one vertex.
    """
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surf)
    locator.BuildLocator()

    seeds = []
    for sample in samples:
        seeds.append(locator.FindClosestPoint(sample))
    return sorted(set(seeds))


def _adjacency(surf):
    """Neighbour ids of every vertex, from the mesh edges."""
    surf.BuildLinks()
    n_points = surf.GetNumberOfPoints()
    neighbours = [set() for _ in range(n_points)]

    cell_points = vtk.vtkIdList()
    for cell_id in range(surf.GetNumberOfCells()):
        surf.GetCellPoints(cell_id, cell_points)
        ids = [cell_points.GetId(i) for i in range(cell_points.GetNumberOfIds())]
        for a in ids:
            for b in ids:
                if a != b:
                    neighbours[a].add(b)
    return neighbours


def GrowBand(surf, seeds, radius):
    """Vertices within `radius` mm of a seed, measured along the surface.

    Growing along the mesh rather than through space is what keeps the band on
    the buccal side: a straight-line radius of a few millimetres would reach the
    lingual surface wherever the ridge is thinner than that.
    """
    points = vtk_to_numpy(surf.GetPoints().GetData())
    neighbours = _adjacency(surf)

    distance = np.full(surf.GetNumberOfPoints(), np.inf)
    queue = []
    for seed in seeds:
        distance[seed] = 0.0
        heapq.heappush(queue, (0.0, seed))

    while queue:
        dist, point_id = heapq.heappop(queue)
        if dist > distance[point_id]:
            continue
        for neighbour in neighbours[point_id]:
            step = float(np.linalg.norm(points[neighbour] - points[point_id]))
            new_dist = dist + step
            if new_dist < distance[neighbour] and new_dist <= radius:
                distance[neighbour] = new_dist
                heapq.heappush(queue, (new_dist, neighbour))

    return distance <= radius


def _tooth_mask(surf):
    """True where a vertex belongs to a tooth crown, False on the gingiva.

    All-False when the mesh carries no segmentation, so the caller keeps the
    whole band rather than losing the patch.
    """
    scalars = None
    for name in ("Universal_ID", "PredictedID", "UniversalID"):
        scalars = surf.GetPointData().GetScalars(name) or surf.GetPointData().GetArray(name)
        if scalars is not None:
            break

    if scalars is None:
        logger.warning("No teeth segmentation on the mesh, the patch is not kept off the crowns")
        return np.zeros(surf.GetNumberOfPoints(), dtype=bool)

    labels = vtk_to_numpy(scalars)
    return np.isin(labels, list(LOWER_TOOTH_LABELS))


def MGLPatch(surf, landmarks, radius=DEFAULT_RADIUS, n_samples=DEFAULT_SAMPLES,
             array_name=MGL_ARRAY_NAME, exclude_teeth=True):
    """Paint the band around the mucogingival line into `array_name`.

    Writes a 0/1 point array shaped like the palatal one, so the registration
    reads it the same way, under a name that says what it is. Returns the surface.

    A radius of 0 leaves no band and no curve either: the array then holds the
    landmarks alone and the registration runs on those points only.
    """
    points = OrderedMGLandmarks(landmarks)
    logger.info(f"Building the MGL patch from {len(points)} landmark(s), radius {radius} mm")

    if radius == 0:
        # Neither a band nor the curve joining the landmarks: the patch is the
        # landmarks themselves, so the ICP runs on those few points alone. Kept
        # as the control case, to measure on real scans what the surface around
        # the mucogingival line brings over the points that carry it.
        seeds = SnapToSurface(surf, points)
        logger.info(f"Height 0: registering on the {len(seeds)} landmark(s) "
                    "alone, without any surface around them")
        inside = np.zeros(surf.GetNumberOfPoints(), dtype=bool)
        inside[seeds] = True
    else:
        samples = SplineThroughLandmarks(points, n_samples)
        seeds = SnapToSurface(surf, samples)
        logger.debug(f"{len(samples)} spline sample(s) snapped onto {len(seeds)} vertex(es)")

        inside = GrowBand(surf, seeds, radius)

    if exclude_teeth:
        on_teeth = _tooth_mask(surf) & inside
        if on_teeth.any():
            what = ("landmark(s) that snapped onto a crown" if radius == 0
                    else "vertex(es) of the band that reached the crowns")
            logger.info(f"Dropping {int(on_teeth.sum())} {what}")
            inside = inside & ~on_teeth

    n_inside = int(inside.sum())
    if n_inside == 0:
        raise ValueError("The MGL patch is empty, the landmarks may not belong to this scan")
    logger.info(f"MGL patch: {n_inside} vertex(es) out of {surf.GetNumberOfPoints()}")

    array = numpy_to_vtk(inside.astype(np.int64))
    array.SetName(array_name)
    surf.GetPointData().AddArray(array)
    return surf
