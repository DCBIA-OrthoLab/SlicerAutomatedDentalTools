# Paint the mucogingival line onto the scan the user gave, so it can be looked
# at by opening that file instead of loading a markups file beside it.
#
# Two arrays are added, and nothing else is touched: the geometry, the teeth
# labels and every other array are written back as they were.
#
#   Bottom_MGL     the band around the line, 0/1 per vertex -- the same array
#                  AREG registers on, so what is seen here is what is used
#   MG_landmarks   the 13 points themselves, 0/1 per vertex, for reading the
#                  line off the mesh without a second file
#
# The band builder below is a copy of AREG_IOS_utils/mgl_patch.py rather than
# an import of it. Each Slicer module loads its own utils, so importing across
# them would make one module's missing dependency take out the other -- the
# same reason ALI and ASO each carry their own dicom.py. What it buys is that
# the band drawn here cannot drift from the band the registration stands on;
# what it costs is that a change to one has to be made in both.
import heapq
import os
import re
import tempfile

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ADTLib.logging_setup import get_logger

logger = get_logger("ALI_IOS_paint")

# Landmark names of the MG model, in arch order.
MGL_ORDER = ['LL6MG', 'LL5MG', 'LL4MG', 'LL3MG', 'LL2MG', 'LL1MG', 'L0MG',
             'LR1MG', 'LR2MG', 'LR3MG', 'LR4MG', 'LR5MG', 'LR6MG']
MGL_ORDER_LEGACY = [name[:-2] for name in MGL_ORDER]

MGL_ARRAY_NAME = "Bottom_MGL"
LANDMARK_ARRAY_NAME = "MG_landmarks"

DEFAULT_RADIUS = 5.0        # mm, half-height of the band around the curve
DEFAULT_SAMPLES = 300       # samples along the spline

# Universal_ID labels of the lower teeth, third molars included: the crowns
# move between two timepoints and must never end up inside the band.
LOWER_TOOTH_LABELS = range(17, 33)

# What a description says when the point is not a plain prediction, and the
# confidence below which the network was, on the corpus it was trained on,
# wrong by 3.8 mm where a sure point is wrong by 1.0 mm. Copied from AREG's
# mgl_patch for the same reason as the band itself: the band drawn here is
# grown from the same landmarks AREG grows its own from, so a point AREG will
# not stand on must not appear to hold this one up either.
DOUBTFUL_MARKS = ("forced", "fallback", "arch fit", "off the aim", "extrapolated")
MIN_CONFIDENCE = 0.785


def IsDoubtful(description):
    """True when a description says the point is not to be trusted."""
    text = (description or "").lower()
    if any(mark in text for mark in DOUBTFUL_MARKS):
        return True
    found = re.search(r"confidence ([0-9.]+)", text)
    return bool(found) and float(found.group(1)) < MIN_CONFIDENCE


def TrustedOnly(landmarks, descriptions):
    """The landmarks AREG would build its band on, out of all of them.

    All of them when too few would be left: a doubtful line still beats none,
    which is the rule AREG applies as well.
    """
    if not descriptions:
        return landmarks
    kept = {name: position for name, position in landmarks.items()
            if not IsDoubtful(descriptions.get(name))}
    if len(kept) < 3:
        return landmarks
    left_out = sorted(set(landmarks) - set(kept))
    if left_out:
        logger.info(f"The band is grown without {', '.join(left_out)}, which "
                    "ALI was not sure of -- the points themselves are still marked")
    return kept


# Formats that can carry a point array at all. An .stl or an .obj holds
# geometry and nothing else, so there is nowhere to paint.
PAINTABLE = (".vtk", ".vtp")


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


def MarkLandmarks(surf, landmarks, array_name=LANDMARK_ARRAY_NAME):
    """Mark the vertex nearest each landmark, as a 0/1 point array."""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surf)
    locator.BuildLocator()

    marked = np.zeros(surf.GetNumberOfPoints(), dtype=np.int64)
    for position in landmarks.values():
        marked[locator.FindClosestPoint(np.asarray(position, dtype=float))] = 1

    array = numpy_to_vtk(marked)
    array.SetName(array_name)
    surf.GetPointData().AddArray(array)
    return int(marked.sum())


def _read(path):
    reader = vtk.vtkXMLPolyDataReader() if path.endswith(".vtp") else vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    if hasattr(reader, "ReadAllScalarsOn"):
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllFieldsOn()
    reader.Update()
    return reader.GetOutput()


def _write(surf, path):
    """Write beside the target and rename over it, so an interrupted run
    cannot leave the user with half a scan."""
    folder = os.path.dirname(os.path.abspath(path)) or "."
    handle, temporary = tempfile.mkstemp(suffix=os.path.splitext(path)[1], dir=folder)
    os.close(handle)
    try:
        writer = (vtk.vtkXMLPolyDataWriter() if path.endswith(".vtp")
                  else vtk.vtkPolyDataWriter())
        writer.SetFileName(temporary)
        writer.SetInputData(surf)
        if hasattr(writer, "SetFileTypeToBinary"):
            writer.SetFileTypeToBinary()
        if not writer.Write():
            raise RuntimeError("the writer reported a failure")
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def PaintScan(path, landmarks, radius=DEFAULT_RADIUS, labels=None,
              descriptions=None):
    """Add the band and the landmarks to the scan at `path`, in place.

    `descriptions` is what ALI wrote about each point. The band is grown from
    the ones it stands behind, exactly as AREG does, so what is drawn here is
    what the registration will use; every landmark is marked either way.

    `labels` is the teeth segmentation to keep the band off the crowns, for a
    scan that does not carry one itself -- ALI segments a copy when the file
    it was given has no labels. It is used only when it has one value per
    vertex of this mesh; a mesh the segmentation re-meshed cannot be matched
    up by index, and the band is then painted without the exclusion rather
    than against the wrong crowns.

    Returns True when the file was written.
    """
    if not path.lower().endswith(PAINTABLE):
        logger.info(f"{os.path.basename(path)} holds no point data ("
                    f"{os.path.splitext(path)[1]}), nothing painted on it")
        return False

    try:
        surf = _read(path)
        if surf is None or surf.GetNumberOfPoints() == 0:
            logger.warning(f"Could not read {path} back to paint it")
            return False

        if labels is not None and len(labels) == surf.GetNumberOfPoints():
            if surf.GetPointData().GetArray("Universal_ID") is None:
                array = numpy_to_vtk(np.asarray(labels, dtype=np.int64))
                array.SetName("Universal_ID")
                surf.GetPointData().AddArray(array)
        elif labels is not None:
            logger.info("The segmentation does not match this mesh vertex for "
                        "vertex, so the band is painted without keeping off the crowns")

        # the band on what AREG would stand on, the marks on every point,
        # so a doubtful one can be seen for what it is rather than vanish
        MGLPatch(surf, TrustedOnly(landmarks, descriptions), radius=radius)
        marked = MarkLandmarks(surf, landmarks)
        _write(surf, path)
        logger.info(f"Painted {MGL_ARRAY_NAME} and {LANDMARK_ARRAY_NAME} "
                    f"({marked} landmark(s)) onto {os.path.basename(path)}")
        return True
    except Exception as error:
        # Painting is there to be looked at; failing to do it must never cost
        # the run the landmarks it just computed.
        logger.warning(f"Could not paint {os.path.basename(path)}: {error}")
        return False
