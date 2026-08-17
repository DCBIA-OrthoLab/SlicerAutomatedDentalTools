# Build the lower registration patch from the mucogingival landmarks predicted
# by ALI_IOS, with a height the user sets tooth by tooth.
#
# The upper arch is registered on a patch drawn over the palate. The mandible
# has no such plateau, but it has the mucogingival line: the 13 MG landmarks run
# along the arch, a B-spline joins them, and the band of surface around that
# curve plays the same role.
#
# Two properties matter for the result to be usable, and a third for it to be
# editable:
#   - every sample of the curve is snapped onto the mesh, because a curve
#     interpolated between landmarks leaves the surface in the concavities
#     between teeth;
#   - the band grows along the surface, never through it, so a buccal patch
#     cannot appear on the lingual side where the ridge is thin;
#   - the height is carried by the landmarks and interpolated in between, so
#     moving one point only reshapes the patch around it.
import logging
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

logger = logging.getLogger("FlexReg_mgl_patch")

# Landmark names of the MG model, in arch order. L0MG is the midline (tooth 25),
# so the right side is shifted by one against the tooth numbers.
MGL_ORDER = ['LL6MG', 'LL5MG', 'LL4MG', 'LL3MG', 'LL2MG', 'LL1MG', 'L0MG',
             'LR1MG', 'LR2MG', 'LR3MG', 'LR4MG', 'LR5MG', 'LR6MG']

# The hand annotations of the training set spoke an older naming for the
# same 13 points, placed symmetrically about the midline: 6 on each side of
# a central point, no L0. One side carries seven names, and that side counts
# the midline point as its "1" (verified against the 24|25 incisor plane on
# the annotated scans: the file with an LL7 has its midline on LL1MG, the
# file with an LR7 on LR1MG). Translating shifts that side's names inward by
# one and renames its "1" to L0MG; nothing is lost, an old file keeps its 13
# points. Which side got the seven varies from file to file, so both markers
# are recognized.
MGL_TRAINING_TABLES = {
    'LL7MG': {
        'LL7MG': 'LL6MG', 'LL6MG': 'LL5MG', 'LL5MG': 'LL4MG', 'LL4MG': 'LL3MG',
        'LL3MG': 'LL2MG', 'LL2MG': 'LL1MG', 'LL1MG': 'L0MG',
        'LR1MG': 'LR1MG', 'LR2MG': 'LR2MG', 'LR3MG': 'LR3MG', 'LR4MG': 'LR4MG',
        'LR5MG': 'LR5MG', 'LR6MG': 'LR6MG',
    },
    'LR7MG': {
        'LR7MG': 'LR6MG', 'LR6MG': 'LR5MG', 'LR5MG': 'LR4MG', 'LR4MG': 'LR3MG',
        'LR3MG': 'LR2MG', 'LR2MG': 'LR1MG', 'LR1MG': 'L0MG',
        'LL1MG': 'LL1MG', 'LL2MG': 'LL2MG', 'LL3MG': 'LL3MG', 'LL4MG': 'LL4MG',
        'LL5MG': 'LL5MG', 'LL6MG': 'LL6MG',
    },
}


def CanonicalLandmarks(landmarks):
    """Landmarks renamed to the canonical naming when the file uses the old one."""
    for marker, table in MGL_TRAINING_TABLES.items():
        if marker not in landmarks:
            continue
        logger.info(f"Old MG naming detected ({marker} present): translating to "
                    f"the canonical naming, their midline {table_centre(table)} "
                    "becomes L0MG")
        translated = {}
        for name, position in landmarks.items():
            canonical = table.get(name)
            if canonical:
                translated[canonical] = position
            else:
                logger.warning(f"Landmark {name} is not an MG name, dropped")
        return translated
    return dict(landmarks)


def table_centre(table):
    """The old name a translation table sends to L0MG."""
    return next(old for old, new in table.items() if new == 'L0MG')


def RecentreNames(landmarks):
    """Keep the counting anchored on the midline when points are missing.

    The 13 points are symmetric positions about the midline, not fixed tooth
    identities: when one is missing mid-arch, the names of the points distal
    to it on its side slide inward by one, so the hole surfaces at the far
    end of that side and never at L0. A missing midline point is replaced by
    the innermost right point (then the innermost left one).
    """
    left = [name for name in MGL_ORDER[:6] if name in landmarks]    # LL6..LL1
    right = [name for name in MGL_ORDER[7:] if name in landmarks]   # LR1..LR6

    mapping = {}
    if 'L0MG' in landmarks:
        mapping['L0MG'] = 'L0MG'
    elif right:
        mapping[right.pop(0)] = 'L0MG'
    elif left:
        mapping[left.pop()] = 'L0MG'
    else:
        return dict(landmarks)

    for old_name, new_name in zip(reversed(left),
                                  ('LL1MG', 'LL2MG', 'LL3MG', 'LL4MG', 'LL5MG', 'LL6MG')):
        mapping[old_name] = new_name
    for old_name, new_name in zip(right,
                                  ('LR1MG', 'LR2MG', 'LR3MG', 'LR4MG', 'LR5MG', 'LR6MG')):
        mapping[old_name] = new_name

    moved = {old: new for old, new in mapping.items() if old != new}
    if moved:
        logger.info(f"Renamed to keep the counting anchored on the midline: {moved}")
    return {new: landmarks[old] for old, new in mapping.items()}

# Name of the point array the patch is written to, matching what AREG_IOS reads.
MGL_ARRAY_NAME = "Bottom_MGL"
MGL_PREVIEW_ARRAY_NAME = "Bottom_MGLPreview"

DEFAULT_HEIGHT = 2.5        # mm of surface on each side of the line
# 0 is meaningful: the band vanishes, the curve between the landmarks goes with
# it, and the landmarks alone are left, so the registration runs on those 13
# points rather than on a patch. That is the control case, kept to measure on
# real scans what the surface around the line brings over the points carrying
# it. A short slider travel keeps every 0.1 mm step wide under the finger.
MIN_HEIGHT = 0.0
MAX_HEIGHT = 5.0
SAMPLES_PER_SEGMENT = 25    # spline samples between two consecutive landmarks

# Universal_ID labels of the lower teeth. The crowns move between the two
# timepoints, so they must never end up inside the patch.
LOWER_TOOTH_LABELS = range(18, 32)


def _controlPoints(path):
    import json
    with open(path) as f:
        data = json.load(f)
    return [point for point in data["markups"][0]["controlPoints"]
            if point.get("position")]


def DoubtfulLandmarks(path):
    """{label: why} for the points ALI was unsure of, read from its json.

    ALI records how it came by each point in the markup description: one it
    forced out of the top pixels, one it fell back on the tooth for, one whose
    tooth was absent and whose cameras were aimed at a guess. Measured over 364
    predictions, those sit a median 4.2 mm off the curve their neighbours draw,
    against 1.2 mm for the rest.

    Empty when dropping them would leave fewer than three points, since a
    doubtful line still beats no line at all.
    """
    points = _controlPoints(path)
    doubtful = {point["label"]: point["description"] for point in points
                if point.get("description")}
    if len(points) - len(doubtful) < 3:
        return {}
    return doubtful


def ReadLandmarks(path, drop_doubtful=True):
    """Read a Slicer markups json as {label: position}.

    The points ALI was unsure of are left out by default: they are the ones
    that pull the mucogingival line out of shape, and the spline spans the
    hole they leave. DoubtfulLandmarks says which those are.
    """
    doubtful = DoubtfulLandmarks(path) if drop_doubtful else {}
    if doubtful:
        logger.info(f"Leaving out {len(doubtful)} landmark(s) ALI was unsure of: "
                    + ", ".join(sorted(doubtful)))
    return {point["label"]: np.array(point["position"], dtype=float)
            for point in _controlPoints(path)
            if point["label"] not in doubtful}


def WriteLandmarks(path, names, positions):
    """Write landmarks as a Slicer markups json, in the file's coordinates.

    The coordinateSystem is declared LPS like the training annotations : the
    positions passed in must therefore be those of the mesh file on disk, not
    the RAS ones Slicer displays. What is written reads back with
    ReadLandmarks and can join the training corpus as it is.
    """
    import json
    content = {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/"
                   "Modules/Loadable/Markups/Resources/Schema/"
                   "markups-schema-v1.0.3.json",
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "coordinateUnits": "mm",
            "controlPoints": [
                {"id": str(index + 1), "label": name,
                 "position": [float(value) for value in position]}
                for index, (name, position) in enumerate(zip(names, positions))
            ],
        }],
    }
    with open(path, 'w') as handle:
        json.dump(content, handle, indent=2)


def OrderedLandmarks(landmarks):
    """MG landmark names and positions in arch order, in the canonical naming.

    Files annotated with the older naming (recognizable by their LL7MG) are
    translated first, so the same name always designates the same tooth
    downstream. Missing teeth are skipped rather than fatal: a scan where ALI
    could not place every point still yields a usable curve.
    """
    landmarks = RecentreNames(CanonicalLandmarks(landmarks))
    names = [name for name in MGL_ORDER if name in landmarks]
    if len(names) < 3:
        raise ValueError(
            f"Fewer than 3 MG landmarks found, expected names such as "
            f"{MGL_ORDER[:3]}, got {sorted(landmarks)}"
        )
    return names, np.array([landmarks[name] for name in names], dtype=float)


def LocalFrames(points):
    """Buccal, tangent and apical unit vectors at each landmark.

    The apical axis is the normal of the plane the landmarks lie in, so it does
    not depend on how the scan happens to be oriented in world space. The
    tangent follows the arch in landmark order, from LL6 towards LR6. The
    buccal axis is perpendicular to both, flipped where needed so it always
    points away from the arch.

    Returns (buccal, tangent, apical) with buccal and tangent of shape (N, 3)
    and apical of shape (3,).
    """
    centre = points.mean(axis=0)
    centred = points - centre
    # smallest singular direction of a ribbon of points is its plane normal
    apical = np.linalg.svd(centred)[2][2]
    apical = apical / np.linalg.norm(apical)

    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent = tangents / np.where(norms < 1e-9, 1.0, norms)

    buccal = np.cross(tangents, apical)
    norms = np.linalg.norm(buccal, axis=1, keepdims=True)
    buccal = buccal / np.where(norms < 1e-9, 1.0, norms)

    outward = centred - np.outer(centred @ apical, apical)
    flip = np.sum(buccal * outward, axis=1) < 0
    buccal[flip] *= -1.0
    return buccal, tangent, apical


def OrientApical(apical, points, tooth_points):
    """Return the apical axis signed away from the crowns.

    The plane normal has an arbitrary sign; the teeth tell which way is up, so
    apical is the direction leading away from them.
    """
    if tooth_points is None or len(tooth_points) == 0:
        return apical
    if np.dot(tooth_points.mean(axis=0) - points.mean(axis=0), apical) > 0:
        return -apical
    return apical


def SplineThroughPoints(points, samples_per_segment=SAMPLES_PER_SEGMENT):
    """Sample a spline passing through `points`.

    Returns (samples, weights) where weights[i] holds, for each sample, how much
    it belongs to each landmark: the sample sits between two landmarks and the
    pair of weights says where. That is what lets a height set on one landmark
    fade into its neighbours instead of stepping.
    """
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(*point)

    spline = vtk.vtkParametricSpline()
    spline.SetPoints(vtk_points)
    spline.ClosedOff()

    n_samples = max(2, (len(points) - 1) * samples_per_segment + 1)
    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(spline)
    source.SetUResolution(n_samples - 1)
    source.Update()

    samples = vtk_to_numpy(source.GetOutput().GetPoints().GetData())

    # Arc position of every sample, expressed in landmark index units. The
    # spline is sampled evenly in its parameter, which passes through the
    # landmarks at regular intervals.
    positions = np.linspace(0.0, len(points) - 1.0, len(samples))
    return samples, positions


def InterpolateHeights(heights, positions):
    """Height at each spline sample, linearly interpolated between landmarks."""
    return np.interp(positions, np.arange(len(heights)), heights)


class MGLPatchBuilder:
    """Turns landmark offsets and heights into a patch, fast enough to drag.

    The per-scan work -- reading the mesh, building the edge graph, locating the
    teeth -- is done once by prepare(). Each compute() then only moves the
    landmarks, redraws the spline and walks the graph, which is what makes a
    live preview possible.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.ready = False
        self.error = None
        self._points = None
        self._graph = None
        self._tooth_mask = None
        self._landmarks = None
        self._names = []

    def prepare(self, polydata, landmarks):
        """Cache what does not change while the user drags. True on success."""
        try:
            self._names, self._landmarks = OrderedLandmarks(landmarks)
        except ValueError as error:
            self.error = str(error)
            self.ready = False
            return False

        self._points = vtk_to_numpy(polydata.GetPoints().GetData())
        self._graph = self._buildGraph(polydata)
        self._tooth_mask = self._buildToothMask(polydata)
        self._locator = vtk.vtkPointLocator()
        self._locator.SetDataSet(polydata)
        self._locator.BuildLocator()

        tooth_points = self._points[self._tooth_mask] if self._tooth_mask.any() else None
        buccal, tangent, apical = LocalFrames(self._landmarks)
        self._buccal = buccal
        self._tangent = tangent
        self._apical = OrientApical(apical, self._landmarks, tooth_points)

        self.ready = True
        self.error = None
        return True

    def _buildGraph(self, polydata):
        """Edge graph of the mesh, one entry per edge.

        Every interior edge belongs to two triangles, and a sparse matrix built
        from the raw list would add those duplicates together, silently doubling
        the length of most edges and halving the reach of the patch.
        """
        from scipy.sparse import coo_matrix

        faces = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        lengths = np.linalg.norm(self._points[edges[:, 0]] - self._points[edges[:, 1]], axis=1)
        n_points = len(self._points)
        return coo_matrix(
            (np.r_[lengths, lengths],
             (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
            shape=(n_points, n_points),
        ).tocsr()

    def _buildToothMask(self, polydata):
        """True where a vertex belongs to a crown, False on the gingiva."""
        point_data = polydata.GetPointData()
        for name in ("Universal_ID", "PredictedID", "UniversalID"):
            scalars = point_data.GetScalars(name) or point_data.GetArray(name)
            if scalars is not None:
                return np.isin(vtk_to_numpy(scalars), list(LOWER_TOOTH_LABELS))

        logger.warning("No teeth segmentation on the mesh, the patch is not kept off the crowns")
        return np.zeros(len(self._points), dtype=bool)

    def names(self):
        return list(self._names)

    def movedLandmarks(self, buccal_offsets, apical_offsets, tangent_offsets=None):
        """Landmark positions after the offsets the user dialled in."""
        moved = self._landmarks.copy()
        moved = moved + self._buccal * np.asarray(buccal_offsets, dtype=float)[:, None]
        moved = moved + np.outer(np.asarray(apical_offsets, dtype=float), self._apical)
        if tangent_offsets is not None:
            moved = moved + self._tangent * np.asarray(tangent_offsets, dtype=float)[:, None]
        return moved

    def snappedLandmarkIds(self, buccal_offsets, apical_offsets, tangent_offsets=None):
        """Vertex id under each moved landmark.

        The coordinate-free form of snappedLandmarks : an id designates the
        same vertex in every copy of the mesh, whatever the coordinate system
        or centering, which is what makes it worth saving into the file.
        """
        moved = self.movedLandmarks(buccal_offsets, apical_offsets, tangent_offsets)
        return [int(self._locator.FindClosestPoint(point)) for point in moved]

    def snappedLandmarks(self, buccal_offsets, apical_offsets, tangent_offsets=None):
        """Moved landmarks, brought back onto the mesh.

        The offsets push the points along straight axes, off the surface; the
        band is built from the curve snapped back onto it, so this is where
        the landmarks effectively act — and where they should be displayed.
        """
        ids = self.snappedLandmarkIds(buccal_offsets, apical_offsets, tangent_offsets)
        return self._points[ids]

    def compute(self, buccal_offsets, apical_offsets, heights, heights_down=None,
                exclude_teeth=True, tangent_offsets=None):
        """Patch labels for the current settings, as a 0/1 array over the mesh.

        Each vertex is reached from the nearest spline sample; the height that
        sample carries is the one it is judged against, so a tall stretch and a
        short stretch can sit side by side on the same curve.

        With `heights_down`, the band is asymmetric: `heights` bounds the crown
        side of the curve and `heights_down` the vestibule side. Left to None,
        both sides use `heights` and the band is the symmetric one of before.

        The landmarks themselves always belong to the patch, and nothing else
        does where both heights are 0: a stretch brought down to 0 keeps its
        landmarks and drops the curve joining them, so heights left at 0 all
        along the arch leave the landmarks alone for the registration to run on.
        """
        from scipy.sparse.csgraph import dijkstra

        moved = self.movedLandmarks(buccal_offsets, apical_offsets, tangent_offsets)
        samples, positions = SplineThroughPoints(moved)
        heights_up = np.asarray(heights, dtype=float)
        heights_down = (heights_up if heights_down is None
                        else np.asarray(heights_down, dtype=float))
        sample_up = InterpolateHeights(heights_up, positions)
        sample_down = InterpolateHeights(heights_down, positions)

        seeds, seed_up, seed_down = [], [], []
        seen = {}
        for sample, up, down in zip(samples, sample_up, sample_down):
            point_id = self._locator.FindClosestPoint(sample)
            if point_id in seen:
                # one vertex can serve several samples: keep the tallest, so a
                # height raised anywhere is never swallowed by a neighbour
                at = seen[point_id]
                seed_up[at] = max(seed_up[at], up)
                seed_down[at] = max(seed_down[at], down)
                continue
            seen[point_id] = len(seeds)
            seeds.append(point_id)
            seed_up.append(up)
            seed_down.append(down)

        seeds = np.asarray(seeds)
        seed_up = np.asarray(seed_up)
        seed_down = np.asarray(seed_down)

        distances, _, sources = dijkstra(
            self._graph, directed=False, indices=seeds,
            limit=float(max(seed_up.max(), seed_down.max())),
            min_only=True, return_predecessors=True,
        )

        reached = np.isfinite(distances)
        inside = np.zeros(len(self._points), dtype=bool)
        if reached.any():
            # sources holds the seed vertex each node was reached from
            seed_index = {seed: i for i, seed in enumerate(seeds)}
            reached_ids = np.flatnonzero(reached)
            source_ids = sources[reached]
            source_at = np.array([seed_index[s] for s in source_ids])
            # Which side of the curve each vertex sits on: the apical axis
            # leads away from the crowns, so a positive component along it is
            # the vestibule side. On the curve itself the component is ~0 and
            # both heights agree well enough for the tie not to matter.
            below = np.sum(
                (self._points[reached_ids] - self._points[source_ids]) * self._apical,
                axis=1) > 0
            allowed = np.where(below, seed_down[source_at], seed_up[source_at])
            # A height of 0 keeps nothing, the curve included: a vertex needs a
            # band to belong to, and the seeds of a stretch left at 0 have none.
            inside[reached_ids] = (distances[reached] <= allowed) & (allowed > 0)

        # The landmarks are what the band is built around, so they stay whatever
        # the heights are. They are also all that is left of a patch flattened
        # to 0 everywhere, which is what makes the registration on the landmarks
        # alone reachable from the sliders.
        landmark_ids = [self._locator.FindClosestPoint(point) for point in moved]
        inside[landmark_ids] = True

        if exclude_teeth:
            inside &= ~self._tooth_mask

        return inside.astype(np.float32), samples

    def toArray(self, labels, name=MGL_ARRAY_NAME):
        """Wrap patch labels as a named vtk array."""
        array = numpy_to_vtk(labels.astype(np.float32), deep=True)
        array.SetName(name)
        return array
