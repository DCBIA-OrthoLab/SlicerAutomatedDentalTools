'''
Live preview of the butterfly patch.

What makes the real patch expensive is the geodesic flood fill of
FlexReg_Method.propagation.Dilation : it needs a GPU and runs in the CLI.
Everything that decides *where* the patch sits -- the arch orientation and the
four tooth centroids -- is pure numpy and does not depend on the ratio/adjust
values, so it is computed once per scan (prepare) and reused for every joystick
move (compute). What is left to redo on each move is the contour and an
approximate fill, which is fast enough to follow the mouse.

The contour is the exact same curve the CLI uses to carve the patch, so what the
preview draws is what Update produces; only the fill is approximated, by a
point-in-polygon test instead of a flood fill on the mesh.
'''

import numpy as np
import vtk

# The array a labelled arch is expected to carry; anything single-component
# stands in when it is absent.
LABEL_PREFERENCE = 'Universal_ID'
from vtk.util.numpy_support import vtk_to_numpy

from .orientation import orientation_matrix

# Canonical frame and band width used by FlexReg_Method.make_butterfly
ORIENT_TARGET = [[-0.5, -0.5, 0], [0, 0, 0], [0.5, -0.5, 0]]
ORIENT_TEETH = ['3', '5', '12', '14']
RADIUS = 0.7

# make_butterfly negates the adjust of the two posterior corners, so that a
# positive value pushes both ends of the patch outwards and grows it. Keep the
# same convention here : these signs must track the CLI.
ADJUST_SIGN = {
    'anterior_left': 1.0,
    'anterior_right': 1.0,
    'posterior_left': -1.0,
    'posterior_right': -1.0,
}

# In the canonical frame : +x is the side of teeth 12/14, +y is anterior,
# z is the occlusal axis (sign resolved per scan in prepare).
CELL = 1.0          # height map resolution, in mm
LIFT = 0.4          # how far the contour floats above the teeth, in mm
NB_POINTS = 120     # samples per contour edge


def _segment(p1, p2, n=NB_POINTS):
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    return p1[None, :2] + t * (p2[:2] - p1[:2])[None, :]


def _mirrored_bezier(start, middle, end, n=NB_POINTS):
    '''
    Quadratic Bezier reflected across the start-end chord, so that it bulges
    away from the centre of the arch. Mirrors compute_bezier_patch in
    FlexReg_Method/make_butterfly.py.
    '''
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    bez = ((1 - t) ** 2) * start[None, :2] + 2 * (1 - t) * t * middle[None, :2] + (t ** 2) * end[None, :2]

    direction = end[:2] - start[:2]
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    proj = (bez - start[None, :2]) @ direction
    return 2 * (np.outer(proj, direction) + start[None, :2]) - bez


def _points_in_polygon(points, polygon):
    '''
    Crossing-number test, vectorised over points. matplotlib ships with Slicer
    and does this in C, so use it when available and keep the numpy version as
    a fallback rather than a hard dependency.
    '''
    try:
        from matplotlib.path import Path
        return Path(polygon).contains_points(points)
    except Exception:
        pass

    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    x1 = polygon[:, 0][None, :]
    y1 = polygon[:, 1][None, :]
    x2 = np.roll(polygon[:, 0], -1)[None, :]
    y2 = np.roll(polygon[:, 1], -1)[None, :]

    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    crossings = np.count_nonzero(straddles & (x < x_cross), axis=1)
    return (crossings % 2) == 1


class ButterflyPreview:
    '''
    Holds the per-scan cache and turns (teeth, ratios, adjusts) into a contour
    and a patch label array, both expressed in the coordinates of the model
    node displayed in the scene.
    '''

    def __init__(self):
        self.clear()

    def clear(self):
        self.ready = False
        self.error = None
        self._inverse = None
        self._centroids = {}
        self._teeth = {}
        self._xy = None
        self._nb_points = 0
        self._grid = None
        self._grid_origin = None
        self._grid_shape = None
        self._occlusal = 1.0
        self._default_z = 0.0

    def prepare(self, polydata, teeth_by_key):
        '''
        Cache what only depends on the scan and the selected teeth : the
        orientation matrix, the four centroids, the projected vertices and a
        coarse height map used to lay the contour on the teeth.
        teeth_by_key maps 'anterior_left', 'anterior_right', 'posterior_left'
        and 'posterior_right' to a tooth number.
        Returns True when the preview can be computed.
        '''
        self.clear()
        teeth_by_key = {key: int(tooth) for key, tooth in teeth_by_key.items()}
        teeth = teeth_by_key.values()

        if polydata is None or polydata.GetNumberOfPoints() == 0:
            self.error = 'no surface loaded'
            return False

        # Universal_ID by preference, then whatever labels the mesh does carry.
        # The server's engine already picks the array this way, and hardcoding
        # one name here made the preview stricter than the run it previews:
        # Crown_Seg writes `PredictedID`, so a mesh it had just labelled was
        # refused by the panel and accepted by the server.
        ids = None
        point_data = polydata.GetPointData()
        for index in range(point_data.GetNumberOfArrays()):
            name = point_data.GetArrayName(index)
            if name == LABEL_PREFERENCE:
                ids = point_data.GetArray(name)
                break
            if ids is None and point_data.GetArray(name).GetNumberOfComponents() == 1:
                ids = point_data.GetArray(name)
        if ids is None:
            self.error = 'the scan carries no per-point tooth labels'
            return False
        self.label_array = ids.GetName()

        try:
            matrix = orientation_matrix(polydata, ORIENT_TARGET, ORIENT_TEETH,
                                        label_array=self.label_array)
        except Exception as error:
            self.error = str(error)
            return False

        if not np.all(np.isfinite(matrix)):
            self.error = 'the arch orientation could not be computed'
            return False

        labels = vtk_to_numpy(ids).ravel().astype(np.int32)
        points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
        oriented = points @ matrix[:3, :3].T + matrix[:3, 3]

        for tooth in teeth:
            selection = labels == int(tooth)
            if not selection.any():
                self.error = f'tooth {int(tooth)} is not segmented'
                return False
            self._centroids[int(tooth)] = oriented[selection].mean(axis=0)

        self._inverse = np.linalg.inv(matrix)
        self._teeth = teeth_by_key
        self._xy = np.ascontiguousarray(oriented[:, :2])
        self._nb_points = oriented.shape[0]

        # Which way the crowns point : the four centroids sit on the occlusal
        # side, so compare them to the bulk of the mesh.
        centroid_z = float(np.mean([c[2] for c in self._centroids.values()]))
        self._occlusal = 1.0 if centroid_z > float(np.median(oriented[:, 2])) else -1.0
        self._default_z = centroid_z

        self._buildHeightMap(oriented)
        self.ready = True
        return True

    def _buildHeightMap(self, oriented):
        '''
        Coarse grid holding, per cell, the height of the outermost surface on
        the occlusal side. Sorting once beats np.maximum.at on 100k points.
        '''
        origin = oriented[:, :2].min(axis=0) - CELL
        extent = oriented[:, :2].max(axis=0) + CELL - origin
        shape = (np.ceil(extent / CELL).astype(np.int64) + 1)

        cells = np.floor((oriented[:, :2] - origin) / CELL).astype(np.int64)
        flat = cells[:, 0] * shape[1] + cells[:, 1]
        height = self._occlusal * oriented[:, 2]

        order = np.lexsort((height, flat))
        flat_sorted = flat[order]
        keep = np.ones(flat_sorted.shape[0], dtype=bool)
        keep[:-1] = flat_sorted[1:] != flat_sorted[:-1]

        grid = np.full(int(shape[0] * shape[1]), np.nan)
        grid[flat_sorted[keep]] = height[order][keep]

        self._grid = grid
        self._grid_origin = origin
        self._grid_shape = shape

    def _heightAt(self, xy):
        cells = np.floor((xy - self._grid_origin) / CELL).astype(np.int64)
        np.clip(cells[:, 0], 0, self._grid_shape[0] - 1, out=cells[:, 0])
        np.clip(cells[:, 1], 0, self._grid_shape[1] - 1, out=cells[:, 1])
        height = self._grid[cells[:, 0] * self._grid_shape[1] + cells[:, 1]]
        return np.where(np.isnan(height), self._occlusal * self._default_z, height)

    def matches(self, teeth_by_key):
        if not self.ready:
            return False
        return self._teeth == {key: int(tooth) for key, tooth in teeth_by_key.items()}

    def centroids(self):
        '''Tooth centroids in the canonical frame, keyed by corner.'''
        return {key: self._centroids[tooth] for key, tooth in self._teeth.items()}

    def landmarks(self, ratios, adjusts, shift=(0.0, 0.0)):
        '''
        The four corners of the patch, in the canonical frame. Reproduces
        butterflyPatch : the ratio is mapped through (1 - r) / 2, then used to
        interpolate between the tooth centroid and the centroid of the tooth
        facing it, and the adjust shifts the centroid along the antero-posterior
        axis with the sign of its corner. A ratio of 1 therefore lands on the
        tooth itself, at the outer edge of the arch, and 0 at mid-arch.
        Keys are 'anterior_left', 'anterior_right', 'posterior_left',
        'posterior_right'.

        shift is (medio-lateral, antero-posterior) in mm, added to every
        centroid at once. Each landmark is an affine combination of two
        centroids whose weights sum to 1, so the same vector comes back out of
        the interpolation : the patch moves, its shape does not change.
        '''
        translation = np.array([float(shift[0]), float(shift[1]), 0.0])

        centroids = {}
        for key, tooth in self._teeth.items():
            centroid = np.array(self._centroids[tooth], dtype=np.float64)
            centroids[key] = (centroid + translation
                              + np.array([0.0, ADJUST_SIGN[key] * float(adjusts[key]), 0.0]))

        facing = {
            'anterior_left': 'anterior_right',
            'anterior_right': 'anterior_left',
            'posterior_left': 'posterior_right',
            'posterior_right': 'posterior_left',
        }

        result = {}
        for key, other in facing.items():
            ratio = (1.0 - float(ratios[key])) / 2.0
            result[key] = (1 - ratio) * centroids[key] + ratio * centroids[other]
        return result

    def contour(self, landmarks):
        '''Closed contour of the patch, in the canonical frame (2D).'''
        anterior_left = landmarks['anterior_left']
        anterior_right = landmarks['anterior_right']
        posterior_left = landmarks['posterior_left']
        posterior_right = landmarks['posterior_right']
        middle_posterior = (posterior_left + posterior_right) / 2

        top = _segment(anterior_left, anterior_right)
        right = _mirrored_bezier(posterior_right, middle_posterior, anterior_right)[::-1]
        bottom = _segment(posterior_right, posterior_left)
        left = _mirrored_bezier(posterior_left, middle_posterior, anterior_left)

        return np.concatenate([top, right, bottom, left], axis=0)

    def fill(self, contour):
        '''
        Approximate patch labels : the mesh vertices whose projection falls
        inside the contour. The CLI instead floods the mesh from the middle
        point, bounded by the same contour, so the two agree except where the
        surface folds back on itself.
        '''
        labels = np.zeros(self._nb_points, dtype=np.float32)

        low = contour.min(axis=0) - RADIUS
        high = contour.max(axis=0) + RADIUS
        inside_box = np.flatnonzero(
            (self._xy[:, 0] >= low[0]) & (self._xy[:, 0] <= high[0])
            & (self._xy[:, 1] >= low[1]) & (self._xy[:, 1] <= high[1])
        )
        if inside_box.size == 0:
            return labels

        coarse = contour[::3]
        inside = _points_in_polygon(self._xy[inside_box], coarse)
        labels[inside_box[inside]] = 1.0
        return labels

    def toScene(self, points):
        '''Canonical frame -> coordinates of the model node in the scene.'''
        points = np.atleast_2d(points)
        return points @ self._inverse[:3, :3].T + self._inverse[:3, 3]

    def compute(self, ratios, adjusts, shift=(0.0, 0.0), with_fill=True):
        '''
        Full preview for one set of values.
        Returns (contour_polydata, labels, landmarks_scene) with the contour
        and the landmarks already expressed in scene coordinates.
        '''
        landmarks = self.landmarks(ratios, adjusts, shift)
        contour = self.contour(landmarks)

        labels = self.fill(contour) if with_fill else None

        height = (self._heightAt(contour) + LIFT) * self._occlusal
        contour_3d = np.column_stack([contour, height])

        order = ('anterior_left', 'anterior_right', 'posterior_right', 'posterior_left')
        corners = np.array([landmarks[key] for key in order])
        corners[:, 2] = (self._heightAt(corners[:, :2]) + LIFT) * self._occlusal

        return (
            self._contourPolyData(self.toScene(contour_3d), self.toScene(corners)),
            labels,
            self.toScene(corners),
        )

    def _contourPolyData(self, contour, corners):
        '''Closed polyline plus a marker on each corner, as a single polydata.'''
        append = vtk.vtkAppendPolyData()

        points = vtk.vtkPoints()
        for point in contour:
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(contour) + 1)
        for index in range(len(contour)):
            lines.InsertCellPoint(index)
        lines.InsertCellPoint(0)

        polyline = vtk.vtkPolyData()
        polyline.SetPoints(points)
        polyline.SetLines(lines)
        append.AddInputData(polyline)

        for corner in corners:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(float(corner[0]), float(corner[1]), float(corner[2]))
            sphere.SetRadius(0.6)
            sphere.SetThetaResolution(12)
            sphere.SetPhiResolution(12)
            sphere.Update()
            append.AddInputData(sphere.GetOutput())

        append.Update()
        return append.GetOutput()
