import os
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from ASO_IOS_utils.utils import LoadJsonLandmarks, ReadSurf
from ASO_IOS_utils.transformation import (
    TranslationDict,
    RotationMatrix,
    ApplyTransform,
    TransformDict,
)
from random import choice
from ADTLib.geometry import VTKMatrixToNumpy  # noqa: F401  (re-exported)
from ADTLib.labels import has_label_array, label_array

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("ASO_IOS_icp")


class ICP:
    def __init__(self, list_icp, option=None) -> None:
        if False in [callable(f) for f in list_icp]:
            raise Exception("objects inside of list_icp are not callable")
        self.list_icp = list_icp
        self.option = option

    def copy(self, source):
        if isinstance(source, (dict, list, np.ndarray)):
            source_copy = source.copy()

        if isinstance(source, vtk.vtkPolyData):
            source_copy = vtk.vtkPolyData()
            source_copy.DeepCopy(source)

        return source_copy

    def pathTo(self, source, target):
        assert os.path.isfile(source) and os.path.isfile(
            target
        ), "source and target are not file"
        assert (
            os.path.splitext(source)[-1] == os.path.splitext(target)[-1]
        ), "source and target dont have the same extension"

        if source.endswith(".json"):
            source = LoadJsonLandmarks(source)
            target = LoadJsonLandmarks(target)

        elif source.endswith(".vtk"):
            source = ReadSurf(source)
            target = ReadSurf(target)

        return source, target

    def run(self, source, target):
        assert type(source) == type(
            target
        ), f"source and target dont have the same type source {type(source)}, target {type(target)}"
        assert self.list_icp != None, "icp method forgot"

        if isinstance(source, str):
            source, target = self.pathTo(source, target)

        source_int = self.copy(source)
        target_int = self.copy(target)

        option = self.option
        if isinstance(option, SelectKey) and isinstance(source, dict):
            # Keep only the landmarks BOTH files carry, which is what the CBCT
            # path has always done: it filters on what is present and then
            # refuses below three. Here `SelectKey` indexed the requested names
            # outright, so one absent name raised KeyError and the whole pair was
            # abandoned -- twelve landmarks asked for, four in the file, nothing
            # registered. Measured on 2026-09-30 on the published Semi-Automated
            # test set.
            wanted = list(option.list_key)
            usable = [key for key in wanted if key in source and key in target]
            dropped = [key for key in wanted if key not in usable]
            if dropped:
                logger.warning(
                    "%d of the %d landmarks asked for are not in both files, so "
                    "they are not used: %s",
                    len(dropped), len(wanted), ", ".join(dropped))
            # Three is not a convention, it is the minimum a 3D rigid
            # registration is determined by: with two, a rotation around the
            # line joining them stays free.
            if len(usable) < 3:
                raise ValueError(
                    "only %d landmark(s) are in both the scan and the reference "
                    "(%s); a registration needs at least 3"
                    % (len(usable), ", ".join(usable) or "none"))
            option = SelectKey(usable)

        if callable(option):
            source_int = option(source_int)
            target_int = option(target_int)

        matrix_final = np.identity(4)

        source_icp = self.copy(source_int)

        for icp in self.list_icp:
            source_icp, matrix = icp(source_icp, target_int)
            matrix_final = matrix_final @ matrix

        dic_out = {
            "source": source,
            "matrix": matrix_final,
            "source_Or": ApplyTransform(source, matrix_final),
            "target": target,
            "source_int": source_int,
            "source_icp": source_icp,
            "target_int": target_int,
        }

        return dic_out


class vtkICP:
    def setup(self, source, target):
        if isinstance(source, list):
            source = {str(i): np.array(source[i]) for i in range(len(source))}
            target = {str(i): np.array(target[i]) for i in range(len(target))}

        if isinstance(source, dict):
            source = DictTovtkPoints(source)
            target = DictTovtkPoints(target)

        return source, target

    def __call__(self, source, target):
        assert type(source) == type(target), "source and target dont have the same type"

        source, target = self.setup(source, target)

        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(target)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(100)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()

        # ============ apply ICP transform ==============
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(source)
        transform_filter.SetTransform(icp)
        transform_filter.Update()

        return source, VTKMatrixToNumpy(icp.GetMatrix())


class InitIcp:
    def setup(self, source, target):
        if isinstance(source, vtk.vtkPolyData):
            source = vtk_to_numpy(source.GetPoints().GetData()).tolist()
            target = vtk_to_numpy(target.GetPoints().GetData()).tolist()

        if isinstance(source, (list, np.ndarray)):
            source = {
                str(i): np.array(source[i], dtype=np.single) for i in range(len(source))
            }
            target = {
                str(i): np.array(target[i], dtype=np.single) for i in range(len(target))
            }

        source, target = SameNumberPoint(source, target)

        return source, target

    def __call__(self, source, target):
        source, target = self.setup(source, target)
        assert len(source) == len(target), "dont have the same of point"
        assert (
            len(source) >= 3
        ), f"source and target dont have enough point source : {source}, length {len(source)}"

        source = {k: source[k] for k in sorted(source)}
        target = {k: target[k] for k in sorted(target)}

        best = self.FindOptimalLandmarks(source, target)

        (
            source_transformed,
            transform_matrix,
        ) = self.InitICP(source, target, BestLMList=best)

        return source_transformed, transform_matrix

    def InitICP(self, source, target, BestLMList=None, search=False):
        transform_list = []
        # TransformMatrix = np.eye(4)
        translation_transform_matrix = np.eye(4)
        rotation_transform_matrix = np.eye(4)

        labels = list(source.keys())
        if BestLMList is not None:
            firstpick, secondpick, thirdpick = (
                BestLMList[0],
                BestLMList[1],
                BestLMList[2],
            )

        # ============ Pick a Random Landmark ==============
        if BestLMList is None:
            firstpick = labels[np.random.randint(0, len(labels))]
            # firstpick = 'LOr'
        # ============ Compute Translation Transform ==============
        T = target[firstpick] - source[firstpick]
        translation_transform_matrix[:3, 3] = T

        # ============ Apply Translation Transform ==============
        source = TranslationDict(source, T)
        # source = TransformDict(source, TranslationTransformMatrix)

        # ============ Pick Another Random Landmark ==============
        if BestLMList is None:
            while True:
                secondpick = labels[np.random.randint(0, len(labels))]
                # secondpick = 'ROr'
                if secondpick != firstpick:
                    break

        # ============ Compute Rotation Angle and Axis ==============
        v1 = np.absolute(source[secondpick] - source[firstpick])
        v2 = np.absolute(target[secondpick] - target[firstpick])
        angle, axis = self.AngleAndAxisVectors(v2, v1)

        # ============ Compute Rotation Transform ==============
        R = RotationMatrix(axis, angle)
        # TransformMatrix[:3, :3] = R
        rotation_transform_matrix[:3, :3] = R

        # ============ Apply Rotation Transform ==============

        source = TransformDict(source, rotation_transform_matrix)

        # ============ Compute Transform Matrix (Rotation + Translation) ==============
        transform_matrix = rotation_transform_matrix @ translation_transform_matrix

        # ============ Pick another Random Landmark ==============
        if BestLMList is None:
            while True:
                thirdpick = labels[np.random.randint(0, len(labels))]
                # thirdpick = 'Ba'
                if thirdpick != firstpick and thirdpick != secondpick:
                    break

        # ============ Compute Rotation Angle and Axis ==============
        v1 = abs(source[thirdpick] - source[firstpick])
        v2 = abs(target[thirdpick] - target[firstpick])
        angle, axis = self.AngleAndAxisVectors(v2, v1)

        # ============ Compute Rotation Transform ==============
        rotation_transform_matrix = np.eye(4)
        R = RotationMatrix(abs(source[secondpick] - source[firstpick]), angle)
        rotation_transform_matrix[:3, :3] = R

        # ============ Apply Rotation Transform ==============

        source = TransformDict(source, rotation_transform_matrix)

        # ============ Compute Transform Matrix (Init ICP) ==============
        transform_matrix = rotation_transform_matrix @ transform_matrix

        if search:
            return (
                firstpick,
                secondpick,
                thirdpick,
                self.ComputeMeanDistance(source, target),
            )

        return source, transform_matrix

    def FindOptimalLandmarks(self, source, target):
        """
        Find the optimal landmarks to use for the Init ICP

        Parameters
        ----------
        source : dict
            source landmarks
        target : dict
            target landmarks

        Returns
        -------
        list
            list of the optimal landmarks
        """

        # remplacer 210 by n*(n-1)*(n-2)   (n)
        dist, l_mlist, ii = [], [], 0
        n = len(source)
        # source used to be reloaded from a .npy on every turn, guarding
        # against a mutation that never happens: InitICP goes through
        # TranslationDict and TransformDict, which copy before transforming.
        while len(dist) < n * (n - 1) * (n - 2) and ii < 2500:
            ii += 1

            firstpick, secondpick, thirdpick, d = self.InitICP(
                source, target, search=True
            )
            if [firstpick, secondpick, thirdpick] not in l_mlist:
                dist.append(d)
                l_mlist.append([firstpick, secondpick, thirdpick])

        return l_mlist[dist.index(min(dist))]

    def ComputeMeanDistance(self, source, target):
        """
        Computes the mean distance between two point sets.

        Parameters
        ----------
        source : dict
            Source landmarks
        target : dict
            Target landmarks

        Returns
        -------
        float
            Mean distance
        """
        distance = 0
        for key in source.keys():
            distance += np.linalg.norm(source[key] - target[key])
        distance /= len(source.keys())
        return distance

    def AngleAndAxisVectors(self, v1, v2):
        """
        Return the angle and the axis of rotation between two vectors

        Parameters
        ----------
        v1 : numpy array
            First vector
        v2 : numpy array
            Second vector

        Returns
        -------
        angle : float
            Angle between the two vectors
        axis : numpy array
            Axis of rotation between the two vectors
        """
        # Compute angle between two vectors
        v1_u = v1 / np.amax(v1)
        v2_u = v2 / np.amax(v2)
        angle = np.arccos(
            np.dot(v1_u, v2_u) / (np.linalg.norm(v1_u) * np.linalg.norm(v2_u))
        )
        cross = lambda x, y: np.cross(x, y)
        axis = cross(v1_u, v2_u)
        # axis = axis / np.linalg.norm(axis)
        return angle, axis


class vtkTeeth:
    def __init__(self, list_teeth, property=None):
        self.property = property
        self.list_teeth = list_teeth

    def CheckLabelSurface(self, surf, property):
        if not self.isLabelSurface(surf, property):
            property = self.GetLabelSurface(surf)
        self.property = property

    def GetLabelSurface(self, surf, preference="Universal_ID"):
        """The numbering array to use: see `ADTLib.labels`.

        Four of the five copies did `continue` where `break` was needed, and
        so returned `Preference` only when it was the last array.
        """
        return label_array(surf, preference)

    def isLabelSurface(self, surf, property):
        return has_label_array(surf, property)


class vtkIterTeeth(vtkTeeth):
    def __init__(self, list_teeth, surf, property=None):
        super().__init__(list_teeth, property)
        self.CheckLabelSurface(surf, property)
        if not self.isLabelSurface(surf, self.property):
            raise NoSegmentationSurf(self.property)
        self.region_id = vtk_to_numpy(surf.GetPointData().GetScalars(self.property))
        self.verts = vtk_to_numpy(surf.GetPoints().GetData())

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter >= len(self.list_teeth):
            raise StopIteration

        verts_crown = np.argwhere(self.region_id == self.list_teeth[self.iter])
        if len(verts_crown) == 0:
            raise ToothNoExist(self.list_teeth[self.iter])

        self.iter += 1
        return np.array(self.verts[verts_crown]), self.list_teeth[self.iter - 1]


class vtkMeanTeeth(vtkTeeth):
    def __init__(self, list_teeth, property=None):
        super().__init__(list_teeth, property)

    def __call__(self, surf):
        dic = {}
        for points, tooth in vtkIterTeeth(
            self.list_teeth, surf, property=self.property
        ):
            dic[str(tooth)] = np.array(np.mean(points, 0).squeeze(0))
        return dic


class vtkMiddleTeeth(vtkTeeth):
    def __init__(self, list_teeth, property=None):
        super().__init__(list_teeth, property)

    def __call__(self, surf):
        dic = {}
        for points, tooth in vtkIterTeeth(
            self.list_teeth, surf, property=self.property
        ):
            dic[str(tooth)] = (
                (np.amax(points, axis=0) + np.amin(points, axis=0)) / 2
            ).squeeze(0)
        return dic


class vtkMeshTeeth(vtkTeeth):
    def __init__(self, list_teeth=None, property=None):
        super().__init__(list_teeth, property)

    def __call__(self, surf):
        self.CheckLabelSurface(surf, self.property)
        region_id = vtk_to_numpy(surf.GetPointData().GetScalars(self.property))
        list_teeth = np.unique(region_id)[1:-1]
        list_points = []
        size = 0

        for points, _ in vtkIterTeeth(list_teeth, surf, property=self.property):
            list_points.append(points)
            size += points.shape[0]

        Points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        labels = vtk.vtkStringArray()
        labels.SetNumberOfValues(size)
        labels.SetName("labels")
        index = 0
        for points in list_points:
            for i in range(points.shape[0]):
                sp_id = Points.InsertNextPoint(points[i, :].squeeze(0))
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(sp_id)
                labels.SetValue(index, str(index))
                index += 1

        output = vtk.vtkPolyData()
        output.SetPoints(Points)
        output.SetVerts(vertices)
        output.GetPointData().AddArray(labels)

        return output


class SelectKey:
    def __init__(self, list_key) -> None:
        self.list_key = list_key

    def __call__(self, input):
        assert isinstance(input, dict)
        out = {}
        for key in self.list_key:
            out[key] = input[key]
        return out


def DictTovtkPoints(dict_landmarks):
    """
    Convert dictionary of landmarks to vtkPoints

    Parameters
    ----------
    dict_landmarks : dict
        Dictionary of landmarks with key as landmark name and value as landmark coordinates\
        Example: {'L1': [0, 0, 0], 'L2': [1, 1, 1], 'L3': [2, 2, 2]}

    Returns
    -------
    vtkPoints
        VTK points object
    """
    Points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    labels = vtk.vtkStringArray()
    labels.SetNumberOfValues(len(dict_landmarks.keys()))
    labels.SetName("labels")

    for i, landmark in enumerate(dict_landmarks.keys()):
        sp_id = Points.InsertNextPoint(dict_landmarks[landmark])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(sp_id)
        labels.SetValue(i, landmark)

    output = vtk.vtkPolyData()
    output.SetPoints(Points)
    output.SetVerts(vertices)
    output.GetPointData().AddArray(labels)

    return output


def vtkSameNumberPoint(source, target):
    source_points = vtk_to_numpy(source.GetPoints().GetData())
    target_points = vtk_to_numpy(target.GetPoints().GetData())

    if source_points.shape[0] > target_points.shape[0]:

        save = np.random.choice(
            np.arange(0, source_points.shape[0]), target_points.shape[0], replace=False
        )
        source_points = source_points[save]
        vpoints = vtk.vtkPoints()
        vpoints.SetNumberOfPoints(source_points.shape[0])
        for i in range(source_points.shape[0]):
            vpoints.SetPoint(i, source_points[i])

        source.SetPoints(vpoints)

    elif source_points.shape[0] < target_points.shape[0]:

        save = np.random.choice(
            np.arange(0, target_points.shape[0]), source_points.shape[0], replace=False
        )
        target_points = target_points[save]

        vpoints = vtk.vtkPoints()
        vpoints.SetNumberOfPoints(source_points.shape[0])
        for i in range(target_points.shape[0]):
            vpoints.SetPoint(i, target_points[i])

        target.SetPoints(vpoints)

    return source, target


def npSameNumberPoint(source, target):

    if source.shape[0] > target.shape[0]:

        save = np.random.choice(
            np.arange(0, source.shape[0]), target.shape[0], replace=False
        )
        source = source[save]

    elif source.shape[0] < target.shape[0]:

        save = np.random.choice(
            np.arange(0, target.shape[0]), source.shape[0], replace=False
        )
        target = target[save]

    return source, target


def ListSameNumberPoint(source, target):
    source, target = npSameNumberPoint(np.array(source), np.array(target))

    return source.tolist(), target.tolist()


def DictSameNumberPoint(source, target):
    if choice(list(target.keys())).isdigit():
        source, target = ListSameNumberPoint(
            list(source.values()), list(target.values())
        )
        source = {str(i): np.array(source[i]) for i in range(len(source))}
        target = {str(i): np.array(target[i]) for i in range(len(target))}
    return source, target


def SameNumberPoint(source, target):
    if isinstance(source, vtk.vtkPolyData):
        source, target = npSameNumberPoint(source, target)

    if isinstance(source, list):
        source, target = ListSameNumberPoint(source, target)

    if isinstance(source, dict):
        source, target = DictSameNumberPoint(source, target)

    if isinstance(source, np.ndarray):
        source, target = npSameNumberPoint(source, target)

    return source, target


class ToothNoExist(Exception):
    def __init__(self, tooth) -> None:
        dic = {
            1: "UR8",
            2: "UR7",
            3: "UR6",
            4: "UR5",
            5: "UR4",
            6: "UR3",
            7: "UR2",
            8: "UR1",
            9: "UL1",
            10: "UL2",
            11: "UL3",
            12: "UL4",
            13: "UL5",
            14: "UL6",
            15: "UL7",
            16: "UL8",
            17: "LL8",
            18: "LL7",
            19: "LL6",
            20: "LL5",
            21: "LL4",
            22: "LL3",
            23: "LL2",
            24: "LL1",
            25: "LR1",
            26: "LR2",
            27: "LR3",
            28: "LR4",
            29: "LR5",
            30: "LR6",
            31: "LR7",
            32: "LR8",
        }
        if isinstance(tooth, int):
            tooth = dic[tooth]
        self.message = f"This tooth {tooth} is not segmented or doesnt exist "
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class NoSegmentationSurf(Exception):
    def __init__(self, property) -> None:
        self.message = f"This surf doesnt have this property {property}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
