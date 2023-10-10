import os
import vtk
import numpy as np
from Method.utils import ReadSurf, LoadJsonLandmarks, VTKMatrixToNumpy
from Method.transformation import ApplyTransform


class ICP:
    def __init__(self, list_icp, option=None) -> None:
        if False in [callable(f) for f in list_icp]:
            raise Exception("objects inside of list_icp are not callable")
        self.list_icp = list_icp
        self.option = option

    def copy(self, source):
        print("type de source : ",type(source))
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
        assert self.list_icp != None, "give icp methode"

        if isinstance(source, str):
            source, target = self.pathTo(source, target)

        source_int = self.copy(source)
        target_int = self.copy(target)

        if callable(self.option):
            source_int = self.option(source_int)
            target_int = self.option(target_int)

        matrix_final = np.identity(4)

        source_icp = self.copy(source_int)
        # print(f'source {source_icp}')
        # print(f'target {target_int}')
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
    def __call__(self, source, target):
        assert type(source) == type(target), "source and target dont have the same type"

        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(target)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(1000)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()

        # ============ apply ICP transform ==============
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(source)
        transformFilter.SetTransform(icp)
        transformFilter.Update()

        return source, VTKMatrixToNumpy(icp.GetMatrix())
