import vtk
import numpy as np

from ADTLib.geometry import VTKMatrixToNumpy  # noqa: F401  (re-exporte)
from ADTLib.io.landmarks import LoadJsonLandmarks  # noqa: F401  (re-exporte)
from ADTLib.io.surface import ReadSurf, WriteSurf  # noqa: F401  (re-exporte)

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("FlexReg_CLI_utils")


def ComputeNormals(surf):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf)
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


def GetColorArray(surf, array_name):
    colored_points = vtk.vtkUnsignedCharArray()
    colored_points.SetName("colors")
    colored_points.SetNumberOfComponents(3)

    normals = surf.GetPointData().GetArray(array_name)

    for pid in range(surf.GetNumberOfPoints()):
        normal = np.array(normals.GetTuple(pid))
        rgb = (normal * 0.5 + 0.5) * 255.0
        colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])
    return colored_points

