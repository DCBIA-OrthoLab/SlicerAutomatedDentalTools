import numpy as np
import vtk
def RotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis : np.array
        Axis of rotation
    theta : float
        Angle of rotation in radians
    
    Returns
    -------
    np.array
        Rotation matrix
    """

    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])






def TransformSurf(surf,matrix):
    assert isinstance(surf,vtk.vtkPolyData)
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.reshape(matrix,16))
    surf = RotateTransform(surf,transform)

    return surf


def RotateTransform(surf, transform):

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()


def ApplyTransform(input, transform):
    if isinstance(input, vtk.vtkPolyData):
        input = TransformSurf(input, transform)

    if isinstance(input, dict):
        input = TransformDict(input, transform)

    if isinstance(input, (list, np.ndarray)):
        input = TransformList(input, transform)

    return input