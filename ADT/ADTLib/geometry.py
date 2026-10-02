"""Matrices, transforms and vectors shared by the surface tools.

Each of these functions existed in two to four **strictly identical** copies,
in `AREG_IOS_utils`, `ASO_IOS_utils`, `FlexReg_Method` and `ASO_CBCT_utils` --
measured line by line, not estimated. The variants that really diverge
(FlexReg's `RotationMatrix`, ALI_IOS's `TransformSurf`) stay where they are:
merging them would take a decision about which one is right, which is not a
refactor.

FlexReg_Method gains `TransformDict` and `TransformList` along the way, which
its `ApplyTransform` called without them being defined there -- a guaranteed
`NameError` as soon as it was handed anything other than a mesh.

Imported from the Conda environment by the CLIs, so nothing here depends on
Slicer or Qt. vtk and numpy are enough, and the callers already have them.
"""
import numpy as np
import vtk


def VTKMatrixToNumpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.

    Parameters
    ----------
    matrix : vtkMatrix4x4
        Matrix to be copied

    Returns
    -------
    numpy array
        Numpy array with the elements of the vtkMatrix4x4
    """
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m

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
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )

def TransformSurf(surf, matrix):
    assert isinstance(surf, vtk.vtkPolyData)
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.reshape(matrix, 16))
    surf = RotateTransform(surf, transform)

    return surf

def TransformList(input, matrix):
    type = np.array
    if isinstance(input, list):
        input = np.array(input)
        type = list

    a = np.ones((input.shape[0], 1))

    input = np.hstack((input, a))
    matrix = matrix[:3, :]
    input = np.matmul(matrix, input.T).T

    if isinstance(type, list):
        input = input.tolist()

    return input

def TransformDict(source, transform):
    """
    Apply a transform matrix to a set of landmarks

    Parameters
    ----------
    source : dict
        Dictionary of landmarks
    transform : np.array
        Transform matrix

    Returns
    -------
    source : dict
        Dictionary of transformed landmarks
    """

    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = transform @ np.append(sourcee[key], 1)
        sourcee[key] = sourcee[key][:3]
    return sourcee

def ApplyTransform(input, transform):
    if isinstance(input, vtk.vtkPolyData):
        input = TransformSurf(input, transform)

    if isinstance(input, dict):
        input = TransformDict(input, transform)

    if isinstance(input, (list, np.ndarray)):
        input = TransformList(input, transform)

    return input

def make_vector(points2, point1):
    perpen = points2[1] - points2[0]
    perpen = perpen / np.linalg.norm(perpen)

    vector1 = points2[0] - point1
    vector1 = vector1 / np.linalg.norm(vector1)

    vector2 = points2[1] - point1
    vector2 = vector2 / np.linalg.norm(vector2)

    normal = np.cross(vector1, vector2)
    normal = normal / np.linalg.norm(normal)

    direction = np.cross(normal, perpen)
    direction = direction / np.linalg.norm(direction)
    return normal, direction

def RotateTransform(surf, transform):

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(surf)
    transform_filter.Update()
    return transform_filter.GetOutput()
