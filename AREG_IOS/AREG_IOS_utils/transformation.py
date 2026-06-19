import SimpleITK as sitk
import numpy as np
import vtk
import os
import logging
import sys
# ===== Logging Configuration =====
logger = logging.getLogger("AREG_IOS_transformation")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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

def read_matrix(tfm_path):
    """
    Reads a 3D affine transformation from a .tfm file and returns a 4x4 numpy matrix.
    """
    transform = sitk.ReadTransform(tfm_path)
    
    if not isinstance(transform, sitk.AffineTransform):
        raise TypeError(f"Transform at {tfm_path} is not an AffineTransform.")

    matrix = np.eye(4)
    matrix[:3, :3] = np.array(transform.GetMatrix()).reshape((3, 3))
    matrix[:3, 3] = np.array(transform.GetTranslation())

    return matrix

def saveMatrixAsTfm(areg_matrix, aso_tfm_path, output_folder, patient_id, suffix, areg_mode):
    if areg_mode == "Auto_IOS":
        try:
            matrix_aso = read_matrix(aso_tfm_path)
        except Exception as e:
            logger.error(f"Error reading ASO matrix for {patient_id}: {e}")
            return
        
        final_matrix = np.linalg.inv(areg_matrix @ np.linalg.inv(matrix_aso))
        composed_path = os.path.join(output_folder, f"{patient_id}_T2_SegOr{suffix}.tfm")
    else:
        final_matrix = np.linalg.inv(areg_matrix)
        composed_path = os.path.join(output_folder, f"{patient_id}_T2_Seg{suffix}.tfm")
    
    transform = sitk.AffineTransform(3)
    rotation = final_matrix[:3, :3].flatten()
    translation = final_matrix[:3, 3]

    transform.SetMatrix(rotation)
    transform.SetTranslation(translation.tolist())

    sitk.WriteTransform(transform, composed_path)
    logger.info(f"Saved composed matrix: {composed_path}")

def RotateTransform(surf, transform):

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()


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


def ApplyTransform(input, transform):
    if isinstance(input, vtk.vtkPolyData):
        input = TransformSurf(input, transform)

    if isinstance(input, dict):
        input = TransformDict(input, transform)

    if isinstance(input, (list, np.ndarray)):
        input = TransformList(input, transform)

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


def ScaleSurf(surf, mean_arr=None, scale_factor=None):

    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    shapedatapoints = surf.GetPoints()

    # calculate bounding box
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    mean_v[0] = (bounds[0] + bounds[1]) / 2.0
    mean_v[1] = (bounds[2] + bounds[3]) / 2.0
    mean_v[2] = (bounds[4] + bounds[5]) / 2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = []
    for i in range(shapedatapoints.GetNumberOfPoints()):
        p = shapedatapoints.GetPoint(i)
        shape_points.append(p)
    shape_points = np.array(shape_points)

    # centering points of the shape
    if mean_arr is None:
        mean_arr = np.array(mean_v)
    shape_points = shape_points - mean_arr

    # Computing scale factor if it is not provided
    if scale_factor is None:
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1 / np.linalg.norm(bounds_max_arr - mean_arr)

    # scale points of the shape by scale factor
    shape_points_scaled = np.multiply(shape_points, scale_factor)

    # assigning scaled points back to shape
    for i in range(shapedatapoints.GetNumberOfPoints()):
        shapedatapoints.SetPoint(i, shape_points_scaled[i])

    surf.SetPoints(shapedatapoints)

    return surf
