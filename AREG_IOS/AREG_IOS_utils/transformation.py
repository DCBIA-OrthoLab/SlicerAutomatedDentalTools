import SimpleITK as sitk
import numpy as np
import vtk
import os
from ADTLib.geometry import ApplyTransform, RotationMatrix, TransformDict, TransformList, TransformSurf  # noqa: F401  (re-exported)
# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("AREG_IOS_transformation")


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
        # The same situation as the T1 copy in AREG_IOS ten lines above this
        # call, which treats it as a warning: ASO produced no orientation
        # transform for this timepoint. Read without checking, it came out as a
        # SimpleITK HDF5 stack five frames deep, logged at ERROR, on a run whose
        # registration had in fact succeeded and written its surfaces. What
        # cannot be written is the COMPOSED transform, there being nothing to
        # compose with -- so that is what the line says now.
        if not aso_tfm_path or not os.path.exists(aso_tfm_path):
            logger.warning(
                "No ASO transform for %s at %s: the registered surfaces are "
                "written, but not the composed %s_T2_SegOr%s.tfm",
                patient_id, aso_tfm_path, patient_id, suffix)
            return
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

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(surf)
    transform_filter.Update()
    return transform_filter.GetOutput()


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
