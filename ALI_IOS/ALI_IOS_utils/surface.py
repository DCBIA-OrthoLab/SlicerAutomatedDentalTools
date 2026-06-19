# Functions for surface loading, scaling, normal calculation, and mesh utilities
import os
import logging
import numpy as np
import sys
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import torch
from monai.transforms import ToTensor

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("ALI_IOS_Surface")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ReadSurf(fileName):
    """Read surface file with error handling."""
    try:
        if not os.path.exists(fileName):
            logger.error(f"Surface file not found: {fileName}")
            raise FileNotFoundError(f"File does not exist: {fileName}")
        
        fname, extension = os.path.splitext(fileName)
        extension = extension.lower()
        logger.debug(f"Reading surface file with extension: {extension}")
        
        if extension == ".vtk":
            reader = vtk.vtkPolyDataReader()
        elif extension == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        elif extension == ".stl":
            reader = vtk.vtkSTLReader()
        elif extension == ".off":
            reader = OFFReader()
        elif extension == ".obj":
            if os.path.exists(fname + ".mtl"):
                obj_import = vtk.vtkOBJImporter()
                obj_import.SetFileName(fileName)
                obj_import.SetFileNameMTL(fname + ".mtl")
                textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
                if os.path.exists(textures_path):
                    obj_import.SetTexturePath(textures_path)
                try:
                    obj_import.Read()
                except Exception as e:
                    logger.error(f"Error reading OBJ file: {e}")
                    raise

                actors = obj_import.GetRenderer().GetActors()
                actors.InitTraversal()
                append = vtk.vtkAppendPolyData()

                for i in range(actors.GetNumberOfItems()):
                    surfActor = actors.GetNextActor()
                    append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())

                append.Update()
                surf = append.GetOutput()
                logger.info(f"Successfully loaded OBJ file with material: {fileName}")
                return surf
            else:
                reader = vtk.vtkOBJReader()
        else:
            logger.error(f"Unsupported file format: {extension}")
            raise ValueError(f"Unsupported file format: {extension}")
        
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
        
        if surf.GetNumberOfPoints() == 0:
            logger.error(f"Surface file is empty: {fileName}")
            raise ValueError("Surface has no points")
        
        logger.info(f"Successfully loaded surface from {fileName} with {surf.GetNumberOfPoints()} points")
        return surf
    except Exception as e:
        logger.error(f"Error reading surface file {fileName}: {e}")
        raise

def ScaleSurf(surf, mean_arr=None, scale_factor=None):
    """Scale surface with error handling."""
    try:
        if surf is None or surf.GetNumberOfPoints() == 0:
            logger.error("Input surface is invalid")
            raise ValueError("Invalid surface")
        
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
        logger.debug(f"Surface scaled successfully with scale factor: {scale_factor}")
        return surf, mean_arr, scale_factor
    except Exception as e:
        logger.error(f"Error scaling surface: {e}")
        raise

def ComputeNormals(surf):
    """Compute normals for surface with error handling."""
    try:
        if surf is None or surf.GetNumberOfPoints() == 0:
            logger.error("Cannot compute normals for invalid surface")
            raise ValueError("Invalid surface")
        
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surf)
        normals.ComputeCellNormalsOff()
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()
        
        result = normals.GetOutput()
        logger.debug(f"Normals computed successfully for {result.GetNumberOfPoints()} points")
        return result
    except Exception as e:
        logger.error(f"Error computing normals: {e}")
        raise

def GetColorArray(surf, array_name):
    """Get color array from surface with error handling."""
    try:
        colored_points = vtk.vtkUnsignedCharArray()
        colored_points.SetName('colors')
        colored_points.SetNumberOfComponents(3)

        normals = surf.GetPointData().GetArray(array_name)
        if normals is None:
            logger.error(f"Array '{array_name}' not found in surface")
            raise ValueError(f"Array not found: {array_name}")

        for pid in range(surf.GetNumberOfPoints()):
            normal = np.array(normals.GetTuple(pid))
            rgb = (normal * 0.5 + 0.5) * 255.0
            colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])
        
        logger.debug(f"Color array created with {colored_points.GetNumberOfTuples()} tuples")
        return colored_points
    except Exception as e:
        logger.error(f"Error getting color array: {e}")
        raise

def GetSurfProp(surf_unit, surf_mean, surf_scale):
    """Get surface properties with error handling."""
    try:
        surf = ComputeNormals(surf_unit)
        color_normals = ToTensor(dtype=torch.float32, device=DEVICE)(
            vtk_to_numpy(GetColorArray(surf, "Normals")) / 255.0
        )
        verts = ToTensor(dtype=torch.float32, device=DEVICE)(vtk_to_numpy(surf.GetPoints().GetData()))
        faces = ToTensor(dtype=torch.int64, device=DEVICE)(
            vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
        )

        region_id = None
        for scalar_name in ["PredictedID", "predictedId", "Universal_ID"]:
            try:
                region_id = torch.tensor(
                    (vtk_to_numpy(surf.GetPointData().GetScalars(scalar_name))),
                    dtype=torch.int64
                )
                logger.debug(f"Found region ID array: {scalar_name}")
                break
            except AttributeError:
                continue

        if region_id is None:
            logger.warning("No region ID array found, using default")
            region_id = torch.zeros(surf.GetNumberOfPoints(), dtype=torch.int64)
        
        region_id = torch.clamp(region_id, min=0)
        logger.debug(f"Surface properties extracted: verts {verts.shape}, faces {faces.shape}")
        return verts.unsqueeze(0), faces.unsqueeze(0), color_normals.unsqueeze(0), region_id.unsqueeze(0)
    except Exception as e:
        logger.error(f"Error getting surface properties: {e}")
        raise

def RemoveExtraFaces(F, num_faces, RI, label):
    """Remove extra faces with error handling."""
    try:
        if not num_faces:
            logger.warning(f"No faces provided for label {label}")
            return []
        
        last_num_faces = []
        for face in num_faces:
            try:
                vertex_color = F.squeeze(0)[int(face.item())]
                for vert in vertex_color:
                    if RI.squeeze(0)[vert] == label:
                        last_num_faces.append(face)
                        break
            except Exception as e:
                logger.warning(f"Error processing face {face}: {e}")
                continue
        
        logger.debug(f"Kept {len(last_num_faces)} out of {len(num_faces)} faces for label {label}")
        return last_num_faces
    except Exception as e:
        logger.error(f"Error removing extra faces: {e}")
        raise

def Upscale(landmark_pos, mean_arr, scale_factor):
    """Upscale landmark position with error handling."""
    try:
        if landmark_pos is None or mean_arr is None:
            logger.error("Invalid input to Upscale")
            raise ValueError("Invalid parameters")
        
        if scale_factor == 0:
            logger.error("Scale factor cannot be zero")
            raise ValueError("Invalid scale factor")
        
        result = (landmark_pos.cpu() / scale_factor) + mean_arr
        logger.debug(f"Landmark upscaled: {result}")
        return result
    except Exception as e:
        logger.error(f"Error upscaling landmark: {e}")
        raise
