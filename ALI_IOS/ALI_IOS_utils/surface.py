# Functions for surface loading, scaling, normal calculation, and mesh utilities
import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import torch
from monai.transforms import ToTensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ReadSurf(fileName):
    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".off":
        reader = OFFReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".obj":
        if os.path.exists(fname + ".mtl"):
            obj_import = vtk.vtkOBJImporter()
            obj_import.SetFileName(fileName)
            obj_import.SetFileNameMTL(fname + ".mtl")
            textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
            if os.path.exists(textures_path):
                obj_import.SetTexturePath(textures_path)
            obj_import.Read()

            actors = obj_import.GetRenderer().GetActors()
            actors.InitTraversal()
            append = vtk.vtkAppendPolyData()

            for i in range(actors.GetNumberOfItems()):
                surfActor = actors.GetNextActor()
                append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())

            append.Update()
            surf = append.GetOutput()

        else:
            reader = vtk.vtkOBJReader()
            reader.SetFileName(fileName)
            reader.Update()
            surf = reader.GetOutput()

    return surf

def ScaleSurf(surf, mean_arr = None, scale_factor = None):
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    shapedatapoints = surf.GetPoints()

    #calculate bounding box
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    mean_v[0] = (bounds[0] + bounds[1])/2.0
    mean_v[1] = (bounds[2] + bounds[3])/2.0
    mean_v[2] = (bounds[4] + bounds[5])/2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = []
    for i in range(shapedatapoints.GetNumberOfPoints()):
        p = shapedatapoints.GetPoint(i)
        shape_points.append(p)
    shape_points = np.array(shape_points)

    #centering points of the shape
    if mean_arr is None:
        mean_arr = np.array(mean_v)
    # print("Mean:", mean_arr)
    shape_points = shape_points - mean_arr

    #Computing scale factor if it is not provided
    if(scale_factor is None):
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1/np.linalg.norm(bounds_max_arr - mean_arr)

    #scale points of the shape by scale factor
    # print("Scale:", scale_factor)
    shape_points_scaled = np.multiply(shape_points, scale_factor)

    #assigning scaled points back to shape
    for i in range(shapedatapoints.GetNumberOfPoints()):
        shapedatapoints.SetPoint(i, shape_points_scaled[i])

    surf.SetPoints(shapedatapoints)

    return surf, mean_arr, scale_factor

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
    colored_points.SetName('colors')
    colored_points.SetNumberOfComponents(3)

    normals = surf.GetPointData().GetArray(array_name)

    for pid in range(surf.GetNumberOfPoints()):
        normal = np.array(normals.GetTuple(pid))
        rgb = (normal*0.5 + 0.5)*255.0
        colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])
    return colored_points

def GetSurfProp(surf_unit, surf_mean, surf_scale):
    surf = ComputeNormals(surf_unit)
    color_normals = ToTensor(dtype=torch.float32, device=DEVICE)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
    verts = ToTensor(dtype=torch.float32, device=DEVICE)(vtk_to_numpy(surf.GetPoints().GetData()))
    faces = ToTensor(dtype=torch.int64, device=DEVICE)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])

    try :
        region_id = torch.tensor((vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID"))),dtype=torch.int64)
    except AttributeError :
        try :
            region_id = torch.tensor((vtk_to_numpy(surf.GetPointData().GetScalars("predictedId"))),dtype=torch.int64)
        except AttributeError:
            region_id = torch.tensor((vtk_to_numpy(surf.GetPointData().GetScalars("Universal_ID"))),dtype=torch.int64)

    region_id = torch.clamp(region_id, min=0)
    return verts.unsqueeze(0), faces.unsqueeze(0), color_normals.unsqueeze(0), region_id.unsqueeze(0)

def RemoveExtraFaces(F,num_faces,RI,label):
    last_num_faces =[]
    for face in num_faces:
        vertex_color = F.squeeze(0)[int(face.item())]
        for vert in vertex_color:
            if RI.squeeze(0)[vert] == label:
                last_num_faces.append(face)
    return last_num_faces

def Upscale(landmark_pos, mean_arr, scale_factor):
    return (landmark_pos.cpu() / scale_factor) + mean_arr
