import os
import vtk
import numpy as np
import json


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


def LoadJsonLandmarks(ldmk_path, full_landmark=True, list_landmark=[]):
    """
    Load landmarks from json file

    Parameters
    ----------
    img : sitk.Image
        Image to which the landmarks belong

    Returns
    -------
    dict
        Dictionary of landmarks

    Raises
    ------
    ValueError
        If the json file is not valid
    """

    with open(ldmk_path) as f:
        data = json.load(f)

    markups = data["markups"][0]["controlPoints"]

    landmarks = {}
    for markup in markups:
        lm_ph_coord = np.array(
            [markup["position"][0], markup["position"][1], markup["position"][2]]
        )
        lm_coord = lm_ph_coord.astype(np.float64)
        landmarks[markup["label"]] = lm_coord

    if not full_landmark:
        out = {}
        for lm in list_landmark:
            out[lm] = landmarks[lm]
        landmarks = out
    return landmarks


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


def WriteSurf(surf, output_folder, name, inname):
    dir, name = os.path.split(name)
    name, extension = os.path.splitext(name)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    writer = vtk.vtkPolyDataWriter()
    # print(os.path.join(output_folder,f"{name}{inname}{extension}"))
    writer.SetFileName(os.path.join(output_folder, f"{name}{inname}{extension}"))
    writer.SetInputData(surf)
    writer.Update()
