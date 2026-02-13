import os
import glob
import vtk
import numpy as np
import json
import SimpleITK as sitk
from vtk.util.numpy_support import vtk_to_numpy
from ASO_IOS_utils.OFFReader import OFFReader


def ReadSurf(path):
    fname, extension = os.path.splitext(os.path.basename(path))
    extension = extension.lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".off":
        reader = OFFReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".obj":
        if os.path.exists(fname + ".mtl"):
            obj_import = vtk.vtkOBJImporter()
            obj_import.SetFileName(path)
            obj_import.SetFileNameMTL(fname + ".mtl")
            textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
            if os.path.exists(textures_path):
                textures_path = os.path.normpath(
                    fname.replace(os.path.basename(fname), "")
                )
                obj_import.SetTexturePath(textures_path)
            else:
                textures_path = os.path.normpath(
                    fname.replace(os.path.basename(fname), "")
                )
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
            reader.SetFileName(path)
            reader.Update()
            surf = reader.GetOutput()

    return surf


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


def WriteSurf(surf, output_folder, name, inname):
    """Write surface to file with proper error handling.
    
    Args:
        surf: VTK polydata surface
        output_folder: Output directory path
        name: Filename (can include path)
        inname: Infix to add to filename (e.g., "Or" -> "A2_SegOr.vtk")
    """
    try:
        dir, name = os.path.split(name)
        name, extension = os.path.splitext(name)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        if extension == ".vtk":
            writer = vtk.vtkPolyDataWriter()
        elif extension == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
        elif extension == ".obj":
            writer = vtk.vtkOBJWriter()
        else:
            # Default to VTK format if extension is not recognized
            extension = ".vtk"
            writer = vtk.vtkPolyDataWriter()
        
        output_path = os.path.join(output_folder, f"{name}{inname}{extension}")
        print(f"DEBUG WriteSurf: output_path = {output_path}")
        print(f"DEBUG WriteSurf: output_folder = {output_folder}")
        print(f"DEBUG WriteSurf: name = {name}, inname = {inname}, extension = {extension}")
        
        writer.SetFileName(output_path)
        writer.SetInputData(surf)
        writer.Update()
        
        # Verify file was created
        if not os.path.exists(output_path):
            raise RuntimeError(f"WriteSurf failed: File {output_path} was not created after writer.Update()")
        
        # Check file size
        file_size = os.path.getsize(output_path)
        print(f"DEBUG WriteSurf: File created successfully: {output_path} ({file_size} bytes)")
            
    except Exception as e:
        print(f"ERROR in WriteSurf: {str(e)}")
        print(f"  Output folder: {output_folder}")
        print(f"  Filename: {name}{inname}{extension}")
        print(f"  Full path attempted: {output_path if 'output_path' in locals() else 'N/A'}")
        raise


def UpperOrLower(path_filename):
    """tell if the file is for upper jaw of lower

    Args:
        path_filename (str): exemple /home/..../landmark_upper.json

    Returns:
        str: Upper or Lower, for the following exemple if Upper
    """
    out = "Lower"
    st = "_U_"
    st2 = "upper"
    filename = os.path.basename(path_filename)
    if st in filename or st2 in filename.lower():
        out = "Upper"
    return out


def search(path, *args):
    """
    Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

    Example:
    args = ('json',['.nii.gz','.nrrd'])
    return:
        {
            'json' : ['path/a.json', 'path/b.json','path/c.json'],
            '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
            '.nrrd.gz' : ['path/c.nrrd']
        }
    """
    arguments = []
    for arg in args:
        if type(arg) == list:
            arguments.extend(arg)
        else:
            arguments.append(arg)
    return {
        key: [
            i
            for i in glob.iglob(
                os.path.normpath("/".join([path, "**", "*"])), recursive=True
            )
            if i.endswith(key)
        ]
        for key in arguments
    }


def PatientNumber(filename):
    number = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    for i in range(len(filename)):
        if filename[i] in number:
            for y in range(i, len(filename)):
                if not filename[y] in number:
                    return int(filename[i:y])


def WriteJsonLandmarks(
    landmarks, output_file, input_file_json, add_innamefile, output_folder
):
    """
    Write the landmarks to a json file

    Parameters
    ----------
    landmarks : dict
        landmarks to write
    output_file : str
        output file name
    """
    # # Load the input image
    dirname, name = os.path.split(output_file)
    name, extension = os.path.splitext(name)
    output_file = os.path.join(output_folder, name + add_innamefile + extension)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(input_file_json, "r") as outfile:
        tempData = json.load(outfile)
    for i in range(len(landmarks)):
        pos = landmarks[tempData["markups"][0]["controlPoints"][i]["label"]]
        tempData["markups"][0]["controlPoints"][i]["position"] = [
            pos[0],
            pos[1],
            pos[2],
        ]
    with open(output_file, "w") as outfile:

        json.dump(tempData, outfile, indent=4)


def listlandmark2diclandmark(list_landmark):
    upper = []
    lower = []
    list_landmark = list_landmark.split(",")
    for landmark in list_landmark:
        if "U" == landmark[0]:
            upper.append(landmark)
        else:
            lower.append(landmark)

    out = {"Upper": upper, "Lower": lower}

    return out


def WritefileError(file, folder_error, message):
    if not os.path.exists(folder_error):
        os.mkdir(folder_error)
    name = os.path.basename(file)
    name, _ = os.path.splitext(name)
    with open(os.path.join(folder_error, f"{name}Error.txt"), "w") as f:
        f.write(message)

def PatientNumber(path):
    matrix_pat = os.path.basename(path).split('_U')[0].split('_L')[0].split('.')[0].replace('.','')
    return matrix_pat


def saveMatrixAsTfm(matrix, output_path):
    assert matrix.shape == (4, 4), "Expected a 4x4 matrix."
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    inverted_matrix = np.linalg.inv(matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(inverted_matrix[:3, :3].flatten())
    transform.SetTranslation(inverted_matrix[:3, 3])
    sitk.WriteTransform(transform, output_path)