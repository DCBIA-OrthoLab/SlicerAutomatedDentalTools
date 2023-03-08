
import os 
import glob 
import vtk
import numpy as np
import json
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
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))
                obj_import.SetTexturePath(textures_path)
            else:
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))                
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


def LoadJsonLandmarks(ldmk_path,full_landmark=True,list_landmark=[]):
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
        lm_ph_coord = np.array([markup["position"][0],markup["position"][1],markup["position"][2]])
        lm_coord = lm_ph_coord.astype(np.float64)
        landmarks[markup["label"]] = lm_coord
    
    if not full_landmark:
        out={}
        for lm in list_landmark:
            out[lm] = landmarks[lm]
        landmarks = out
    return landmarks







def WriteSurf(surf, output_folder,name,inname):
    dir, name = os.path.split(name)
    name, extension = os.path.splitext(name)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif extension =='.obj':
        writer = vtk.vtkWriter()
    writer.SetFileName(os.path.join(output_folder,f"{name}{inname}{extension}"))
    writer.SetInputData(surf)
    writer.Update()






def UpperOrLower(path_filename):
    """tell if the file is for upper jaw of lower

    Args:
        path_filename (str): exemple /home/..../landmark_upper.json

    Returns:
        str: Upper or Lower, for the following exemple if Upper
    """
    out = 'Lower'
    st = '_U_'
    st2= 'upper'
    filename = os.path.basename(path_filename)
    if st in filename or st2 in filename.lower():
        out ='Upper'
    return out




def search(path,*args):
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
    arguments=[]
    for arg in args:
        if type(arg) == list:
            arguments.extend(arg)
        else:
            arguments.append(arg)
    return {key: [i for i in glob.iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)] for key in arguments}




def PatientNumber(filename):
    number = ['1','2','3','4','5','6','7','8','9','0']
    for i in range(len(filename)):
        if filename[i] in number:
            for y in range(i,len(filename)):
                if not filename[y] in number:
                    return int(filename[i:y])





def WriteJsonLandmarks(landmarks,output_file,input_file_json,add_innamefile,output_folder):
    '''
    Write the landmarks to a json file
    
    Parameters
    ----------
    landmarks : dict
        landmarks to write
    output_file : str
        output file name
    '''
    # # Load the input image
    # spacing, origin = LoadImage(input_file)
    dirname , name  = os.path.split(output_file)
    name, extension = os.path.splitext(name)
    output_file = os.path.join(output_folder,name+add_innamefile+extension)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    

    with open(input_file_json, 'r') as outfile:
        tempData = json.load(outfile)
    for i in range(len(landmarks)):
        pos = landmarks[tempData['markups'][0]['controlPoints'][i]['label']]
        # pos = (pos + abs(inorigin)) * inspacing
        tempData['markups'][0]['controlPoints'][i]['position'] = [pos[0],pos[1],pos[2]]
    with open(output_file, 'w') as outfile:

        json.dump(tempData, outfile, indent=4)




def listlandmark2diclandmark(list_landmark):
    upper =[]
    lower=[]
    list_landmark=list_landmark.split(',')
    for landmark in list_landmark:
        if 'U' == landmark[0]:
            upper.append(landmark)
        else :
            lower.append(landmark)

    out ={'Upper':upper,'Lower':lower}

    return out



def WritefileError(file,folder_error,message):
    if not os.path.exists(folder_error):
        os.mkdir(folder_error)
    name = os.path.basename(file)
    name , _ = os.path.splitext(name)
    with open(os.path.join(folder_error,f'{name}Error.txt'),'w') as f:
        f.write(message)













