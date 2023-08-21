import vtk
import os,time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Apply_matrix_utils.OFFReader import OFFReader
from Apply_matrix_utils.General_tools import search
from Apply_matrix_utils.Matrix_tools import ReadMatrix





def ReadSurf(path:str):
    '''
    Read surface and return it
    '''
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


def WriteSurf(surf, output_folder:str,name:str,inname:str)->None:
        '''
        input -> surf : the surface to save, output_folder : the path where to solve the surface
        name : the name of the file to save, inname : the suffix to add at the name

        Save new surface with the right extension.
        '''
        dir, name = os.path.split(name)
        name, extension = os.path.splitext(name)

        out_path = os.path.dirname(output_folder)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        if extension == '.vtk':
            writer = vtk.vtkPolyDataWriter()
        elif extension == '.vtp':
            writer = vtk.vtkXMLPolyDataWriter()
        elif extension =='.obj':
            writer = vtk.vtkWriter()
        writer.SetFileName(os.path.join(out_path,f"{name}{inname}{extension}"))
        writer.SetInputData(surf)
        writer.Update()


def RotateTransform(surf, transform):
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()

def TransformSurf(surf,matrix):
    '''
    Apply the matrix to the surface
    '''
    assert isinstance(surf,vtk.vtkPolyData)
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.reshape(matrix,16))
    surf = RotateTransform(surf,transform)

    return surf


def GetPatientsVTK(file_path:str,matrix_path:str):
    """
        Return a dictionnary with the patients names for the key. Their .nii.gz files and their matrix.
        exemple :
        input : file_path matrix_path
        output : 
        ('patient1': {'scan':[file_path_1_patient1.nii.gz,file_path_2_patient1.nii.gz],'matrix':[matrix_path_1_patient1.tfm,matrix_path_2_patient1.tfm]})
    """
    patients = {}
    files = []

    if Path(file_path).is_dir():
        files_original = search(file_path,'.vtk','.vtp','.stl','.off','.obj')
        files = []
        for i in range(len(files_original['.vtk'])):
            files.append(files_original['.vtk'][i])

        for i in range(len(files_original['.vtp'])):
            files.append(files_original['.vtp'][i])

        for i in range(len(files_original['.stl'])):
            files.append(files_original['.stl'][i])
        
        for i in range(len(files_original['.off'])):
            files.append(files_original['.off'][i])

        for i in range(len(files_original['.obj'])):
            files.append(files_original['.obj'][i])

        for i in range(len(files)):
            file = files[i]

            file_pat = (os.path.basename(file)).split('_Seg')[0].split('_seg')[0].split('_Scan')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('.')[0]
            for i in range(50):
                file_pat=file_pat.split('_T'+str(i))[0]

            if file_pat not in patients.keys():
                patients[file_pat] = {}
                patients[file_pat]['scan'] = []
                patients[file_pat]['matrix'] = []
            patients[file_pat]['scan'].append(file)
    
    else : 
        name, extension = os.path.splitext(file_path)
        if extension ==".vtk" or extension ==".vtp" or extension ==".stl" or extension ==".off" or extension ==".obj" :
            files = [file_path]
            file_pat = os.path.basename(file_path).split('_Seg')[0].split('_seg')[0].split('_Scan')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('.')[0].replace('.','')
            for i in range(50):
                file_pat=file_pat.split('_T'+str(i))[0]

            if file_pat not in patients.keys():
                patients[file_pat] = {}
                patients[file_pat]['scan'] = []
                patients[file_pat]['matrix'] = []
            patients[file_pat]['scan'].append(file_path)    

    
    if Path(matrix_path).is_dir():
        matrixes_original = search(matrix_path,'.npy','.h5','.tfm','.mat','.txt')
        matrixes = []

        for i in range(len(matrixes_original['.npy'])):
            matrixes.append(matrixes_original['.npy'][i])
        
        for i in range(len(matrixes_original['.h5'])):
            matrixes.append(matrixes_original['.h5'][i])

        for i in range(len(matrixes_original['.tfm'])):
            matrixes.append(matrixes_original['.tfm'][i])

        for i in range(len(matrixes_original['.mat'])):
            matrixes.append(matrixes_original['.mat'][i])

        for i in range(len(matrixes_original['.txt'])):
            matrixes.append(matrixes_original['.txt'][i])
            
        for i in range(len(matrixes)):
            matrix = matrixes[i]
            matrix_pat = os.path.basename(matrix).split('_Left')[0].split('_left')[0].split('_Right')[0].split('_right')[0].split('_T1')[0].split('_T2')[0].split('_MA')[0]

            for i in range(50):
                matrix_pat=matrix_pat.split('_T'+str(i))[0]

            if matrix_pat in patients.keys():
                patients[matrix_pat]['matrix'].append(matrix)

    else : 
        for key in patients.keys() :
            patients[key]['matrix'].append(matrix_path)

    return patients,len(files)



def ApplyMatrixVTK(patients:list,keys:list,input_path:str, out_path:str, num_worker=0, shared_list=None,logPath=None,idx=0,suffix=""):
    """
        Process the files in patients by applying their matrix and saved them
    """
    for key in keys:
        try:
            for scan in patients[key]["scan"] :
                surf = ReadSurf(scan)
                outpath = scan.replace(os.path.normpath(input_path),os.path.normpath(out_path))

                for matrix_path in patients[key]["matrix"] :
                    matrix = ReadMatrix(matrix_path)
                    new_surf=TransformSurf(surf,matrix)

                    matrix_name = os.path.basename(matrix_path).split('.tfm')[0].split('.h5')[0].split('.npy')[0].split('.mat')[0].split('.txt')[0].split(key)[1]
                    WriteSurf(new_surf,outpath,scan,suffix+matrix_name)

                shared_list[num_worker] += 1

                    
           
          
        except KeyError:
            print(f"Patient {key} not have either scan or matrix")
            shared_list[num_worker] += 1
            continue

        time.sleep(0.5)


def CheckSharedListVTK(shared_list:list,maxvalue:int,logPath:str,idxProcess)->None:
    """
        Update the log files for the progress bar when a new files has been processed
    """
    for i in tqdm(range(maxvalue)):
        while sum(shared_list) < i+1:
            time.sleep(0.1)
        with open(logPath,'r+') as log_f :
            idxProcess.acquire()
            log_f.write(str(idxProcess.value))
        idxProcess.value +=1
        idxProcess.release()
        time.sleep(0.5)