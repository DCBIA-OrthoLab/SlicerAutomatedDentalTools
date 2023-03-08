'''
8888888 888b     d888 8888888b.   .d88888b.  8888888b.  88888888888  .d8888b.  
  888   8888b   d8888 888   Y88b d88P" "Y88b 888   Y88b     888     d88P  Y88b 
  888   88888b.d88888 888    888 888     888 888    888     888     Y88b.      
  888   888Y88888P888 888   d88P 888     888 888   d88P     888      "Y888b.   
  888   888 Y888P 888 8888888P"  888     888 8888888P"      888         "Y88b. 
  888   888  Y8P  888 888        888     888 888 T88b       888           "888 
  888   888   "   888 888        Y88b. .d88P 888  T88b      888     Y88b  d88P 
8888888 888       888 888         "Y88888P"  888   T88b     888      "Y8888P"  
'''

import json
import numpy as np
import vtk
import SimpleITK as sitk
import glob
import os
import shutil

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkIterativeClosestPointTransform,
    vtkPolyData
)
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter


cross = lambda x,y:np.cross(x,y) # to avoid unreachable code error on np.cross function


'''
888             d8888 888b    888 8888888b.  888b     d888        d8888 8888888b.  888    d8P   .d8888b.  
888            d88888 8888b   888 888  "Y88b 8888b   d8888       d88888 888   Y88b 888   d8P   d88P  Y88b 
888           d88P888 88888b  888 888    888 88888b.d88888      d88P888 888    888 888  d8P    Y88b.      
888          d88P 888 888Y88b 888 888    888 888Y88888P888     d88P 888 888   d88P 888d88K      "Y888b.   
888         d88P  888 888 Y88b888 888    888 888 Y888P 888    d88P  888 8888888P"  8888888b        "Y88b. 
888        d88P   888 888  Y88888 888    888 888  Y8P  888   d88P   888 888 T88b   888  Y88b         "888 
888       d8888888888 888   Y8888 888  .d88P 888   "   888  d8888888888 888  T88b  888   Y88b  Y88b  d88P 
88888888 d88P     888 888    Y888 8888888P"  888       888 d88P     888 888   T88b 888    Y88b  "Y8888P" 
'''
def MergeJson(data_dir,extension='MERGED'):
    """
    Create one MERGED json file per scans from all the different json files (Upper, Lower...)
    """

    normpath = os.path.normpath("/".join([data_dir, '**', '']))
    json_file = [i for i in sorted(glob.iglob(normpath, recursive=True)) if i.endswith('.json')]

    # ==================== ALL JSON classified by patient  ====================
    dict_list = {}
    for file in json_file:
        patient = '_'.join(file.split('/')[-3:-1])+'#'+file.split('/')[-1].split('.')[0].split('_lm')[0]+'_lm'
        if patient not in dict_list:
            dict_list[patient] = []
        dict_list[patient].append(file)

    # ==================== MERGE JSON  ====================``
    for key, files in dict_list.items():
        file1 = files[0]
        with open(file1, 'r') as f:
            data1 = json.load(f)
            data1["@schema"] = "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#"
        for i in range(1,len(files)):
            with open(files[i], 'r') as f:
                data = json.load(f)
            data1['markups'][0]['controlPoints'].extend(data['markups'][0]['controlPoints'])
        outpath = os.path.normpath("/".join(files[0].split('/')[:-1]))        # Write the merged json file
        with open(outpath+'/'+key.split('#')[1] + '_'+ extension +'.mrk.json', 'w') as f: 
            json.dump(data1, f, indent=4)

    # ==================== DELETE UNUSED JSON  ====================
    for key, files in dict_list.items():
        for file in files:
            if extension not in os.path.basename(file):
                os.remove(file)    

def LoadJsonLandmarks(ldmk_path, ldmk_list=None):
    """
    Load landmarks from json file
    
    Parameters
    ----------
    ldmk_path : str
        Path to the json file
    gold : bool, optional
        If True, load gold standard landmarks, by default False
    
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
        try:
            lm_ph_coord = np.array([markup["position"][0],markup["position"][1],markup["position"][2]])
            #lm_coord = ((lm_ph_coord - origin) / spacing).astype(np.float16)
            lm_coord = lm_ph_coord.astype(np.float64)
            landmarks[markup["label"]] = lm_coord
        except IndexError:
            continue
    if ldmk_list is not None:
        return {key:landmarks[key] for key in ldmk_list if key in landmarks.keys()}
    
    return landmarks

def FindOptimalLandmarks(source,target,nb_lmrk):
    '''
    Find the optimal landmarks to use for the Init ICP
    
    Parameters
    ----------
    source : dict
        source landmarks
    target : dict
        target landmarks
    
    Returns
    -------
    list
        list of the optimal landmarks
    '''
    dist, LMlist,ii = [],[],0
    while len(dist) < (nb_lmrk*(nb_lmrk-1)*(nb_lmrk-2)) and ii < 2500:
        ii+=1
        firstpick,secondpick,thirdpick, d = InitICP(source,target, Print=False, search=True)
        if [firstpick,secondpick,thirdpick] not in LMlist:
            dist.append(d)
            LMlist.append([firstpick,secondpick,thirdpick])
    return LMlist[dist.index(min(dist))]


def search(self,path,*args):
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

def WriteJsonLandmarks(landmarks, input_json_file ,output_file):
    '''
    Write the landmarks to a json file
    
    Parameters
    ----------
    landmarks : dict
        landmarks to write
    output_file : str
        output file name
    '''
    with open(input_json_file, 'r') as outfile:
        tempData = json.load(outfile)
    for i in range(len(landmarks)):
        pos = landmarks[tempData['markups'][0]['controlPoints'][i]['label']]
        # pos = (pos + abs(inorigin)) * inspacing
        tempData['markups'][0]['controlPoints'][i]['position'] = [pos[0],pos[1],pos[2]]
    if not os.path.exists(output_file):
        shutil.copy(input_json_file,output_file)
    with open(output_file, 'w') as outfile:
        json.dump(tempData, outfile, indent=4)

def GenControlePoint(landmarks):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark,data in landmarks.items():
        id+=1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [data[0], data[1], data[2]],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "defined"
        }
        lm_lst.append(controle_point)

    return lm_lst

def WriteJson(landmarks,out_path):
    false = False
    true = True
    file = {
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
    "markups": [
        {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": false,
            "labelFormat": "%N-%d",
            "controlPoints": GenControlePoint(landmarks),
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.0],
                "color": [0.5, 0.5, 0.5],
                "selectedColor": [0.26666666666666669, 0.6745098039215687, 0.39215686274509806],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 2.0,
                "glyphType": "Sphere3D",
                "glyphScale": 2.0,
                "glyphSize": 5.0,
                "useGlyphScale": true,
                "sliceProjection": false,
                "sliceProjectionUseFiducialColor": true,
                "sliceProjectionOutlinedBehindSlicePlane": false,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": false,
                "snapMode": "toVisibleSurface"
            }
        }
    ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=4)

    f.close

def GetDistances(landmark_dic):
    """Compute all distances between all landmarks for a single file"""
    distances = {}

    landmarks = landmark_dic.keys()

    for lm in landmarks:
        distances[lm] = {}
        for lm2 in landmarks:
            if lm != lm2:
                distances[lm][lm2] = np.linalg.norm(landmark_dic[lm] - landmark_dic[lm2])

    return distances

def GetDistDifference(gold_dic, input_dic):
    """Compare the distances between all landmarks for two files in a dictionary"""
    gold = GetDistances(gold_dic)
    test = GetDistances(input_dic)

    differences = {}

    for lm in test.keys():
        differences[lm] = {}
        for lm2 in test[lm].keys():
            differences[lm][lm2] = abs(gold[lm][lm2] - test[lm][lm2])

    return differences

def GetDirections(landmark_dic):
    """Compute all directions between all landmarks for a single file"""
    directions = {}

    landmarks = landmark_dic.keys()

    for lm in landmarks:
        directions[lm] = {}
        for lm2 in landmarks:
            if lm != lm2:
                directions[lm][lm2] = landmark_dic[lm] - landmark_dic[lm2]

    return directions

def GetDirDifference(gold_dic, input_dic):
    """Compare the angular differences between all landmarks for two files in a dictionary"""
    gold = GetDirections(gold_dic)
    test = GetDirections(input_dic)

    angular_diff = {}

    for lm in test.keys():
        angular_diff[lm] = {}
        for lm2 in test[lm].keys():
            angular_diff[lm][lm2] = np.arccos(np.dot(gold[lm][lm2], test[lm][lm2]) / (np.linalg.norm(gold[lm][lm2])* np.linalg.norm(test[lm][lm2])))

    return angular_diff

def GetCount(differences, max_diff=1, min_diff=0):
    """Count, for each landmark, the number of landmarks that are too far from it"""
    landmark_count = {}

    for lm in differences.keys():
        count = 0
        for lm2 in differences[lm].keys():
            if differences[lm][lm2] > max_diff:
                count += 1
            if differences[lm][lm2] < min_diff:
                count -= 1
        landmark_count[lm] = count

    return landmark_count

def SumCount(distance_count, direction_count):
    """Sum the number count of distance and direction"""
    sum_count = {}

    for lm in distance_count.keys():
        sum_count[lm] = distance_count[lm] + direction_count[lm]

    return sum_count

def GetLandmarkToRemove(input_path,gold_path):
    """Get the list of landmark that should be removed from the landmark dictionary based on the difference with the gold standard"""

    test_ldmk = LoadJsonLandmarks(input_path)
    gold_ldmk = LoadJsonLandmarks(gold_path)

    dist_diff = GetDistDifference(gold_ldmk ,test_ldmk)
    dir_diff = GetDirDifference(gold_ldmk ,test_ldmk)

    dist_count = GetCount(dist_diff,max_diff=15)
    dir_count = GetCount(dir_diff,max_diff=0.4,min_diff=0.1)

    tot_count = SumCount(dist_count, dir_count)

    removed_landmarks = []
    for lm in tot_count.keys():
        if tot_count[lm] > len(test_ldmk):
            removed_landmarks.append(lm)

    return removed_landmarks

'''
888b     d888 8888888888 88888888888 8888888b.  8888888  .d8888b.   .d8888b.  
8888b   d8888 888            888     888   Y88b   888   d88P  Y88b d88P  Y88b 
88888b.d88888 888            888     888    888   888   888    888 Y88b.      
888Y88888P888 8888888        888     888   d88P   888   888         "Y888b.   
888 Y888P 888 888            888     8888888P"    888   888            "Y88b. 
888  Y8P  888 888            888     888 T88b     888   888    888       "888 
888   "   888 888            888     888  T88b    888   Y88b  d88P Y88b  d88P 
888       888 8888888888     888     888   T88b 8888888  "Y8888P"   "Y8888P" 
'''

def ComputeMeanDistance(source, target):
    """
    Computes the mean distance between two point sets.
    
    Parameters
    ----------
    source : dict
        Source landmarks
    target : dict
        Target landmarks
    
    Returns
    -------
    float
        Mean distance
    """
    distance = 0
    for key in source.keys():
        distance += np.linalg.norm(source[key] - target[key])
    distance /= len(source.keys())
    return distance

'''
888     888 88888888888 8888888 888       .d8888b.  
888     888     888       888   888      d88P  Y88b 
888     888     888       888   888      Y88b.      
888     888     888       888   888       "Y888b.   
888     888     888       888   888          "Y88b. 
888     888     888       888   888            "888 
Y88b. .d88P     888       888   888      Y88b  d88P 
 "Y88888P"      888     8888888 88888888  "Y8888P"                                                   
'''

def SortDict(input_dict):
    """
    Sorts a dictionary by key
    
    Parameters
    ----------
    input_dict : dict
        Dictionary to be sorted
    
    Returns
    -------
    dict
        Sorted dictionary
    """
    return {k: input_dict[k] for k in sorted(input_dict)}

def PrintMatrix(transform):
    """
    Prints a matrix
    
    Parameters
    ----------
    transform : vtk.vtkMatrix4x4
        Matrix to be printed
    """
    for i in range(4):
        print(transform.GetElement(i,0), transform.GetElement(i,1), transform.GetElement(i,2), transform.GetElement(i,3))
    print()

'''
888     888 88888888888 888    d8P       .d8888b.  88888888888 888     888 8888888888 8888888888 
888     888     888     888   d8P       d88P  Y88b     888     888     888 888        888        
888     888     888     888  d8P        Y88b.          888     888     888 888        888        
Y88b   d88P     888     888d88K          "Y888b.       888     888     888 8888888    8888888    
 Y88b d88P      888     8888888b            "Y88b.     888     888     888 888        888        
  Y88o88P       888     888  Y88b             "888     888     888     888 888        888        
   Y888P        888     888   Y88b      Y88b  d88P     888     Y88b. .d88P 888        888        
    Y8P         888     888    Y88b      "Y8888P"      888      "Y88888P"  888        888  
'''                                                                                   

def ConvertToVTKPoints(dict_landmarks):
    """
    Convert dictionary of landmarks to vtkPoints
    
    Parameters
    ----------
    dict_landmarks : dict
        Dictionary of landmarks with key as landmark name and value as landmark coordinates\
        Example: {'L1': [0, 0, 0], 'L2': [1, 1, 1], 'L3': [2, 2, 2]}

    Returns
    -------
    vtkPoints
        VTK points object
    """
    Points = vtkPoints()
    Vertices = vtkCellArray()
    labels = vtk.vtkStringArray()
    labels.SetNumberOfValues(len(dict_landmarks.keys()))
    labels.SetName("labels")

    for i,landmark in enumerate(dict_landmarks.keys()):
        sp_id = Points.InsertNextPoint(dict_landmarks[landmark])
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(sp_id)
        labels.SetValue(i, landmark)
        
    output = vtkPolyData()
    output.SetPoints(Points)
    output.SetVerts(Vertices)
    output.GetPointData().AddArray(labels)

    return output

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


'''
8888888  .d8888b.  8888888b.       .d8888b.  88888888888 888     888 8888888888 8888888888 
  888   d88P  Y88b 888   Y88b     d88P  Y88b     888     888     888 888        888        
  888   888    888 888    888     Y88b.          888     888     888 888        888        
  888   888        888   d88P      "Y888b.       888     888     888 8888888    8888888    
  888   888        8888888P"          "Y88b.     888     888     888 888        888        
  888   888    888 888                  "888     888     888     888 888        888        
  888   Y88b  d88P 888            Y88b  d88P     888     Y88b. .d88P 888        888        
8888888  "Y8888P"  888             "Y8888P"      888      "Y88888P"  888        888 
'''
def ICP_Transform(source, target):
    """
    Create the VTK ICP transform with source and target
    """
    # ============ create source points ==============
    source = ConvertToVTKPoints(source)

    # ============ create target points ==============
    target = ConvertToVTKPoints(target)

    # ============ create ICP transform ==============
    icp = vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(1000)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    # print("Number of iterations: {}".format(icp.GetNumberOfIterations()))

    # ============ apply ICP transform ==============
    transformFilter = vtkTransformPolyDataFilter()
    transformFilter.SetInputData(source)
    transformFilter.SetTransform(icp)
    transformFilter.Update()

    return icp

def InitICP(source,target, Print=False, BestLMList=None, search=False):
    """
    Do some initialisation transforms (1 translation and 2 rotations to make the ICP even more efficient
    """

    TransformList = []
    TranslationTransformMatrix = np.eye(4)
    RotationTransformMatrix = np.eye(4)

    labels = list(source.keys())
    if BestLMList is not None:
        firstpick, secondpick, thirdpick = BestLMList[0], BestLMList[1], BestLMList[2]
        if Print:
            print("Best Landmarks are: {},{},{}".format(firstpick, secondpick, thirdpick))
    # print("Mean Distance:{:.2f}".format(ComputeMeanDistance(source, target)))

    # ============ Pick a Random Landmark ==============
    if BestLMList is None:
        firstpick = labels[np.random.randint(0, len(labels))]

    if Print:
        print("First pick: {}".format(firstpick))

    # ============ Compute Translation Transform ==============
    T = target[firstpick] - source[firstpick]
    TranslationTransformMatrix[:3, 3] = T
    Translationsitk = sitk.TranslationTransform(3)
    Translationsitk.SetOffset(T.tolist())
    TransformList.append(Translationsitk)
    # ============ Apply Translation Transform ==============
    source = ApplyTranslation(source,T)
    # source = ApplyTransform(source, TranslationTransformMatrix)

    # print("Mean Distance:{:.2f}".format(ComputeMeanDistance(source, target)))

    # ============ Pick Another Random Landmark ==============
    if BestLMList is None:
        while True:
            secondpick = labels[np.random.randint(0, len(labels))]
            # secondpick = 'ROr'
            if secondpick != firstpick:
                break
    if Print:
        print("Second pick: {}".format(secondpick))

    # ============ Compute Rotation Angle and Axis ==============
    v1 = abs(source[secondpick] - source[firstpick])
    v2 = abs(target[secondpick] - target[firstpick])
    angle,axis = AngleAndAxisVectors(v2, v1)

    # print("Angle: {:.4f}".format(angle))
    # print("Angle: {:.2f}°".format(angle*180/np.pi))

    # ============ Compute Rotation Transform ==============
    R = RotationMatrix(axis,angle)
    # TransformMatrix[:3, :3] = R
    RotationTransformMatrix[:3, :3] = R
    Rotationsitk = sitk.VersorRigid3DTransform()
    Rotationsitk.SetMatrix(R.flatten().tolist())
    TransformList.append(Rotationsitk)
    # ============ Apply Rotation Transform ==============
    # source = ApplyRotation(source,R)
    source = ApplyTransform(source, RotationTransformMatrix)

    # print("Mean Distance:{:.2f}".format(ComputeMeanDistance(source, target)))
    # print("Rotation:\n{}".format(R))
    
    # ============ Compute Transform Matrix (Rotation + Translation) ==============
    TransformMatrix = RotationTransformMatrix @ TranslationTransformMatrix

    # ============ Pick another Random Landmark ==============
    if BestLMList is None:
        while True:
            thirdpick = labels[np.random.randint(0, len(labels))]
            # thirdpick = 'Ba'
            if thirdpick != firstpick and thirdpick != secondpick:
                break
    if Print:
        print("Third pick: {}".format(thirdpick))
    
    # ============ Compute Rotation Angle and Axis ==============
    v1 = abs(source[thirdpick] - source[firstpick])
    v2 = abs(target[thirdpick] - target[firstpick])
    angle,axis = AngleAndAxisVectors(v2, v1)
    # print("Angle: {:.4f}".format(angle))

    # ============ Compute Rotation Transform ==============
    RotationTransformMatrix = np.eye(4)
    R = RotationMatrix(abs(source[secondpick] - source[firstpick]),angle)
    RotationTransformMatrix[:3, :3] = R
    Rotationsitk = sitk.VersorRigid3DTransform()
    Rotationsitk.SetMatrix(R.flatten().tolist())
    TransformList.append(Rotationsitk)
    # ============ Apply Rotation Transform ==============
    # source = ApplyRotation(source,R)
    source = ApplyTransform(source, RotationTransformMatrix)

    # ============ Compute Transform Matrix (Init ICP) ==============
    TransformMatrix = RotationTransformMatrix @ TransformMatrix

    if Print:
        print("Mean Distance:{:.2f}".format(ComputeMeanDistance(source, target)))
    
    # return source
    if search:
        return firstpick,secondpick,thirdpick, ComputeMeanDistance(source, target)

    return source, TransformMatrix, TransformList

def ICP(input_file,input_json_file,gold_file,gold_json_file,list_landmark):
    # Check if some landmarks are not well located
    ldmk_to_remove = GetLandmarkToRemove(input_json_file,gold_json_file)
    if len(ldmk_to_remove) > 0:
        print("Patient {} --> Landmark not used:{}".format(os.path.basename(input_file).split('.'),ldmk_to_remove))
        list_landmark = [lm for lm in list_landmark if lm not in ldmk_to_remove]
    
    if len(list_landmark) <= 3:
        return None,None
    
    # Read input files
    input_image = sitk.ReadImage(input_file)
    # print('input spacing:',input_image.GetSpacing())
    gold_image = sitk.ReadImage(gold_file)
    # print('gold spacing:',gold_image.GetSpacing())
    source = LoadJsonLandmarks(input_json_file,list_landmark)
    nb_lmrk = len(source.keys())

    target = LoadJsonLandmarks(gold_json_file,list_landmark)
    target = {key:target[key] for key in source.keys()} # If source and target don't have the same number of landmarks

    # Make sure the landmarks are in the same order
    source = SortDict(source)
    source_orig = source.copy()
    target = SortDict(target)

    # save the source and target landmarks arrays
    script_dir = os.path.dirname(__file__)

    # Apply Init ICP with only the best landmarks
    source_transformed, TransformMatrix, TransformList = InitICP(source,target, Print=False, BestLMList=FindOptimalLandmarks(source,target,nb_lmrk))
    
    # Apply ICP
    icp = ICP_Transform(source_transformed,target) 
    TransformMatrixBis = VTKMatrixToNumpy(icp.GetMatrix())

    # Split the transform matrix into translation and rotation simpleitk transform
    TransformMatrixsitk = sitk.VersorRigid3DTransform()
    TransformMatrixsitk.SetTranslation(TransformMatrixBis[:3, 3].tolist())
    try:
        TransformMatrixsitk.SetMatrix(TransformMatrixBis[:3, :3].flatten().tolist())
    except RuntimeError:
        print('Error: The rotation matrix is not orthogonal')
        mat = TransformMatrixBis[:3, :3]
        print(mat)
        print('det:', np.linalg.det(mat))
        print('AxA^T:', mat @ mat.T)
    TransformList.append(TransformMatrixsitk)


    # Compute the final transform (inverse all the transforms)
    TransformSITK = sitk.CompositeTransform(3)
    for i in range(len(TransformList)-1,-1,-1):
        TransformSITK.AddTransform(TransformList[i])

    TransformSITK = TransformSITK.GetInverse()
    # Write the transform to a file
    # sitk.WriteTransform(TransformSITK, 'data/output/transform.tfm')

    TransformMatrixFinal = TransformMatrixBis @ TransformMatrix
    # print(TransformMatrixFinal)

    # Apply the final transform matrix
    source_transformed = ApplyTransform(source_transformed,TransformMatrixBis)
    
    source = ApplyTransform(source_orig,TransformMatrixFinal)
        
    # Resample the source image with the final transform 
    output = ResampleImage(input_image, gold_image, transform=TransformSITK)
    return output,source_transformed

'''
 .d8888b.  8888888 88888888888 888    d8P       .d8888b.  88888888888 888     888 8888888888 8888888888 
d88P  Y88b   888       888     888   d8P       d88P  Y88b     888     888     888 888        888        
Y88b.        888       888     888  d8P        Y88b.          888     888     888 888        888        
 "Y888b.     888       888     888d88K          "Y888b.       888     888     888 8888888    8888888    
    "Y88b.   888       888     8888888b            "Y88b.     888     888     888 888        888        
      "888   888       888     888  Y88b             "888     888     888     888 888        888        
Y88b  d88P   888       888     888   Y88b      Y88b  d88P     888     Y88b. .d88P 888        888        
 "Y8888P"  8888888     888     888    Y88b      "Y8888P"      888      "Y88888P"  888        888 
'''

def ResampleImage(image, target, transform):
    '''
    Resample image using SimpleITK
    
    Parameters
    ----------
    image : SimpleITK.Image
        Image to be resampled
    target : SimpleITK.Image
        Target image
    transform : SimpleITK transform
        Transform to be applied to the image.
        
    Returns
    -------
    SimpleITK image
        Resampled image.
    '''
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(target)
    resample.SetTransform(transform)
    resample.SetInterpolator(sitk.sitkLinear)
    orig_size = np.array(image.GetSize(), dtype=np.int)
    ratio = np.array(image.GetSpacing())/np.array(target.GetSpacing())
    new_size = orig_size*(ratio)+0.5
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetDefaultPixelValue(0)

    # Set New Origin
    orig_origin = np.array(image.GetOrigin())
    # apply transform to the origin
    orig_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    new_center = np.array(target.TransformContinuousIndexToPhysicalPoint(np.array(target.GetSize())/2.0))
    new_origin = orig_origin - orig_center + new_center
    resample.SetOutputOrigin(new_origin)
    
    return resample.Execute(image)

'''
88888888888 8888888b.         d8888 888b    888  .d8888b.  8888888888 .d88888b.  8888888b.  888b     d888  .d8888b.  
    888     888   Y88b       d88888 8888b   888 d88P  Y88b 888       d88P" "Y88b 888   Y88b 8888b   d8888 d88P  Y88b 
    888     888    888      d88P888 88888b  888 Y88b.      888       888     888 888    888 88888b.d88888 Y88b.      
    888     888   d88P     d88P 888 888Y88b 888  "Y888b.   8888888   888     888 888   d88P 888Y88888P888  "Y888b.   
    888     8888888P"     d88P  888 888 Y88b888     "Y88b. 888       888     888 8888888P"  888 Y888P 888     "Y88b. 
    888     888 T88b     d88P   888 888  Y88888       "888 888       888     888 888 T88b   888  Y8P  888       "888 
    888     888  T88b   d8888888888 888   Y8888 Y88b  d88P 888       Y88b. .d88P 888  T88b  888   "   888 Y88b  d88P 
    888     888   T88b d88P     888 888    Y888  "Y8888P"  888        "Y88888P"  888   T88b 888       888  "Y8888P" 
'''

def ApplyTranslation(source,transform):
    '''
    Apply translation to source dictionary of landmarks

    Parameters
    ----------
    source : Dictionary
        Dictionary containing the source landmarks.
    transform : numpy array
        Translation to be applied to the source.
    
    Returns
    -------
    Dictionary
        Dictionary containing the translated source landmarks.
    '''
    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = sourcee[key] + transform
    return sourcee

def ApplyTransform(source,transform):
    '''
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
    '''
    # Translation = transform[:3,3]
    # Rotation = transform[:3,:3]
    # for key in source.keys():
    #     source[key] = Rotation @ source[key] + Translation
    # return source

    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = transform @ np.append(sourcee[key],1)
        sourcee[key] = sourcee[key][:3]
    return sourcee

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
    import math
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def AngleAndAxisVectors(v1, v2):
    '''
    Return the angle and the axis of rotation between two vectors
    
    Parameters
    ----------
    v1 : numpy array
        First vector
    v2 : numpy array
        Second vector
    
    Returns
    -------
    angle : float
        Angle between the two vectors
    axis : numpy array
        Axis of rotation between the two vectors
    '''
    # Compute angle between two vectors
    v1_u = v1 / np.amax(v1)
    v2_u = v2 / np.amax(v2)
    angle = np.arccos(np.dot(v1_u, v2_u) / (np.linalg.norm(v1_u) * np.linalg.norm(v2_u)))
    axis = cross(v1_u, v2_u)
    #axis = axis / np.linalg.norm(axis)
    return angle,axis

"""
8888888888 8888888 888      8888888888  .d8888b.  
888          888   888      888        d88P  Y88b 
888          888   888      888        Y88b.      
8888888      888   888      8888888     "Y888b.   
888          888   888      888            "Y88b. 
888          888   888      888              "888 
888          888   888      888        Y88b  d88P 
888        8888888 88888888 8888888888  "Y8888P"
"""

def ExtractFilesFromFolder(folder_path, scan_extension, lm_extension=None, gold=False):
    """Create list of files that are in folder with adequate extension"""

    scan_files = []
    json_files = []
    normpath = os.path.normpath("/".join([folder_path, '**', '']))
    for file in sorted(glob.iglob(normpath, recursive=True)):
        if lm_extension is not None:
            if os.path.isfile(file) and True in [ext in file for ext in lm_extension]:
                json_files.append(file)
        if os.path.isfile(file) and True in [ext in file for ext in scan_extension]:
            scan_files.append(file)
    
    if gold:
        return scan_files[0], json_files[0]
    else:
        return sorted(scan_files), sorted(json_files)

def GetPatients(scan_files,json_files):
    """To associate scan and json files to every patient in input folder of SEMI ASO"""

    patients = {}

    for i in range(len(scan_files)):
        patient = os.path.basename(scan_files[i]).split('_Or')[0].split('_OR')[0].split('_scan')[0].split("_Scanreg")[0].split('.')[0].split('Scan')[0]
        
        if patient not in patients.keys():
            patients[patient] = {"scan":scan_files[i],"json":""}
        else:
            patients[patient]["scan"] = scan_files[i]

        patientjson = os.path.basename(json_files[i]).split('_Or')[0].split('_OR')[0].split('_lm')[0].split("_Scanreg")[0].split('.')[0].split('_scan')[0].split('Scan')[0]
        if patientjson not in patients.keys():
            patients[patientjson] = {"scan":"","json":json_files[i]}
        else:
            patients[patientjson]["json"] = json_files[i]

    return patients