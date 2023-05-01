#!/usr/bin/env python-real

"""
AUTOMATIC LANDMARK IDENTIFICATION IN CBCT SCANS (ALI_CBCT)

Authors :
- Maxime Gillot (UoM)
- Baptiste Baquero (UoM)

"""






#### ##     ## ########   #######  ########  ######## 
 ##  ###   ### ##     ## ##     ## ##     ##    ##    
 ##  #### #### ##     ## ##     ## ##     ##    ##    
 ##  ## ### ## ########  ##     ## ########     ##    
 ##  ##     ## ##        ##     ## ##   ##      ##    
 ##  ##     ## ##        ##     ## ##    ##     ##    
#### ##     ## ##         #######  ##     ##    ##    


#region IMPORTS

import glob
import sys
import os
import time
import json
import shutil
from collections import deque


from slicer.util import pip_install, pip_uninstall


import SimpleITK as sitk
import numpy as np

try:
    import itk
except ImportError:
    pip_install('itk -q')
    import itk

try:
    import torch
except ImportError:
    pip_install('torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -q')
    import torch

try:
    import dicom2nifti
except ImportError:
    pip_install('dicom2nifti -q')
    import dicom2nifti

from torch import nn
import torch.nn.functional as F

pip_uninstall('monai -q')
pip_install('monai==0.7.0 -q')

from monai.data import (
    DataLoader,
    Dataset,
    SmartCacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    AddChannel,
    Compose,
    CropForegroundd,
    LoadImage,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandShiftIntensityd,
    ScaleIntensityd,
    ScaleIntensity,
    Spacingd,
    Spacing,
    Rotate90d,
    RandRotate90d,
    ToTensord,
    ToTensor,
    SaveImaged,
    SaveImage,
    RandCropByLabelClassesd,
    Lambdad,
    CastToTyped,
    SpatialCrop,
    BorderPadd,
    RandAdjustContrastd,
    HistogramNormalized,
    NormalizeIntensityd,
    BorderPad,
)

from monai.networks.nets.densenet import (
    DenseNet
)


#endregion

##     ##    ###    ########  ####    ###    ########  ##       ########  ######  
##     ##   ## ##   ##     ##  ##    ## ##   ##     ## ##       ##       ##    ## 
##     ##  ##   ##  ##     ##  ##   ##   ##  ##     ## ##       ##       ##       
##     ## ##     ## ########   ##  ##     ## ########  ##       ######    ######  
 ##   ##  ######### ##   ##    ##  ######### ##     ## ##       ##             ## 
  ## ##   ##     ## ##    ##   ##  ##     ## ##     ## ##       ##       ##    ## 
   ###    ##     ## ##     ## #### ##     ## ########  ######## ########  ######  


#region GLOBAL VARIABLES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUP_LABELS = {
    'CB' : ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4'],

    'U' : ['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R'],

    'L' : ['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R'],

    'CI' : ['UR3OIP','UL3OIP','UR3RIP','UL3RIP']
}





LABEL_GROUPES = {}
LABELS = []
for group,labels in GROUP_LABELS.items():
    for label in labels:
        LABEL_GROUPES[label] = group
        LABELS.append(label)
        # LABELS_TO_TRAIN.append(label)

# print(len(LABELS))

SCALE_KEYS = ['1','0-3']


MOVEMENT_MATRIX_6 = np.array([
    [1,0,0],  # MoveUp
    [-1,0,0], # MoveDown
    [0,1,0],  # MoveBack
    [0,-1,0], # MoveFront
    [0,0,1],  # MoveLeft
    [0,0,-1], # MoveRight
])
MOVEMENT_ID_6 = [
    "Up",
    "Down",
    "Back",
    "Front",
    "Left",
    "Right"
]

MOVEMENTS = {
    "id" : MOVEMENT_ID_6,
    "mat" : MOVEMENT_MATRIX_6

}


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def GetTargetOutputFromAction(mov_mat,action):
    target = np.zeros((1,len(mov_mat)))[0]
    target[action] = 1
    return target


#endregion


######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######  
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ## 
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##       
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######  
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ## 
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ## 
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######  

#region FUNCTIONS

def CorrectHisto(filepath,outpath,min_porcent=0.01,max_porcent = 0.95,i_min=-1500, i_max=4000):

    print("Correcting scan contrast :", filepath)
    input_img = sitk.ReadImage(filepath) 
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(input_img)


    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min
    # print(img_min,img_max,img_range)

    definition = 1000
    histo = np.histogram(img,definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i> max_porcent, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    res_low = list(map(lambda i: i> min_porcent, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min

    res_min = max(res_min,i_min)
    res_max = min(res_max,i_max)


    # print(res_min,res_min)

    img = np.where(img > res_max, res_max,img)
    img = np.where(img < res_min, res_min,img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output

def ResampleImage(input,size,spacing,origin,direction,interpolator,VectorImageType):
        ResampleType = itk.ResampleImageFilter[VectorImageType, VectorImageType]

        resampleImageFilter = ResampleType.New()
        resampleImageFilter.SetOutputSpacing(spacing.tolist())
        resampleImageFilter.SetOutputOrigin(origin)
        resampleImageFilter.SetOutputDirection(direction)
        resampleImageFilter.SetInterpolator(interpolator)
        resampleImageFilter.SetSize(size)
        resampleImageFilter.SetInput(input)
        resampleImageFilter.Update()

        resampled_img = resampleImageFilter.GetOutput()
        return resampled_img


def SetSpacing(filepath,output_spacing=[0.5, 0.5, 0.5],outpath=-1):
    """
    Set the spacing of the image at the wanted scale 

    Parameters
    ----------
    filePath
     path of the image file 
    output_spacing
     whanted spacing of the new image file (default : [0.5, 0.5, 0.5])
    outpath
     path to save the new image
    """

    print("Resample :", filepath, ", with spacing :", output_spacing)
    img = itk.imread(filepath)
    # arr_img = itk.GetArrayFromImage(img)
    # print(np.min(arr_img),np.max(arr_img))
    # arr_img = np.where(arr_img < 2500, arr_img,2500)
    # print(np.min(arr_img),np.max(arr_img))

    # img_rescale = itk.GetImageFromArray(arr_img)

    spacing = np.array(img.GetSpacing())
    output_spacing = np.array(output_spacing)

    if not np.array_equal(spacing,output_spacing):

        size = itk.size(img)
        scale = spacing/output_spacing

        output_size = (np.array(size)*scale).astype(int).tolist()
        output_origin = img.GetOrigin()

        #Find new origin
        output_physical_size = np.array(output_size)*np.array(output_spacing)
        input_physical_size = np.array(size)*spacing
        output_origin = np.array(output_origin) - (output_physical_size - input_physical_size)/2.0

        img_info = itk.template(img)[1]
        pixel_type = img_info[0]
        pixel_dimension = img_info[1]

        VectorImageType = itk.Image[pixel_type, pixel_dimension]

        if True in [seg in os.path.basename(filepath) for seg in ["seg","Seg"]]:
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        else:
            InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Scan with spacing :", output_spacing)

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,output_size,output_spacing,output_origin,img.GetDirection(),interpolator,VectorImageType)

        if outpath != -1:
            itk.imwrite(resampled_img, outpath)
        return resampled_img

    else:
        # print("Already at the wanted spacing")
        if outpath != -1:
            itk.imwrite(img, outpath)
        return img


def GenEnvironmentLst(patient_dic ,env_type, padding = 1, device = DEVICE):
    environement_lst = []
    for patient,data in patient_dic.items():
        print(f"{bcolors.OKCYAN}Generating Environement for the patient: {bcolors.OKBLUE}{patient}{bcolors.ENDC}")
        env = env_type(
            patient_id = patient,
            device = device,
            padding = padding,
            verbose = False,
        )
        env.LoadImages(data["scans"])
        environement_lst.append(env)
    return environement_lst



def GenControlePoint(groupe_data):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark,data in groupe_data.items():
        id+=1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [data["x"], data["y"], data["z"]],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "defined"
        }
        lm_lst.append(controle_point)

    return lm_lst

def WriteJson(lm_lst,out_path):
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
            "controlPoints": lm_lst,
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

def OUT_WARNING():
    print(f"{bcolors.WARNING}WARNING : Agent trying to go in a none existing space {bcolors.ENDC}")


def GetAgentLst(agents_param):
    print("-- Generating agents --")

    agent_lst = []
    for label in args["landmarks"]:
        print(f"{bcolors.OKCYAN}Generating Agent for the lamdmark: {bcolors.OKBLUE}{label}{bcolors.ENDC}")
        agt = agents_param["type"](
            targeted_landmark=label,
            movements = agents_param["movements"],
            scale_keys = agents_param["scale_keys"],
            FOV=agents_param["FOV"],
            start_pos_radius = agents_param["spawn_rad"],
            speed_per_scale = agents_param["speed_per_scale"],
            verbose = agents_param["verbose"]
        )
        agent_lst.append(agt)

    print(f"{bcolors.OKGREEN}{len(agent_lst)} agent successfully generated. {bcolors.ENDC}")

    return agent_lst

def GetBrain(dir_path):
    brainDic = {}
    normpath = os.path.normpath("/".join([dir_path, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and ".pth" in img_fn:
            lab = os.path.basename(os.path.dirname(os.path.dirname(img_fn)))
            num = os.path.basename(os.path.dirname(img_fn))
            if lab in brainDic.keys():
                brainDic[lab][num] = img_fn
            else:
                network = {num : img_fn}
                brainDic[lab] = network

    return brainDic


#endregion






#region CLASSES

######## ##    ## ##     ## #### ########   #######  ##    ## ##     ## ######## ##    ## ######## 
##       ###   ## ##     ##  ##  ##     ## ##     ## ###   ## ###   ### ##       ###   ##    ##    
##       ####  ## ##     ##  ##  ##     ## ##     ## ####  ## #### #### ##       ####  ##    ##    
######   ## ## ## ##     ##  ##  ########  ##     ## ## ## ## ## ### ## ######   ## ## ##    ##    
##       ##  ####  ##   ##   ##  ##   ##   ##     ## ##  #### ##     ## ##       ##  ####    ##    
##       ##   ###   ## ##    ##  ##    ##  ##     ## ##   ### ##     ## ##       ##   ###    ##    
######## ##    ##    ###    #### ##     ##  #######  ##    ## ##     ## ######## ##    ##    ##    


class Environement :
    def __init__(
        self,
        patient_id,
        padding,
        device,
        correct_contrast = False,
        verbose = False,

    ) -> None:
        """
        Args:
            images_path : path of the image with all the different scale,
            landmark_fiducial : path of the fiducial list linked with the image,
        """
        self.patient_id = patient_id
        self.padding = padding.astype(np.int16)
        self.device = device
        self.verbose = verbose
        self.transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist())])
        # self.transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist()),ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)])

        self.scale_nbr = 0

        # self.transform = Compose([AddChannel(),BorderPad(spatial_border=self.padding.tolist())])
        self.available_lm = []

        self.data = {}

        self.predicted_landmarks = {}


    def LoadImages(self,images_path):

        scales = []

        for scale_id,path in images_path.items():
            data = {"path":path}
            img = sitk.ReadImage(path)
            img_ar = sitk.GetArrayFromImage(img)
            data["image"] = torch.from_numpy(self.transform(img_ar)).type(torch.int16)

            data["spacing"] = np.array(img.GetSpacing())
            origin = img.GetOrigin()
            data["origin"] = np.array([origin[2],origin[1],origin[0]])
            data["size"] = np.array(np.shape(img_ar))

            data["landmarks"] = {}

            self.data[scale_id] = data
            self.scale_nbr += 1

            

    def LoadJsonLandmarks(self,fiducial_path):
        # print(fiducial_path)
        # test = []

        with open(fiducial_path) as f:
            data = json.load(f)

        markups = data["markups"][0]["controlPoints"]
        for markup in markups:
            if markup["label"] not in LABELS:
                print(fiducial_path)
                print(f"{bcolors.WARNING}WARNING : {markup['label']} is an unusual landmark{bcolors.ENDC}")
            # test.append(markup["label"])
            mark_pos = markup["position"]
            lm_ph_coord = np.array([mark_pos[2],mark_pos[1],mark_pos[0]])
            self.available_lm.append(markup["label"])
            for scale,scale_data in self.data.items():
                lm_coord = ((lm_ph_coord+ abs(scale_data["origin"]))/scale_data["spacing"]).astype(np.int16)
                scale_data["landmarks"][markup["label"]] = lm_coord

        # print(test)


    def SavePredictedLandmarks(self,scale_key,out_path=None):
        img_path = self.data[scale_key]["path"]
        print(f"Saving predicted landmarks for patient{self.patient_id} at scale {scale_key}")

        ref_origin = self.data[scale_key]["origin"]
        ref_spacing = self.data[scale_key]["spacing"]
        physical_origin = abs(ref_origin/ref_spacing)

        # print(ref_origin,ref_spacing,physical_origin)

        landmark_dic = {}
        for landmark,pos in self.predicted_landmarks.items():

            real_label_pos = (pos-physical_origin)*ref_spacing
            real_label_pos = [real_label_pos[2],real_label_pos[1],real_label_pos[0]]
            # print(real_label_pos)
            if LABEL_GROUPES[landmark] in landmark_dic.keys():
                landmark_dic[LABEL_GROUPES[landmark]].append({"label": landmark, "coord":real_label_pos})
            else:landmark_dic[LABEL_GROUPES[landmark]] = [{"label": landmark, "coord":real_label_pos}]


        # print(landmark_dic)

        for group,list in landmark_dic.items():

            id = self.patient_id.split(".")[0]
            json_name = f"{id}_lm_Pred_{group}.mrk.json"

            if out_path is not None:
                file_path = os.path.join(out_path,json_name)
            else:
                file_path = os.path.join(os.path.dirname(img_path),json_name)
            groupe_data = {}
            for lm in list:
                groupe_data[lm["label"]] = {"x":lm["coord"][0],"y":lm["coord"][1],"z":lm["coord"][2]}

            lm_lst = GenControlePoint(groupe_data)
            WriteJson(lm_lst,file_path)

    def ResetLandmarks(self):
        for scale in self.data.keys():
            self.data[scale]["landmarks"] = {}

        self.available_lm = []

    def LandmarkIsPresent(self,landmark):
        return landmark in self.available_lm

    def GetLandmarkPos(self,scale,landmark):
        return self.data[scale]["landmarks"][landmark]

    def GetL2DistFromLandmark(self, scale, position, target):
        label_pos = self.GetLandmarkPos(scale,target)
        return np.linalg.norm(position-label_pos)**2

    def GetZone(self,scale,center,crop_size):
        cropTransform = SpatialCrop(center.tolist() + self.padding,crop_size)
        rescale = ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)
        crop = cropTransform(self.data[scale]["image"])
        # print(tor ch.max(crop))
        crop = rescale(crop).type(torch.float32)
        return crop

    def GetRewardLst(self,scale,position,target,mvt_matrix):
        agent_dist = self.GetL2DistFromLandmark(scale,position,target)
        get_reward = lambda move : agent_dist - self.GetL2DistFromLandmark(scale,position + move,target)
        reward_lst = list(map(get_reward,mvt_matrix))
        return reward_lst
    
    def GetRandomPoses(self,scale,target,radius,pos_nbr):
        if scale == SCALE_KEYS[0]:
            porcentage = 0.2 #porcentage of data around landmark
            centered_pos_nbr = int(porcentage*pos_nbr)
            rand_coord_lst = self.GetRandomPosesInAllScan(scale,pos_nbr-centered_pos_nbr)
            rand_coord_lst += self.GetRandomPosesArounfLabel(scale,target,radius,centered_pos_nbr)
        else:
            # print("RANDOOOOOOM AROUND LABEL")
            rand_coord_lst = self.GetRandomPosesArounfLabel(scale,target,radius,pos_nbr)

        return rand_coord_lst

    def GetRandomPosesInAllScan(self,scale,pos_nbr):
        max_coord = self.data[scale]["size"]
        get_rand_coord = lambda x: np.random.randint(1, max_coord, dtype=np.int16)
        rand_coord_lst = list(map(get_rand_coord,range(pos_nbr)))
        return rand_coord_lst
    
    def GetRandomPosesArounfLabel(self,scale,target,radius,pos_nbr):
        min_coord = [0,0,0]
        max_coord = self.data[scale]["size"]
        landmark_pos = self.GetLandmarkPos(scale,target)

        get_random_coord = lambda x: landmark_pos + np.random.randint([1,1,1], radius*2) - radius

        rand_coords = map(get_random_coord,range(pos_nbr))

        correct_coord = lambda coord: np.array([min(max(coord[0],min_coord[0]),max_coord[0]),min(max(coord[1],min_coord[1]),max_coord[1]),min(max(coord[2],min_coord[2]),max_coord[2])])
        rand_coords = list(map(correct_coord,rand_coords))

        return rand_coords

    def GetSampleFromPoses(self,scale,target,pos_lst,crop_size,mvt_matrix):

        get_sample = lambda coord : {
            "state":self.GetZone(scale,coord,crop_size),
            "target": np.argmax(self.GetRewardLst(scale,coord,target,mvt_matrix))
            }
        sample_lst = list(map(get_sample,pos_lst))

        return sample_lst

    def GetSpacing(self,scale):
        return self.data[scale]["spacing"]

    def GetSize(self,scale):
        return self.data[scale]["size"]

    def AddPredictedLandmark(self,lm_id,lm_pos):
        # print(f"Add landmark {lm_id} at {lm_pos}")
        self.predicted_landmarks[lm_id] = lm_pos

    def __str__(self):
        print(self.patient_id)
        for scale in self.data.keys():
            print(f"{scale}")
            print(self.data[scale]["spacing"])
            print(self.data[scale]["origin"])
            print(self.data[scale]["size"])
            print(self.data[scale]["landmarks"])
        return ""


   ###     ######   ######## ##    ## ######## 
  ## ##   ##    ##  ##       ###   ##    ##    
 ##   ##  ##        ##       ####  ##    ##    
##     ## ##   #### ######   ## ## ##    ##    
######### ##    ##  ##       ##  ####    ##    
##     ## ##    ##  ##       ##   ###    ##    
##     ##  ######   ######## ##    ##    ##    

class Agent :
    def __init__(
        self,
        targeted_landmark,
        movements,
        scale_keys,
        brain = None,
        environement = None,
        FOV = [32,32,32],
        start_pos_radius = 20,
        shortmem_size = 10,
        speed_per_scale = [2,1],
        verbose = False
    ) -> None:
    
        self.target = targeted_landmark
        self.scale_keys = scale_keys
        self.environement = environement
        self.scale_state = 0
        self.start_pos_radius = start_pos_radius
        self.start_position = np.array([0,0,0], dtype=np.int16)
        self.position = np.array([0,0,0], dtype=np.int16)
        self.FOV = np.array(FOV, dtype=np.int16)
        
        self.movement_matrix = movements["mat"]
        self.movement_id = movements["id"]

        self.brain = brain
        self.shortmem_size = shortmem_size

        self.verbose = verbose


        self.search_atempt = 0
        self.speed_per_scale = speed_per_scale
        self.speed = self.speed_per_scale[0]


    def SetEnvironement(self, environement): 
        self.environement = environement
        position_mem = []
        position_shortmem = []
        for i in range(environement.scale_nbr):
            position_mem.append([])
            position_shortmem.append(deque(maxlen=self.shortmem_size))
        self.position_mem = position_mem
        self.position_shortmem = position_shortmem

    def SetBrain(self,brain): self.brain = brain

    def ClearShortMem(self):
        for mem in self.position_shortmem:
            mem.clear()

    def GoToScale(self,scale=0):
        self.position = (self.position*(self.environement.GetSpacing(self.scale_keys[self.scale_state])/self.environement.GetSpacing(self.scale_keys[scale]))).astype(np.int16)
        self.scale_state = scale
        self.search_atempt = 0
        self.speed = self.speed_per_scale[scale]

    def SetPosAtCenter(self):
        self.position = self.environement.GetSize(self.scale_keys[self.scale_state])/2

    def SetRandomPos(self):
        if self.scale_state == 0:
            rand_coord = np.random.randint(1, self.environement.GetSize(self.scale_keys[self.scale_state]), dtype=np.int16)
            self.start_position = rand_coord
            # rand_coord = self.environement.GetLandmarkPos(self.scale_keys[self.scale_state],self.target)
        else:
            rand_coord = np.random.randint([1,1,1], self.start_pos_radius*2) - self.start_pos_radius
            rand_coord = self.start_position + rand_coord
            rand_coord = np.where(rand_coord<0, 0, rand_coord)
            rand_coord = rand_coord.astype(np.int16)

        self.position = rand_coord


    def GetState(self):
        state = self.environement.GetZone(self.scale_keys[self.scale_state] ,self.position,self.FOV)
        return state

    def UpScale(self):
        scale_changed = False
        if self.scale_state < self.environement.scale_nbr-1:
            self.GoToScale(self.scale_state + 1)
            scale_changed = True
            self.start_position = self.position
        # else:
        #     OUT_WARNING()
        return scale_changed

    def PredictAction(self):
        return self.brain.Predict(self.scale_state,self.GetState())
        
    def Move(self, movement_idx):
        new_pos = self.position + self.movement_matrix[movement_idx]*self.speed
        if new_pos.all() > 0 and (new_pos < self.environement.GetSize(self.scale_keys[self.scale_state])).all():
            self.position = new_pos
            # if self.verbose:
            #     print("Moving ", self.movement_id[movement_idx])
        else:
            OUT_WARNING()
            self.ClearShortMem()
            self.SetRandomPos()
            self.search_atempt +=1

    def Train(self, data, dim):
        if self.verbose:
            print(f"{bcolors.OKCYAN}Training agent :{bcolors.OKBLUE}{self.target}{bcolors.ENDC}")
        self.brain.Train(data,dim)

    def Validate(self, data,dim):
        if self.verbose:
            print(f"{bcolors.OKCYAN}Validating agent :{bcolors.OKBLUE}{self.target}{bcolors.ENDC}")
        return self.brain.Validate(data,dim)

    def SavePos(self):
        self.position_mem[self.scale_state].append(self.position)
        self.position_shortmem[self.scale_state].append(self.position)

    def Focus(self,start_pos):
        explore_pos = np.array(
            [
                [1,0,0],
                [-1,0,0],
                [0,1,0],
                [0,-1,0],
                [0,0,1],
                [0,0,-1]
            ],
            dtype=np.int16
        )
        radius = 4
        final_pos = np.array([0,0,0], dtype=np.float64)
        for pos in explore_pos:
            found = False
            self.position_shortmem[self.scale_state].clear()
            self.position = start_pos + radius*pos
            while  not found:
                action = self.PredictAction()
                self.Move(action)
                if self.Visited():
                    found = True
                self.SavePos()
            final_pos += self.position
        return final_pos/len(explore_pos)

    def Search(self):
        # if self.verbose:
        tic = time.time()
        print("Searching landmark :",self.target)
        self.GoToScale()
        self.SetPosAtCenter()
        # self.SetRandomPos()
        self.SavePos()
        found = False
        tot_step = 0
        while not found and time.time()-tic < 15:
            # action = self.environement.GetBestMove(self.scale_state,self.position,self.target)
            tot_step+=1
            action = self.PredictAction()
            self.Move(action)
            if self.Visited():
                found = True
            self.SavePos()
            if found:
                if self.verbose:
                    print("Landmark found at scale :",self.scale_state)
                    print("Agent pos = ", self.position)
                    if self.environement.LandmarkIsPresent(self.target):
                        print("Landmark pos = ", self.environement.GetLandmarkPos(self.scale_keys[self.scale_state],self.target))
                scale_changed = self.UpScale()
                found = not scale_changed
            if self.search_atempt > 2:
                print(self.target, "landmark not found")
                self.search_atempt = 0
                return -1

        if not found: # Took too much time
            print(self.target, "landmark not found")
            self.search_atempt = 0
            return -1
        
        final_pos = self.Focus(self.position)
        print("Result :", final_pos)
        self.environement.AddPredictedLandmark(self.target,final_pos)
        return tot_step

    def Visited(self):
        visited = False
        # print(self.position, self.position_shortmem[self.scale_state],)
        for previous_pos in self.position_shortmem[self.scale_state]:
            if np.array_equal(self.position,previous_pos):
                visited = True
        return visited


########  ########     ###    #### ##    ## 
##     ## ##     ##   ## ##    ##  ###   ## 
##     ## ##     ##  ##   ##   ##  ####  ## 
########  ########  ##     ##  ##  ## ## ## 
##     ## ##   ##   #########  ##  ##  #### 
##     ## ##    ##  ##     ##  ##  ##   ### 
########  ##     ## ##     ## #### ##    ## 


class Brain:
    def __init__(
        self,
        network_type,
        network_scales,
        device,
        in_channels,
        out_channels,
        model_dir = "",
        model_name = "",
        run_dir = "",
        learning_rate = 1e-4,
        batch_size = 10,
        generate_tensorboard = False,
        verbose = False
    ) -> None:
        self.network_type = network_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.verbose = verbose
        self.generate_tensorboard = generate_tensorboard
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        networks = []
        global_epoch = []
        epoch_losses = []
        validation_metrics = []
        models_dirs = []

        writers = []
        optimizers = []
        best_metrics = []
        best_epoch = []

        self.network_scales = network_scales

        for n,scale in enumerate(network_scales):
            net = network_type(
                in_channels = in_channels,
                out_channels = out_channels,
            )
            net.to(self.device)
            networks.append(net)

            # num_param = sum(p.numel() for p in net.parameters())
            # print("Number of parameters :",num_param)
            # summary(net,(1,64,64,64))

            epoch_losses.append([0])
            validation_metrics.append([])
            best_metrics.append(0)
            global_epoch.append(0)
            best_epoch.append(0)

            if not model_dir == "":
                dir_path = os.path.join(model_dir,scale)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                models_dirs.append(dir_path)
            


        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizers = optimizers
        self.writers = writers

        self.networks = networks
        # self.networks = [networks[0]]
        self.epoch_losses = epoch_losses
        self.validation_metrics = validation_metrics
        self.best_metrics = best_metrics
        self.global_epoch = global_epoch
        self.best_epoch = best_epoch

        self.model_dirs = models_dirs
        self.model_name = model_name


    def ResetNet(self,n):
        net = self.network_type(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
        )
        net.to(self.device)
        self.networks[n] = net

        self.epoch_losses[n] = [0]
        self.validation_metrics[n] = []
        self.best_metrics[n] = 0
        self.global_epoch[n] = 0
        self.best_epoch[n] = 0


    def Predict(self,dim,state):
        network = self.networks[dim]
        network.eval()
        with torch.no_grad():
            input = torch.unsqueeze(state,0).type(torch.float32).to(self.device)
            x = network(input)
        return torch.argmax(x)

    def LoadModels(self,model_lst):
        # for scale,network in model_lst.items():
        #     print("Loading model", scale)
        #     net.load_state_dict(torch.load(model_lst[n],map_location=self.device))

        for n,net in enumerate(self.networks):
            print("Loading model", model_lst[self.network_scales[n]])
            net.load_state_dict(torch.load(model_lst[self.network_scales[n]],map_location=self.device))


##    ## ######## ######## ##      ##  #######  ########  ##    ##  ######  
###   ## ##          ##    ##  ##  ## ##     ## ##     ## ##   ##  ##    ## 
####  ## ##          ##    ##  ##  ## ##     ## ##     ## ##  ##   ##       
## ## ## ######      ##    ##  ##  ## ##     ## ########  #####     ######  
##  #### ##          ##    ##  ##  ## ##     ## ##   ##   ##  ##         ## 
##   ### ##          ##    ##  ##  ## ##     ## ##    ##  ##   ##  ##    ## 
##    ## ########    ##     ###  ###   #######  ##     ## ##    ##  ######  


class DNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 6,
    ) -> None:
        super(DNet, self).__init__()

        self.featNet = DenseNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=in_channels,
            growth_rate = 34,
            block_config = (6, 12, 24, 16),
        )

        self.dens = DN(
            in_channels = in_channels,
            out_channels = out_channels
        )

    def forward(self,x):
        x = self.featNet(x)
        x = self.dens(x)
        return x


class DN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels: int = 6,
    ) -> None:
        super(DN, self).__init__()

        self.fc0 = nn.Linear(in_channels,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_channels)

        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self,x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = x #F.softmax(self.fc3(x), dim=1)
        return output


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

def convertdicom2nifti(input_folder,output_folder=None):
    patients_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder,folder)) and folder != 'NIFTI']

    if output_folder is None:
        output_folder = os.path.join(input_folder,'NIFTI')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for patient in patients_folders:
        if not os.path.exists(os.path.join(output_folder,patient+".nii.gz")):    
            print("Converting patient: {}...".format(patient))
            current_directory = os.path.join(input_folder,patient)
            try:
                reader = sitk.ImageSeriesReader()
                sitk.ProcessObject_SetGlobalWarningDisplay(False)
                dicom_names = reader.GetGDCMSeriesFileNames(current_directory)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                sitk.ProcessObject_SetGlobalWarningDisplay(True)
                sitk.WriteImage(image, os.path.join(output_folder,os.path.basename(current_directory)+'.nii.gz'))
            except RuntimeError:
                dicom2nifti.convert_directory(current_directory,output_folder)
                nifti_file = search(output_folder,'nii.gz')['nii.gz'][0]
                os.rename(nifti_file,os.path.join(output_folder,patient+".nii.gz"))

#endregion




##     ##    ###    #### ##    ## 
###   ###   ## ##    ##  ###   ## 
#### ####  ##   ##   ##  ####  ## 
## ### ## ##     ##  ##  ## ## ## 
##     ## #########  ##  ##  #### 
##     ## ##     ##  ##  ##   ### 
##     ## ##     ## #### ##    ## 

#region Main

def main(input):
    print("Reading : ",args["input"])
    print("Selected spacings : ", args["spacing"])


    scale_spacing = args["spacing"]

    # If input in DICOM Format --> CONVERT THEM INTO NIFTI
    if args["isDCMInput"]:
        convertdicom2nifti(args['input'])

    patients = {}
    if os.path.isfile(args["input"]):  
        basename = os.path.basename(args["input"])
        patients[basename] = {"scan": args["input"], "scans":{}}
    
    else:
        normpath = os.path.normpath("/".join([args["input"], '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            #  print(img_fn)
            basename = os.path.basename(img_fn)

            if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:

                if basename not in patients.keys():
                    patients[basename] = {"scan": img_fn, "scans":{}}




    temp_fold = args["temp_fold"]

    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)


    print(f"""<filter-progress>{0}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)
    print(f"""<filter-progress>{2}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)
    print(f"""<filter-progress>{0}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)


    for patient,data in patients.items():

        scan = data["scan"]

        scan_name = patient.split(".")

        tempPath = os.path.join(temp_fold, patient)

        if not os.path.exists(tempPath):
            CorrectHisto(scan, tempPath,0.01, 0.99)


        for sp in scale_spacing:
            new_name = ""
            spac = str(sp).replace(".","-")
            for i,element in enumerate(scan_name):
                if i == 0:
                    new_name = scan_name[0] + "_scan_sp" + spac
                else:
                    new_name += "." + element
            
            outpath = os.path.join(temp_fold,new_name)
            if not os.path.exists(outpath):
                SetSpacing(tempPath,[sp,sp,sp],outpath)
            patients[patient]["scans"][spac] = outpath
        
        print(f"""<filter-progress>{1}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)


    # print("Patients : ",patients)


    # #####################################
    #  Init_param
    # #####################################


    image_dim = len(args["agent_FOV"]) # Dimention of the images 2D or 3D
    agent_FOV = args["agent_FOV"] # FOV of the agent

    # batch_size = args.batch_size # Batch size for the training
    # dataset_size = args.dataset_size # Size of the dataset to generate for the training

    SCALE_KEYS = [str(scale).replace('.','-') for scale in scale_spacing]




    environement_lst = GenEnvironmentLst(patient_dic = patients,env_type = Environement, padding =  np.array(agent_FOV)/2 + 1, device = DEVICE)
    # environement_lst = GetEnvironmentLst(environments_param)

    # environement_lst[0].SavePredictedLandmarks(multi_scale_keys[0])

    # return

    agents_param = {
        "type" : Agent,
        "FOV" : agent_FOV,
        "movements" : MOVEMENTS,
        "scale_keys" : SCALE_KEYS,
        "spawn_rad" : args["spawn_radius"],
        "speed_per_scale" : args["speed_per_scale"],
        "verbose" : False,
    }

    agent_lst = GetAgentLst(agents_param)


    # agent_lst = GetAgentLst(agents_param)
    brain_lst = GetBrain(args["dir_models"])
    # print( brain_lst)
    # environement_lst = [environement_lst[0]]
    # agent_lst = [agent_lst[0]]

    trainsitionLayerSize = 1024


    # for agent in agent_lst:
    #     brain = Brain(
    #         network_type = DNet,
    #         network_scales = SCALE_KEYS,
    #         # model_dir = dir_path,
    #         # model_name = target,
    #         device = DEVICE,
    #         in_channels = trainsitionLayerSize,
    #         out_channels = len(MOVEMENTS["id"]),
    #         batch_size= 1,
    #         generate_tensorboard=False,
    #         verbose=True
    #         )
    #     brain.LoadModels(brain_lst[agent.target])
    #     agent.SetBrain(brain)


    print(f"""<filter-progress>{0}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)
    print(f"""<filter-progress>{2}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)
    print(f"""<filter-progress>{0}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)


    start_time = time.time()


    tot_step = 0
    fails = {}
    outPath = args["output_dir"]
    for environment in environement_lst:
        print(environment.patient_id)
        # print(environment)
        for agent in agent_lst:
            brain = Brain(
                network_type = DNet,
                network_scales = SCALE_KEYS,
                # model_dir = dir_path,
                # model_name = target,
                device = DEVICE,
                in_channels = trainsitionLayerSize,
                out_channels = len(MOVEMENTS["id"]),
                batch_size= 1,
                generate_tensorboard=False,
                verbose=True
                )
            brain.LoadModels(brain_lst[agent.target])
            agent.SetBrain(brain)
            agent.SetEnvironement(environment)
            search_result = agent.Search()
            agent.SetBrain(None)
            if search_result == -1:
                fails[agent.target] = fails.get(agent.target,0) + 1
            else:
                tot_step += search_result
            # PlotAgentPath(agent)
            print(f"""<filter-progress>{1}</filter-progress>""")
            sys.stdout.flush()
            time.sleep(0.5)
            print(f"""<filter-progress>{0}</filter-progress>""")
            sys.stdout.flush()
        
        outputdir = outPath
        if args["save_in_folder"]:
            outputdir = outPath + "/" + environment.patient_id.split(".")[0] + "_landmarks"
            # print("Output dir :",outputdir)
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
        environment.SavePredictedLandmarks(SCALE_KEYS[-1],outputdir)
    
    print("Total steps:",tot_step)    
    end_time = time.time()
    print('prediction time :' , end_time-start_time)
        

    for lm, nbr in fails.items():
        print(f"Fails for {lm} : {nbr}/{len(environement_lst)}")


    try:
        shutil.rmtree(temp_fold)
    except OSError as e:
        print("Error: %s : %s" % (temp_fold, e.strerror))


if __name__ == "__main__":

    print("Starting")
    print(sys.argv)

    args = {
        "input": sys.argv[1],
        "dir_models": sys.argv[2],
        "landmarks": sys.argv[3].split(" "),
        "save_in_folder": sys.argv[4] == "true",
        "output_dir": sys.argv[5],
        "temp_fold" : sys.argv[6],
        "isDCMInput": sys.argv[7] == "true",

        "spacing": [1,0.3],
        "speed_per_scale": [1,1],
        "agent_FOV":[64,64,64],
        "spawn_radius" : 10,

    }


    # args["temp_fold"] = temp_dir

    # args = {
    #     # "input": "/home/luciacev/Desktop/TEST/TEST_ALI/MG_test_scan.nii.gz",
    #     "input" : "/home/luciacev/Desktop/TEST/TEST_ALI",
    #     "dir_models": "/home/luciacev/Desktop/Maxime_Gillot/Data/ALI_CBCT/MODELS",
    #     "landmarks": ["B","Gn","ANS","S"],
    #     "save_in_folder": True,
    #     "output_dir": "/home/luciacev/Desktop/TEST/TEST_ALI/Out",

    #     "spacing": [1,0.3],
    #     "speed_per_scale": [1,1],
    #     "agent_FOV":[64,64,64],
    #     "temp_fold" : "..",
    #     "spawn_radius" : 10,
    # }

    main(args)

#endregion



