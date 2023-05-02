#!/usr/bin/env python-real
"""
AMASSS_CLI.py (https://github.com/Maxlo24/Slicer_Automatic_Tools/blob/main/AMASSS_CLI/AMASSS_CLI.py)
Autors : Maxime Gillot (CPE Lyon & UoM), Baptiste Baquero (CPE Lyon & UoM)

Info : This script is used to perform automatic segmentation of a CBCT scan using AMASSS.

This file was developed by Maxime Gillot (CPE Lyon & UoM), Baptiste Baquero (CPE Lyon & UoM)
and was supported by NIDCR R01 024450, AA0F Dewel Memorial Biomedical Research award and by
Research Enhancement Award Activity 141 from the University of the Pacific, Arthur A. Dugoni School of Dentistry.
"""


#region Imports
print("Importing librairies...")

import time
import os
import shutil
import glob
import sys


# try:
#     import argparse
# except ImportError:
#     pip_install('argparse')
#     import argparse


# print(sys.argv)


from slicer.util import pip_install, pip_uninstall

# from slicer.util import pip_uninstall
# # pip_uninstall('torch torchvision torchaudio') 

# pip_uninstall('monai')

# try :
#     import logic
# except ImportError:


try:
    import torch
except ImportError:
    pip_install('torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -q')
    import torch

try:
    import nibabel
except ImportError:
    pip_install('nibabel -q')
    import nibabel

try:
    import einops
except ImportError:
    pip_install('einops -q')
    import einops

try:
    import dicom2nifti
except ImportError:
    pip_install('dicom2nifti -q')
    import dicom2nifti

#region try import
pip_uninstall('monai -q')
pip_install('monai==0.7.0 -q')
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    Dataset,
)

from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    Spacingd,
    ToTensord,

)

from monai.inferers import sliding_window_inference

import SimpleITK as sitk
import vtk
import numpy as np
try :
    import itk 
except ImportError:
    pip_install('itk -q')
    import itk


try:
    import cc3d
except ImportError:
    pip_install('connected-components-3d==3.9.1 -q')
    import cc3d

 #endregion



# endregion

#region Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

TRANSLATE ={
  "Mandible" : "MAND",
  "Maxilla" : "MAX",
  "Cranial-base" : "CB",
  "Cervical-vertebra" : "CV",
  "Root-canal" : "RC",
  "Mandibular-canal" : "MCAN",
  "Upper-airway" : "UAW",
  "Skin" : "SKIN",
  "Teeth" : "TEETH",
  "Cranial Base (Mask)" : "CBMASK",
  "Mandible (Mask)" : "MANDMASK",
  "Maxilla (Mask)" : "MAXMASK",
}
NTRANSLATE = {v: v for k, v in TRANSLATE.items()}

LABELS = {

    "LARGE":{
        "MAND" : 1,
        "CB" : 2,
        "UAW" : 3,
        "MAX" : 4,
        "CV" : 5,
        "SKIN" : 6,
        "CBMASK" : 7,
        "MANDMASK" : 8,
        "MAXMASK" : 9,
    },
    "SMALL":{
        "MAND" : 1,
        "RC" : 2,
        "MAX" : 4,
    }
}


LABEL_COLORS = {
    1: [216, 101, 79],
    2: [128, 174, 128],
    3: [0, 0, 0],
    4: [230, 220, 70],
    5: [111, 184, 210],
    6: [172, 122, 101],
}

NAMES_FROM_LABELS = {"LARGE":{}, "SMALL":{}}
for group,data in LABELS.items():
    for k,v in data.items():
        NAMES_FROM_LABELS[group][v] = NTRANSLATE[k]


MODELS_GROUP = {
    "LARGE": {
        "FF":
        {
            "MAND" : 1,
            "CB" : 2,
            "UAW" : 3,
            "MAX" : 4,
            "CV" : 5,
        },
        "SKIN":
        {
            "SKIN" : 1,
        },
        "CBMASK":{
            "CBMASK" : 1,
        },
        "MANDMASK":
        {
            "MANDMASK" : 1,
        },
        "MAXMASK":
        {
            "MAXMASK" : 1,
        },
    },


    "SMALL": {
        "HD-MAND":
        {
            "MAND" : 1
        },
        "HD-MAX":
        {
            "MAX" : 1
        },
        "RC":        
        {
            "RC" : 1
        },
    },
}

#endregion

#region Functions

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


def Create_UNETR(input_channel, label_nbr,cropSize):

    model = UNETR(
        in_channels=input_channel,
        out_channels=label_nbr,
        img_size=cropSize,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        # feature_size=32,
        # hidden_size=1024,
        # mlp_dim=4096,
        # num_heads=16,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.05,
    )
    return model




def CreatePredTransform(spacing):
    pred_transforms = Compose(
        [
            LoadImaged(keys=["scan"]),
            AddChanneld(keys=["scan"]),
            ScaleIntensityd(
                keys=["scan"],minv = 0.0, maxv = 1.0, factor = None
            ),
            Spacingd(keys=["scan"],pixdim=spacing),
            ToTensord(keys=["scan"]),
        ]
    )
    return pred_transforms


def Create_SwinUNETR(input_channel, label_nbr,cropSize):

    model = SwinUNETR(
        img_size=cropSize,
        in_channels=input_channel,
        out_channels=label_nbr,
        feature_size=48,
        # drop_rate=0.0,
        # attn_drop_rate=0.0,
        # dropout_path_rate=0.0,
        use_checkpoint=True,
    )

    return model

def SavePrediction(img,ref_filepath, outpath, output_spacing):

    # print("Saving prediction for : ", ref_filepath)

    # print(data)

    ref_img = sitk.ReadImage(ref_filepath) 



    output = sitk.GetImageFromArray(img)
    output.SetSpacing(output_spacing)
    output.SetDirection(ref_img.GetDirection())
    output.SetOrigin(ref_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)



def CleanScan(file_path):
    input_img = sitk.ReadImage(file_path) 


    closing_radius = 2
    output = sitk.BinaryDilate(input_img, [closing_radius] * input_img.GetDimension())
    output = sitk.BinaryFillhole(output)
    output = sitk.BinaryErode(output, [closing_radius] * output.GetDimension())

    labels_in = sitk.GetArrayFromImage(input_img)
    out, N = cc3d.largest_k(
        labels_in, k=1, 
        connectivity=26, delta=0,
        return_N=True,
    )
    output = sitk.GetImageFromArray(out)
    # closed = sitk.GetArrayFromImage(output)

    # stats = cc3d.statistics(out)
    # mand_bbox = stats['bounding_boxes'][1]
    # rng_lst = []
    # mid_lst = []
    # for slices in mand_bbox:
    #     rng = slices.stop-slices.start
    #     mid = (2/3)*rng+slices.start
    #     rng_lst.append(rng)
    #     mid_lst.append(mid)

    # merge_slice = int(mid_lst[0])
    # out = np.concatenate((out[:merge_slice,:,:],closed[merge_slice:,:,:]),axis=0)
    # output = sitk.GetImageFromArray(out)

    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_path)
    writer.Execute(output)


def CleanArray(seg_arr,radius):
    input_img = sitk.GetImageFromArray(seg_arr)
    output = sitk.BinaryDilate(input_img, [radius] * input_img.GetDimension())
    output = sitk.BinaryFillhole(output)
    output = sitk.BinaryErode(output, [radius] * output.GetDimension())

    labels_in = sitk.GetArrayFromImage(output)
    out, N = cc3d.largest_k(
        labels_in, k=1, 
        connectivity=26, delta=0,
        return_N=True,
    )

    return out


def SetSpacingFromRef(filepath,refFile,interpolator = "NearestNeighbor",outpath=-1):
    """
    Set the spacing of the image the same as the reference image 

    Parameters
    ----------
    filepath
      image file 
    refFile
     path of the reference image 
    interpolator
     Type of interpolation 'NearestNeighbor' or 'Linear'
    outpath
     path to save the new image
    """

    img = itk.imread(filepath)
    ref = itk.imread(refFile)

    img_sp = np.array(img.GetSpacing()) 
    img_size = np.array(itk.size(img))

    ref_sp = np.array(ref.GetSpacing())
    ref_size = np.array(itk.size(ref))
    ref_origin = ref.GetOrigin()
    ref_direction = ref.GetDirection()

    Dimension = 3
    InputPixelType = itk.D

    InputImageType = itk.Image[InputPixelType, Dimension]

    reader = itk.ImageFileReader[InputImageType].New()
    reader.SetFileName(filepath)
    img = reader.GetOutput()

    # reader2 = itk.ImageFileReader[InputImageType].New()
    # reader2.SetFileName(refFile)
    # ref = reader2.GetOutput()

    if not (np.array_equal(img_sp,ref_sp) and np.array_equal(img_size,ref_size)):
        img_info = itk.template(img)[1]
        Ipixel_type = img_info[0]
        Ipixel_dimension = img_info[1]

        ref_info = itk.template(ref)[1]
        Opixel_type = ref_info[0]
        Opixel_dimension = ref_info[1]

        OVectorImageType = itk.Image[Opixel_type, Opixel_dimension]
        IVectorImageType = itk.Image[Ipixel_type, Ipixel_dimension]

        if interpolator == "NearestNeighbor":
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[InputImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        elif interpolator == "Linear":
            InterpolatorType = itk.LinearInterpolateImageFunction[InputImageType, itk.D]
            # print("Rescale Scan with spacing :", output_spacing)

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,ref_size.tolist(),ref_sp,ref_origin,ref_direction,interpolator,InputImageType,InputImageType)

        output = ItkToSitk(resampled_img)
        output = sitk.Cast(output, sitk.sitkInt16)

        # if img_sp[0] > ref_sp[0]:
        closing_radius = 2
        MedianFilter = sitk.MedianImageFilter()
        MedianFilter.SetRadius(closing_radius)
        output = MedianFilter.Execute(output)


        if outpath != -1:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(outpath)
            writer.Execute(output)
                # itk.imwrite(resampled_img, outpath)
        return output

    else:
        output = ItkToSitk(img)
        output = sitk.Cast(output, sitk.sitkInt16)
        if outpath != -1:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(outpath)
            writer.Execute(output)
        return output


def ResampleImage(input,size,spacing,origin,direction,interpolator,IVectorImageType,OVectorImageType):
        ResampleType = itk.ResampleImageFilter[IVectorImageType, OVectorImageType]

        # print(input)

        resampleImageFilter = ResampleType.New()
        resampleImageFilter.SetInput(input)
        resampleImageFilter.SetOutputSpacing(spacing.tolist())
        resampleImageFilter.SetOutputOrigin(origin)
        resampleImageFilter.SetOutputDirection(direction)
        resampleImageFilter.SetInterpolator(interpolator)
        resampleImageFilter.SetSize(size)
        resampleImageFilter.Update()

        resampled_img = resampleImageFilter.GetOutput()
        return resampled_img


def ItkToSitk(itk_img):
    new_sitk_img = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_img), isVector=itk_img.GetNumberOfComponentsPerPixel()>1)
    new_sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
    new_sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
    new_sitk_img.SetDirection(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten())
    return new_sitk_img


def SavePredToVTK(file_path,temp_folder,smoothing, out_folder, model_size,isSegmentInput=False):
    print("Generating VTK for ", file_path)

    img = sitk.ReadImage(file_path) 
    img_arr = sitk.GetArrayFromImage(img)


    present_labels = []
    for label in range(np.max(img_arr)):
        if label+1 in img_arr:
            present_labels.append(label+1)

    for i in present_labels:
        label = i
        seg = np.where(img_arr == label, 1,0)

        output = sitk.GetImageFromArray(seg)

        output.SetOrigin(img.GetOrigin())
        output.SetSpacing(img.GetSpacing())
        output.SetDirection(img.GetDirection())
        output = sitk.Cast(output, sitk.sitkInt16)

        temp_path = temp_folder +f"/tempVTK_{label}.nrrd"
        # print(temp_path)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(temp_path)
        writer.Execute(output)

        surf = vtk.vtkNrrdReader()
        surf.SetFileName(temp_path)
        surf.Update()
        # print(surf)

        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputConnection(surf.GetOutputPort())
        dmc.GenerateValues(100, 1, 100)

        # LAPLACIAN smooth
        SmoothPolyDataFilter = vtk.vtkSmoothPolyDataFilter()
        SmoothPolyDataFilter.SetInputConnection(dmc.GetOutputPort())
        SmoothPolyDataFilter.SetNumberOfIterations(smoothing)
        SmoothPolyDataFilter.SetFeatureAngle(120.0)
        SmoothPolyDataFilter.SetRelaxationFactor(0.6)
        SmoothPolyDataFilter.Update()

        model = SmoothPolyDataFilter.GetOutput()

        color = vtk.vtkUnsignedCharArray() 
        color.SetName("Colors") 
        color.SetNumberOfComponents(3) 
        color.SetNumberOfTuples( model.GetNumberOfCells() )
            
        for i in range(model.GetNumberOfCells()):
            color_tup=LABEL_COLORS[label]
            color.SetTuple(i, color_tup)

        model.GetCellData().SetScalars(color)


        # model.GetPointData().SetS

        # SINC smooth
        # smoother = vtk.vtkWindowedSincPolyDataFilter()
        # smoother.SetInputConnection(dmc.GetOutputPort())
        # smoother.SetNumberOfIterations(30)
        # smoother.BoundarySmoothingOff()
        # smoother.FeatureEdgeSmoothingOff()
        # smoother.SetFeatureAngle(120.0)
        # smoother.SetPassBand(0.001)
        # smoother.NonManifoldSmoothingOn()
        # smoother.NormalizeCoordinatesOn()
        # smoother.Update()

        # print(SmoothPolyDataFilter.GetOutput())

        # outputFilename = "Test.vtk"
        if not isSegmentInput:
            if len(present_labels)>1:
                outpath = out_folder + "/VTK files/" + os.path.basename(file_path).split('.')[0].split('_MERGED')[0] + f"_{NAMES_FROM_LABELS[model_size][label]}_model.vtk"
            else:
                outpath = out_folder + "/VTK files/" + os.path.basename(file_path).split('.')[0].split('-')[0] + "_model.vtk"
        else:
            if len(present_labels)>1:
                outpath = out_folder + "/"+ os.path.basename(file_path).split("_Seg")[0].split('_MERGED')[0] + "_VTK/" + os.path.basename(file_path).split('.')[0].split('_MERGED')[0].split('_Seg')[0] + f"_{NAMES_FROM_LABELS[model_size][label]}_model.vtk"
            else:
                outpath = out_folder + "/"+ os.path.basename(file_path).split("-Seg")[0] + "_model.vtk"              
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))
        Write(model, outpath)

def Write(vtkdata, output_name):
	outfilename = output_name
	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(vtkdata)
	polydatawriter.Write()


def MergeSeg(seg_path_dic,out_path,seg_order):
    merge_lst = []
    for id in seg_order:
        if id in seg_path_dic.keys():
            merge_lst.append(seg_path_dic[id])

    first_img = sitk.ReadImage(merge_lst[0])
    main_seg = sitk.GetArrayFromImage(first_img)
    for i in range(len(merge_lst)-1):
        label = i+2
        img = sitk.ReadImage(merge_lst[i+1])
        seg = sitk.GetArrayFromImage(img)
        main_seg = np.where(seg==1,label,main_seg)

    output = sitk.GetImageFromArray(main_seg)
    output.SetSpacing(first_img.GetSpacing())
    output.SetDirection(first_img.GetDirection())
    output.SetOrigin(first_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_path)
    writer.Execute(output)
    return output

def SaveSeg(file_path, spacing ,seg_arr, input_path,temp_path, outputdir,temp_folder, save_vtk, smoothing = 5, model_size= "LARGE"):

    print("Saving segmentation for ", file_path)

    SavePrediction(seg_arr,input_path,temp_path,output_spacing = spacing)
    # if clean_seg:
    #     CleanScan(temp_path)
    SetSpacingFromRef(
        temp_path,
        input_path,
        # "Linear",
        outpath=file_path
        )

    if save_vtk:
        SavePredToVTK(file_path,temp_folder, smoothing, out_folder=outputdir,model_size=model_size)

def CropSkin(skin_seg_arr, thickness):


    skin_img = sitk.GetImageFromArray(skin_seg_arr)
    skin_img = sitk.BinaryFillhole(skin_img)

    eroded_img = sitk.BinaryErode(skin_img, [thickness] * skin_img.GetDimension())

    skin_arr = sitk.GetArrayFromImage(skin_img)
    eroded_arr = sitk.GetArrayFromImage(eroded_img)

    croped_skin = np.where(eroded_arr==1, 0, skin_arr)

    out, N = cc3d.largest_k(
        croped_skin, k=1, 
        connectivity=26, delta=0,
        return_N=True,
    )


    return out
    
    

def GenerateMask(skin_seg_arr, radius):

    seg_arr = sitk.GetImageFromArray(skin_seg_arr)

    dilate_arr = sitk.BinaryDilate(seg_arr, [radius] * seg_arr.GetDimension())
    eroded_arr = sitk.BinaryErode(dilate_arr, [radius] * seg_arr.GetDimension())

    out = sitk.GetArrayFromImage(eroded_arr)

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

#region Main
def main(args):
    print("Start")

    isSegmentInput = args["isSegmentInput"]

    temp_fold = args["temp_fold"]
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)

    if not isSegmentInput:

        # region Read data
        cropSize = [128,128,128]
        # cropSize = [96,96,96]




        # Find available models in folder
        available_models = {}
        print("Loading models from", args["dir_models"])
        normpath = os.path.normpath("/".join([args["dir_models"], '**', '']))
        for img_fn in glob.iglob(normpath, recursive=True):
            #  print(img_fn)
            basename = os.path.basename(img_fn)
            if basename.endswith(".pth"):
                model_id = basename.split("_")[1]
                if model_id == "Mask":
                    model_id = basename.split("_")[2] + "MASK"
                available_models[model_id] = img_fn

        print("Available models:", available_models)

        # Choose models to use
        MODELS_DICT = {}
        models_to_use = {}
        # models_ID = []  
        if args["high_def"]:
            model_size = "SMALL"
            MODELS_DICT = MODELS_GROUP["SMALL"]
            spacing = [0.16,0.16,0.32]

        else:
            model_size = "LARGE"
            MODELS_DICT = MODELS_GROUP["LARGE"]
            spacing = [0.4,0.4,0.4]


        for model_id in MODELS_DICT.keys():
            if model_id in available_models.keys():
                for struct in args["skul_structure"]:
                    if struct in MODELS_DICT[model_id].keys():
                        if model_id not in models_to_use.keys():
                            models_to_use[model_id] = available_models[model_id]


                # if True in [ for struct in args.skul_structure]:



        print(models_to_use)

        # If input in DICOM Format --> CONVERT THEM INTO NIFTI
        if args["isDCMInput"]:
            convertdicom2nifti(args['input'])


        # load data
        data_list = []


        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)


        number_of_scans = 0
        if os.path.isfile(args["input"]):  
            print("Loading scan :", args["input"])
            img_fn = args["input"]
            basename = os.path.basename(img_fn)
            new_path = os.path.join(temp_fold,basename)
            temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
            if not os.path.exists(new_path):
                CorrectHisto(img_fn, new_path,0.01, 0.99)
            # new_path = img_fn
            data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})
            number_of_scans += 1

        else:

            scan_dir = args["input"]
            print("Loading data from",scan_dir )
            normpath = os.path.normpath("/".join([scan_dir, '**', '']))
            for img_fn in sorted(glob.iglob(normpath, recursive=True)):
                #  print(img_fn)
                basename = os.path.basename(img_fn)

                if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                    if not True in [txt in basename for txt in ["_Pred","seg","Seg",'Mask','MASK']]:
                        number_of_scans += 1


            counter = 0
            for img_fn in sorted(glob.iglob(normpath, recursive=True)):
                #  print(img_fn)
                basename = os.path.basename(img_fn)

                if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                    if not True in [txt in basename for txt in ["_Pred","seg","Seg",'Mask','MASK']]:
                        new_path = os.path.join(temp_fold,basename)
                        temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
                        if not os.path.exists(new_path):
                            CorrectHisto(img_fn, new_path,0.01, 0.99)
                        data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})
                        counter += 1
                        print(f"""<filter-progress>{1}</filter-progress>""")
                        sys.stdout.flush()
                        time.sleep(0.5)
                        print(f"""<filter-progress>{0}</filter-progress>""")
                        sys.stdout.flush()
                        time.sleep(0.5)



        # print(f"""<filter-progress>{0.99}</filter-progress>""")
        # sys.stdout.flush()
        # time.sleep(0.5)

        #endregion

        # region prepare data

        pred_transform = CreatePredTransform(spacing)

        pred_ds = Dataset(
            data=data_list, 
            transform=pred_transform, 
        )
        pred_loader = DataLoader(
            dataset=pred_ds,
            batch_size=1, 
            shuffle=False, 
            num_workers=args["nbr_CPU_worker"], 
            pin_memory=True
        )
        # endregion


        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)


        startTime = time.time()
        seg_not_to_clean = ["CV","RC"]


        with torch.no_grad():
            for step, batch in enumerate(pred_loader):

                #region PREDICTION

                input_img, input_path,temp_path = (batch["scan"].to(DEVICE), batch["name"],batch["temp_path"])

                image = input_path[0]
                print("Working on :",image)
                baseName = os.path.basename(image)
                scan_name= baseName.split(".")
                # print(baseName)
                pred_id = "_XXXX-Seg_"+ args["prediction_ID"]

                if "_scan" in baseName:
                    pred_name = baseName.replace("_scan",pred_id)
                elif "_Scan" in baseName:
                    pred_name = baseName.replace("_Scan",pred_id)
                else:
                    pred_name = ""
                    for i,element in enumerate(scan_name):
                        if i == 0:
                            pred_name += element + pred_id
                        else:
                            pred_name += "." + element

                outputdir = args["output_dir"]
                if args["save_in_folder"]:
                    outputdir += "/" + scan_name[0] + "_" + "SegOut"
                    print("Output dir :",outputdir)

                    if not os.path.exists(outputdir):
                        os.makedirs(outputdir)
                else:
                    outputdir = os.path.dirname(image)
                    

                prediction_segmentation = {}



                for model_id,model_path in models_to_use.items():

                    net = Create_UNETR(
                        input_channel = 1,
                        label_nbr= len(MODELS_DICT[model_id].keys()) + 1,
                        cropSize=cropSize
                    ).to(DEVICE)
                    
                    # net = Create_SwinUNETR(
                    #     input_channel = 1,
                    #     label_nbr= len(MODELS_DICT[model_id].keys()) + 1,
                    #     cropSize=cropSize
                    # ).to(DEVICE)

                    print("Loading model", model_path)
                    net.load_state_dict(torch.load(model_path,map_location=DEVICE))
                    net.eval()


                    val_outputs = sliding_window_inference(input_img, cropSize, args["nbr_GPU_worker"], net,overlap=args["precision"])

                    pred_data = torch.argmax(val_outputs, dim=1).detach().cpu().type(torch.int16)

                    segmentations = pred_data.permute(0,3,2,1)

                    # print("Segmentations shape :",segmentations.shape)

                    seg = segmentations.squeeze(0)

                    seg_arr = seg.numpy()[:]



                    for struct, label in MODELS_DICT[model_id].items():
                    
                        sep_arr = np.where(seg_arr == label, 1,0)

                        if (struct == "SKIN"):
                            sep_arr = CropSkin(sep_arr,5)
                            # sep_arr = GenerateMask(sep_arr,20)
                        elif not True in [struct == id for id in seg_not_to_clean]:
                            sep_arr = CleanArray(sep_arr,2)

                        prediction_segmentation[struct] = sep_arr

                        print(f"""<filter-progress>{1}</filter-progress>""")
                        sys.stdout.flush()
                        time.sleep(0.5)
                        print(f"""<filter-progress>{0}</filter-progress>""")
                        sys.stdout.flush()
                        time.sleep(0.5)


                #endregion

                # print(f"""<filter-progress>{1}</filter-progress>""")
                # sys.stdout.flush()
                # time.sleep(0.5)
                # print(f"""<filter-progress>{0}</filter-progress>""")
                # sys.stdout.flush()


                #region ===== SAVE RESULT =====

                seg_to_save = {}
                for struct in args["skul_structure"]:
                    seg_to_save[struct] = prediction_segmentation[struct]

                save_vtk = args["gen_vtk"]

                if "SEPARATE" in args["merge"] or len(args["skul_structure"]) == 1:
                    for struct,segmentation in seg_to_save.items():
                        file_path = os.path.join(outputdir,pred_name.replace('XXXX',struct))
                        SaveSeg(
                            file_path = file_path,
                            spacing = spacing,
                            seg_arr=segmentation,
                            input_path=input_path[0],
                            outputdir=outputdir,
                            temp_path=temp_path[0],
                            temp_folder=temp_fold,
                            save_vtk=args["gen_vtk"],
                            smoothing=args["vtk_smooth"],
                            model_size=model_size
                        )
                        save_vtk = False

                if "MERGE" in args["merge"] and len(args["skul_structure"]) > 1:
                    print("Merging")
                    file_path = os.path.join(outputdir,pred_name.replace('XXXX',"MERGED"))
                    merged_seg = np.zeros(seg_arr.shape)
                    for struct in args["merging_order"]:
                        if struct in seg_to_save.keys():
                            merged_seg = np.where(seg_to_save[struct] == 1, LABELS[model_size][struct], merged_seg)
                    SaveSeg(
                        file_path = file_path,
                        spacing = spacing,
                        seg_arr=merged_seg,
                        input_path=input_path[0],
                        outputdir=outputdir,
                        temp_path=temp_path[0],
                        temp_folder=temp_fold,
                        save_vtk=save_vtk,
                        model_size=model_size
                    )
                    

                # print(f"""<filter-progress>{1}</filter-progress>""")
                # sys.stdout.flush()
                # time.sleep(0.5)
                # print(f"""<filter-progress>{0}</filter-progress>""")
                # sys.stdout.flush()

                #endregion

        # print(f"""<filter-progress>{1}</filter-progress>""")
        # sys.stdout.flush()
        # time.sleep(0.5)
        # print(f"""<filter-progress>{0}</filter-progress>""")
        # sys.stdout.flush()
        # time.sleep(0.5)

    if isSegmentInput:   
        
        startTime = time.time()
        
        data = []

        number_of_scans = 0
        if os.path.isfile(args["input"]):  
            print("Loading scan :", args["input"])
            data.append(args["input"])
            number_of_scans += 1
        else:
            scan_dir = args["input"]
            print("Loading data from",scan_dir )
            normpath = os.path.normpath("/".join([scan_dir, '**', '']))
            for img_fn in sorted(glob.iglob(normpath, recursive=True)):
                #  print(img_fn)
                basename = os.path.basename(img_fn)

                if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                    if True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                        data.append(img_fn)
                        number_of_scans += 1
        print("NOMBRE DE SEG:",number_of_scans)

        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        
        for seg in data:
            SavePredToVTK(file_path=seg,temp_folder=temp_fold,smoothing=args["vtk_smooth"],out_folder=args["output_dir"],model_size="LARGE",isSegmentInput=isSegmentInput)
            print(f"""<filter-progress>{1}</filter-progress>""")
            sys.stdout.flush()
            time.sleep(0.5)
            print(f"""<filter-progress>{0}</filter-progress>""")
            sys.stdout.flush()
            time.sleep(0.5)
    try:
        shutil.rmtree(temp_fold)
    except OSError as e:
        print("Error: %s : %s" % (temp_fold, e.strerror))

    print("Done in %.2f seconds" % (time.time() - startTime))


#endregion

#region argparse
if __name__ == "__main__":

    print("Starting")
    print(sys.argv)
    args = {
        "input": sys.argv[1],
        "dir_models": sys.argv[2],
        "high_def": sys.argv[3] == "true",
        "skul_structure": sys.argv[4].split(" "),
        "merge": sys.argv[5].split(" "),
        "gen_vtk": sys.argv[6] == "true",
        "save_in_folder": sys.argv[7] == "true",
        "output_dir": sys.argv[8],
        "precision": int(sys.argv[9]) / 100,
        "vtk_smooth": int(sys.argv[10]),
        "prediction_ID": sys.argv[11],
        "nbr_GPU_worker": int(sys.argv[12]),
        "nbr_CPU_worker": int(sys.argv[13]),
        "temp_fold" : sys.argv[14],
        "isSegmentInput" : sys.argv[15] == "true",
        "isDCMInput": sys.argv[16] == "true",

        "merging_order": ["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC","CBMASK","MANDMASK","MAXMASK"],

    }
    
    # print(args)


    # args["temp_fold"] = temp_dir

    # args = {
    #     "input": '/home/luciacev/Desktop/TEST/TEST_ALI/ALI_CBCT/AnaJ_Scan_T1_OR.gipl.gz',
    #     "dir_models": '/home/luciacev/Desktop/Maxime_Gillot/Data/AMASSS/FULL_FACE_MODELS',
    #     "high_def": False,
    #     "skul_structure": ["SKIN","CV","UAW","CB","MAX","MAND"],
    #     "merge": ["MERGE"],
    #     "gen_vtk": True,
    #     "save_in_folder": True,
    #     "output_dir": '/home/luciacev/Desktop/TEST/TEST_ALI/ALI_CBCT/',
    #     "precision": 0.5,
    #     "vtk_smooth": 5,
    #     "prediction_ID": "Pred",
    #     "nbr_GPU_worker": 5,
    #     "nbr_CPU_worker": 5,
    #     "temp_fold" : "/home/luciacev/Documents/Slicer_temp_AMASSS",

    #     "merging_order": ["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC"],
    # }


    print(args)




    # print(f"""<filter-progress>{0.5}</filter-progress>""")
    # sys.stdout.flush()
    # for i in range(2):
    #     print(f"""<filter-progress>{(50*(i+1))/100}</filter-progress>""")
    #     sys.stdout.flush()
    #     time.sleep(0.1)



    # parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input_group = parser.add_argument_group('directory')

    # input_group.add_argument('-id','--dir', type=str, help='Path to the scans folder', default='/app/data/scans')
    # input_group.add_argument('-if','--file', type=str, help='Path to the scan', default=None)
    # input_group.add_argument('-dm', '--dir_models', type=str, help='Folder with the models', required=True)

    # input_group.add_argument('--temp_fold', type=str, help='temporary folder', default='..')

    # input_group.add_argument('-ss', '--skul_structure', nargs="+", type=str, help='Skul structure to segment', default=["MAND","UAW"])
    # input_group.add_argument('-hd', '--high_def', type=bool, help='Use high definition models', default=False)

    # input_group.add_argument('-m', '--merge',  nargs="+", type=str, help='merge the segmentations', default=["MERGE","SEPARATE"])

    # input_group.add_argument('-sf', '--save_in_folder', type=bool, help='Save the output in one folder', default=True)

    # input_group.add_argument('-vtk', '--gen_vtk', type=bool, help='Genrate vtk file', default=True)
    # input_group.add_argument('--vtk_smooth', type=int, help='Smoothness of the vtk', default=5)


    # input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[128,128,128])
    # input_group.add_argument('-pr', '--precision', type=float, help='precision of the prediction', default=0.5)

    # input_group.add_argument('-mo','--merging_order',nargs="+", type=str, help='order of the merging', default=["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC"])

    # input_group.add_argument('-ncw', '--nbr_CPU_worker', type=int, help='Number of worker', default=5)
    # input_group.add_argument('-ngw', '--nbr_GPU_worker', type=int, help='Number of worker', default=5)

    # args = parser.parse_args()

    # print(args.skul_structure)

    main(args)


#endregion

