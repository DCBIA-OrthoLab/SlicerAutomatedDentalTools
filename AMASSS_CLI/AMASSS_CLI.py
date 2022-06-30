#!/usr/bin/env python-real


#region Imports
print("Importing librairies...")

import time
import os
import shutil
import random
import glob
import sys


# try:
#     import argparse
# except ImportError:
#     pip_install('argparse')
#     import argparse


# print(sys.argv)


# from slicer.util import pip_install

# # from slicer.util import pip_uninstall
# # pip_uninstall('torch torchvision torchaudio') 

# pip_install('--upgrade pip')


try:
    import torch
except ImportError:
    pip_install('torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
    import torch


#region try import
try :
    from monai.networks.nets import UNETR
except ImportError:
    pip_install('monai==0.7.0')
    pip_install('nibabel')
    pip_install('einops')

    from monai.networks.nets import UNETR

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
)

from monai.inferers import sliding_window_inference



try:
    import nibabel
except ImportError:
    pip_install('nibabel')
    import nibabel


try:
    import SimpleITK as sitk
except ImportError:
    pip_install('SimpleITK==2.1.1')
    import SimpleITK as sitk

try:
    import itk
except ImportError:
    pip_install('itk==5.2.1')
    import itk

try:
    import vtk
except ImportError:
    pip_install('vtk==9.1.0')
    import vtk

try:
    import numpy as np
except ImportError:
    pip_install('numpy==1.22.3')
    import numpy as np



try:
    import cc3d
except ImportError:
    pip_install('connected-components-3d==3.9.1')
    import cc3d

 #endregion



# endregion

#region Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRANSLATE ={
  "Mandible" : "MAND",
  "Maxilla" : "MAX",
  "Cranial base" : "CB",
  "Cervical vertebra" : "CV",
  "Root canal" : "RC",
  "Mandibular canal" : "MCAN",
  "Upper-airway" : "UAW",
  "Skin" : "SKIN",
  "Teeth" : "TEETH"
}

INV_TRANSLATE = {}
for k,v in TRANSLATE.items():
    INV_TRANSLATE[v] = k

LABELS = {
    "MAND" : 1,
    "CB" : 2,
    "UAW" : 3,
    "MAX" : 4,
    "CV" : 5,
    "SKIN" : 6,
    
}

NAMES_FROM_LABELS = {}
for k,v in LABELS.items():
    NAMES_FROM_LABELS[v] = INV_TRANSLATE[k]


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
        }
    },


    "SMALL": {
        "HD_MAND":
        {
            "MAND" : 1
        },
        "HD_MAX":
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


def SavePredToVTK(file_path,temp_folder,smoothing, out_folder):
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
        if len(present_labels) > 1:
            outpath = out_folder + "/VTK files/" + os.path.basename(file_path).split('.')[0] + f"_{NAMES_FROM_LABELS[label]}_model.vtk"
        else:
            outpath = out_folder + "/VTK files/" + os.path.basename(file_path).split('.')[0] + f"_model.vtk"
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))
        Write(SmoothPolyDataFilter.GetOutput(), outpath)

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

def SaveSeg(file_path, spacing ,seg_arr, clean_seg , input_path,temp_path, outputdir,temp_folder, save_vtk, smoothing = 5):

    print("Saving segmentation for ", file_path)

    SavePrediction(seg_arr,input_path,temp_path,output_spacing = spacing)
    if clean_seg:
        CleanScan(temp_path)
    SetSpacingFromRef(
        temp_path,
        input_path,
        # "Linear",
        outpath=file_path
        )

    if save_vtk:
        SavePredToVTK(file_path,temp_folder, smoothing, out_folder=outputdir)

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


#endregion

#region Main
def main(args):
    print("Start")

    # region Read data
    cropSize = [128,128,128]

    temp_fold = os.path.join(args["temp_fold"], "temp")
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)


    # Find available models in folder
    available_models = {}
    print("Loading models from", args["dir_models"])
    normpath = os.path.normpath("/".join([args["dir_models"], '**', '']))
    for img_fn in glob.iglob(normpath, recursive=True):
        #  print(img_fn)
        basename = os.path.basename(img_fn)
        if basename.endswith(".pth"):
            model_id = basename.split("_")[1]
            available_models[model_id] = img_fn

    # Choose models to use
    MODELS_DICT = {}
    models_to_use = {}
    # models_ID = []  
    if args["high_def"]:
        # model_size = "SMALL"
        MODELS_DICT = MODELS_GROUP["SMALL"]
        spacing = [0.16,0.16,0.32]

    else:
        # model_size = "LARGE"
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

    # load data
    data_list = []


    number_of_scans = 0
    if os.path.isfile(args["input"]):  
        print("Loading scan :", args["input"])
        img_fn = args["input"]
        basename = os.path.basename(img_fn)
        new_path = os.path.join(temp_fold,basename)
        temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
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
                if not True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                    number_of_scans += 1

        print(f"""<filter-progress>{200}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()

        counter = 0
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            #  print(img_fn)
            basename = os.path.basename(img_fn)

            if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                if not True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                    new_path = os.path.join(temp_fold,basename)
                    temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
                    CorrectHisto(img_fn, new_path,0.01, 0.99)
                    data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})
                    counter += 1
                    print(f"""<filter-progress>{(counter/number_of_scans)}</filter-progress>""")
                    sys.stdout.flush()



    #endregion

    # region prepare data

    number_of_scans

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

    startTime = time.time()
    seg_not_to_clean = ["CV","RC","SKIN"]

    print(f"""<filter-progress>{300}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)

    print(f"""<filter-progress>{0}</filter-progress>""")
    sys.stdout.flush()

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
                

            prediction_segmentation = {}

            for model_id,model_path in models_to_use.items():

                net = Create_UNETR(
                    input_channel = 1,
                    label_nbr= len(MODELS_DICT[model_id].keys()) + 1,
                    cropSize=cropSize
                ).to(DEVICE)
                

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

                    prediction_segmentation[struct] = sep_arr

            #endregion

            #region ===== SAVE RESULT =====

            seg_to_save = {}
            for struct in args["skul_structure"]:
                seg_to_save[struct] = prediction_segmentation[struct]

            save_vtk = args["gen_vtk"]

            if "SEPARATE" in args["merge"]:
                for struct,segmentation in seg_to_save.items():
                    file_path = os.path.join(outputdir,pred_name.replace('XXXX',struct))
                    SaveSeg(
                        file_path = file_path,
                        spacing = spacing,
                        seg_arr=segmentation,
                        clean_seg= not True in [struct == id for id in seg_not_to_clean],
                        input_path=input_path[0],
                        outputdir=outputdir,
                        temp_path=temp_path[0],
                        temp_folder=temp_fold,
                        save_vtk=args["gen_vtk"],
                        smoothing=args["vtk_smooth"]
                    )
                    save_vtk = False

            if "MERGE" in args["merge"] and len(args["skul_structure"]) > 1:
                print("Merging")
                file_path = os.path.join(outputdir,pred_name.replace('XXXX',"MERGED"))
                merged_seg = np.zeros(seg_arr.shape)
                for struct in args["merging_order"]:
                    if struct in seg_to_save.keys():
                        merged_seg = np.where(seg_to_save[struct] == 1, LABELS[struct], merged_seg)
                SaveSeg(
                    file_path = file_path,
                    spacing = spacing,
                    seg_arr=merged_seg,
                    clean_seg= not True in [struct == id for id in seg_not_to_clean],
                    input_path=input_path[0],
                    outputdir=outputdir,
                    temp_path=temp_path[0],
                    temp_folder=temp_fold,
                    save_vtk=save_vtk,
                )

            print(f"""<filter-progress>{(step+1/number_of_scans)}</filter-progress>""")
            sys.stdout.flush()
            #endregion
                            
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

        "merging_order": ["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC"],

        "temp_fold" : "..",
        "nbr_GPU_worker": 5,
        "nbr_CPU_worker": 5,
    }

    # args = {
    #     "input": '/home/luciacev/Desktop/REQUESTED_SEG/1_T1_scan_or.nii.gz',
    #     "dir_models": '/home/luciacev/Desktop/Maxime_Gillot/Data/AMASSS/FULL_FACE_MODELS',
    #     "high_def": False,
    #     "skul_structure": ["SKIN","CV","UAW","CB","MAX","MAND"],
    #     "merge": ["MERGE"],
    #     "gen_vtk": True,
    #     "save_in_folder": True,
    #     "output_dir": '/home/luciacev/Desktop/REQUESTED_SEG/CranialBaseSegmentation',
    #     "precision": 0.5,
    #     "vtk_smooth": 5,
    #     "prediction_ID": "Pred",

    #     "merging_order": ["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC"],

    #     "temp_fold" : "..",
    #     "nbr_GPU_worker": 5,
    #     "nbr_CPU_worker": 5,
    # }


    # print(args)




    # print(f"""<filter-progress>{300}</filter-progress>""")
    # sys.stdout.flush()
    # time.sleep(0.5)

    # for i in range(20):
    #     print(f"""<filter-progress>{(5*i)/100}</filter-progress>""")
    #     # print(f"""<filter-progress>{-3}</filter-progress>""")

    #     sys.stdout.flush()
    #     time.sleep(0.2)

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

