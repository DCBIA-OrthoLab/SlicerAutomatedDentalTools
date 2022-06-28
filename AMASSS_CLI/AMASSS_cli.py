#!/use/bin/env python-real

#region Imports
print("Importing librairies...")

from numpy import spacing
# from slicer.util import pip_install

#region try import
try :
    from monai.networks.nets import UNETR
except ImportError:
    pip_install('monai==0.7.0')
    from monai.networks.nets import UNETR

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
    import torch
except ImportError:
    pip_install('torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113')
    import torch

try:
    import cc3d
except ImportError:
    pip_install('connected-components-3d==3.9.1')
    import cc3d

#endregion

import time
import os
import shutil
import random
import string
import glob

import argparse

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

#endregion

#region Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LABELS = {
    "MAND" : 1,
    "CB" : 2,
    "UAW" : 3,
    "MAX" : 4,
    "CV" : 5,
    "SKIN" : 6,
    
    "RC" : 2,
    "TEETH" : 5,
}

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

    # print("Saving prediction to : ", outpath)

    # print(data)

    ref_img = sitk.ReadImage(ref_filepath) 



    output = sitk.GetImageFromArray(img)
    output.SetSpacing(output_spacing)
    output.SetDirection(ref_img.GetDirection())
    output.SetOrigin(ref_img.GetOrigin())

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

    for i in range(np.max(img_arr)):
        label = i+1
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
        outpath = out_folder + "/VTK files/" + os.path.basename(file_path).split('.')[0] + f"_label-{label}.vtk"
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

def SaveSeg(file_path, spacing ,seg_arr, clean_seg , input_path,temp_path, outputdir,temp_folder):

    SavePrediction(seg_arr,input_path,temp_path,output_spacing = spacing)
    if clean_seg:
        CleanScan(temp_path)
    SetSpacingFromRef(
        temp_path,
        input_path,
        # "Linear",
        outpath=file_path
        )

    if args.gen_vtk:
        SavePredToVTK(file_path,temp_folder, args.vtk_smooth, out_folder=outputdir)

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
    
#endregion

#region Main
def main(args):

    # region Read data
    cropSize = args.crop_size

    temp_fold = os.path.join(args.temp_fold, "temp")
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)


    # Find available models in folder
    available_models = {}
    print("Loading models from", args.dir_models)
    normpath = os.path.normpath("/".join([args.dir_models, '**', '']))
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
    if args.high_def:
        # model_size = "SMALL"
        MODELS_DICT = MODELS_GROUP["SMALL"]
        spacing = [0.16,0.16,0.32]

    else:
        # model_size = "LARGE"
        MODELS_DICT = MODELS_GROUP["LARGE"]
        spacing = [0.4,0.4,0.4]


    for model_id in MODELS_DICT.keys():
        if model_id in available_models.keys():
            for struct in args.skul_structure:
                if struct in MODELS_DICT[model_id].keys():
                    if model_id not in models_to_use.keys():
                        models_to_use[model_id] = available_models[model_id]


            # if True in [ for struct in args.skul_structure]:



    print(models_to_use)

    # load data
    data_list = []

    if args.file:
        print("Loading scan :", args.file)
        img_fn = args.file
        basename = os.path.basename(img_fn)
        new_path = os.path.join(temp_fold,basename)
        temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
        CorrectHisto(img_fn, new_path,0.01, 0.99)
        # new_path = img_fn
        data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})

    else:
        scan_dir = args.dir
        print("Loading data from",scan_dir )
        normpath = os.path.normpath("/".join([scan_dir, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            #  print(img_fn)
            basename = os.path.basename(img_fn)

            if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                if not True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                    new_path = os.path.join(temp_fold,basename)
                    temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
                    CorrectHisto(img_fn, new_path,0.01, 0.99)
                    data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})



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
        num_workers=args.nbr_CPU_worker, 
        pin_memory=True
    )
    # endregion

    startTime = time.time()
    seg_not_to_clean = ["CV","RC","SKIN"]

    with torch.no_grad():
        for step, batch in enumerate(pred_loader):

            #region PREDICTION

            input_img, input_path,temp_path = (batch["scan"].to(DEVICE), batch["name"],batch["temp_path"])

            image = input_path[0]
            print("Working on :",image)
            baseName = os.path.basename(image)
            scan_name= baseName.split(".")
            # print(baseName)
            pred_id = "_XXXX-Seg_Pred"

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

            if args.save_in_folder:
                outputdir = os.path.dirname(input_path[0]) + "/" + scan_name[0] + "_" + "SegOut"
                print("Output dir :",outputdir)

                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
            else :
                outputdir = os.path.dirname(input_path[0])

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

                val_outputs = sliding_window_inference(input_img, cropSize, args.nbr_GPU_worker, net,overlap=args.precision)

                pred_data = torch.argmax(val_outputs, dim=1).detach().cpu().type(torch.int16)

                segmentations = pred_data.permute(0,3,2,1)

                seg = segmentations.squeeze(0)

                seg_arr = seg.numpy()[:]



                for struct, label in MODELS_DICT[model_id].items():
                
                    sep_arr = seg_arr.where(seg_arr == label, 1,0)

                    if (struct == "SKIN"):
                        sep_arr = CropSkin(sep_arr,5)

                    prediction_segmentation[struct] = sep_arr

            #endregion

            #region Save predictions
            if "SEPARATE" in args.merge:
                for struct,segmentation in prediction_segmentation.items():
                    file_path = os.path.join(outputdir,pred_name.replace('XXXX',struct))
                    SaveSeg(
                        file_path = file_path,
                        spacing = spacing,
                        seg_arr=segmentation,
                        clean_seg= not True in [struct == id for id in seg_not_to_clean],
                        input_path=input_path[0],
                        outputdir=outputdir,
                        temp_path=temp_path[0],
                        temp_folder=temp_fold
                    )

            if "MERGE" in args.merge:
                print("Merging")
                file_path = os.path.join(outputdir,pred_name.replace('XXXX',"MERGED"))
                merged_seg = np.zeros(seg_arr.shape)
                for struct in args.merging_order:
                    if struct in prediction_segmentation.keys():
                        merged_seg = np.where(prediction_segmentation[struct] == 1, LABELS[struct], merged_seg)
                SaveSeg(
                    file_path = file_path,
                    spacing = spacing,
                    seg_arr=merged_seg,
                    clean_seg= not True in [struct == id for id in seg_not_to_clean],
                    input_path=input_path[0],
                    outputdir=outputdir,
                    temp_path=temp_path[0],
                    temp_folder=temp_fold
                )
            #endregion
                            
    try:
        shutil.rmtree(temp_fold)
    except OSError as e:
        print("Error: %s : %s" % (temp_fold, e.strerror))

    print("Done in %.2f seconds" % (time.time() - startTime))


#endregion

#region argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group('directory')

    input_group.add_argument('-id','--dir', type=str, help='Path to the scans folder', default='/app/data/scans')
    input_group.add_argument('-if','--file', type=str, help='Path to the scan', default=None)
    input_group.add_argument('-dm', '--dir_models', type=str, help='Folder with the models', required=True)

    input_group.add_argument('--temp_fold', type=str, help='temporary folder', default='..')

    input_group.add_argument('-ss', '--skul_structure', nargs="+", type=str, help='Skul structure to segment', default=["MAND","SKIN"])
    input_group.add_argument('-hd', '--high_def', type=bool, help='Use high definition models', default=False)

    input_group.add_argument('-m', '--merge',  nargs="+", type=str, help='merge the segmentations', default=["MERGE","SEPARATE"])

    input_group.add_argument('-sf', '--save_in_folder', type=bool, help='Save the output in one folder', default=True)

    input_group.add_argument('-vtk', '--gen_vtk', type=bool, help='Genrate vtk file', default=True)
    input_group.add_argument('--vtk_smooth', type=int, help='Smoothness of the vtk', default=5)


    input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size', default=[128,128,128])
    input_group.add_argument('-pr', '--precision', type=float, help='precision of the prediction', default=0.5)

    input_group.add_argument('-mo','--merging_order',nargs="+", type=str, help='order of the merging', default=["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC"])

    input_group.add_argument('-ncw', '--nbr_CPU_worker', type=int, help='Number of worker', default=5)
    input_group.add_argument('-ngw', '--nbr_GPU_worker', type=int, help='Number of worker', default=5)

    args = parser.parse_args()
    main(args)


#endregion

