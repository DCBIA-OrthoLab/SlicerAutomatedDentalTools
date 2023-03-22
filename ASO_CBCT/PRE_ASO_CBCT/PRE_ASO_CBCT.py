#!/usr/bin/env python-real

import argparse
import SimpleITK as sitk
import sys,os,time
import numpy as np
import slicer

from slicer.util import pip_install

try:
    import torch
except ImportError:
    pip_install('torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
    import torch

fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)

from ASO_CBCT_utils import (ExtractFilesFromFolder, DenseNet, AngleAndAxisVectors, RotationMatrix, PreASOResample)
# from ASO_CBCT_utils.ResamplePreASO import PreASOResample
# from ASO_CBCT_utils.utils import ExtractFilesFromFolder, AngleAndAxisVectors, RotationMatrix
# from ASO_CBCT_utils.Net import DenseNet

def ResampleImage(image, transform):
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
    resample.SetReferenceImage(image)
    resample.SetTransform(transform)
    resample.SetInterpolator(sitk.sitkLinear)
    orig_size = np.array(image.GetSize(), dtype=int)
    ratio = 1
    new_size = orig_size * ratio
    new_size = np.ceil(new_size).astype(int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetDefaultPixelValue(0)

    # Set New Origin
    orig_origin = np.array(image.GetOrigin())
    # apply transform to the origin
    orig_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    # new_center = np.array(target.TransformContinuousIndexToPhysicalPoint(np.array(target.GetSize())/2.0))
    new_origin = orig_origin - orig_center
    resample.SetOutputOrigin(new_origin)

    return resample.Execute(image)
    
def main(args):

    input_dir, out_dir, smallFOV = args.input[0], args.output_folder[0], args.SmallFOV[0] == 'True'

    # RESAMPLE BEFORE USING MODELS
    temp_folder = args.temp_folder[0]

    # Small and Large FOV difference
    if smallFOV:  # Small FOV input
        spacingFOV = 0.69
        ckpt_file = 'SmallFOV.ckpt'
        
    else: # Large FOV input
        spacingFOV = 1.45
        ckpt_file = 'LargeFOV.ckpt'

    PreASOResample(input_dir,temp_folder,spacing=spacingFOV) # /!\ large and small FOV choice for spacing choice /!\

    CosSim = torch.nn.CosineSimilarity() # /!\ if loss < 0.1 dont apply rotation /!\
    Loss = lambda x,y: 1 - CosSim(torch.Tensor(x),torch.Tensor(y))
    
    ckpt_path = os.path.join(args.model_folder[0],ckpt_file) # /!\ large and small FOV choice to include /!\ 

    model = DenseNet.load_from_checkpoint(checkpoint_path = ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
    
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    input_files, _ = ExtractFilesFromFolder(input_dir, scan_extension)

    for i in range(len(input_files)):
        
        input_file = input_files[i]
        
        img = sitk.ReadImage(input_file)
        
        # Translation to center volume
        T = - np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        translation = sitk.TranslationTransform(3)
        translation.SetOffset(T.tolist())
        
        goal = np.array((0.0,0.0,1.0)) # Direction vector for good orientation

        img_temp = sitk.ReadImage(os.path.join(temp_folder,os.path.basename(input_file).split('.')[0]+'.nii.gz'))
        array = sitk.GetArrayFromImage(img_temp)
        scan = torch.Tensor(array).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            directionVector_pred = model(scan.to(device))
        directionVector_pred = directionVector_pred.cpu().numpy()
        
        if Loss(directionVector_pred,goal) > 1 and not smallFOV: # When angle is large enough to apply orientation modification
            #                                    /!\ only to LargeFOV /!\
            angle, axis = AngleAndAxisVectors(goal,directionVector_pred[0])
            Rotmatrix = RotationMatrix(axis,angle)

            rotation = sitk.VersorRigid3DTransform()
            Rotmatrix = np.linalg.inv(Rotmatrix)
            rotation.SetMatrix(Rotmatrix.flatten().tolist())
            
            TransformList = [translation,rotation]
            
            # Compute the final transform (inverse all the transforms)
            TransformSITK = sitk.CompositeTransform(3)
            for i in range(len(TransformList)-1,-1,-1):
                TransformSITK.AddTransform(TransformList[i])
            TransformSITK = TransformSITK.GetInverse()
            
            img_out = ResampleImage(img,TransformSITK)
            
        else: # When angle is too little --> only the center translation is applied

            img_trans = ResampleImage(img,translation.GetInverse())
            img_out = img_trans
        
        # Write Scan
        dir_scan = os.path.dirname(input_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_scan):
            os.makedirs(dir_scan)
        
        file_outpath = os.path.join(dir_scan,os.path.basename(input_file))
        if not os.path.exists(file_outpath):
            sitk.WriteImage(img_out, file_outpath)

        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)

if __name__ == "__main__":
    
    print("PRE ASO")

    parser = argparse.ArgumentParser()

    parser.add_argument('input',nargs=1)
    parser.add_argument('output_folder',nargs=1)
    parser.add_argument('model_folder',nargs=1)
    parser.add_argument('SmallFOV',nargs=1)
    parser.add_argument('temp_folder',nargs=1)

    args = parser.parse_args()
    
    main(args)