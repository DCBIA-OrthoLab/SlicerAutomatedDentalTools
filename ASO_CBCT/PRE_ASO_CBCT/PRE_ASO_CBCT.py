#!/usr/bin/env python-real

import argparse
import SimpleITK as sitk
import sys,os,time
import numpy as np

fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)

from ASO_CBCT_utils import (ExtractFilesFromFolder, AngleAndAxisVectors, RotationMatrix, PreASOResample, convertdicom2nifti)

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

    input_dir, out_dir, smallFOV, isDCMInput = os.path.normpath(args.input[0]), os.path.normpath(args.output_folder[0]), args.SmallFOV[0] == 'true', args.DCMInput[0] == 'true'
    
    if isDCMInput:
        convertdicom2nifti(input_dir)
   
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
    parser.add_argument('DCMInput',nargs=1)

    args = parser.parse_args()
    
    main(args)