import SimpleITK as sitk
import os
import pandas as pd
import argparse

def get_nifti_info(file_path,output_resample):
    # Read the NIfTI file
    image = sitk.ReadImage(file_path)

    # Get information
    info = {
        "in": file_path, 
        "out" : file_path.replace(os.path.dirname(file_path),output_resample),
        "size": image.GetSize(),
        "Spacing": image.GetSpacing(),
    }

    return info

def create_csv(input:str,output_resample:str,output_csv:str,name_csv:str):
    if not os.path.exists(output_resample):
        os.makedirs(output_resample)
        
    if not os.path.exists(output_csv):
        os.makedirs(output_csv)
        
    input_folder = input
    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))

    # Get nifti info for every nifti file
    nifti_info = []
    for file in nifti_files:
        info = get_nifti_info(file,output_resample)
        nifti_info.append(info)

    # Créez un seul DataFrame avec toutes les informations
    df = pd.DataFrame(nifti_info)
    outpath = os.path.join(output_csv,name_csv)
    df.to_csv(outpath, index=False)

    return outpath


