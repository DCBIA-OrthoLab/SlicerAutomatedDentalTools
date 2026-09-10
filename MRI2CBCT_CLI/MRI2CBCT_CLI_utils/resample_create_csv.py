import SimpleITK as sitk
import os
import pandas as pd
import argparse
import sys
import logging

# ===== Logging Configuration =====
logger = logging.getLogger("MRI2CBCT_CLI_utils_resample_csv")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_nifti_info(file_path,output_resample):
    """
    Retrieves information about a nifti file and prepares the output path for resampling.

    Arguments:
    file_path (str): Path to the input nifti file.
    output_resample (str): Path to the folder to save resampled nifti files.
    """
    
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
    """
    Creates a CSV file with information about nifti files in the input folder, resampling them if needed.

    Arguments:
    input (str): Path to the input folder containing nifti files.
    output_resample (str): Path to the folder to save resampled nifti files.
    output_csv (str): Path to the folder to save the output CSV file.
    name_csv (str): Name of the output CSV file.
    """

    if not os.path.exists(output_resample):
        os.makedirs(output_resample)
        
    if not os.path.exists(output_csv):
        os.makedirs(output_csv)
        
    input_folder = input
    # Same formats as the rest of the pipeline (PRE_ASO_CBCT, ALI_CBCT, AMASSS...).
    # Restricting this to .nii dropped .nrrd cohorts here without a word, and every
    # later step then reported "0 file" on an input folder that was not empty.
    # SimpleITK, used by get_nifti_info below, reads all of them.
    scan_extensions = (".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz")
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(scan_extensions):
                nifti_files.append(os.path.join(root, file))

    # Get nifti info for every nifti file
    nifti_info = []
    for file in nifti_files:
        info = get_nifti_info(file,output_resample)
        nifti_info.append(info)

    # Create only one DataFrame with all informations
    df = pd.DataFrame(nifti_info)
    outpath = os.path.join(output_csv,name_csv)
    df.to_csv(outpath, index=False)

    return outpath


