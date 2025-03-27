#!/usr/bin/env python-real

import argparse
import os
import re
import shutil
from pathlib import Path

import sys
fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)
from MRI2CBCT_CLI_utils import approximation, get_transformation, crop_volume

def create_folder(folder):
    """
    Creates a folder if it does not already exist.

    Arguments:
    folder (str): Path of the folder to create.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def run_script_first_approximation(cbct_folder, mri_folder, output_folder, temp_folder):
    """
    Approximates CBCT images to MRI images and saves the resulting images.

    Args:
        cbct_folder (str): Path to the folder containing CBCT images
        mri_folder (str): Path to the folder containing MRI images
        output_folder (str): Path to the folder where output images will be saved
        temp_folder (str): Path to the temporary folder for intermediate files

    Returns:
        str: Path to the folder containing the approximated images.
    """
    temp_mri = os.path.join(temp_folder, 'mri/')
    temp_cbct = os.path.join(temp_folder, 'cbct/')
    os.makedirs(temp_mri, exist_ok=True)
    os.makedirs(temp_cbct, exist_ok=True)
    
    first_approximation_folder = os.path.join(output_folder, "first_approximation")
    create_folder(first_approximation_folder)
    
    approximation(cbct_folder, mri_folder, first_approximation_folder, temp_mri, temp_cbct)
    return first_approximation_folder

def run_script_get_transformation(mean_folder, cbct_folder, output_folder):
    """
    Generates the registration of the CBCT images to the mean CBCT and saves the results.

    Args:
        mean_folder (str): Path to the folder containing the mean CBCT image.
        cbct_folder (str): Path to the folder containing the CBCT images.
        output_folder (str): Path to the folder where the registered images will be saved.

    Returns:
        str: Path to the folder containing the registered images.
    """
    
    transformation_folder = os.path.join(output_folder, "mean_registration")
    create_folder(transformation_folder)
    get_transformation(mean_folder, cbct_folder, transformation_folder)
    return transformation_folder

def delete_folder(folder_path):
    """
    Deletes a folder if it exists.

    Arguments:
    folder_path (str): Path of the folder to create.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"The folder '{folder_path}' has been deleted successfully.")
    else:
        print(f"The folder '{folder_path}' does not exist.")
        
def run_script_crop_volumes(ROI_file, transformation_folder, first_approximation_folder, cbct_folder, output_folder):
    """
    Crops the CBCT volumes and MRI volumes based on the ROI and saves the results.

    Args:
        ROI_file (str): Path to the file containing the ROI.
        transformation_folder (str): Path to the folder containing the transformation files.
        first_approximation_folder (str): Path to the folder containing the approximated images.
        mri_folder (str): Path to the folder containing the MRI images.
        output_folder (str): Path to the folder where the cropped images will be saved.

    Returns:
        str: Path to the folder containing the cropped images.
    """
    
    cropped_cbct_folder = os.path.join(output_folder, "cropped_cbct")
    create_folder(cropped_cbct_folder)
    crop_volume(ROI_file, transformation_folder, first_approximation_folder, cbct_folder, cropped_cbct_folder)

def main():
    parser = argparse.ArgumentParser(description="Run multiple Python scripts with arguments")
    parser.add_argument('cbct_folder', type=str, help="Folder containing original CBCT images.")
    parser.add_argument('mri_folder', type=str, help="Folder containing original MRI images.")
    parser.add_argument('output_folder', type=str, help="Folder containing the outputs of the approximation.")
    parser.add_argument('temp_dir', type=str, help="Temporary directory for intermediate files.")
    parser.add_argument('tempo_fold', type=str, help="Indicate to keep the temporary fold or not")
    args = parser.parse_args()
    
    # Approximate MRI to CBCT
    first_approximation_folder = run_script_first_approximation(args.cbct_folder, args.mri_folder, args.output_folder, args.temp_dir)
    
    # if args.tempo_fold=="false":
    #     delete_folder(temp_folder)

if __name__ == "__main__":
    print("Debug: MRI2CBCT_APPROX module is being loaded")

    main()