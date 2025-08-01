#!/usr/bin/env python-real

import argparse
import os
import re
import shutil

import sys
fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)
from MRI2CBCT_CLI_utils import invert_mri_intensity, normalize, apply_mask_f, registration

def create_folder(folder):
    """
    Creates a folder if it does not already exist.

    Arguments:
    folder (str): Path of the folder to create.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def run_script_inverse_mri(mri_folder, folder_general):
    """
    Inverts the intensity of MRI images and saves the results.

    Arguments:
    mri_folder (str): Folder containing MRI files.
    folder_general (str): General folder for output.
    """
    
    folder_mri_inverse = os.path.join(folder_general,"a01_MRI_inv")
    create_folder(folder_mri_inverse)
    invert_mri_intensity(mri_folder, folder_mri_inverse, "inv")
    return folder_mri_inverse

def run_script_normalize_percentile(file_type,input_folder, folder_general, upper_percentile, lower_percentile, max_norm, min_norm):
    """
    Normalizes images based on specified percentiles and saves the results.

    Arguments:
    file_type (str): Type of files to normalize ('MRI' or other).
    input_folder (str): Folder containing the input files.
    folder_general (str): General folder for output.
    upper_percentile (float): Upper percentile for normalization.
    lower_percentile (float): Lower percentile for normalization.
    max_norm (float): Maximum value for normalization.
    min_norm (float): Minimum value for normalization.
    """
    
    if file_type=="MRI":
        output_folder_norm_general = os.path.join(folder_general,"a2_MRI_inv_norm")
    else :
        output_folder_norm_general = os.path.join(folder_general,"b2_CBCT_norm")
    create_folder(output_folder_norm_general)
    
    output_folder_norm = os.path.join(output_folder_norm_general,f"percentile=[{lower_percentile},{upper_percentile}]_norm=[{min_norm},{max_norm}]")
    create_folder(output_folder_norm)
    
    normalize(input_folder, output_folder_norm,upper_percentile,lower_percentile,min_norm, max_norm)
    return output_folder_norm


def run_script_apply_mask(cbct_folder, cbct_label2,folder_general, suffix,upper_percentile, lower_percentile, max_norm, min_norm, is_mri=False):
    """
    Applies a mask to CBCT images and saves the normalized results.

    Arguments:
    cbct_folder (str): Folder containing CBCT files.
    cbct_label2 (str): Folder containing the segmentation labels.
    folder_general (str): General folder for output.
    suffix (str): Suffix for the output files.
    upper_percentile (float): Upper percentile for normalization.
    lower_percentile (float): Lower percentile for normalization.
    max_norm (float): Maximum value for normalization.
    min_norm (float): Minimum value for normalization.
    is_mri (bool): Whether the input files are MRI files (default: False).
    """
    print("folder_general : ",folder_general)
    if is_mri:
        mask_folder = os.path.join(folder_general,"a3_MRI_inv_norm_mask",f"percentile=[{lower_percentile},{upper_percentile}]_norm=[{min_norm},{max_norm}]")
    else:
        mask_folder = os.path.join(folder_general,"b3_CBCT_norm_mask_l2",f"percentile=[{lower_percentile},{upper_percentile}]_norm=[{min_norm},{max_norm}]")
    create_folder(mask_folder)
    apply_mask_f(folder_path=cbct_folder, seg_folder=cbct_label2, folder_output=mask_folder, suffix=suffix, seg_label=1)
    return mask_folder

def run_script_AREG_MRI_folder(cbct_folder, cbct_mask_folder,mri_folder,mri_original_folder,folder_general,mri_lower_p,mri_upper_p,mri_min_norm,mri_max_norm,cbct_lower_p,cbct_upper_p,cbct_min_norm,cbct_max_norm):
    """
    Runs the registration script for MRI and CBCT folders, applying normalization and percentile adjustments.

    Arguments:
    cbct_folder (str): Folder containing CBCT files.
    cbct_mask_folder (str): Folder containing CBCT mask files.
    mri_folder (str): Folder containing MRI files.
    mri_original_folder (str): Folder containing original MRI files.
    folder_general (str): General folder for output.
    mri_lower_p (float): Lower percentile for MRI normalization.
    mri_upper_p (float): Upper percentile for MRI normalization.
    mri_min_norm (float): Minimum value for MRI normalization.
    mri_max_norm (float): Maximum value for MRI normalization.
    cbct_lower_p (float): Lower percentile for CBCT normalization.
    cbct_upper_p (float): Upper percentile for CBCT normalization.
    cbct_min_norm (float): Minimum value for CBCT normalization.
    cbct_max_norm (float): Maximum value for CBCT normalization.
    """
    
    output_folder = os.path.join(folder_general,f"mri=inv+norm[{mri_min_norm},{mri_max_norm}]+p[{mri_lower_p},{mri_upper_p}]_cbct=norm[{cbct_min_norm},{cbct_max_norm}]+p[{cbct_lower_p},{cbct_upper_p}]+mask")
    create_folder(output_folder)
    registration(cbct_folder,mri_folder,cbct_mask_folder,output_folder,mri_original_folder)
    return cbct_mask_folder

def extract_values(input_string):
    """
    Extracts 8 integers from the input string and returns them as a tuple.

    Arguments:
    input_string (str): String containing the integers.
    """
    
    numbers = re.findall(r'\d+', input_string)

    numbers = list(map(int, numbers))
    
    if len(numbers) != 8:
        raise ValueError("The input need to contains 8 numbers")
    
    a, b, c, d, e, f, g, h = numbers
    
    return a, b, c, d, e, f, g, h

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"The folder '{folder_path}' has been deleted successfully.")
    else:
        print(f"The folder '{folder_path}' does not exist.")

def main():
    parser = argparse.ArgumentParser(description="Run multiple Python scripts with arguments")
    parser.add_argument('folder_general', type=str, help="Folder general where to make all the output")
    parser.add_argument('mri_folder', type=str, help="Folder containing original MRI images.")
    parser.add_argument('cbct_folder', type=str, help="Folder containing original CBCT images.")
    parser.add_argument('cbct_label2', type=str, help="Folder containing CBCT masks.")
    parser.add_argument('normalization', type=str, help="Folder containing CBCT masks.")
    parser.add_argument('tempo_fold', type=str, help="Indicate to keep the temporary fold or not")
    args = parser.parse_args()
    
    mri_min_norm, mri_max_norm, mri_lower_p, mri_upper_p, cbct_min_norm, cbct_max_norm, cbct_lower_p, cbct_upper_p = extract_values(args.normalization)
    
    total_steps = 6
    current_step = 0
    
    # MRI
    folder_mri_inverse = run_script_inverse_mri(args.mri_folder, args.folder_general)
    current_step += 1
    print(f"<filter-progress>{current_step/total_steps}</filter-progress>")
    sys.stdout.flush()
    
    input_path_norm_mri = run_script_normalize_percentile("MRI",folder_mri_inverse, args.folder_general, upper_percentile=mri_upper_p, lower_percentile=mri_lower_p, max_norm=mri_max_norm, min_norm=mri_min_norm)
    current_step += 1
    print(f"<filter-progress>{current_step/total_steps}</filter-progress>")
    sys.stdout.flush()
    
    input_path_mri_norm_mask = run_script_apply_mask(input_path_norm_mri, args.cbct_label2, args.folder_general, "mask", upper_percentile=mri_upper_p, lower_percentile=mri_lower_p, max_norm=mri_max_norm, min_norm=mri_min_norm, is_mri=True)
    current_step += 1
    print(f"<filter-progress>{current_step/total_steps}</filter-progress>")
    sys.stdout.flush()

    # CBCT
    output_path_norm_cbct = run_script_normalize_percentile("CBCT",args.cbct_folder, args.folder_general, upper_percentile=cbct_upper_p, lower_percentile=cbct_lower_p, max_norm=cbct_max_norm, min_norm=cbct_min_norm)
    current_step += 1
    print(f"<filter-progress>{current_step/total_steps}</filter-progress>")
    sys.stdout.flush()
    
    input_path_cbct_norm_mask = run_script_apply_mask(output_path_norm_cbct,args.cbct_label2,args.folder_general,"mask",upper_percentile=cbct_upper_p, lower_percentile=cbct_lower_p, max_norm=cbct_max_norm, min_norm=cbct_min_norm, is_mri=False)
    current_step += 1
    print(f"<filter-progress>{current_step/total_steps}</filter-progress>")
    sys.stdout.flush()
    
    # REG
    run_script_AREG_MRI_folder(cbct_folder=args.cbct_folder,cbct_mask_folder=input_path_cbct_norm_mask,mri_folder=input_path_mri_norm_mask,mri_original_folder=args.mri_folder,folder_general=args.folder_general,mri_lower_p=mri_lower_p,mri_upper_p=mri_upper_p,mri_min_norm=mri_min_norm,mri_max_norm=mri_max_norm,cbct_lower_p=cbct_lower_p,cbct_upper_p=cbct_upper_p,cbct_min_norm=cbct_min_norm,cbct_max_norm=cbct_max_norm)
    current_step += 1
    print(f"<filter-progress>{current_step/total_steps}</filter-progress>")
    sys.stdout.flush()
    
    
    if args.tempo_fold=="false":
        delete_folder(folder_mri_inverse)
        delete_folder(input_path_norm_mri)
        delete_folder(os.path.dirname(input_path_norm_mri))
        delete_folder(input_path_mri_norm_mask)
        delete_folder(os.path.dirname(input_path_mri_norm_mask))
        delete_folder(output_path_norm_cbct)
        delete_folder(os.path.dirname(output_path_norm_cbct))
        delete_folder(input_path_cbct_norm_mask)
        delete_folder(os.path.dirname(input_path_cbct_norm_mask))
    
    

if __name__ == "__main__":
    main()
