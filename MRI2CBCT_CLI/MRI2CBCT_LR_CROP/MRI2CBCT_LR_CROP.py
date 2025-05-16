#!/usr/bin/env python-real

import argparse
import os
import time
import sys
import glob
fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from MRI2CBCT_CLI_utils import crop_mri, crop_cbct

            
def main(input_folder,output_folder, is_cbct=False):
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all .nii and .nii.gz files
    files = glob.glob(os.path.join(input_folder, "*.nii")) + glob.glob(os.path.join(input_folder, "*.nii.gz"))
    total_patients = len(files)
    patient_count = 0

    print(f"[INFO] Found {total_patients} file(s) in {input_folder}.")

    for img_path in files:
        try:
            if is_cbct:
                crop_cbct(img_path, output_folder)
            else:
                crop_mri(img_path, output_folder)

            patient_count += 1
            progress = patient_count / total_patients
            print(f"<filter-progress>{progress}</filter-progress>")
            sys.stdout.flush()
            time.sleep(0.2)

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('input_folder_CBCT', type=str, help='Input path')
    parser.add_argument('input_folder_MRI', type=str, help='Input path')
    parser.add_argument('output_folder', type=str, help='Output path')
    args = parser.parse_args()

    if os.path.isdir(args.input_folder_CBCT):
        cbct_output_folder = os.path.join(args.output_folder, "CBCT")
        main(args.input_folder_CBCT, cbct_output_folder, is_cbct=True)
        
    if os.path.isdir(args.input_folder_MRI):
        mri_output_folder = os.path.join(args.output_folder, "MRI")
        main(args.input_folder_MRI, mri_output_folder, is_cbct=False)