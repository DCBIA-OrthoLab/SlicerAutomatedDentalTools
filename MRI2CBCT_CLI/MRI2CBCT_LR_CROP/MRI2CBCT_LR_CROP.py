#!/usr/bin/env python-real

import argparse
import os
import time
import glob

import sys

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.progress_protocol import emit_fraction

logger = get_logger("MRI2CBCT_CLI_LR_Crop")

# realpath, not __file__: this CLI sits in a sub-folder, so it is registered
# through a flat folder of symlinks into the source tree. __file__ then names
# the link, whose parent holds no MRI2CBCT_CLI_utils. Resolving first lands in
# MRI2CBCT_CLI either way; a built install has the package on sys.path already.
fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

sys.path.append(fpath)

from MRI2CBCT_CLI_utils import crop_mri, crop_cbct

            
def main(input_folder,output_folder, is_cbct=False):
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all .nii and .nii.gz files
    files = glob.glob(os.path.join(input_folder, "*.nii")) + glob.glob(os.path.join(input_folder, "*.nii.gz"))
    total_patients = len(files)
    patient_count = 0

    logger.info(f"[INFO] Found {total_patients} file(s) in {input_folder}.")

    for img_path in files:
        try:
            if is_cbct:
                crop_cbct(img_path, output_folder)
            else:
                crop_mri(img_path, output_folder)

            patient_count += 1
            progress = patient_count / total_patients
            emit_fraction(progress)
            sys.stdout.flush()
            time.sleep(0.2)

        except Exception as e:
            logger.error(f"[ERROR] Failed to process {img_path}: {e}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('input_folder_CBCT', type=str, help='Input path')
    parser.add_argument('input_folder_MRI', type=str, help='Input path')
    parser.add_argument('input_folder_Seg', type=str, help='Input path')
    parser.add_argument('output_folder', type=str, help='Output path')
    args = parser.parse_args()

    if os.path.isdir(args.input_folder_CBCT):
        cbct_output_folder = os.path.join(args.output_folder, "CBCT")
        main(args.input_folder_CBCT, cbct_output_folder, is_cbct=True)
        
    if os.path.isdir(args.input_folder_MRI):
        mri_output_folder = os.path.join(args.output_folder, "MRI")
        main(args.input_folder_MRI, mri_output_folder, is_cbct=False)
        
    if os.path.isdir(args.input_folder_Seg):
        seg_output_folder = os.path.join(args.output_folder, "Seg")
        main(args.input_folder_Seg, seg_output_folder, is_cbct=True)