#!/usr/bin/env python-real

import subprocess
import argparse
import os
import re

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def run_script(script_name, args):
    command = ['python', script_name] + args
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"Running {script_name} with arguments {args}")
    print("Output:\n", result.stdout)
    if result.stderr:
        print("Errors:\n", result.stderr)

def run_script_inverse_mri(mri_folder, folder_general):
    folder_mri_inverse = os.path.join(folder_general,"a01_MRI_inv")
    create_folder(folder_mri_inverse)
    script_name = 'mri_inverse.py'
    args = [
        f"--path_folder={mri_folder}",
        f"--folder_output={folder_mri_inverse}",
        f"--suffix=inv",
    ]
    run_script(script_name, args)
    return folder_mri_inverse

def run_script_normalize_percentile(file_type,input_folder, folder_general, upper_percentile, lower_percentile, max_norm, min_norm):
    script_name = 'normalize_percentile.py'
    if file_type=="MRI":
        output_folder_norm = os.path.join(folder_general,"a2_MRI_inv_norm")
    else :
        output_folder_norm = os.path.join(folder_general,"b2_CBCT_norm")
    create_folder(output_folder_norm)
    args = [
        f"--input_folder={input_folder}",
        f"--output_folder={output_folder_norm}",
        f"--upper_percentile={upper_percentile}",
        f"--lower_percentile={lower_percentile}",
        f"--max_norm={max_norm}",
        f"--min_norm={min_norm}"
    ]
    run_script(script_name, args)
    output_path_norm = os.path.join(output_folder_norm,f"test_percentile=[{lower_percentile},{upper_percentile}]_norm=[{min_norm},{max_norm}]")
    return output_path_norm


def run_script_apply_mask(cbct_folder, cbct_label2,folder_general, suffix,upper_percentile, lower_percentile, max_norm, min_norm):
    script_name = 'apply_mask_folder.py'
    cbct_mask_folder = os.path.join(folder_general,"b3_CBCT_norm_mask:l2",f"test_percentile=[{lower_percentile},{upper_percentile}]_norm=[{min_norm},{max_norm}]")
    create_folder(cbct_mask_folder)
    args = [
        f"--folder_path={cbct_folder}",
        f"--seg_folder={cbct_label2}",
        f"--folder_output={cbct_mask_folder}",
        f"--suffix={suffix}",
        f"--seg_label={1}"
    ]
    run_script(script_name, args)
    return cbct_mask_folder

def run_script_AREG_MRI_folder(cbct_folder, cbct_mask_folder,mri_folder,mri_original_folder,folder_general,mri_lower_p,mri_upper_p,mri_min_norm,mri_max_norm,cbct_lower_p,cbct_upper_p,cbct_min_norm,cbct_max_norm):
    script_name = 'AREG_MRI_folder.py'
    output_folder = os.path.join(folder_general,"z01_output",f"a01_mri:inv+norm[{mri_min_norm},{mri_max_norm}]+p[{mri_lower_p},{mri_upper_p}]_cbct:norm[{cbct_min_norm},{cbct_max_norm}]+p[{cbct_lower_p},{cbct_upper_p}]+mask")
    create_folder(output_folder)
    args = [
        f"--cbct_folder={cbct_folder}",
        f"--cbct_original_folder=None",
        f"--cbct_mask_folder={cbct_mask_folder}",
        f"--cbct_seg_folder=None",
        f"--mri_folder={mri_folder}",
        f"--mri_original_folder={mri_original_folder}",
        f"--mri_mask_folder=None",
        f"--mri_seg_folder=None",
        f"--output_folder={output_folder}"
    ]
    run_script(script_name, args)
    return cbct_mask_folder

def extract_values(input_string):
    # Utiliser une expression régulière pour extraire tous les chiffres
    numbers = re.findall(r'\d+', input_string)
    
    # Convertir les nombres extraits en entiers
    numbers = list(map(int, numbers))
    
    # Vérifier qu'il y a exactement 8 nombres
    if len(numbers) != 8:
        raise ValueError("L'entrée doit contenir exactement 8 chiffres.")
    
    # Assigner les nombres à des variables séparées
    a, b, c, d, e, f, g, h = numbers
    
    return a, b, c, d, e, f, g, h

def main():
    parser = argparse.ArgumentParser(description="Run multiple Python scripts with arguments")
    parser.add_argument('folder_general', type=str, help="Folder general where to make all the output")
    parser.add_argument('mri_folder', type=str, help="Folder containing original MRI images.")
    parser.add_argument('cbct_folder', type=str, help="Folder containing original CBCT images.")
    parser.add_argument('cbct_label2', type=str, help="Folder containing CBCT masks.")
    parser.add_argument('normalization', type=str, help="Folder containing CBCT masks.")
    args = parser.parse_args()
    print("normalization : ",args.normalization)
    print("type normalization : ",type(args.normalization))
    
    mri_min_norm, mri_max_norm, mri_lower_p, mri_upper_p, cbct_min_norm, cbct_max_norm, cbct_lower_p, cbct_upper_p = extract_values(args.normalization)
    print(f"mri_lower_p: {mri_lower_p}")
    print(f"mri_upper_p: {mri_upper_p}")
    print(f"mri_min_norm: {mri_min_norm}")
    print(f"mri_max_norm: {mri_max_norm}")
    
    print(f"cbct_lower_p: {cbct_lower_p}")
    print(f"cbct_upper_p: {cbct_upper_p}")
    print(f"cbct_min_norm: {cbct_min_norm}")
    print(f"cbct_max_norm: {cbct_max_norm}")
    
    # Appel des fonctions une par une
    # mri_lower_p = 10
    # mri_upper_p = 95
    # mri_min_norm = 0
    # mri_max_norm = 100

    # cbct_lower_p = 10
    # cbct_upper_p = 95
    # cbct_min_norm = 0
    # cbct_max_norm = 150
    # # MRI
    # folder_mri_inverse = run_script_inverse_mri(args.mri_folder, args.folder_general)
    # input_path_norm_mri = run_script_normalize_percentile("MRI",folder_mri_inverse, args.folder_general, upper_percentile=mri_upper_p, lower_percentile=mri_lower_p, max_norm=mri_max_norm, min_norm=mri_min_norm)

    # # CBCT
    # output_path_norm_cbct = run_script_normalize_percentile("CBCT",args.cbct_folder, args.folder_general, upper_percentile=cbct_upper_p, lower_percentile=cbct_lower_p, max_norm=cbct_max_norm, min_norm=cbct_min_norm)
    # input_path_cbct_norm_mask = run_script_apply_mask(output_path_norm_cbct,args.cbct_label2,args.folder_general,"mask",upper_percentile=cbct_upper_p, lower_percentile=cbct_lower_p, max_norm=cbct_max_norm, min_norm=cbct_min_norm)
    
    # # REG
    # run_script_AREG_MRI_folder(cbct_folder=args.cbct_folder,cbct_mask_folder=input_path_cbct_norm_mask,mri_folder=input_path_norm_mri,mri_original_folder=args.mri_folder,folder_general=args.folder_general,mri_lower_p=mri_lower_p,mri_upper_p=mri_upper_p,mri_min_norm=mri_min_norm,mri_max_norm=mri_max_norm,cbct_lower_p=cbct_lower_p,cbct_upper_p=cbct_upper_p,cbct_min_norm=cbct_min_norm,cbct_max_norm=cbct_max_norm)
    
    

if __name__ == "__main__":
    main()
