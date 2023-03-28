#!/usr/bin/env python-real

import sys
import os
import time
import argparse
import SimpleITK as sitk

fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)

from ASO_CBCT_utils import ICP,ExtractFilesFromFolder,MergeJson,WriteJson,GetPatients

def main(args):
       
    scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
    lm_extension = ['.json']

    list_landmark = args.list_landmark[0].split(' ')
    input_dir, gold_dir, out_dir = args.input[0], args.gold_folder[0], args.output_folder[0]
    
    MergeJson(input_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    input_files, input_json_files = ExtractFilesFromFolder(input_dir, scan_extension, lm_extension)
    
    patients = GetPatients(input_files,input_json_files)

    gold_file, gold_json_file = ExtractFilesFromFolder(gold_dir, scan_extension, lm_extension, gold=True)
    
    for patient,data in patients.items():
        
        input_file, input_json_file = data["scan"],data["json"]

        output, source_transformed = ICP(input_file,input_json_file,gold_file,gold_json_file,list_landmark)
        
        if output is None:
            print("ICP failed for patient: ",patient)
            continue

        # Write JSON
        dir_json = os.path.dirname(input_json_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_json):
            os.makedirs(dir_json)
        json_path = os.path.join(dir_json,patient+'_lm_'+args.add_inname[0]+'.mrk.json')

        if not os.path.exists(json_path):
            WriteJson(source_transformed,json_path)

        # Write Scan
        dir_scan = os.path.dirname(input_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_scan):
            os.makedirs(dir_scan)
        
        file_outpath = os.path.join(dir_scan,patient+'_'+args.add_inname[0]+'.nii.gz')
        if not os.path.exists(file_outpath):
            sitk.WriteImage(output, file_outpath)

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
    
    print("Starting")
    print(sys.argv)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('input',nargs=1)
    parser.add_argument('gold_folder',nargs=1)
    parser.add_argument('output_folder',nargs=1)
    parser.add_argument('add_inname',nargs=1)
    parser.add_argument('list_landmark',nargs=1)
    
    args = parser.parse_args()

    main(args)
