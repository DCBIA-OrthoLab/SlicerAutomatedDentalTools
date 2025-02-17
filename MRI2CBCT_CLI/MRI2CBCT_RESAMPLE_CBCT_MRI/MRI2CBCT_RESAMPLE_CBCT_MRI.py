#!/usr/bin/env python-real

import csv
import argparse
import os

import sys
fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from MRI2CBCT_CLI_utils import create_csv, resample_images
import csv


def run_resample(img=None, dir=None, csv=None, csv_column='image', csv_root_path=None, csv_use_spc=0,
                     csv_column_spcx=None, csv_column_spcy=None, csv_column_spcz=None, ref=None, size=None,
                     img_spacing=None, spacing=None, origin=None, linear=False, center=0, fit_spacing=False,
                     iso_spacing=False, image_dimension=2, pixel_dimension=1, rgb=False, ow=1, out="./out.nrrd",
                     out_ext=None):
    args = {
        'img': img,
        'dir': dir,
        'csv': csv,
        'csv_column': csv_column,
        'csv_root_path': csv_root_path,
        'csv_use_spc': csv_use_spc,
        'csv_column_spcx': csv_column_spcx,
        'csv_column_spcy': csv_column_spcy,
        'csv_column_spcz': csv_column_spcz,
        'ref': ref,
        'size': size,
        'img_spacing': img_spacing,
        'spacing': spacing,
        'origin': origin,
        'linear': linear,
        'center': center,
        'fit_spacing': fit_spacing,
        'iso_spacing': iso_spacing,
        'image_dimension': image_dimension,
        'pixel_dimension': pixel_dimension,
        'rgb': rgb,
        'ow': ow,
        'out': out,
        'out_ext': out_ext,
    }
    resample_images(args)

def transform_size(size_str):
    """
    Transforms a string '[x,y,z]' into 'x y z' with x, y, z as integers.

    :param size_str: String in the format '[x,y,z]'
    :return: String in the format 'x y z'
    """
    # Remove the brackets and split by comma
    size_list = size_str.strip('[]').split(',')
    
    # Convert each element to int and join with space
    size_transformed = ' '.join(map(str, map(int, size_list)))
    
    return size_transformed
            
def main(input_folder,output_folder,resample_size,spacing,iso_spacing,is_seg=False):
    csv_path = create_csv(input_folder,output_folder,output_csv=output_folder,name_csv="resample_csv.csv")
    
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            size_file = tuple(map(int, row["size"].strip("()").split(",")))
            spacing_file = tuple(map(float, row["Spacing"].strip("()").split(",")))
            input_path = row["in"]
            out_path = row["out"]
            linear = False if is_seg else True
            
            if resample_size != "None" and spacing=="None" :
                run_resample(img=input_path,out=out_path,size=list(map(int, resample_size.split(','))),fit_spacing=True,center=1,iso_spacing=False,linear=linear,image_dimension=3,pixel_dimension=1,rgb=False,ow=0)
            elif resample_size == "None" and spacing!="None" :
                run_resample(img=input_path,out=out_path,spacing=list(map(float, spacing.split(','))),size=[size_file[0],size_file[1],size_file[2]],fit_spacing=False,center=1,iso_spacing=False,linear=linear,image_dimension=3,pixel_dimension=1,rgb=False,ow=0)
            elif resample_size != "None" and spacing!="None" :
                run_resample(img=input_path,out=out_path,spacing=list(map(float, spacing.split(','))),size=list(map(int, resample_size.split(','))),fit_spacing=True,center=1,iso_spacing=False,linear=linear,image_dimension=3,pixel_dimension=1,rgb=False,ow=0)
            
    delete_csv(csv_path)
    
def delete_csv(file_path):
    """Delete a CSV file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted successfully.")
        else:
            print(f"File {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while trying to delete the file {file_path}: {e}")

            


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('input_folder_MRI', type=str, help='Input path')
    parser.add_argument('input_folder_T2_MRI', type=str, help='Input path for T2_MRI')
    parser.add_argument('input_folder_CBCT', type=str, help='Input path')
    parser.add_argument('input_folder_T2_CBCT', type=str, help='Input path for T2_CBCT')
    parser.add_argument('input_folder_Seg', type=str, help='Input path for Seg')
    parser.add_argument('input_folder_T2_Seg', type=str, help='Input path for T2_Seg')
    parser.add_argument('output_folder', type=str, help='Output path')
    parser.add_argument('resample_size', type=str, help='size_resample')
    parser.add_argument('spacing', type=str, help='size_resample')
    args = parser.parse_args()


    if os.path.isdir(args.input_folder_MRI):
        mri_output_folder = os.path.join(args.output_folder, "MRI")
        main(args.input_folder_MRI,mri_output_folder,args.resample_size,args.spacing,iso_spacing=True)
    if os.path.isdir(args.input_folder_T2_MRI):
        main(args.input_folder_T2_MRI,mri_output_folder,args.resample_size,args.spacing,iso_spacing=True)
        
    if os.path.isdir(args.input_folder_CBCT):
        cbct_output_folder = os.path.join(args.output_folder, "CBCT")
        main(args.input_folder_CBCT,cbct_output_folder,args.resample_size,args.spacing,iso_spacing=False)
    if os.path.isdir(args.input_folder_T2_CBCT):
        main(args.input_folder_T2_CBCT,cbct_output_folder,args.resample_size,args.spacing,iso_spacing=False)
        
    if os.path.isdir(args.input_folder_Seg):
        seg_output_folder = os.path.join(args.output_folder, "Seg")
        main(args.input_folder_Seg,seg_output_folder,args.resample_size,args.spacing,iso_spacing=False, is_seg=True)
    if os.path.isdir(args.input_folder_T2_Seg):
        main(args.input_folder_T2_Seg,seg_output_folder,args.resample_size,args.spacing,iso_spacing=False, is_seg=True)