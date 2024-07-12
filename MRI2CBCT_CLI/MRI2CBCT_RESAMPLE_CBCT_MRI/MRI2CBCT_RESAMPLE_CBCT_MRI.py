#!/usr/bin/env python-real

import subprocess
import csv
import argparse
import os
# from MRI2CBCT_CLI import MRI2CBCT_CLI_utils
# from MRI2CBCT_CLI_utils import (
#     create_csv,
#     resample_images,
    
# )
import sys
fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from MRI2CBCT_CLI_utils import create_csv, resample_images
import csv

# THE PACING IS CHOOSEN RUNING CALCUL_SPACING_MEAN.PY DONC LA MOYENNE DES SPACING QUE ON A

def run_resample(args):
    # Remplacez ceci par le chemin vers votre fichier CSV
    csv_file_path = args.csv
    # Ouvrir le fichier CSV en lecture
    #RESAMPLE
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Boucle sur chaque ligne du fichier CSV
        for row in csv_reader:
            # Afficher les informations de chaque ligne
            size = tuple(map(int, row["size"].strip("()").split(",")))
            input_path = row["in"]
            out_path = row["out"]
            print(size[1])
            print(f'Image d\'entrée: {row["in"]}, Image de sortie: {row["out"]}, Taille: {size}')
            # command = [f"python3 {args.python_file} --img \"{input_path}\" --out \"{out_path}\" --size 768 576 768 --spacing 0.3 0.3 0.3 --center False --linear False --fit_spacing True --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
            # command = [f"python3 {args.python_file} --img \"{input_path}\" --out \"{out_path}\" --size 443 443 119 --spacing 0.3 0.3 0.3 --center False --linear False --fit_spacing True --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
            command = [f"python3 {args.python_file} --img \"{input_path}\" --out \"{out_path}\" --size 443 443 443  --fit_spacing True --center 0 --iso_spacing 1 --linear False --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
            subprocess.run(command,shell=True)
            
def create_args(img=None, dir=None, csv=None, csv_column='image', csv_root_path=None, csv_use_spc=0,
                csv_column_spcx=None, csv_column_spcy=None, csv_column_spcz=None, ref=None, size=None,
                img_spacing=None, spacing=None, origin=None, linear=False, center=0, fit_spacing=False,
                iso_spacing=False, image_dimension=2, pixel_dimension=1, rgb=False, ow=1, out="./out.nrrd",
                out_ext=None):
    parser = argparse.ArgumentParser(description='Resample an image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    in_group = parser.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--img', type=str, help='image to resample')
    in_group.add_argument('--dir', type=str, help='Directory with image to resample')
    in_group.add_argument('--csv', type=str, help='CSV file with column img with paths to images to resample')

    csv_group = parser.add_argument_group('CSV extra parameters')
    csv_group.add_argument('--csv_column', type=str, default='image', help='CSV column name (Only used if flag csv is used)')
    csv_group.add_argument('--csv_root_path', type=str, default=None, help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')
    csv_group.add_argument('--csv_use_spc', type=int, default=0, help='Use the spacing information in the csv instead of the image')
    csv_group.add_argument('--csv_column_spcx', type=str, default=None, help='Column name in csv')
    csv_group.add_argument('--csv_column_spcy', type=str, default=None, help='Column name in csv')
    csv_group.add_argument('--csv_column_spcz', type=str, default=None, help='Column name in csv')

    transform_group = parser.add_argument_group('Transform parameters')
    transform_group.add_argument('--ref', type=str, help='Reference image. Use an image as reference for the resampling', default=None)
    transform_group.add_argument('--size', nargs="+", type=int, help='Output size, -1 to leave unchanged', default=None)
    transform_group.add_argument('--img_spacing', nargs="+", type=float, default=None, help='Use this spacing information instead of the one in the image')
    transform_group.add_argument('--spacing', nargs="+", type=float, default=None, help='Output spacing')
    transform_group.add_argument('--origin', nargs="+", type=float, default=None, help='Output origin')
    transform_group.add_argument('--linear', type=bool, help='Use linear interpolation.', default=False)
    transform_group.add_argument('--center', type=int, help='Center the image in the space', default=0)
    transform_group.add_argument('--fit_spacing', type=bool, help='Fit spacing to output', default=False)
    transform_group.add_argument('--iso_spacing', type=bool, help='Same spacing for resampled output', default=False)

    img_group = parser.add_argument_group('Image parameters')
    img_group.add_argument('--image_dimension', type=int, help='Image dimension', default=2)
    img_group.add_argument('--pixel_dimension', type=int, help='Pixel dimension', default=1)
    img_group.add_argument('--rgb', type=bool, help='Use RGB type pixel', default=False)

    out_group = parser.add_argument_group('Output parameters')
    out_group.add_argument('--ow', type=int, help='Overwrite', default=1)
    out_group.add_argument('--out', type=str, help='Output image/directory', default="./out.nrrd")
    out_group.add_argument('--out_ext', type=str, help='Output extension type', default=None)

    # Manually set the args
    print(transform_size(size))
    args = parser.parse_args(args=[
        '--img', img,
        '--size', "119 443 443",
        '--linear', str(linear),
        '--center', str(center),
        '--fit_spacing', str(fit_spacing),
        '--iso_spacing', str(iso_spacing),
        '--image_dimension', str(image_dimension),
        '--pixel_dimension', str(pixel_dimension),
        '--rgb', str(rgb),
        '--ow', str(ow),
        '--out', out,
    ])
    return args

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
            
def main(args):
    csv_path = create_csv(args.input_folder,args.output_folder,output_csv=args.output_folder,name_csv="resample_csv.csv")
    
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Boucle sur chaque ligne du fichier CSV
        for row in csv_reader:
            # Afficher les informations de chaque ligne
            size = tuple(map(int, row["size"].strip("()").split(",")))
            input_path = row["in"]
            out_path = row["out"]
            print(size[1])
            print(f'Image d\'entrée: {row["in"]}, Image de sortie: {row["out"]}, Taille: {size}')
            # command = [f"python3 {args.python_file} --img \"{input_path}\" --out \"{out_path}\" --size 768 576 768 --spacing 0.3 0.3 0.3 --center False --linear False --fit_spacing True --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
            # command = [f"python3 {args.python_file} --img \"{input_path}\" --out \"{out_path}\" --size 443 443 119 --spacing 0.3 0.3 0.3 --center False --linear False --fit_spacing True --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
            # command = [f"python3 {args.python_file} --img \"{input_path}\" --out \"{out_path}\" --size 443 443 443  --fit_spacing True --center 0 --iso_spacing 1 --linear False --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
            args_resample = create_args(img=input_path,out=out_path,size=args.resample_size,fit_spacing=True,center=0,iso_spacing=1,linear=False,image_dimension=3,pixel_dimension=1,rgb=False,ow=0)
            print("args resample : ",args_resample)
            break
            # subprocess.run(command,shell=True)
            resample_images(args_resample)
            
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
    # SIZE AND SPACING TO RESAMPLE ARE HARD WRITTEN IN THE LINE 24
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('input_folder', type=str, help='Input path')
    parser.add_argument('output_folder', type=str, help='Output path')
    parser.add_argument('resample_size', type=str, help='size_resample')
    # /home/luciacev/Documents/Gaelle/MultimodelRegistration/resample/resample.py
    args = parser.parse_args()


    main(args)