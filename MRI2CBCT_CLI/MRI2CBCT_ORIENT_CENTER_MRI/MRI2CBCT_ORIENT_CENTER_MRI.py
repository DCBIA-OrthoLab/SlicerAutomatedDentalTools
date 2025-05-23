#!/usr/bin/env python-real

import os
import sys
import time
import SimpleITK as sitk
import argparse

def extract_id(filename):
    """
    Extracts and returns the ID from a filename, removing common NIfTI extensions.
    
    Parameters:
        filename (str): The filename from which to extract the ID.
    
    Returns:
        str: The extracted ID without the extension.
    """
    # Remove the extension using os.path.splitext
    type_file = 0
    base = os.path.splitext(filename)[0]
    # If the file has a double extension (commonly .nii.gz), remove the second extension
    if base.endswith('.nii'):
        base = os.path.splitext(base)[0]
        type_file=1
    
    
    return base,type_file

def calculate_new_origin(image):
    """
    Calculate the new origin to center the image in the Slicer viewport across all axes.
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    # Calculate the center offset for each axis
    new_origin = [(size[i] * spacing[i]) / 2 for i in range(len(size))]
    new_origin = [new_origin[2],-new_origin[0],new_origin[1]] # FOR MRI
    # new_origin = [-new_origin[0]*1.5,new_origin[1],-new_origin[2]*0.5] # FOR CBCT
    # new_origin = [-new_origin[0]*1,new_origin[1],-new_origin[2]*1] # SAVE INSIDE BUT NOT CENTER
    return tuple(new_origin)

def modify_image_properties(nifti_file_path, new_direction, output_file_path=None, acquisition_z_spacing=3.0):
    """
    Read a NIfTI file, change its Direction and optionally center and save the modified image.
    """
    image = sitk.ReadImage(nifti_file_path)
    # Set the new direction
    image.SetDirection(new_direction)
    spacing = list(image.GetSpacing())
    
    # Update only Z spacing (index 2)
    if acquisition_z_spacing != "None":
        spacing[2] = float(acquisition_z_spacing)
        image.SetSpacing(tuple(spacing))

    # Calculate and set the new origin
    new_origin = calculate_new_origin(image)
    image.SetOrigin(new_origin)

    if output_file_path:
        sitk.WriteImage(image, output_file_path)
        print(f"Modified image saved to {output_file_path}")

    return image

def main(args):
    new_direction = tuple(map(float, args.direction.split(',')))  # Assumes direction as comma-separated values
    input_folder = args.input_folder
    output_folder = args.output_folder if args.output_folder else input_folder  # Default to input folder if no output folder is provided.

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))

    # Process each file
    total_patients = len(nifti_files)
    patient_count = 0
    for file_path in nifti_files:
        filename = os.path.basename(file_path)
        file_id,type_file = extract_id(filename)
        if type_file==0:
            output_file_path = os.path.join(output_folder, f"{file_id}_OR.nii")
        else :
            output_file_path = os.path.join(output_folder, f"{file_id}_OR.nii.gz")
        modify_image_properties(file_path, new_direction, output_file_path, args.acquisition_z_spacing)
        
        if total_patients > 0:
            patient_count += 1
            progress = patient_count / total_patients
            print(f"<filter-progress>{progress}</filter-progress>")
            sys.stdout.flush()
            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify NIfTI file directions and center them.")
    parser.add_argument('input_folder', default = '.', help='Path to the input folder containing NIfTI files.')
    parser.add_argument('direction', default = "-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0",  help='New direction for the NIfTI files, specified as a comma-separated string of floats. ')
    parser.add_argument('output_folder', default = '.', help='Path to the output folder where modified NIfTI files will be saved.')
    parser.add_argument('acquisition_z_spacing', default = "3.0", help='New Z spacing for the NIfTI files.')
    args = parser.parse_args()
    main(args)

# USE THIS DIRECTION FOR MRI : "0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0"
# FOR CBCT : "1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0"



#  "-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0"