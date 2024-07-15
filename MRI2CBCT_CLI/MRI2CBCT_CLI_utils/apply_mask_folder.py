import SimpleITK as sitk
import os
import argparse
import numpy as np


def MaskedImage(fixed_image_path, fixed_seg_path, folder_output, suffix, SegLabel=None):
    """Mask the fixed image with the fixed segmentation and write it to a file"""
    # print("fixed_seg_path : ",fixed_seg_path)
    # print("fixed_image_path : ",fixed_image_path)
    fixed_image_sitk = sitk.ReadImage(fixed_image_path)
    fixed_seg_sitk = sitk.ReadImage(fixed_seg_path)
    # fixed_seg_sitk.SetOrigin(fixed_image_sitk.GetOrigin())

    fixed_image_masked = applyMask(fixed_image_sitk, fixed_seg_sitk, label=SegLabel)
    if fixed_image_masked=="failed":
        print("failed process on : ",fixed_image_sitk)
        return 
    
    # Write the masked image
    base_name, ext = os.path.splitext(fixed_image_path)
    if base_name.endswith('.nii'):  # Case for .nii.gz
        ext = '.nii.gz'
    
    file_name = os.path.basename(fixed_image_path)
    file_name_without_ext = os.path.splitext(os.path.splitext(file_name)[0])[0]

    # Construction du chemin de fichier de sortie
    output_path = os.path.join(folder_output, f"{file_name_without_ext}_{suffix}{ext}")
    print("-"*100)
    print("output_path : ",output_path)
    sitk.WriteImage(sitk.Cast(fixed_image_masked, sitk.sitkInt16), output_path)

    return output_path


def applyMask(image, mask, label):
    """Apply a mask to an image."""
    try : 
        array = sitk.GetArrayFromImage(mask)
        if label is not None and label in np.unique(array):
            array = np.where(array == label, 1, 0)
            mask = sitk.GetImageFromArray(array)
            mask.CopyInformation(image)
    except KeyError as e :
        print(e)
        return "failed"

    return sitk.Mask(image, mask)


def find_segmentation_file(image_file, seg_folder):
    """Find the corresponding segmentation file for a given image file."""
    base_name = os.path.basename(image_file)
    patient_id = base_name.split('_CBCT')[0].split('_MR')[0]
    # print("patient_id : ",patient_id)
    # print("base_name : ",base_name)
    
    for seg_file in os.listdir(seg_folder):
        if seg_file.startswith(patient_id) and seg_file.endswith('_label2.nii.gz'):
            return os.path.join(seg_folder, seg_file)
    
    return None


def process_folder(folder_path, seg_folder, folder_output, suffix, seg_label):
    """Process all files in the specified folder."""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')) and ('_CBCT' in file or '_MR' in file):
                fixed_image_path = os.path.join(root, file)
                fixed_seg_path = find_segmentation_file(file, seg_folder)
                
                if fixed_seg_path:
                    # print(f"Processing file: {fixed_image_path}")
                    try :
                        MaskedImage(fixed_image_path, fixed_seg_path, folder_output, suffix, seg_label)
                        # print(f"Segmentation file for {fixed_image_path} succedeed.")
                    except KeyError as e:
                        print(f"Segmentation file for {fixed_image_path} failed.")
                        print(e)
                        continue
                else:
                    print(f"Segmentation file for {fixed_image_path} not found.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply segmentation mask to all MRI files in a folder.")
    parser.add_argument("--folder_path", type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a3_Registration_closer_all/b2_CBCT_norm/test_percentile=[10,95]_norm=[0,75]", help="The path to the folder containing the MRI files.")
    parser.add_argument("--seg_folder", type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a3_Registration_closer_all/d0_CBCT_seg_sep/label_2", help="The path to the segmentation file.")
    parser.add_argument("--folder_output", type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a3_Registration_closer_all/b3_CBCT_inv_norm_mask:l2/a03_test_percentile=[10,95]_norm=[0,75]", help="The path to the output folder for the masked files.")
    parser.add_argument("--suffix", type=str, default="mask", help="The suffix to add to the output filenames.")
    parser.add_argument("--seg_label", type=int, default=1, help="Label of the segmentation.")

    args = parser.parse_args()
    
    if not os.path.exists(args.folder_output):
        os.makedirs(args.folder_output)
        
    process_folder(args.folder_path, args.seg_folder, args.folder_output, args.suffix, args.seg_label)
