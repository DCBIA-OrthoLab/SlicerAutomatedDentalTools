import SimpleITK as sitk
import os
import argparse

def invert_mri_intensity(path_folder, folder_output, suffix):
    # Check if the output folder exists, if not create it
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    
    # Iterate through the files in the input folder
    for filename in os.listdir(path_folder):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            filepath = os.path.join(path_folder, filename)
            image = sitk.ReadImage(filepath)

            # Convert the image to a numpy array to manipulate intensities
            image_array = sitk.GetArrayFromImage(image)

            # Find the maximum intensity value in the image
            max_intensity = image_array.max()

            # Invert the intensities while keeping the background (where intensity is 0) unchanged
            inverted_image_array = max_intensity - image_array
            inverted_image_array[image_array == 0] = 0

            # Convert the inverted numpy array back to a SimpleITK image
            inverted_image = sitk.GetImageFromArray(inverted_image_array)

            # Copy the original image information (such as spacing, origin, etc.) to the inverted image
            inverted_image.CopyInformation(image)

            # Generate the new filename with the suffix
            base_name, ext = os.path.splitext(filename)
            if base_name.endswith('.nii'):  # Case for .nii.gz
                base_name, ext2 = os.path.splitext(base_name)
                ext = ext2 + ext

            output_filename = os.path.join(folder_output, f"{base_name}_{suffix}{ext}")

            # Save the inverted image
            sitk.WriteImage(inverted_image, output_filename)

            print(f"Inversion completed for {filename}, saved as {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert the intensity of MRI images while keeping the background at 0.")
    parser.add_argument("--path_folder", type=str, help="The path to the folder containing the MRI files", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a3_Registration_closer_all/a0_MRI")
    parser.add_argument("--folder_output", type=str, help="The path to the output folder for the inverted files",default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a3_Registration_closer_all/a1_MRI_inv")
    parser.add_argument("--suffix", type=str, help="The suffix to add to the output filenames",default="inv")

    args = parser.parse_args()
    invert_mri_intensity(args.path_folder, args.folder_output, args.suffix)
