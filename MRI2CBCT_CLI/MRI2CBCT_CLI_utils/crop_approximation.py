import os
import torch
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch.nn.functional as F
from sklearn.model_selection import ParameterSampler
from torchreg import AffineRegistration
from .nmi import NMI

def save_as_nifti(moving_tensor, static_path, output_path):
    """
    Saves a tensor as a NIfTI file.
    
    Args:
        moving_tensor (torch.Tensor): PyTorch tensor to save as NIfTI
        static_path (str): Path to the reference NIfTI file to get the affine and header information
        output_path (str): Path to save the NIfTI file
    """
    # Load the reference nifti file to get the affine and header information
    static_nifti = nib.load(static_path)
    
    # Create a new Nifti1Image using the numpy data and the affine from the reference image
    try:
        new_nifti = nib.Nifti1Image(moving_tensor.cpu().numpy(), static_nifti.affine, static_nifti.header)
    except:
        new_nifti = nib.Nifti1Image(moving_tensor, static_nifti.affine, static_nifti.header)
    
    # Save the new NIfTI image to disk
    print("Saved registered image to:", output_path)
    nib.save(new_nifti, output_path)

def save_affine_transform(matrix, output_path):
    """
    Save the 4x4 transformation matrix as an affine transform in SimpleITK and save to a file.

    Args:
        matrix (numpy.ndarray): The 4x4 affine transformation matrix.
        output_path (str): The path to save the .tfm file.
    """
    # Create a SimpleITK affine transform (3D)
    affine_transform = sitk.AffineTransform(3)

    # Extract the 3x3 rotation/scale matrix and the translation vector
    rotation_scale_matrix = matrix[:3, :3].flatten().tolist()  # Convert to a list for SimpleITK
    translation_vector = matrix[:3, 3].tolist()  # Translation part

    # Set the matrix and translation in the SimpleITK affine transform
    affine_transform.SetMatrix(rotation_scale_matrix)
    affine_transform.SetTranslation(translation_vector)

    # Save the transform to a file
    sitk.WriteTransform(affine_transform, output_path)
    print(f"Affine transformation matrix saved to {output_path}")

def compute_transform_matrix(fixed_volume_affine, moving_volume_affine):
    """
    Compute the transformation matrix to register the fixed volume to the moving volume.

    Args:
        fixed_volume (NifTi.affine): Affine of the fixed volume.
        moving_volume (NifTi.affine): Affine of the moving volume.

    Returns:
        matrix: Transformation matrix to register the fixed volume to the moving volume.
    """
    # Get origin, spacing, and direction from both volumes
    fixed_origin = np.array(fixed_volume_affine.GetOrigin())
    fixed_spacing = np.array(fixed_volume_affine.GetSpacing())
    fixed_direction = np.array(fixed_volume_affine.GetDirection()).reshape(3, 3)
    
    moving_origin = np.array(moving_volume_affine.GetOrigin())
    moving_spacing = np.array(moving_volume_affine.GetSpacing())
    moving_direction = np.array(moving_volume_affine.GetDirection()).reshape(3, 3)
    
    # Create a scaling matrix from the spacing (this will ensure voxel size differences are handled)
    fixed_scaling_matrix = np.diag(fixed_spacing)
    moving_scaling_matrix = np.diag(moving_spacing)
    
    # Compute the rigid transformation (rotation and translation)
    # Rotation is derived from the direction matrix
    rotation_matrix = np.dot(fixed_direction, np.linalg.inv(moving_direction))
    
    # Adjust translation to account for rotation and scaling
    scaled_moving_origin = np.dot(moving_scaling_matrix, moving_origin)
    scaled_fixed_origin = np.dot(fixed_scaling_matrix, fixed_origin)
    
    # Translation is derived from the difference in the scaled origins
    translation_vector = scaled_fixed_origin - np.dot(rotation_matrix, scaled_moving_origin)
    
    # Compose the transformation matrix: rotation + translation
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    
    return transformation_matrix

def pad_image_to_match(image, target_shape):
    """
    Pads the input image symmetrically to match the target shape.

    Args:
        image (SimpleITK.Image): The input image to be padded.
        target_shape (tuple): The target shape to match (z, y, x).

    Returns:
        SimpleITK.Image: The padded image with the target shape.
    """
    current_shape = np.array(image.GetSize())
    target_shape = np.array(target_shape)

    # Calculate the padding required on each side
    padding_needed = (target_shape - current_shape) / 2
    padding_lower = np.floor(padding_needed).astype(int)
    padding_upper = np.ceil(padding_needed).astype(int)

    # Ensure padding is non-negative
    padding_lower = np.maximum(padding_lower, 0)
    padding_upper = np.maximum(padding_upper, 0)

    # Apply padding using SimpleITK's ConstantPad method
    padded_image = sitk.ConstantPad(image, list(map(int, padding_lower)), list(map(int, padding_upper)), 0.0)
    return padded_image

def convert_tensor_to_sitk(tensor, origin, spacing, direction):
    """
    Convert a PyTorch tensor to a SimpleITK image.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        origin (tuple): The origin for the SimpleITK image.
        spacing (tuple): The spacing for the SimpleITK image.
        direction (tuple): The direction for the SimpleITK image.
        
    Returns:
        SimpleITK.Image: The converted SimpleITK image with set metadata.
    """
    # Convert tensor to NumPy array
    np_array = tensor.cpu().numpy()

    # Convert NumPy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(np_array)

    # Set metadata (origin, spacing, direction)
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetDirection(direction)

    return sitk_image

def normalize_image(image_sitk):
    """
    Normalize the SimpleITK image to the range [0, 1] using min-max normalization.

    Args:
        image_sitk (SimpleITK.Image): The input image.

    Returns:
        SimpleITK.Image: The normalized image.
    """
    # Convert to a NumPy array for normalization
    image_np = sitk.GetArrayFromImage(image_sitk)
    epsilon = 1e-8
    image_np_normalized = (image_np - image_np.min()) / (image_np.max() - image_np.min() + epsilon)

    # Convert back to SimpleITK image
    normalized_sitk = sitk.GetImageFromArray(image_np_normalized)
    normalized_sitk.CopyInformation(image_sitk)  # Keep the same metadata
    return normalized_sitk

def get_transformation(cbct_folder, mean_folder, output_folder):
    """
    Automatically crop CBCT images around the TMJ.

    Args:
        cbct_folder (str): Path to the folder containing CBCT images.
        mean_folder (str): Path to the folder containing the mean CBCT image.
        output_folder (str): Path to the folder where output images and transformations will be saved.
    """
    # Generate output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Define the parameter grid for hyperparameter search
    param_grid = {
        'learning_rate_rigid': np.logspace(-5, -3, 10),   # Learning rate for rigid registration
        'sigma_rigid': np.logspace(-2, -1, 4)             # Sigma for rigid NMI
    }

    # Number of parameter combinations to sample
    n_samples = 40
    param_sampler = ParameterSampler(param_grid, n_iter=n_samples)
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if "_CBCT_" in cbct_file and (cbct_file.endswith(".nii") or cbct_file.endswith(".nii.gz")):
                patient_id = cbct_file.split("_CBCT_")[0]
                cbct_path = os.path.join(root, cbct_file)
                mean_path = os.path.join(mean_folder, 'mean_image.nii.gz')

                if mean_path:
                    best_loss = float('inf')  # Initialize to track the best loss
                    best_params = None        # Initialize to store the best parameter combination
                    for params in param_sampler:
                        # Load images using nibabel
                        moving_nii = nib.load(cbct_path)
                        static_nii = nib.load(mean_path)

                        # Get the NumPy data from nibabel images
                        moving_np = moving_nii.get_fdata()
                        static_np = static_nii.get_fdata()

                        # Normalize the images
                        epsilon = 1e-8
                        moving_np_normalized = (moving_np - moving_np.min()) / (moving_np.max() - moving_np.min() + epsilon)
                        static_np_normalized = (static_np - static_np.min()) / (static_np.max() - static_np.min() + epsilon)

                        # Pad the moving image (CBCT) to match the dimensions of the mean image
                        static_shape = static_np_normalized.shape
                        moving_padded_np = np.pad(
                            moving_np_normalized,
                            [(max(0, (s - ms) // 2), max(0, (s - ms + 1) // 2)) for s, ms in zip(static_shape, moving_np_normalized.shape)],
                            mode='constant',
                            constant_values=0
                        )

                        print(f"\nUsing {device.upper()} -- Registering CBCT: {cbct_path} with mean: {mean_path}")

                        # Convert NumPy arrays to torch tensors
                        moving_normed = torch.from_numpy(moving_padded_np).float().to(device)
                        static_normed = torch.from_numpy(static_np_normalized).float().to(device)

                        print(f"Testing parameters: {params}")

                        # Initialize NMI loss function for rigid registration
                        nmi_loss_function_rigid = NMI(intensity_range=None, nbins=64, sigma=params['sigma_rigid'], use_mask=False)

                        # Initialize AffineRegistration for Rigid registration
                        reg_rigid = AffineRegistration(scales=(4, 2), iterations=(100, 30), is_3d=True, 
                                                       learning_rate=params['learning_rate_rigid'], verbose=True, 
                                                       dissimilarity_function=nmi_loss_function_rigid.metric, optimizer=torch.optim.Adam, 
                                                       with_translation=True, with_rotation=True, with_zoom=False, with_shear=False, 
                                                       align_corners=True, interp_mode="trilinear", padding_mode='zeros')

                        # Perform rigid registration
                        moved_image = reg_rigid(moving_normed[None, None], static_normed[None, None])
                        moved_image = moved_image[0, 0]

                        # Compute the final loss
                        final_loss = -nmi_loss_function_rigid.metric(moved_image[None, None], static_normed[None, None])
                        print(f"Final Loss (NMI): {final_loss}")

                        # Check if this is the best loss so far
                        if final_loss < best_loss and final_loss > 1e-5:
                            best_loss = final_loss
                            best_params = params
                            print(f"New best parameters found with loss: {best_loss}")

                            # Save the registered image using your save_as_nifti function
                            output_path = os.path.join(output_folder, f'{patient_id}_CBCT_transform.nii.gz')
                            save_as_nifti(moved_image, mean_path, output_path)

                            # Print the transformation matrix for debugging
                            transform_matrix = compute_transform_matrix(static_nii.affine, moving_nii.affine)
                            print("Transformation Matrix from Volume A to Volume B:")
                            print(transform_matrix)

                    # Print the best result at the end
                    print(f"Best parameters: {best_params}")
                    print(f"Best NMI loss: {best_loss}")
                    
def crop_volume(ROI_file, transformation_folder, first_approximation_folder, cbct_folder, cropped_cbct_folder):
    """
    Crops the CBCT volumes and first approximated MRIs based on the ROI and saves the results.

    Args:
        ROI_file (str): Path to the file containing the ROI.
        transformation_folder (str): Path to the folder containing the transformation files.
        first_approximation_folder (str): Path to the folder containing the approximated MRI images.
        cbct_folder (str): Path to the folder containing the CBCT images.
        cropped_cbct_folder (str): Path to the folder where the cropped images will be saved.
    """
    # Load the ROI file
    roi_image = sitk.ReadImage(ROI_file)

    # Iterate over the files in the first approximation folder
    for root, _, files in os.walk(first_approximation_folder):
        for mri_file in files:
            if mri_file.endswith(".nii") or mri_file.endswith(".nii.gz"):
                patient_id = mri_file.split("_MR_registered.nii.gz")[0]
                mri_path = os.path.join(root, mri_file)
                cbct_path = os.path.join(cbct_folder, f"{patient_id}_CBCT_Crop.nii.gz")
                transformation_path = os.path.join(transformation_folder, f"{patient_id}_transform.tfm")

                if not os.path.exists(transformation_path):
                    print(f"Transformation file missing for {patient_id}")
                    continue

                if not os.path.exists(cbct_path):
                    print(f"CBCT file missing for {patient_id}")
                    continue

                # Apply the transformation to the ROI
                transform = sitk.ReadTransform(transformation_path)
                transformed_roi = sitk.Resample(roi_image, transform)

                # Load the MRI and CBCT images
                mri_image = sitk.ReadImage(mri_path)
                cbct_image = sitk.ReadImage(cbct_path)

                # Crop MRI using the transformed ROI
                mri_cropped = sitk.RegionOfInterest(
                    mri_image,
                    transformed_roi.GetSize(),
                    transformed_roi.GetOrigin()
                )
                mri_output_path = os.path.join(cropped_cbct_folder, f"{patient_id}_MR_approx_crop.nii.gz")
                sitk.WriteImage(mri_cropped, mri_output_path)

                # Crop CBCT using the transformed ROI
                cbct_cropped = sitk.RegionOfInterest(
                    cbct_image,
                    transformed_roi.GetSize(),
                    transformed_roi.GetOrigin()
                )
                cbct_output_path = os.path.join(cropped_cbct_folder, f"{patient_id}_CBCT_approx_crop.nii.gz")
                sitk.WriteImage(cbct_cropped, cbct_output_path)

                print(f"Cropped and saved images for {patient_id}")

def main():
    parser = argparse.ArgumentParser(description='Register CBCT images with the mean CBCT.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mean_folder', type=str, help='Path to the folder containing the mean CBCT')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()

    get_transformation(args.cbct_folder, args.mean_folder, args.output_folder) 
    
if __name__ == "__main__":
    main()