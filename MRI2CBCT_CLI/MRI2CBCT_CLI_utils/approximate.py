import os
import torch
import shutil
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from .nmi import NMI
from pathlib import Path
from torchreg import AffineRegistration
from sklearn.model_selection import ParameterSampler
from .approx_utils import get_corresponding_file, downsample, prealign_mri_to_cbct, resample_image, sitk_to_nib, convert_transform_for_slicer


def approximation(cbct_folder, mri_folder, output_folder, temp_mri, temp_cbct, progress_callback=None):
    """
    Main function to perform registration of CBCT images with corresponding MRI images.

    Args:
        cbct_folder (str): Path to the folder containing CBCT images.
        mri_folder (str): Path to the folder containing MRI images.
        output_folder (str): Path to save the registered images.
        temp_mri (str): Temporary folder for MRI images.
        temp_cbct (str): Temporary folder for CBCT images.
    """
    if not os.path.isdir(cbct_folder): raise ValueError(f"CBCT folder does not exist: {cbct_folder}")
    if not os.path.isdir(mri_folder): raise ValueError(f"MRI folder does not exist: {mri_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    param_grid = {
        'learning_rate': np.linspace(1e-5, 1e-3, 7),
        'sigma': np.linspace(1e-2, 1e-1, 2)
    }
    param_sampler = ParameterSampler(param_grid, n_iter=14)

    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if "_CBCT_" in cbct_file and (cbct_file.endswith(".nii") or cbct_file.endswith(".nii.gz")):
                patient_id = cbct_file.split("_CBCT_")[0]
                cbct_path = os.path.join(root, cbct_file)
                mri_path = get_corresponding_file(mri_folder, patient_id, "_MR_")

                if not mri_path:
                    print(f"No corresponding MRI file found for: {cbct_file}")
                    continue
                
                print(f"Treating patient: {patient_id}")
                output_path = os.path.join(output_folder, f'{patient_id}_MR_registered.tfm')

                moving_nii, static_nii, prealign_transform, mri_spacing = prealign_mri_to_cbct(mri_path, cbct_path)
                
                nib.save(moving_nii, os.path.join(temp_mri, f"{patient_id}_prealigned.nii.gz"))
                          
                static_spacing = tuple(float(s) for s in static_nii.header.get_zooms())      
                moving_sitk = resample_image(moving_nii, static_spacing, sitk.sitkLinear)
                static_sitk = resample_image(static_nii, static_spacing, sitk.sitkLinear)
                
                nib.save(sitk_to_nib(moving_sitk), os.path.join(temp_mri, os.path.basename(mri_path)))
                nib.save(sitk_to_nib(static_sitk), os.path.join(temp_cbct, os.path.basename(cbct_path)))
                
                # Convert to PyTorch tensors
                moving_data = sitk.GetArrayFromImage(moving_sitk)
                static_data = sitk.GetArrayFromImage(static_sitk)
                moving = torch.from_numpy(moving_data).float().to(device)
                static = torch.from_numpy(static_data).float().to(device)
                
                moving = moving.max() - moving
                moving[moving == 0] = 0

                moving_normed = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
                static_normed = (static - static.min()) / (static.max() - static.min() + 1e-8)

                best_loss = float('inf')
                best_transform = None
                
                for params in param_sampler:
                    # print(f"\n\033[1mUsing {device.upper()} -- Registering MRI: {temp_mri} to CBCT: {temp_cbct}")
                    # print(f"Testing parameters: {params}\033[0m")

                    nmi_loss_function = NMI(intensity_range=None, nbins=32, sigma=params['sigma'], use_mask=False)
                    reg = AffineRegistration(scales=(4, 2), iterations=(500, 100), is_3d=True, 
                                                    learning_rate=params['learning_rate'],
                                                    dissimilarity_function=nmi_loss_function,
                                                    optimizer=torch.optim.Adam, verbose=False,
                                                    with_translation=True, with_rotation=True, 
                                                    with_zoom=False, with_shear=False,
                                                    interp_mode="trilinear", padding_mode='zeros')
                    moved_image = reg(moving_normed[None, None], static_normed[None, None])
                    moved_image = moved_image[0,0]
                    
                    transform_matrix = reg.get_affine().cpu().numpy()
                    
                    if transform_matrix.shape[0] == 1:
                        transform_matrix = transform_matrix.squeeze(0)
                    transform_matrix = np.vstack([transform_matrix, [0, 0, 0, 1]])
                    final_transform = prealign_transform @ transform_matrix
                    
                    moved_downsampled, static_downsampled = downsample(moved_image, static_normed, scale_factor=0.5)
                    loss = nmi_loss_function(moved_downsampled[None, None], static_downsampled[None, None])

                    if loss < best_loss:
                        best_loss = loss
                        best_transform = final_transform
                        # print(f"New best parameters found with loss: {best_loss}")
                        
                if best_transform is not None:
                    best_transform_lps = convert_transform_for_slicer(best_transform)
                    best_transform_lps_inv = np.linalg.inv(best_transform_lps)
                    
                    rotation = best_transform_lps_inv[:3, :3].flatten().tolist()
                    translation = best_transform_lps_inv[:3, 3].tolist()
                    
                    sitk_transform = sitk.AffineTransform(3)
                    sitk_transform.SetMatrix(rotation)
                    sitk_transform.SetTranslation(translation)

                    sitk.WriteTransform(sitk_transform, output_path)
                    print(f"Saved transformation to {output_path}\n\n")

            else: 
                print(f"CBCT file {cbct_file} does not match the expected format: {patient_id}_CBCT_xx.nii.gz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    parser.add_argument('--del_temp', action='store_true', help='Delete temporary files after processing')
    args = parser.parse_args()
    
    temp_folder = str(Path(args.mri_folder).parent) + '/temp/'
    temp_mri_folder_path = temp_folder + 'mri/'
    temp_cbct_folder_path = temp_folder + 'cbct/'
    os.makedirs(temp_mri_folder_path, exist_ok=True)
    os.makedirs(temp_cbct_folder_path, exist_ok=True)

    approximation(args.cbct_folder, args.mri_folder, args.output_folder, temp_mri_folder_path, temp_cbct_folder_path)
    
    if args.del_temp:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            print(f"Temporary folder {temp_folder} deleted.")
        else:
            print(f"Temporary folder {temp_folder} does not exist.")