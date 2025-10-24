import os
import sys
import time
import torch
import shutil
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from torchreg import AffineRegistration
from sklearn.model_selection import ParameterGrid

from MRI2CBCT_CLI_utils.nmi import NMI
from MRI2CBCT_CLI_utils.approx_utils import get_corresponding_file, downsample, prealign_mri_to_cbct, resample_image, sitk_to_nib, convert_transform_for_slicer


def approximation(cbct_folder, mri_folder, output_folder):
    """
    Main function to perform registration of CBCT images with corresponding MRI images.

    Args:
        cbct_folder (str): Path to the folder containing CBCT images.
        mri_folder (str): Path to the folder containing MRI images.
        output_folder (str): Path to save the registered images.
    """

    if not os.path.isdir(cbct_folder): raise ValueError(f"CBCT folder does not exist: {cbct_folder}")
    if not os.path.isdir(mri_folder): raise ValueError(f"MRI folder does not exist: {mri_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    param_grid = {
        'learning_rate': np.linspace(1e-5, 1e-3, 7),
        'sigma': np.linspace(1e-2, 1e-1, 2)
    }
    param_sampler = ParameterGrid(param_grid)
    
    patient_count = 0
    total_patients = sum(1 for root, _, files in os.walk(cbct_folder)
                         for f in files if "CBCT" in f and (f.endswith(".nii") or f.endswith(".nii.gz")))

    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if "CBCT" in cbct_file and (cbct_file.endswith(".nii") or cbct_file.endswith(".nii.gz")):
                patient_id = cbct_file.split("CBCT")[0]
                cbct_path = os.path.join(root, cbct_file)
                mri_path = get_corresponding_file(mri_folder, patient_id, "MRI")

                if not mri_path:
                    print(f"No corresponding MRIs file found for: {cbct_file}")
                    continue
                
                print(f"Treating patient: {patient_id}")
                output_path = os.path.join(output_folder, f'{patient_id}_MRI_approximate.tfm')

                moving_nii, static_nii, prealign_transform, mri_spacing = prealign_mri_to_cbct(mri_path, cbct_path)
                          
                static_spacing = tuple(float(s) for s in static_nii.header.get_zooms())      
                moving_sitk = resample_image(moving_nii, static_spacing, sitk.sitkLinear)
                static_sitk = resample_image(static_nii, static_spacing, sitk.sitkLinear)
                
                # Convert to PyTorch tensors
                moving_data = sitk.GetArrayFromImage(moving_sitk)
                static_data = sitk.GetArrayFromImage(static_sitk)
                moving = torch.from_numpy(moving_data).float().to(device)
                static = torch.from_numpy(static_data).float().to(device)
                

                bg = (moving == 0)
                moving = moving.max() - moving
                moving[bg] = 0

                moving_normed = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
                static_normed = (static - static.min()) / (static.max() - static.min() + 1e-8)

                best_loss = float('inf')
                best_transform = None
                
                for params in param_sampler:
                    # print(f"\n\033[1mUsing {device.upper()} -- Registering MRI: {mri_path} to CBCT: {cbct_path}")
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
                    final_transform = transform_matrix @ prealign_transform
                    
                    moved_downsampled, static_downsampled = downsample(moved_image, static_normed, scale_factor=0.5)
                    loss = nmi_loss_function(moved_downsampled[None, None], static_downsampled[None, None])

                    loss_value = float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float(loss)
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_transform = final_transform
                        
                if best_transform is not None:
                    best_transform_lps = convert_transform_for_slicer(best_transform)
                    
                    rotation = best_transform_lps[:3, :3].flatten().tolist()
                    translation = best_transform_lps[:3, 3].tolist()
                    
                    sitk_transform = sitk.AffineTransform(3)
                    sitk_transform.SetMatrix(rotation)
                    sitk_transform.SetTranslation(translation)

                    sitk.WriteTransform(sitk_transform, output_path)
                    print(f"Saved transformation to {output_path}\n\n")
                    
                    # === Apply transform to MRI and save transformed volume ===
                    # Reload original MRI to preserve spacing/origin
                    original_mri_sitk = sitk.ReadImage(mri_path)
                    cbct_ref = sitk.ReadImage(cbct_path)

                    sitk_transform_inv = sitk.AffineTransform(sitk_transform).GetInverse()

                    transformed_mri = sitk.Resample(
                        original_mri_sitk,
                        cbct_ref,
                        sitk_transform_inv,
                        sitk.sitkLinear,
                        0.0,
                        original_mri_sitk.GetPixelID()
                    )
                    
                    original = sitk.GetArrayFromImage(original_mri_sitk)
                    transform = sitk.GetArrayFromImage(transformed_mri)

                    print(np.amax(original))
                    print(np.amin(original))
                    print("val IRM")
                    print(np.amax(transform))
                    print(np.amin(transform))

                    # Define volume output path in same folder as transform
                    mri_out_filename = os.path.basename(mri_path).replace(".nii", "_approximate.nii")
                    mri_out_path = os.path.join(output_folder, mri_out_filename)
                    sitk.WriteImage(transformed_mri, mri_out_path)
                    print(f"Saved transformed MRI to {mri_out_path}\n\n")
                    
                    patient_count += 1
                    if total_patients > 0:
                        progress = patient_count / total_patients
                        print(f"<filter-progress>{progress}</filter-progress>")
                        sys.stdout.flush()
                        time.sleep(0.5)

            else: 
                print(f"CBCT file {cbct_file} does not match the expected format: {patient_id}_CBCT_xx.nii.gz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    args = parser.parse_args()

    approximation(args.cbct_folder, args.mri_folder, args.output_folder)