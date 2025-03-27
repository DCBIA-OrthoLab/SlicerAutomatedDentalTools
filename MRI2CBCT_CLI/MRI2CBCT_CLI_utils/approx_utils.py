import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch.nn.functional as F
from nibabel.processing import resample_from_to

def get_corresponding_file(folder, patient_id, modality):
    """
    Gets the corresponding file for a patient in a folder.

    Args:
        folder (str): Path to the folder containing the files
        patient_id (str): ID of the patient
        modality (str): Modality of the file

    Returns:
        str: Path to the corresponding file if exists, None otherwise
    """
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and (file.endswith(".nii.gz") or file.endswith(".nii")):
                return os.path.join(root, file)
    return None

def downsample(moved, static, scale_factor=0.5, mode='trilinear', align_corners=False):
    """
    Downsamples two 3D images (moved and static) by the specified scale factor.
    
    Args:
        moved (torch.Tensor): The moved image tensor of shape [D, H, W].
        static (torch.Tensor): The static image tensor of shape [D, H, W].
        scale_factor (float): The downsampling factor (e.g., 0.5 to halve each dimension).
        mode (str): Interpolation mode to use (default is 'trilinear' for 3D images).
        align_corners (bool): Passed to F.interpolate (default is False).
        
    Returns:
        tuple: A tuple (moved_downsampled, static_downsampled) of downsampled tensors.
    """
    # Add batch and channel dimensions: [N, C, D, H, W]
    moved_batch = moved.unsqueeze(0).unsqueeze(0)
    static_batch = static.unsqueeze(0).unsqueeze(0)
    
    # Downsample using interpolation.
    moved_down = F.interpolate(moved_batch, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    static_down = F.interpolate(static_batch, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    
    # Remove the added dimensions so that the result is again [D, H, W]
    moved_down = moved_down.squeeze(0).squeeze(0)
    static_down = static_down.squeeze(0).squeeze(0)
    
    return moved_down, static_down

def prealign_mri_to_cbct(mri_path, cbct_path, order=1):
    """
    Pre-aligns the MRI to CBCT by applying an affine transformation, then resamples both images to a common shape.

    Parameters
    ----------
    mri_path : str
        Path to the MRI NIfTI file.
    cbct_path : str
        Path to the CBCT NIfTI file.
    order : int, optional
        Interpolation order (default is 1 for trilinear interpolation).

    Returns
    -------
    moving_resampled : nibabel.Nifti1Image
        The MRI image pre-aligned and resampled to the CBCT space.
    static_resampled : nibabel.Nifti1Image
        The CBCT image resampled to the same shape as the MRI.
    """
    
    # Load MRI and CBCT NIfTI images
    moving_nii = nib.load(mri_path)
    static_nii = nib.load(cbct_path)
    moving_nii = nib.as_closest_canonical(moving_nii)
    static_nii = nib.as_closest_canonical(static_nii)

    # Get affines
    A_mri = moving_nii.affine
    A_cbct = static_nii.affine

    # Extract rotation matrices (ignores scaling/shear)
    def get_rotation(affine):
        R = affine[:3, :3]
        # Orthogonalize using SVD
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt

    R_mri = get_rotation(A_mri)
    R_cbct = get_rotation(A_cbct)

    # Compute rotation correction
    R_correction = R_cbct @ R_mri.T

    # Compute translation correction
    T_correction = A_cbct[:3, 3] - R_correction @ A_mri[:3, 3]

    # Build rigid transformation matrix
    prealign_transform = np.eye(4)
    prealign_transform[:3, :3] = R_correction
    prealign_transform[:3, 3] = T_correction

    # Apply transformation (preserves MRI's original voxel spacing)
    moving_aligned = apply_affine_to_nifti(moving_nii, prealign_transform)

    # Resample to common shape
    moving_resampled, static_resampled = resample_to_match(moving_aligned, static_nii, order=order)
    
    return moving_resampled, static_resampled, prealign_transform, moving_nii.header.get_zooms()

def apply_affine_to_nifti(nifti_img, affine):
    """
    Applies an affine transformation to a NIfTI image.

    Parameters
    ----------
    nifti_img : nibabel.Nifti1Image
        The input NIfTI image to be transformed.
    affine : np.ndarray
        A 4x4 affine transformation matrix.

    Returns
    -------
    transformed_img : nibabel.Nifti1Image
        The transformed NIfTI image.
    """
    new_affine = affine @ nifti_img.affine  # Apply transformation to affine
    return nib.Nifti1Image(nifti_img.get_fdata(), new_affine, header=nifti_img.header)

def resample_to_match(moving_img, static_img, order=1):
    """
    Resample the moving and static images to a common shape,
    using for each axis the maximum of the two sizes.
    
    Both images will keep their original affine (and hence spacing and origin)
    but the returned images will have a shape that is the element‐wise maximum of the
    two input shapes. Voxels that fall outside the original image are filled with 0.
    
    Parameters
    ----------
    moving_img : nibabel.Nifti1Image
        The moving image.
    static_img : nibabel.Nifti1Image
        The static image.
    order : int, optional
        The interpolation order (default is 1, i.e. trilinear interpolation).
    
    Returns
    -------
    moving_resampled, static_resampled : tuple of nibabel.Nifti1Image
        The moving and static images resampled to the same shape.
    """

    shape_moving = np.array(moving_img.shape)
    shape_static = np.array(static_img.shape)
    target_shape = tuple(np.maximum(shape_moving, shape_static))
    
    target_moving = (target_shape, moving_img.affine)
    target_static = (target_shape, static_img.affine)
    
    print(f"Resampling images to shape {target_shape}")
    
    moving_resampled = resample_from_to(moving_img, target_moving, order=order)
    static_resampled = resample_from_to(static_img, target_static, order=order)
    
    return moving_resampled, static_resampled

def convert_transform_for_slicer(transform_matrix):
    """
    Converts a 4x4 transformation matrix from RAS (Nibabel/NumPy) to LPS (ITK/Slicer).
    """
    # Convert rotation part (3x3)
    R = transform_matrix[:3, :3]
    
    # Convert translation vector
    T = transform_matrix[:3, 3]

    # Apply RAS to LPS conversion
    ras_to_lps = np.diag([-1, -1, 1])  # Flip X and Y axes

    # Convert rotation matrix
    R_lps = ras_to_lps @ R @ np.linalg.inv(ras_to_lps)

    # Convert translation vector
    T_lps = ras_to_lps @ T  # Flip X and Y components

    # Build new transformation matrix
    transform_lps = np.eye(4)
    transform_lps[:3, :3] = R_lps
    transform_lps[:3, 3] = T_lps

    return transform_lps

def resample_image(nib_img, spacing, interpolator=sitk.sitkLinear):
    """
    Resample a NiBabel image to a given size and spacing while preserving orientation.

    Parameters:
    ----------
    nib_img : nibabel.Nifti1Image
        The input image in NiBabel format.
    size : tuple
        Target size of the image (width, height, depth).
    spacing : tuple
        Target voxel spacing.
    interpolator : sitk interpolator
        The interpolation method (default is sitkLinear).

    Returns:
    --------
    resampled_sitk : sitk.Image
        The resampled image in sitk format.
    """

    # Convert NiBabel image to SimpleITK
    sitk_img = nib_to_sitk(nib_img)

    # Initialize resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(spacing)
    resample.SetSize(sitk_img.GetSize())
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetInterpolator(interpolator)

    # Apply resampling
    resampled_sitk = resample.Execute(sitk_img)

    return resampled_sitk

def sitk_to_nib(sitk_img):
    """Convert a SimpleITK image back to NiBabel while preserving spacing and orientation."""
    array = sitk.GetArrayFromImage(sitk_img)  # Extract image data
    spacing = sitk_img.GetSpacing()
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    origin = np.array(sitk_img.GetOrigin())

    # Construct the affine matrix to include spacing
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)  # Apply spacing to direction
    affine[:3, 3] = origin  # Set translation

    # Reverse Z-Y-X axis order (SimpleITK stores images in flipped order)
    array = np.moveaxis(array, [0, 1, 2], [2, 1, 0])

    return nib.Nifti1Image(array, affine)

def nib_to_sitk(nib_img):
    """Convert a NiBabel image to a SimpleITK image while preserving orientation correctly."""
    img_data = nib_img.get_fdata()
    affine = nib_img.affine

    # Extract spacing
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # Flip axes to match SimpleITK ordering (Z, Y, X)
    img_data = np.moveaxis(img_data, source=[0, 1, 2], destination=[2, 1, 0])

    # Convert image data to SimpleITK
    sitk_img = sitk.GetImageFromArray(img_data)  # SimpleITK expects (Z, Y, X) ordering

    # Set spacing and origin correctly
    sitk_img.SetSpacing(tuple(spacing))
    sitk_img.SetOrigin(tuple(affine[:3, 3]))

    # Fix direction matrix before setting it in SimpleITK
    direction = affine[:3, :3].flatten()
    fixed_direction = orthogonalize_direction_matrix(direction)
    sitk_img.SetDirection(fixed_direction)

    return sitk_img

def orthogonalize_direction_matrix(direction):
    """Ensure the direction matrix is strictly orthogonal."""
    direction = np.array(direction).reshape(3, 3)
    U, _, Vt = np.linalg.svd(direction)  # Singular Value Decomposition
    orthogonal_direction = np.dot(U, Vt)  # Reconstruct an orthogonal matrix
    return tuple(orthogonal_direction.flatten())