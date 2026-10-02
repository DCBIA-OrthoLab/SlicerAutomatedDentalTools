import os
import numpy as np
import nibabel as nib


# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("MRI2CBCT_CLI_utils_approx")

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

def compute_rotation_correction(mri_path, cbct_path):
    """
    Extracts the rotation-only correction between an MRI and a CBCT, ignoring
    translation: orthogonalizes the rotation part of each affine (via SVD) and
    returns the rotation that takes the MRI's orientation onto the CBCT's.

    Parameters
    ----------
    mri_path : str
        Path to the MRI NIfTI file.
    cbct_path : str
        Path to the CBCT NIfTI file.

    Returns
    -------
    R_correction : np.ndarray
        3x3 rotation matrix, in world (RAS) space.
    """
    moving_nii = nib.as_closest_canonical(nib.load(mri_path))
    static_nii = nib.as_closest_canonical(nib.load(cbct_path))

    def get_rotation(affine):
        R = affine[:3, :3]
        U, _, vt = np.linalg.svd(R)
        return U @ vt

    r_mri = get_rotation(moving_nii.affine)
    r_cbct = get_rotation(static_nii.affine)
    return r_cbct @ r_mri.T


def world_center_of_mass(nifti_img):
    """
    Center of mass of the nonzero voxels of a NIfTI image, in world (RAS) mm.

    Parameters
    ----------
    nifti_img : nibabel.Nifti1Image

    Returns
    -------
    np.ndarray
        3-element world-space (RAS) point.
    """
    ijk_mean = np.array(np.nonzero(nifti_img.get_fdata() > 0)).mean(axis=1)
    return nifti_img.affine[:3, :3] @ ijk_mean + nifti_img.affine[:3, 3]
