import SimpleITK as sitk
import os
import numpy as np


# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("MRI2CBCT_CLI_utils_LR_Crop")

def crop_mri(img_path, output_folder):
    image = sitk.ReadImage(img_path)
    size = image.GetSize()
    split_z = size[2] // 2

    left = sitk.RegionOfInterest(image, [size[0], size[1], split_z], [0, 0, 0])
    right = sitk.RegionOfInterest(image, [size[0], size[1], size[2] - split_z], [0, 0, split_z])

    filename = os.path.basename(img_path)
    base, ext = os.path.splitext(filename)
    if base.endswith(".nii"):  # Handle .nii.gz
        base = base[:-4]
        ext = ".nii.gz"

    out_left = os.path.join(output_folder, f"{base}_cropLeft{ext}")
    out_right = os.path.join(output_folder, f"{base}_cropRight{ext}")

    sitk.WriteImage(left, out_left)
    sitk.WriteImage(right, out_right)

    logger.info(f"[MRI] Saved: {out_left}")
    logger.info(f"[MRI] Saved: {out_right}")


def crop_cbct(img_path, output_folder):
    image = sitk.ReadImage(img_path)
    size = image.GetSize()
    split_x = size[0] // 2

    left = sitk.RegionOfInterest(image, [split_x, size[1], size[2]], [0, 0, 0])
    right = sitk.RegionOfInterest(image, [size[0] - split_x, size[1], size[2]], [split_x, 0, 0])
    
    direction = image.GetDirection()
    direction_matrix = np.array(direction).reshape(3, 3)
    if direction_matrix[0, 0] < 0:  # X axis is flipped then need to swap
        logger.info("[INFO] Flipped X-axis detected. Swapping left and right labels.")
        left, right = right, left

    filename = os.path.basename(img_path)
    base, ext = os.path.splitext(filename)
    if base.endswith(".nii"):
        base = base[:-4]
        ext = ".nii.gz"

    # TODO: Check why left is right and why right is left
    out_left = os.path.join(output_folder, f"{base}_cropRight{ext}")
    out_right = os.path.join(output_folder, f"{base}_cropLeft{ext}")

    sitk.WriteImage(left, out_left)
    sitk.WriteImage(right, out_right)

    logger.info(f"[CBCT] Saved: {out_left}")
    logger.info(f"[CBCT] Saved: {out_right}")