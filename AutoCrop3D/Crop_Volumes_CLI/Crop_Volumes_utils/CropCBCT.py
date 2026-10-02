import SimpleITK as sitk
import numpy as np
import os,json

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("Autocrop3D_CropCBCT")
#import multiprocessing as mp

def Crop(scan_list, input_path, roi_path, output_path, suffix_namefile ):
    '''
    !!! UNUSED  !!! This code is directly in the CLI of the extension

    Function to crop Scan with a Region Of Interest
    Input: Dictionnary with the Path of the Files and key, Input Path,
            Path of the ROI, Output Path, Suffix for the files

    Output: Cropped Scan in the folder OutputPath
    '''

    for key,data in scan_list.items():
        for patient_path in data:
            patient = os.path.basename(patient_path).split('_Scan')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('.')[0]

            scan_out_path = output_path+"/"+patient+suffix_namefile+key

            img = sitk.ReadImage(patient_path)

            str_patient = str(patient)
            logger.info(f"working on patient: {str_patient}")
            ROI = json.load(open(roi_path))['markups'][0]
            roi_center = np.array(ROI['center'])
            roi_size = np.array(ROI['size'])

            lower = roi_center - roi_size / 2
            upper = roi_center + roi_size / 2

            lower = np.array(img.TransformPhysicalPointToContinuousIndex(lower)).astype(int)
            upper = np.array(img.TransformPhysicalPointToContinuousIndex(upper)).astype(int)

            # Crop the image
            crop_image = img[lower[0]:upper[0],
                            lower[1]:upper[1],
                            lower[2]:upper[2]]

            try:
                sitk.WriteImage(crop_image,scan_out_path)
            except Exception:
                logger.error("Error for patient: "+str(patient))



