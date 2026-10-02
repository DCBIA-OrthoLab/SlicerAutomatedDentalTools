#!/usr/bin/env python-real

import argparse
import SimpleITK as sitk



from Crop_Volumes_utils.FilesType import Search, ChangeKeyDict
from Crop_Volumes_utils.GenerateVTKfromSeg import convertNiftiToVTK
import numpy as np
import os,json

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
import sys

logger = get_logger("AutoCrop3D_CLI")

def main(args)-> None:
    """
    Crop a Region of Interest on files with the extension .nii.gz .nrrd.gz .gipl.gz
    Input:  scan_files_path,
            path_ROI_file,
            output_path,
            suffix,
            box_Size, #checkbox in UI
            logPath # For the progress bar in UI


    """
    path_input = args.scan_files_path
    roi_path = args.path_ROI_file
    output_path = args.output_path
    suffix_namefile = args.suffix
    original_size = args.box_Size

    with open(args.logPath,'w') as log_f:
        # clear log file
        log_f.truncate(0)
    index =0
    scan_list = Search(path_input, ".nii.gz",".nii",".nrrd.gz",".nrrd",".gipl.gz",".gipl")

    # Include case with a folder of ROI corresponding to a folder of scans
    roi_list = Search(roi_path,".mrk.json")

    if len(roi_list['.mrk.json']) >1:
        roi_dict = ChangeKeyDict(roi_list)

    for key,data in scan_list.items():

        for patient_path in data:
            patient = os.path.basename(patient_path).split('_Scan')[0].split('_scan')[0].split('_Seg')[0].split('_seg')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('.')[0]

            img = sitk.ReadImage(patient_path)

            if len(roi_list['.mrk.json']) >1:
                try:
                    roi_path = roi_dict[patient]
                except Exception:
                    logger.warning('No ROI for patient:'+str(patient))
                    continue

            ROI = json.load(open(roi_path))['markups'][0]
            roi_center = np.array(ROI['center'])
            roi_size = np.array(ROI['size'])

            lower = roi_center - roi_size / 2
            upper = roi_center + roi_size / 2

            lower = np.array(img.TransformPhysicalPointToContinuousIndex(lower)).astype(int)
            upper = np.array(img.TransformPhysicalPointToContinuousIndex(upper)).astype(int)

            for i in range(3):
                if lower[i] > upper[i]:
                    lower[i], upper[i] = upper[i], lower[i]
            # Bounds checking
            img_size = img.GetSize()
            lower = [max(0, l) for l in lower]

            upper = [min(img_size[i], u) for i, u in enumerate(upper)]

            # Crop the image

            # copy img to apply changes
            img_blank = sitk.Image(img.GetSize(), img.GetPixelID())
            img_blank.CopyInformation(img)

            img_blank_arr = sitk.GetArrayFromImage(img_blank)

            # Coord of the ROI in the blank image
            size_roi = [int(upper[0]-lower[0]),int(upper[1]-lower[1]),int(upper[2]-lower[2])]
            start_coord = [int(lower[0]),int(lower[1]),int(lower[2])]
            end_coord = [start_coord[0]+size_roi[0],start_coord[1]+size_roi[1],start_coord[2]+size_roi[2]]

            # Get only the ROI
            img_roi = img[lower[0]:upper[0],
                            lower[1]:upper[1],
                            lower[2]:upper[2]]

            if original_size=='True':
                img_roi_arr = sitk.GetArrayFromImage(img_roi)

                # GetArrayFromImage return a numpy array with the shape (z,y,x)
                # Put Pixel Value in the blank image
                img_blank_arr[start_coord[2]:end_coord[2],
                                start_coord[1]:end_coord[1],
                                start_coord[0]:end_coord[0]] = img_roi_arr

                img_crop = sitk.GetImageFromArray(img_blank_arr)
                img_crop.CopyInformation(img_blank)

            else:
                img_crop = img_roi

            # Create the output path
            # relative_path = all folder to get to the file we want in the input
            relative_path = os.path.relpath(patient_path,path_input)
            filename_interm = os.path.basename(patient_path).split('.')[0]
            filename = filename_interm + "_"+ suffix_namefile + key

            vtk_filename = filename_interm + "_" + suffix_namefile + "_vtk.vtk"
            scan_out_path = os.path.join(output_path,relative_path).replace(os.path.basename(relative_path),filename)

            vtk_out_path = os.path.join(output_path,relative_path).replace(os.path.basename(relative_path),vtk_filename)

            os.makedirs(os.path.dirname(scan_out_path), exist_ok=True)

            try:

                sitk.WriteImage(img_crop,scan_out_path)

            except Exception:
                logger.error("Error for patient: "+str(patient))
                logger.error('The error says: '+str(sys.exc_info()[0]))
                logger.error('Lower: '+str(lower))
                logger.error('Upper: '+str(upper))
                logger.error('Lower[2]:'+str(lower[2]))
                logger.error('Upper[2]:'+str(upper[2]))

            with open(args.logPath,'r+') as log_f :
                    log_f.write(str(index))

            if "seg" in scan_out_path.lower():
                try :
                    convertNiftiToVTK(scan_out_path,vtk_out_path)
                except Exception:
                    logger.debug("Could not convert the segmentation to VTK", exc_info=True)

            index+=1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('scan_files_path',type=str)
    parser.add_argument('path_ROI_file',type=str)
    parser.add_argument("output_path",type=str)
    parser.add_argument('suffix',type=str)
    parser.add_argument('box_Size',type=str)
    parser.add_argument('logPath',type=str)


    args = parser.parse_args()


    main(args)