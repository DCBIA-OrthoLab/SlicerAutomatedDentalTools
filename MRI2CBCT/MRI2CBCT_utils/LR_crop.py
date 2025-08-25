from .Method import Method
from .utils_CBCT import GetDictPatients, GetPatients
import os, sys

import SimpleITK as sitk
import numpy as np

from glob import iglob
import slicer
import time
import qt
import platform
import re


class LR_CROP_MRI2CBCT(Method):
    def __init__(self, widget):
        super().__init__(widget)
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        documents = qt.QStandardPaths.writableLocation(documentsLocation)

    def getGPUUsage(self):
        if platform.system() == "Darwin":
            return 1
        else:
            return 5

    def NumberScan(self, scan_folder_t1: str, scan_folder_t2: str):
        return len(GetDictPatients(scan_folder_t1, scan_folder_t2))
    
    def getModelUrl(self):
        return {
        }

    def TestScan(self, scan_folder: str):
        extensions = ['.nii', '.nii.gz', '.nrrd']
        if scan_folder!="None" :
            found_files = self.search(scan_folder, extensions)
            if any(found_files[ext] for ext in extensions):
                return True, ""
            else:
                return False, "No files to run has been found in the "    
        return True,""
        
    def TestProcess(self, **kwargs) -> str:
        out = ""
        ok = True
        
        if kwargs["input_folder_CBCT"] == "":
            out += "Please select an input folder for CBCT scans\n"
            ok = False
            
        if kwargs["input_folder_MRI"] == "":
            out += "Please select an input folder for MRI scans\n"
            ok = False
            
        if kwargs["input_folder_Seg"] == "":
            out += "Please select an input folder for Segmentation scans\n"
            ok = False
            
        if kwargs["output_folder"] == "":
            out += "Please select an output folder\n"
            ok = False
            
        if out == "":
            out = None

        return ok,out
    
    def Process(self, **kwargs):
        list_process=[]
    
        MRI2CBCT_LR_CROP = slicer.modules.mri2cbct_lr_crop
        parameter_mri2cbct_lr_crop = {
            "input_folder_CBCT": kwargs["input_folder_CBCT"],
            "input_folder_MRI": kwargs["input_folder_MRI"],
            "input_folder_Seg": kwargs["input_folder_Seg"],
            "output_folder": kwargs["output_folder"]
        }
        
        list_process.append(
            {
                "Process": MRI2CBCT_LR_CROP,
                "Parameter": parameter_mri2cbct_lr_crop,
                "Module": "MRI2CBCT_LR_CROP",
            }
        )
           
        return list_process
    
  