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


class Registration_MRI2CBCT(Method):
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


    def TestScan(self, scan_folder: str):
        extensions = ['.nii', '.nii.gz', '.nrrd']
        found_files = self.search(scan_folder, extensions)
        if any(found_files[ext] for ext in extensions):
            return True, ""
        else:
            return False, "No files to run has been found in the input folder"
        
    def CheckNormalization(self, norm: str):
        mri_min_norm, mri_max_norm, mri_lower_p, mri_upper_p = norm[0][0]
        cbct_min_norm, cbct_max_norm, cbct_lower_p, cbct_upper_p = norm[0][1]
        
        ok = True
        messages = []

        if mri_max_norm <= mri_min_norm:
            ok = False
            messages.append("MRI normalization max must be greater than min")
        if mri_upper_p <= mri_lower_p:
            ok = False
            messages.append("MRI percentile max must be greater than min")
        
        if cbct_max_norm <= cbct_min_norm:
            ok = False
            messages.append("CBCT normalization max must be greater than min")
        if cbct_upper_p <= cbct_lower_p:
            ok = False
            messages.append("CBCT percentile max must be greater than min")
        
        message = "\n".join(messages)
        
        return ok, message 
        
    def TestProcess(self, **kwargs) -> str:
        out = ""
        ok = True

        if kwargs["folder_general"] == "":
            out += "Please select an input folder for CBCT scans\n"
            ok = False

        if kwargs["mri_folder"] == "":
            out += "Please select an input folder for MRI scans\n"
            ok = False

        if kwargs["cbct_folder"] == "":
            out += "Please select an input folder for CBCT scans\n"
            ok = False
            
        if kwargs["cbct_label2"] == "":
            out += "Please select an input folder for CBCT segmentation\n"
            ok = False
            
        if kwargs["normalization"] == "":
            out += "Please select some values for the normalization\n"
            ok = False
            
        if out == "":
            out = None

        return ok,out
    
    def Process(self, **kwargs):
        list_process=[]
    
        MRI2CBCT_RESAMPLE_REG = slicer.modules.mri2cbct_reg
        parameter_mri2cbct_reg = {
            "folder_general": kwargs["folder_general"],
            "mri_folder": kwargs["mri_folder"],
            "cbct_folder": kwargs["cbct_folder"],
            "cbct_label2": kwargs["cbct_label2"],
            "normalization" : kwargs["normalization"],
            "tempo_fold" : kwargs["tempo_fold"]
        }
        
        list_process.append(
            {
                "Process": MRI2CBCT_RESAMPLE_REG,
                "Parameter": parameter_mri2cbct_reg,
                "Module": "Resample files",
            }
        )
           
        return list_process
    
  
