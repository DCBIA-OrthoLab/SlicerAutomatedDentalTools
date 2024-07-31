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


class Process_MRI(Method):
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
        
        
    def TestProcess(self, **kwargs) -> str:
        out = ""
        ok = True

        if kwargs["input_folder"] == "":
            out += "Please select an input folder for CBCT scans\n"
            ok = False

        if kwargs["output_folder"] == "":
            out += "Please select an output folder\n"
            ok = False

        if kwargs["direction"] == "":
            out += "Please select a new direction for X,Y and Z\n"
            ok = False
            
        if out == "":
            out = None

        return ok,out
    
    def Process(self, **kwargs):
        list_process=[]
        # MRI2CBCT_ORIENT_CENTER_MRI
        MRI2CBCT_ORIENT_CENTER_MRI = slicer.modules.mri2cbct_orient_center_mri
        parameter_mri2cbct_orient_center_mri = {
            "input_folder": kwargs["input_folder"],
            "direction": kwargs["direction"],
            "output_folder": kwargs["output_folder"]
        }
        
        list_process.append(
            {
                "Process": MRI2CBCT_ORIENT_CENTER_MRI,
                "Parameter": parameter_mri2cbct_orient_center_mri,
                "Module": "Orientation and Centering of the MRI",
            }
        )
           
        return list_process
    
  
