from .Method import Method
from .utils_CBCT import GetDictPatients, GetPatients
import os, sys

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import urllib.request

from glob import iglob
import slicer
import time
import qt
import platform
import re


class TMJ_CROP_MRI2CBCT(Method):
    def __init__(self, widget):
        super().__init__(widget)
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documentsLocation)

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
    
    def TestModel(self, model_folder: str):
        checkpoint_path = Path(model_folder) / "fold_0" / "checkpoint_final.pth"
        dataset_json_path = Path(model_folder) / "dataset.json"
        plans_json_path = Path(model_folder) / "plans.json"

        if not checkpoint_path.exists():
            return False, f"Checkpoint file not found: {checkpoint_path}"
        if not dataset_json_path.exists():
            return False, f"Dataset JSON file not found: {dataset_json_path}"
        if not plans_json_path.exists():
            return False, f"Plans JSON file not found: {plans_json_path}"

        return True, ""
        
    def TestProcess(self, **kwargs) -> str:
        out = ""
        ok = True
        
        if kwargs["cbct_folder"] == "":
            out += "Please select an input folder for CBCT scans\n"
            ok = False
            
        if kwargs["mri_folder"] == "":
            out += "Please select an input folder for MRI scans\n"
            ok = False
            
        if kwargs["seg_folder"] == "":
            out += "Please select an input folder for Segmentation scans\n"
            ok = False
            
        if kwargs["output_folder"] == "":
            out += "Please select an output folder\n"
            ok = False
        
        if kwargs["model_folder"] == "":
            out += "Please select a folder for nnUNet model\n"
            ok = False
            
        if out == "":
            out = None

        return ok,out
    
    def Process(self, **kwargs):
        path_tmp = slicer.util.tempDirectory()
        os.makedirs(path_tmp, exist_ok=True)
        os.makedirs(kwargs["output_folder"], exist_ok=True)
        
        list_process=[]
    
        MRI2CBCT_TMJ_CROP = slicer.modules.mri2cbct_tmj_crop
        parameter_mri2cbct_tmj_crop = {
            "cbct_folder": kwargs["cbct_folder"],
            "mri_folder": kwargs["mri_folder"],
            "seg_folder": kwargs["seg_folder"],
            "output_folder": kwargs["output_folder"],
            "model_folder": kwargs["model_folder"],
            "tmp_folder": path_tmp
        }
        
        list_process.append(
            {
                "Process": MRI2CBCT_TMJ_CROP,
                "Parameter": parameter_mri2cbct_tmj_crop,
                "Module": "MRI2CBCT TMJ Crop",
            }
        )
        
        return list_process
    
  