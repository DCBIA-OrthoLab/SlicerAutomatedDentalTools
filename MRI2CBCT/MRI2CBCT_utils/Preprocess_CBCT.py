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


class Process_CBCT(Method):
    def __init__(self, widget):
        super().__init__(widget)
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        documents = qt.QStandardPaths.writableLocation(documentsLocation)
        self.tempAMASSS_folder = os.path.join(
            documents, slicer.app.applicationName + "_temp_AMASSS"
        )

    def getGPUUsage(self):
        if platform.system() == "Darwin":
            return 1
        else:
            return 5

    def NumberScan(self, scan_folder_t1: str, scan_folder_t2: str):
        return len(GetDictPatients(scan_folder_t1, scan_folder_t2))



    def TestModel(self, model_folder: str,lineEdit:str) -> str:
        if lineEdit == "lineEditSegCBCT":
            if len(super().search(model_folder, "pth")["pth"]) == 0:
                return "Folder must have models for mask segmentation"
            else:
                return None


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

        if kwargs["input_t1_folder"] == "":
            out += "Please select an input folder for CBCT scans\n"
            ok = False

        if kwargs["folder_output"] == "":
            out += "Please select an output folder\n"
            ok = False

        if kwargs["model_folder_1"] == "":
            out += "Please select a folder for segmentation models\n"
            ok = False
            
        if out == "":
            out = None

        return ok,out

    def getModelUrl(self):
        return {
            "Segmentation": {
                "Full Face Models": "https://github.com/lucanchling/AMASSS_CBCT/releases/download/v1.0.2/AMASSS_Models.zip",
                "Mask Models": "https://github.com/lucanchling/AMASSS_CBCT/releases/download/v1.0.2/Masks_Models.zip",
            },
            "Orientation": {
                "PreASO": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_preASOmodels/PreASOModels.zip",
                "Occlusal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Occlusal_Midsagittal_Plane.zip",
                "Frankfurt Horizontal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Frankfurt_Horizontal_Midsagittal_Plane.zip",
            },
        }


    def Process(self, **kwargs):
        centered_T1 = kwargs["folder_output"] + "CBCT_Center"
        centered_T1 = os.path.join(kwargs["folder_output"], "_CBCT_Center")
        parameter_pre_aso = {
            "input": kwargs["input_t1_folder"],
            "output_folder": centered_T1,
            "model_folder": os.path.join(
                kwargs["slicerDownload"], "Models", "Orientation", "PreASO"
            ),
            "SmallFOV": False,
            "temp_folder": "../",
            "DCMInput": kwargs["isDCMInput"],
        }

        PreOrientProcess = slicer.modules.pre_aso_cbct
        list_process = [
            {
                "Process": PreOrientProcess,
                "Parameter": parameter_pre_aso,
                "Module": "PRE_ASO_CBCT",
                # "Display": DisplayASOCBCT(nb_scan),
            }
        ]



        # AMASSS PROCESS - SEGMENTATION
        AMASSSProcess = slicer.modules.amasss_cli
        parameter_amasss_seg_t1 = {
            "inputVolume": centered_T1,
            "modelDirectory": kwargs["model_folder_1"],
            "highDefinition": False,
            "skullStructure": "CB",
            "merge": "MERGE" if kwargs["merge_seg"] else "SEPARATE",
            "genVtk": True,
            "save_in_folder": True,
            "output_folder": os.path.join(kwargs["folder_output"],'CBCT_Segmentation'),
            "precision": 50,
            "vtk_smooth": 5,
            "prediction_ID": "Pred",
            "gpu_usage": self.getGPUUsage(),
            "cpu_usage": 1,
            "temp_fold": self.tempAMASSS_folder,
            "SegmentInput": False,
            "DCMInput": False,
        }
        
        list_process.append(
            {
                "Process": AMASSSProcess,
                "Parameter": parameter_amasss_seg_t1,
                "Module": "AMASSS_CLI",
                # "Display": DisplayAMASSS(nb_scan, len(full_seg_struct)),
            }
        )
           
        return list_process
