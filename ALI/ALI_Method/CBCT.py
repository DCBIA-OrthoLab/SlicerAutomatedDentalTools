from ALI_Method.Method import Method
from ALI_Method.Progress import (
    DisplayALICBCT,
)
import os, sys

import SimpleITK as sitk
import numpy as np

from glob import iglob
import slicer
import time
import qt
import platform


class Auto_CBCT(Method):
    def __init__(self, widget):
        super().__init__(widget)

    def getGPUUsage(self):
        if platform.system() == "Darwin":
            return 1
        else:
            return 5

    def NumberScan(self, scan_folder: str):
        scan_extensions = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        files = []
        for ext in scan_extensions:
            files.extend(self.search(scan_folder, ext)[ext])
        return len(files)
    
    def NumberLandmark(self, landmarks: str):
        import re
        cleaned = re.sub(r"[\[\]\"']", "", landmarks)
        teeth_list = re.split(r"[,\s]+", cleaned)
        teeth_list = [t for t in teeth_list if t]
        return len(teeth_list)

    def getReferenceList(self):
        return {
            "Occlusal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Occlusal_Midsagittal_Plane.zip",
            "Frankfurt Horizontal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Frankfurt_Horizontal_Midsagittal_Plane.zip",
        }

    def TestReference(self, ref_folder: str):
        out = None
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        lm_extension = [".json"]

        if self.NumberScan(ref_folder) == 0:
            out = "The selected folder must contain scans"

        if self.NumberScan(ref_folder) > 1:
            out = "The selected folder must contain only 1 case"

        return None

    def TestModel(self, model_folder: str, lineEditName) -> str:

        if lineEditName == "lineEditModelPath":
            if len(super().search(model_folder, "pth")["pth"]) == 0:
                return "Folder must have models for mask segmentation"
            else:
                return None

    def TestProcess(self, **kwargs) -> str:
        out = ""
        
        if kwargs["input_folder"] == "":
            out += "Please select an input folder for T1 scans\n"

        if kwargs["output_dir"] == "":
            out += "Please select an output folder\n"

        if kwargs["dir_models"] == "":
            out += "Please select a folder for segmentation models\n"

        if out == "":
            out = None

        return out

    def getModelUrl(self):
        return {
            "Segmentation": {
                "Full Face Models": "https://github.com/lucanchling/AMASSS_CBCT/releases/download/v1.0.2/AMASSS_Models.zip",
                "Mask Models": "https://github.com/lucanchling/AMASSS_CBCT/releases/download/v1.0.2/Masks_Models.zip",
            },
            
            "Landmark": {
                "Cranial Base": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Cranial_Base.zip",
                "Lower Bones 1": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Bones_1.zip",
                "Lower Bones 2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Bones_2.zip",
                "Lower Left Teeth": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Left_Teeth.zip",
                "Lower_Right_Teeth": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Right_Teeth.zip",
                "Upper Bones v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Bones_v2.zip",
                "Upper Left Teeth v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Left_Teeth_v2.zip",
                "Upper Right Teeth v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Right_Teeth_v2.zip",
            }
        }

    def getALIModelList(self):
        return (
            "ALIModels",
            "https://github.com/lucanchling/ALI_CBCT/releases/download/models_v01/",
        )

    def ProperPrint(self, notfound_list):
        dic = {"scan": "Scan", "seg": "Segmentation"}
        return "\n".join(dic.get(key, key) for key in notfound_list)

    def TestScan(self, scan_folder: str):
        if self.NumberScan(scan_folder) == 0:
            return "Please select a folder with valid scan files"
        return None


    def DicLandmark(self):
        return {
            "Regions of Reference for Registration": [
                "Cranial Base",
                "Mandible",
                "Maxilla",
            ],
            "AMASSS Segmentation": [
                "Cranial Base",
                "Cervical Vertebra",
                "Mandible",
                "Maxilla",
                "Skin",
                "Upper Airway",
            ],
        }

    def TranslateModels(self, listeModels, mask=False):
        dicTranslate = {
            "Models": {
                "Mandible": "MAND",
                "Maxilla": "MAX",
                "Cranial Base": "CB",
                "Cervical Vertebra": "CV",
                "Root Canal": "RC",
                "Mandibular Canal": "MCAN",
                "Upper Airway": "UAW",
                "Skin": "SKIN",
            },
            "Masks": {
                "Cranial Base": "CBMASK",
                "Mandible": "MANDMASK",
                "Maxilla": "MAXMASK",
            },
        }

        translate = ""
        for i, model in enumerate(listeModels):
            if i < len(listeModels) - 1:
                if mask:
                    translate += dicTranslate["Masks"][model] + " "
                else:
                    translate += dicTranslate["Models"][model] + " "
            else:
                if mask:
                    translate += dicTranslate["Masks"][model]
                else:
                    translate += dicTranslate["Models"][model]

        return translate

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None

    def getTestFileList(self):
        return (
            "ALI_test_scan",
            "https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/MG_test_scan.nii.gz",
        )

    def Process(self, **kwargs):
        
        path_tmp = slicer.util.tempDirectory()
        os.makedirs(path_tmp, exist_ok=True)
        os.makedirs(kwargs["output_dir"], exist_ok=True)
        
        parameter_ali = {
            "input": kwargs["input_folder"],
            "dir_models": kwargs["dir_models"],
            "lm_type": kwargs["lm_type"].split(" "),
            "output_dir": kwargs["output_dir"],
            "temp_fold": path_tmp,
            "DCMInput": kwargs["DCMInput"],
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
            
        }
        
        print('-' * 70)
        print("parameter ali : ", parameter_ali)
        print('-' * 70)
        
        ALIProcess = slicer.modules.ali_cbct
        
        number_scan = self.NumberScan(
            kwargs["input_folder"]
        )
        number_lm = self.NumberLandmark(
            kwargs["lm_type"]
        )
        
        
        list_process = [
            {
                "Process": ALIProcess,
                "Parameter": parameter_ali,
                "Module": "ALI_CBCT",
                "Display": DisplayALICBCT(
                    number_lm, number_scan
                ),
            },
        ]

        return list_process