from ALI_Method.Method import Method
from ALI_Method.Progress import (
    DisplayALICBCT,
)
import os

# --- LOGGING CONFIGURATION ---
from ADTLib.logging_setup import get_logger

logger = get_logger("ALI_CBCT_Process")


import slicer
import platform
from ADTLib.model_registry import ADT_MODELS
from ADTLib.model_registry import (
    AMASSS_CBCT_UPSTREAM, AMASSS_TEST_SCAN, ASO_CBCT_GOLD)
import re


class Auto_CBCT(Method):
    def __init__(self, widget):
        super().__init__(widget)

    def getGPUUsage(self):
        if platform.system() == "Darwin":
            return 1
        else:
            return 5

    def NumberScan(self, scan_folder: str):
        if not scan_folder:
            return 0
        scan_extensions = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        files = []
        for ext in scan_extensions:
            files.extend(self.search(scan_folder, ext)[ext])
        return len(files)
    
    def NumberLandmark(self, landmarks: str):
        cleaned = re.sub(r"[\[\]\"']", "", landmarks)
        teeth_list = re.split(r"[,\s]+", cleaned)
        teeth_list = [t for t in teeth_list if t]
        return len(teeth_list)

    def getReferenceList(self):
        return {
            "Occlusal and Midsagittal Plane": f"{ASO_CBCT_GOLD}/Occlusal_Midsagittal_Plane.zip",
            "Frankfurt Horizontal and Midsagittal Plane": f"{ASO_CBCT_GOLD}/Frankfurt_Horizontal_Midsagittal_Plane.zip",
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

    def TestModel(self, model_folder: str, line_edit_name) -> str:

        if line_edit_name == "lineEditModelPath":
            if len(super().search(model_folder, "pth")["pth"]) == 0:
                return "Folder must have models for mask segmentation"
            else:
                return None

    def TestProcess(self, request) -> str:
        out = ""
        
        if request.input_folder == "":
            out += "Please select an input folder for T1 scans\n"

        if request.output_folder == "":
            out += "Please select an output folder\n"

        if request.model_folder == "":
            out += "Please select a folder for segmentation models\n"

        if out == "":
            out = None

        return out

    def getModelUrl(self):
        return {
            "Segmentation": {
                "Full Face Models": f"{AMASSS_CBCT_UPSTREAM}/AMASSS_Models.zip",
                "Mask Models": f"{AMASSS_CBCT_UPSTREAM}/Masks_Models.zip",
            },
            
            "Landmark": {
                "Cranial Base": f"{ADT_MODELS}/Cranial_Base.zip",
                "Lower Bones 1": f"{ADT_MODELS}/Lower_Bones_1.zip",
                "Lower Bones 2": f"{ADT_MODELS}/Lower_Bones_2.zip",
                "Lower Left Teeth": f"{ADT_MODELS}/Lower_Left_Teeth.zip",
                "Lower_Right_Teeth": f"{ADT_MODELS}/Lower_Right_Teeth.zip",
                "Upper Bones v2": f"{ADT_MODELS}/Upper_Bones_v2.zip",
                "Upper Left Teeth v2": f"{ADT_MODELS}/Upper_Left_Teeth_v2.zip",
                "Upper Right Teeth v2": f"{ADT_MODELS}/Upper_Right_Teeth_v2.zip",
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

    def TranslateModels(self, liste_models, mask=False):
        dic_translate = {
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
        for i, model in enumerate(liste_models):
            if i < len(liste_models) - 1:
                if mask:
                    translate += dic_translate["Masks"][model] + " "
                else:
                    translate += dic_translate["Models"][model] + " "
            else:
                if mask:
                    translate += dic_translate["Masks"][model]
                else:
                    translate += dic_translate["Models"][model]

        return translate

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None

    def getTestFileList(self):
        """The test scan: a .nii.gz, not an archive."""
        return (
            "ALI_test_scan",
            AMASSS_TEST_SCAN,
        )

    def Process(self, request):
        
        path_tmp = slicer.util.tempDirectory()
        os.makedirs(path_tmp, exist_ok=True)
        os.makedirs(request.output_folder, exist_ok=True)
        
        parameter_ali = {
            "input": request.input_folder,
            "dir_models": request.model_folder,
            "lm_type": request.lm_type.split(" "),
            "output_dir": request.output_folder,
            "temp_fold": path_tmp,
            "DCMInput": request.is_dicom_input,
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
            
        }
        
        logger.debug("=" * 70)
        logger.debug(f"Processing parameters: {parameter_ali}")
        logger.debug("=" * 70)
        
        ali_process = slicer.modules.ali_cbct
        
        number_scan = self.NumberScan(
            request.input_folder
        )
        number_lm = self.NumberLandmark(
            request.lm_type
        )
        
        
        list_process = [
            {
                "Process": ali_process,
                "Parameter": parameter_ali,
                "Module": "ALI_CBCT",
                "Display": DisplayALICBCT(
                    number_lm, number_scan
                ),
            },
        ]

        return list_process