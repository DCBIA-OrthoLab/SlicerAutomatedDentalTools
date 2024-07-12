from utils.Method import Method
from utils.utils_CBCT import GetDictPatients, GetPatients
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
        self.tempALI_folder = os.path.join(
            documents, slicer.app.applicationName + "_temp_ALI"
        )

    def getGPUUsage(self):
        if platform.system() == "Darwin":
            return 1
        else:
            return 5

    def NumberScan(self, scan_folder_t1: str, scan_folder_t2: str):
        return len(GetDictPatients(scan_folder_t1, scan_folder_t2))

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

    def TestCheckbox(self, dic_checkbox):
        list_landmark = self.CheckboxisChecked(dic_checkbox)[
            "Regions of Reference for Registration"
        ]

        out = None
        if len(list_landmark) == 0:
            out = "Please select a Registration Type\n"
        return out

    def TestModel(self, model_folder: str, lineEditName) -> str:

        if lineEditName == "lineEditSegCBCT":
            if len(super().search(model_folder, "pth")["pth"]) == 0:
                return "Folder must have models for mask segmentation"
            else:
                return None

        # if lineEditName == 'lineEditModelAli':
        #     if len(super().search(model_folder,'pth')['pth']) == 0:
        #         return 'Folder must have ALI models files'
        #     else:
        #         return None

    def TestProcess(self, **kwargs) -> str:
        out = ""

        testcheckbox = self.TestCheckbox(kwargs["dic_checkbox"])
        if testcheckbox is not None:
            out += testcheckbox

        if kwargs["input_t1_folder"] == "":
            out += "Please select an input folder for T1 scans\n"

        if kwargs["input_t2_folder"] == "":
            out += "Please select an input folder for T2 scans\n"

        if kwargs["folder_output"] == "":
            out += "Please select an output folder\n"

        if kwargs["add_in_namefile"] == "":
            out += "Please select an extension for output files\n"

        if kwargs["model_folder_1"] == "":
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
            "Orientation": {
                "PreASO": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_preASOmodels/PreASOModels.zip",
                "Occlusal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Occlusal_Midsagittal_Plane.zip",
                "Frankfurt Horizontal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Frankfurt_Horizontal_Midsagittal_Plane.zip",
            },
        }

    def getALIModelList(self):
        return (
            "ALIModels",
            "https://github.com/lucanchling/ALI_CBCT/releases/download/models_v01/",
        )

    def ProperPrint(self, notfound_list):
        dic = {
            "scanT1": "T1 scan",
            "scanT2": "T2 scan",
            "segT1": "T1 segmentation",
            "segT2": "T2 segmentation",
        }
        out = ""
        if "scanT1" in notfound_list and "scanT2" in notfound_list:
            out += "T1 and T2 scans\n"
        elif "segT1" in notfound_list and "segT2" in notfound_list:
            out += "T1 and T2 segmentations\n"
        else:
            for notfound in notfound_list:
                out += dic[notfound] + " "
        return out

    def TestScan(
        self,
        scan_folder_t1: str,
        scan_folder_t2: str,
        liste_keys=["scanT1", "scanT2", "segT1"],
    ):
        out = ""
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        if self.NumberScan(scan_folder_t1, scan_folder_t2) == 0:
            return "Please Select folder with scans"

        patients = GetDictPatients(scan_folder_t1, scan_folder_t2)
        for patient, data in patients.items():
            not_found = [key for key in liste_keys if key not in data.keys()]
            if len(not_found) != 0:
                out += (
                    f"Patient {patient} does not have {self.ProperPrint(not_found)}\n"
                )

        if out == "":  # If no errors
            out = None

        return out

    def GetSegmentationLabel(self, seg_folder):
        seg_label = []
        patients = GetPatients(seg_folder)
        seg_path = patients[list(patients.keys())[0]]["segT1"]
        seg = sitk.ReadImage(seg_path)
        seg_array = sitk.GetArrayFromImage(seg)
        labels = np.unique(seg_array)
        for label in labels:
            if label != 0 and label not in seg_label:
                seg_label.append(label)
        return seg_label

    def CheckboxisChecked(self, diccheckbox: dict, in_str=False):
        listchecked = {key: [] for key in diccheckbox.keys()}
        for key, checkboxs in diccheckbox.items():
            for checkbox in checkboxs:
                if checkbox.isChecked():
                    listchecked[key] += [checkbox.text]

        return listchecked

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
            "Semi-Automated",
            "https://github.com/lucanchling/Areg_CBCT/releases/download/TestFiles/SemiAuto.zip",
        )

    def Process(self, **kwargs):
        centered_T1 = kwargs["input_t1_folder"] + "_Center"
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
                "Module": "Centering CBCT",
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
            "output_folder": kwargs["folder_output"],
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
                "Module": "AMASSS_CBCT Segmentation of CBCT",
                # "Display": DisplayAMASSS(nb_scan, len(full_seg_struct)),
            }
        )
           
        return list_process
