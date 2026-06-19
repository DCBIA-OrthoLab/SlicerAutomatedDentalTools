from AREG_Method.Method import Method
from AREG_Method.Progress import DisplayAREGIOSCBCT, DisplayALICBCT,DisplayASOIOS,DisplayASOCBCT,DisplayCrownSeg,DisplayALIIOS
import webbrowser
import os
import slicer
import json
import time
import qt
import csv
import platform

import logging
import sys
# ===== Logging Configuration =====
logger = logging.getLogger("AREG_Method_IOSCBCT")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class IOSCBCT(Method):
    def __init__(self, widget):
        super().__init__(widget)

    def NumberScan(self, scan_folder_t1: str, scan_folder_t2: str):
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        dic = super().search(scan_folder_t2, scan_extension)
        lenscan = 0
        for key in scan_extension:
            lenscan += len(dic[key])
        return lenscan

    def PatientScanLandmark(self, dic, scan_extension, lm_extension):
        patients = {}

        for extension, files in dic.items():
            for file in files:
                file_name = os.path.basename(file).split(".")[0]
                patient = (
                    file_name.split("_scan")[0].split("_Scanreg")[0].split("_lm")[0]
                )

                if patient not in patients.keys():
                    patients[patient] = {"dir": os.path.dirname(file), "lmrk": []}
                if extension in scan_extension:
                    patients[patient]["scan"] = file
                if extension in lm_extension:
                    patients[patient]["lmrk"].append(file)

        return patients

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

        return out

    def TestCheckbox(self, dic_checkbox):
        list_landmark = self.CheckboxisChecked(dic_checkbox)
        out = None
        if len(list_landmark) < 3:
            out = "Please select at least 3 landmarks\n"
        return out

    def TestModel(self, model_folder: str, lineEditName) -> str:

        if lineEditName == "lineEditModelSegOr":
            if len(super().search(model_folder, "ckpt")["ckpt"]) == 0:
                return "Folder must have Pre ASO models files"
            else:
                return None

        if lineEditName == "lineEditModelAli":
            if len(super().search(model_folder, "pth")["pth"]) == 0:
                return "Folder must have ALI models files"
            else:
                return None

    def TestProcess(self, **kwargs) -> str:
        out = ""

        if kwargs["input_t1_folder"] == "":
            out += "Please select an input folder for IOS scans\n"

        if kwargs["input_t2_folder"] == "":
            out += "Please select an input folder for CBCT scans\n"
            
        if kwargs["folder_output"] == "":
            out += "Please select an output folder\n"

        if kwargs["add_in_namefile"] == "":
            out += "Please select a suffix\n"

        if out == "":
            out = None

        return out

    def getSegOrModelList(self):
        return (
            "PreASOModels",
            "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_preASOmodels/PreASOModels.zip",
        )

    def getALIModelList(self):
        return (
            "ALIModels",
            "https://github.com/lucanchling/ALI_CBCT/releases/download/models_v01/",
        )

    def DicLandmark(self):
        return {"Landmark": ["Cranial Base", "Mandible", "Maxilla"]}

    def Sugest(self):
        return ["Ba", "S", "N", "RPo", "LPo", "ROr", "LOr"]

    def CheckboxisChecked(self, diccheckbox: dict, in_str=False):
        out = ""
        listchecked = []
        if not len(diccheckbox) == 0:
            for checkboxs in diccheckbox.values():
                for checkbox in checkboxs:
                    if checkbox.isChecked():
                        listchecked.append(checkbox.text)
        if in_str:
            listchecked_str = ""
            for i, lm in enumerate(listchecked):
                if i < len(listchecked) - 1:
                    listchecked_str += lm + " "
                else:
                    listchecked_str += lm
            return listchecked_str

        return listchecked


class Semi_IOSCBCT(IOSCBCT):
    def getTestFileList(self):
        return (
            "Semi-Automated-Registration",
            "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/AREG_IOSCBCT/TestFile.zip",
        )

    def TestScan(self,scan_folder_t1: str,scan_folder_t2: str,mask_folder_t1: str = None) -> str:
        return None

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None
    
    def TestProcess(self, **kwargs) -> str:
        out = ""

        if kwargs["input_t1_folder"] == "":
            out += "Please select an input folder for IOS scans\n"

        if kwargs["input_t2_folder"] == "":
            out += "Please select an input folder for CBCT scans\n"
            
        if kwargs["folder_output"] == "":
            out += "Please select an output folder\n"

        if kwargs["model_folder_2"] == "":
            out += "Please select a CBCT Landmarks model folder\n"

        if kwargs["model_folder_3"] == "":
            out += "Please select an IOS Landmarks model folder\n"

        if kwargs["add_in_namefile"] == "":
            out += "Please select a suffix\n"

        if out == "":
            out = None

        return out
    
    def getModelUrl(self):
        return {
            "CBCT": {
                "Cranial Base": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Cranial_Base.zip",
                "Lower Bones 1": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Bones_1.zip",
                "Lower Bones 2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Bones_2.zip",
                "Lower Left Teeth": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Left_Teeth.zip",
                "Lower_Right_Teeth": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Right_Teeth.zip",
                "Upper Bones v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Bones_v2.zip",
                "Upper Left Teeth v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Left_Teeth_v2.zip",
                "Upper Right Teeth v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Right_Teeth_v2.zip",
        },
            "IOS": "https://github.com/baptistebaquero/ALIDDM/releases/download/v1.0.3/Models.zip",
        }
    
    def getReferenceList(self):
        return None
    
    def is_wsl(self):
        return platform.system() == "Linux" and "microsoft" in platform.release().lower()
    
    def create_csv(self, input_dir, name_csv):
        file_path = os.path.abspath(__file__)
        folder_path = os.path.dirname(file_path)
        csv_file = os.path.join(folder_path, f"{name_csv}.csv")
        with open(csv_file, 'w', newline='') as fichier:
            writer = csv.writer(fichier)
            writer.writerow(["surf"])

            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".vtk") or file.endswith(".stl"):
                        if platform.system() != "Windows" and not self.is_wsl():
                            writer.writerow([os.path.join(root, file)])
                        else:
                            norm_file_path = os.path.normpath(os.path.join(root, file))
                            writer.writerow([self.windows_to_linux_path(norm_file_path)])
        return csv_file

    def Process(self, **kwargs):

        nb_scan = self.NumberScan(kwargs["input_t1_folder"],kwargs["input_t2_folder"])
        
        slicer_path = slicer.app.applicationDirPath()
        dentalmodelseg_path = os.path.join(slicer_path,"..","lib","Python","bin","dentalmodelseg")

        surf = "None"
        input_csv = "None"
        vtk_folder = "None"
        if os.path.isfile(kwargs["input_t1_folder"]):
            extension = os.path.splitext(self.input)[1]
            if extension == ".vtk" or extension == ".stl":
              surf = kwargs["input_t1_folder"]
              
        elif os.path.isdir(kwargs["input_t1_folder"]):
          input_csv = self.create_csv(kwargs["input_t1_folder"],"liste_csv_file")
          vtk_folder = kwargs["input_t1_folder"]

        seg_ios_folder_path = os.path.join(kwargs["folder_output"],"Seg IOS")
        os.makedirs(seg_ios_folder_path, exist_ok=True)

        parameter_seg = {
            "surf": surf,
            "input_csv": input_csv,
            "out": seg_ios_folder_path,
            "overwrite": "0",
            "model": "latest",
            "crown_segmentation": "0",
            "array_name": "Universal_ID",
            "fdi": 0,
            "suffix": "Seg",
            "vtk_folder": vtk_folder,
            "dentalmodelseg_path": dentalmodelseg_path
        }

        logger.info(f"Parameter CrownSegmentation :  {parameter_seg}")
        SegProcess_IOS = slicer.modules.crownsegmentationcli
        
        list_process = [
            {
                "Process": SegProcess_IOS,
                "Parameter": parameter_seg,
                "Module": "CrownSegmentationcli",
                "Display": DisplayCrownSeg(
                    nb_scan, kwargs["logPath"],"Segmentation Patient"
                ),
            }]
        
        temp_ali_cbct_folder = slicer.util.tempDirectory()
        cbct_landmarks_folder_path = os.path.join(kwargs["folder_output"],"CBCT Landmarks")
        os.makedirs(cbct_landmarks_folder_path, exist_ok=True)

        parameter_ali_cbct = {
            "input": kwargs["input_t2_folder"],
            "dir_models": kwargs["model_folder_2"],
            "lm_type": "'LL1O','LL3O','LL6O','LR1O','LR3O','LR6O','UL1O','UL3O','UL6O','UR1O','UR3O','UR6O'",
            "output_dir": cbct_landmarks_folder_path,
            "temp_fold": temp_ali_cbct_folder,
            "DCMInput": kwargs["isDCMInput"],
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
        }
        
        logger.info(f"Parameter ALI_CBCT :  {parameter_ali_cbct}")
        ALIProcess_CBCT = slicer.modules.ali_cbct

        list_process.append(
            {
                "Process": ALIProcess_CBCT,
                "Parameter": parameter_ali_cbct,
                "Module": "ALI_CBCT",
                "Display": DisplayALICBCT(
                    12, nb_scan
                ),
            },
        )
        
        temp_ali_ios_folder = os.path.join(slicer.util.tempDirectory(), "process.log")
        ios_landmarks_folder_path = os.path.join(kwargs["folder_output"],"IOS Landmarks")
        os.makedirs(ios_landmarks_folder_path, exist_ok=True)

        parameter_ali_ios = {
            "input": seg_ios_folder_path,
            "dir_models": kwargs["model_folder_3"],
            "lm_type": "'O'",
            "teeth": "LL1 LL3 LL6 LR1 LR3 LR6 UL1 UL3 UL6 UR1 UR3 UR6'",
            "output_dir": ios_landmarks_folder_path,
            "image_size": "224",
            "blur_radius": "0",
            "faces_per_pixel": "1",
            "log_path": temp_ali_ios_folder
        }

        logger.info(f"Parameter ALI_IOS :  {parameter_ali_ios}")

        ALIProcess_IOS = slicer.modules.ali_ios
    
        list_process.append({
                "Process": ALIProcess_IOS,
                "Parameter": parameter_ali_ios,
                "Module": "ALI_IOS",
                "Display": DisplayALIIOS(
                    12, nb_scan
                ),
            })
        
        registered_ios_folder_path = os.path.join(kwargs["folder_output"],"Registered IOS")
        os.makedirs(registered_ios_folder_path, exist_ok=True)

        parameter_areg_IOSCBCT = {
            "IOS_folder": os.path.join(seg_ios_folder_path,"liste_csv_file_Seg"),
            "CBCT_folder": kwargs["input_t2_folder"],
            "IOS_lm_folder": ios_landmarks_folder_path,
            "CBCT_lm_folder": cbct_landmarks_folder_path,
            "output": registered_ios_folder_path
        }
        logger.info(f"Parameter reg: {parameter_areg_IOSCBCT}")

        AREGProcess = slicer.modules.areg_ioscbct

        list_process.append(
            {
                "Process": AREGProcess,
                "Parameter": parameter_areg_IOSCBCT,
                "Module": "AREG IOSCBCT",
                "Display": DisplayAREGIOSCBCT(0),
            }
        )
        return list_process
    

class Reg_IOSCBCT(IOSCBCT):
    def getTestFileList(self):
        return (
            "Registration",
            "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/AREG_IOSCBCT/RegTestFiles.zip",
        )

    def TestScan(self,scan_folder_t1: str,scan_folder_t2: str,mask_folder_t1: str = None) -> str:
        return None

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None
    
    def TestProcess(self, **kwargs) -> str:
        out = ""

        if kwargs["input_t1_folder"] == "":
            out += "Please select an input folder for IOS scans\n"

        if kwargs["input_t2_folder"] == "":
            out += "Please select an input folder for CBCT scans\n"

        if kwargs["input_t1_mask"] == "":
            out += "Please select an input folder for IOS Landmarks\n"

        if kwargs["input_t2_landmarks"] == "":
            out += "Please select an input folder for CBCT Landmarks\n"
            
        if kwargs["folder_output"] == "":
            out += "Please select an output folder\n"

        if kwargs["add_in_namefile"] == "":
            out += "Please select a suffix\n"

        if out == "":
            out = None

        return out
    
    def getModelUrl(self):
        return None
    
    def getReferenceList(self):
        return None

    def Process(self, **kwargs):

        parameter_areg_IOSCBCT = {
            "IOS_folder": kwargs["input_t1_folder"],
            "CBCT_folder": kwargs["input_t2_folder"],
            "IOS_lm_folder": kwargs["input_t1_mask"],
            "CBCT_lm_folder": kwargs["input_t2_landmarks"],
            "output": kwargs["folder_output"]
        }
        logger.info(f"Parameter reg: {parameter_areg_IOSCBCT}")

        AREGProcess = slicer.modules.areg_ioscbct

        list_process = [
            {
                "Process": AREGProcess,
                "Parameter": parameter_areg_IOSCBCT,
                "Module": "AREG IOSCBCT",
                "Display": DisplayAREGIOSCBCT(0),
            }
        ]
        return list_process


class Auto_IOSCBCT(IOSCBCT):
    def getTestFileList(self):
        return (
            "Fully-Automated-Registration",
            "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/AREG_IOSCBCT/TestFile.zip",
        )

    def TestScan(self,scan_folder_t1: str,scan_folder_t2: str,mask_folder_t1: str = None) -> str:
        return None

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None
    
    def TestProcess(self, **kwargs) -> str:
        out = ""

        if kwargs["input_t1_folder"] == "":
            out += "Please select an input folder for IOS scans\n"

        if kwargs["input_t2_folder"] == "":
            out += "Please select an input folder for CBCT scans\n"
            
        if kwargs["folder_output"] == "":
            out += "Please select an output folder\n"

        if kwargs["model_folder_1"] == "":
            out += "Please select an Orientation model folder\n"

        if kwargs["model_folder_2"] == "":
            out += "Please select a CBCT Landmarks model folder\n"

        if kwargs["model_folder_3"] == "":
            out += "Please select an IOS Landmarks model folder\n"

        if kwargs["add_in_namefile"] == "":
            out += "Please select a suffix\n"

        if out == "":
            out = None

        return out
    
    def getModelUrl(self):
        return {
            "Orientation": {
                "PreASO": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_preASOmodels/PreASOModels.zip",
                "Occlusal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Occlusal_Midsagittal_Plane.zip",
                "Frankfurt Horizontal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Frankfurt_Horizontal_Midsagittal_Plane.zip",
                "IOS":"https://github.com/HUTIN1/ASO/releases/download/v1.0.0/Gold_file.zip"
            },
            "CBCT": {
                "Cranial Base": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Cranial_Base.zip",
                "Lower Bones 1": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Bones_1.zip",
                "Lower Bones 2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Bones_2.zip",
                "Lower Left Teeth": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Left_Teeth.zip",
                "Lower_Right_Teeth": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Right_Teeth.zip",
                "Upper Bones v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Bones_v2.zip",
                "Upper Left Teeth v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Left_Teeth_v2.zip",
                "Upper Right Teeth v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Right_Teeth_v2.zip",
        },
            "IOS": "https://github.com/baptistebaquero/ALIDDM/releases/download/v1.0.3/Models.zip",
        }
    
    def getReferenceList(self):
        return {
            "Occlusal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Occlusal_Midsagittal_Plane.zip",
            "Frankfurt Horizontal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Frankfurt_Horizontal_Midsagittal_Plane.zip",
        }

    def IsLower(self, folder_path_or_file_list):
        words_lower = ["lower", "_l", "l_", "mandibule", "md"]

        if isinstance(folder_path_or_file_list, str) and os.path.isdir(folder_path_or_file_list):
            list_files = self.search(folder_path_or_file_list, ".vtk", ".stl")
            all_files = list_files[".vtk"] + list_files[".stl"]
        else:
            all_files = folder_path_or_file_list

        for file in all_files:
            name = os.path.basename(file).lower()
            if any(word in name for word in words_lower):
                return True

        return False
    
    def ReferenceLandmarks(self, name_reference):
        correspondance = {
            "Occlusal and Midsagittal Plane": ("IF ANS PNS UR1O UR6O UL6O", 6),
            "Frankfurt Horizontal and Midsagittal Plane": ("N S Ba RPo LPo LOr ROr", 7),
        }

        return correspondance[name_reference]
        
    def format_lm_string(self, lm_str: str) -> str:
        """
        Convert a space-separated string of landmarks into a string format like:
        "'Ba', 'LPo', 'N', 'RPo', 'S', 'LOr', 'ROr'"
        """
        lms = lm_str.strip().split()
        return ", ".join(f"'{lm}'" for lm in lms)
    
    def is_wsl(self):
        return platform.system() == "Linux" and "microsoft" in platform.release().lower()
    
    def create_csv(self, input_dir, name_csv):
        file_path = os.path.abspath(__file__)
        folder_path = os.path.dirname(file_path)
        csv_file = os.path.join(folder_path, f"{name_csv}.csv")
        with open(csv_file, 'w', newline='') as fichier:
            writer = csv.writer(fichier)
            writer.writerow(["surf"])

            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".vtk") or file.endswith(".stl"):
                        if platform.system() != "Windows" and not self.is_wsl():
                            writer.writerow([os.path.join(root, file)])
                        else:
                            norm_file_path = os.path.normpath(os.path.join(root, file))
                            writer.writerow([self.windows_to_linux_path(norm_file_path)])
        return csv_file

    def Process(self, **kwargs):

        nb_scan = self.NumberScan(kwargs["input_t1_folder"],kwargs["input_t2_folder"])

        resample_folder_path = os.path.join(kwargs["folder_output"],"CBCT Resampled")
        os.makedirs(resample_folder_path, exist_ok=True)

        parameter_resample_cbct = {
            "input_folder_MRI": "None",
            "input_folder_T2_MRI": "None",
            "input_folder_CBCT": kwargs["input_t2_folder"],
            "input_folder_T2_CBCT": "None",
            "input_folder_Seg": "None",
            "input_folder_T2_Seg": "None",
            "output_folder": resample_folder_path,
            "resample_size": "None",
            "spacing": [0.3,0.3,0.3],
            "center": "True"
        }

        logger.info(f"Parameter Resample_CBCT : {parameter_resample_cbct}")
        ResampleProcess_CBCT = slicer.modules.mri2cbct_resample_cbct_mri

        
        
        list_process = [
            {
                "Process": ResampleProcess_CBCT,
                "Parameter": parameter_resample_cbct,
                "Module": "CBCT Resampling",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            }
        ]

        pre_aso_cbct_folder_path = os.path.join(kwargs["folder_output"],"PRE ASO CBCT")
        os.makedirs(pre_aso_cbct_folder_path, exist_ok=True)
        temp_pre_aso_folder = slicer.util.tempDirectory()

        parameter_pre_aso_cbct = {
            "input": os.path.join(resample_folder_path,"CBCT"),
            "output_folder": pre_aso_cbct_folder_path,
            "model_folder": os.path.join(kwargs["model_folder_1"], "PreASO"),
            "SmallFOV": False,
            "temp_folder": temp_pre_aso_folder,
            "DCMInput": kwargs["isDCMInput"],
        }

        list_lmrk_str, nb_landmark = self.ReferenceLandmarks(kwargs["OrientReference"])
        temp_ali_cbct_aso_folder = slicer.util.tempDirectory()

        parameter_ali_cbct = {
            "input": pre_aso_cbct_folder_path,
            "dir_models": kwargs["model_folder_2"],
            "lm_type": self.format_lm_string(list_lmrk_str),
            "output_dir": pre_aso_cbct_folder_path,
            "temp_fold": temp_ali_cbct_aso_folder,
            "DCMInput": False,
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
        }

        oriented_cbct_folder_path = os.path.join(kwargs["folder_output"],"Oriented CBCT")
        os.makedirs(oriented_cbct_folder_path, exist_ok=True)

        parameter_semi_aso_cbct = {
            "input": pre_aso_cbct_folder_path,
            "gold_folder": os.path.join(kwargs["model_folder_1"], kwargs["OrientReference"]),
            "output_folder": oriented_cbct_folder_path,
            "add_inname": "Or",
            "list_landmark": list_lmrk_str,
        }

        logger.info(f"Parameter PRE_ASO_CBCT :  {parameter_pre_aso_cbct}")
        logger.info(f"Parameter ALI_CBCT :  {parameter_ali_cbct}")
        logger.info(f"Parameter SEMI_ASO_CBCT : {parameter_semi_aso_cbct}")
        
        PreOrientProcess_CBCT = slicer.modules.pre_aso_cbct
        ALIProcess_CBCT = slicer.modules.ali_cbct
        OrientProcess_CBCT = slicer.modules.semi_aso_cbct
        
        list_process.append(
            {
                "Process": PreOrientProcess_CBCT,
                "Parameter": parameter_pre_aso_cbct,
                "Module": "PRE_ASO_CBCT",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            })
        list_process.append({
                "Process": ALIProcess_CBCT,
                "Parameter": parameter_ali_cbct,
                "Module": "ALI_CBCT",
                "Display": DisplayALICBCT(
                    nb_landmark, nb_scan
                ),
            })
        list_process.append({
                "Process": OrientProcess_CBCT,
                "Parameter": parameter_semi_aso_cbct,
                "Module": "SEMI_ASO_CBCT",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            })
        
        slicer_path = slicer.app.applicationDirPath()
        dentalmodelseg_path = os.path.join(slicer_path,"..","lib","Python","bin","dentalmodelseg")

        surf = "None"
        input_csv = "None"
        vtk_folder = "None"
        if os.path.isfile(kwargs["input_t1_folder"]):
            extension = os.path.splitext(self.input)[1]
            if extension == ".vtk" or extension == ".stl":
              surf = kwargs["input_t1_folder"]
              
        elif os.path.isdir(kwargs["input_t1_folder"]):
          input_csv = self.create_csv(kwargs["input_t1_folder"],"liste_csv_file")
          vtk_folder = kwargs["input_t1_folder"]

        seg_ios_folder_path = os.path.join(kwargs["folder_output"],"Seg IOS")
        os.makedirs(seg_ios_folder_path, exist_ok=True)

        pre_aso_ios_folder_path = os.path.join(kwargs["folder_output"],"PRE ASO IOS")
        os.makedirs(pre_aso_ios_folder_path, exist_ok=True)

        parameter_seg = {
            "surf": surf,
            "input_csv": input_csv,
            "out": seg_ios_folder_path,
            "overwrite": "0",
            "model": "latest",
            "crown_segmentation": "0",
            "array_name": "Universal_ID",
            "fdi": 0,
            "suffix": "Seg",
            "vtk_folder": vtk_folder,
            "dentalmodelseg_path": dentalmodelseg_path
        }
        
        path_error = os.path.join(pre_aso_ios_folder_path, "Error")

        parameter_pre_aso_ios = {
            "input": seg_ios_folder_path,
            "gold_folder": os.path.join(kwargs["model_folder_1"],"IOS"),
            "output_folder": pre_aso_ios_folder_path,
            "add_inname": "Or",
            "list_teeth": "UR6,UR4,UL4,UL6",
            "occlusion": "true" if self.IsLower(kwargs["input_t1_folder"]) else "false",
            "jaw": "Upper",
            "folder_error": path_error,
            "log_path": kwargs["logPath"],
        }

        logger.info(f"Parameter CrownSegmentation :  {parameter_seg}")
        logger.info(f"Parameter PRE_ASO_IOS :  {parameter_pre_aso_ios}")

        PreOrientProcess_IOS = slicer.modules.pre_aso_ios
        SegProcess_IOS = slicer.modules.crownsegmentationcli
        OrientProcess_IOS = slicer.modules.semi_aso_ios
        
        
        list_process.append(
            {
                "Process": SegProcess_IOS,
                "Parameter": parameter_seg,
                "Module": "CrownSegmentationcli",
                "Display": DisplayCrownSeg(
                    nb_scan, kwargs["logPath"],"Segmentation Patient"
                ),
            })
        list_process.append({
                "Process": PreOrientProcess_IOS,
                "Parameter": parameter_pre_aso_ios,
                "Module": "PRE_ASO_IOS",
                "Display": DisplayASOIOS(
                    nb_scan, kwargs["logPath"],"Orient IOS Patient"
                ),
            })
        
        temp_ali_cbct_folder = slicer.util.tempDirectory()
        cbct_landmarks_folder_path = os.path.join(kwargs["folder_output"],"CBCT Landmarks")
        os.makedirs(cbct_landmarks_folder_path, exist_ok=True)

        parameter_ali_cbct_2 = {
            "input": oriented_cbct_folder_path,
            "dir_models": kwargs["model_folder_2"],
            "lm_type": "'LL1O','LL3O','LL6O','LR1O','LR3O','LR6O','UL1O','UL3O','UL6O','UR1O','UR3O','UR6O'",
            "output_dir": cbct_landmarks_folder_path,
            "temp_fold": temp_ali_cbct_folder,
            "DCMInput": kwargs["isDCMInput"],
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
        }
        
        logger.info(f"Parameter ALI_CBCT :  {parameter_ali_cbct_2}")

        list_process.append(
            {
                "Process": ALIProcess_CBCT,
                "Parameter": parameter_ali_cbct_2,
                "Module": "ALI_CBCT",
                "Display": DisplayALICBCT(
                    12, nb_scan
                ),
            },
        )
        
        temp_ali_ios_folder = os.path.join(slicer.util.tempDirectory(), "process.log")
        ios_landmarks_folder_path = os.path.join(kwargs["folder_output"],"IOS Landmarks")
        os.makedirs(ios_landmarks_folder_path, exist_ok=True)

        parameter_ali_ios = {
            "input": pre_aso_ios_folder_path,
            "dir_models": kwargs["model_folder_3"],
            "lm_type": "'O'",
            "teeth": "LL1 LL3 LL6 LR1 LR3 LR6 UL1 UL3 UL6 UR1 UR3 UR6'",
            "output_dir": ios_landmarks_folder_path,
            "image_size": "224",
            "blur_radius": "0",
            "faces_per_pixel": "1",
            "log_path": temp_ali_ios_folder
        }

        logger.info(f"Parameter ALI_IOS :  {parameter_ali_ios}")

        ALIProcess_IOS = slicer.modules.ali_ios
    
        list_process.append({
                "Process": ALIProcess_IOS,
                "Parameter": parameter_ali_ios,
                "Module": "ALI_IOS",
                "Display": DisplayALIIOS(
                    12, nb_scan
                ),
            })
        
        registered_ios_folder_path = os.path.join(kwargs["folder_output"],"Registered IOS")
        os.makedirs(registered_ios_folder_path, exist_ok=True)

        parameter_areg_IOSCBCT = {
            "IOS_folder": pre_aso_ios_folder_path,
            "CBCT_folder": oriented_cbct_folder_path,
            "IOS_lm_folder": ios_landmarks_folder_path,
            "CBCT_lm_folder": cbct_landmarks_folder_path,
            "output": registered_ios_folder_path
        }
        logger.info(f"Parameter reg: {parameter_areg_IOSCBCT}")

        AREGProcess = slicer.modules.areg_ioscbct

        list_process.append(
            {
                "Process": AREGProcess,
                "Parameter": parameter_areg_IOSCBCT,
                "Module": "AREG IOSCBCT",
                "Display": DisplayAREGIOSCBCT(0),
            }
        )
        return list_process
