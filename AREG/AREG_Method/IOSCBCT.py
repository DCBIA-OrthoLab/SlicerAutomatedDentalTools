from AREG_Method.Method import Method, FindDentalModelSeg
from AREG_Method import Review
from AREG_Method.Progress import DisplayAREGIOSCBCT, DisplayALICBCT,DisplayASOIOS,DisplayASOCBCT,DisplayCrownSeg,DisplayALIIOS
import os
import slicer
import csv
import platform

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.model_registry import ADT_MODELS
from ADTLib.naming import patient_id as read_patient_id, AREG_IOSCBCT_MARKERS
from ADTLib.model_registry import ALIDDM, AREG_IOSCBCT_MODELS, ASO_CBCT_GOLD, ASO_CBCT_PRE, ASO_IOS_GOLD

logger = get_logger("AREG_Method_IOSCBCT")


class IOSCBCT(Method):
    # --- interface description (see `Method`) ---
    stacked_page = 4
    scan_type = "IOSCBCT"
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
                    read_patient_id(file_name, AREG_IOSCBCT_MARKERS))

                if patient not in patients.keys():
                    patients[patient] = {"dir": os.path.dirname(file), "lmrk": []}
                if extension in scan_extension:
                    patients[patient]["scan"] = file
                if extension in lm_extension:
                    patients[patient]["lmrk"].append(file)

        return patients

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

        return out

    def TestCheckbox(self, dic_checkbox):
        list_landmark = self.CheckboxisChecked(dic_checkbox)
        out = None
        if len(list_landmark) < 3:
            out = "Please select at least 3 landmarks\n"
        return out

    def TestModel(self, model_folder: str, line_edit_name) -> str:

        if line_edit_name == "lineEditModelSegOr":
            if len(super().search(model_folder, "ckpt")["ckpt"]) == 0:
                return "Folder must have Pre ASO models files"
            else:
                return None

        if line_edit_name == "lineEditModelAli":
            if len(super().search(model_folder, "pth")["pth"]) == 0:
                return "Folder must have ALI models files"
            else:
                return None

    def TestProcess(self, request) -> str:
        out = ""

        if request.input_t1_folder == "":
            out += "Please select an input folder for IOS scans\n"

        if request.input_t2_folder == "":
            out += "Please select an input folder for CBCT scans\n"
            
        if request.output_folder == "":
            out += "Please select an output folder\n"

        if request.add_in_namefile == "":
            out += "Please select a suffix\n"

        if out == "":
            out = None

        return out

    def getSegOrModelList(self):
        return (
            "PreASOModels",
            f"{ASO_CBCT_PRE}/PreASOModels.zip",
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
            f"{AREG_IOSCBCT_MODELS}/TestFile.zip",
        )

    def TestScan(self,scan_folder_t1: str,scan_folder_t2: str,mask_folder_t1: str = None) -> str:
        return None

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None
    
    def TestProcess(self, request) -> str:
        out = ""

        if request.input_t1_folder == "":
            out += "Please select an input folder for IOS scans\n"

        if request.input_t2_folder == "":
            out += "Please select an input folder for CBCT scans\n"
            
        if request.output_folder == "":
            out += "Please select an output folder\n"

        if request.model_folder_2 == "":
            out += "Please select a CBCT Landmarks model folder\n"

        if request.model_folder_3 == "":
            out += "Please select an IOS Landmarks model folder\n"

        if request.add_in_namefile == "":
            out += "Please select a suffix\n"

        if out == "":
            out = None

        return out
    
    def getModelUrl(self):
        return {
            "CBCT": {
                "Cranial Base": f"{ADT_MODELS}/Cranial_Base.zip",
                "Lower Bones 1": f"{ADT_MODELS}/Lower_Bones_1.zip",
                "Lower Bones 2": f"{ADT_MODELS}/Lower_Bones_2.zip",
                "Lower Left Teeth": f"{ADT_MODELS}/Lower_Left_Teeth.zip",
                "Lower_Right_Teeth": f"{ADT_MODELS}/Lower_Right_Teeth.zip",
                "Upper Bones v2": f"{ADT_MODELS}/Upper_Bones_v2.zip",
                "Upper Left Teeth v2": f"{ADT_MODELS}/Upper_Left_Teeth_v2.zip",
                "Upper Right Teeth v2": f"{ADT_MODELS}/Upper_Right_Teeth_v2.zip",
        },
            "IOS": f"{ALIDDM}/Models.zip",
        }
    
    def getReferenceList(self):
        return None
    
    def is_wsl(self):
        return platform.system() == "Linux" and "microsoft" in platform.release().lower()
    
    def create_csv(self, input_dir, name_csv):
        # Written next to the module until now, which needs the extension
        # install to be writable. The basename is what CrownSegmentation names
        # its output subfolder after, so only the folder changes here.
        csv_file = os.path.join(slicer.util.tempDirectory(), f"{name_csv}.csv")
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

    def getReviewSteps(self, request) -> list:
        """Pauses this mode can offer, in the order the run reaches them."""
        return Review.stepsFor([
            "ios_segmented",
            "cbct_landmarks_registration",
            "ios_landmarks",
            "ioscbct_registration",
        ])

    def Process(self, request):

        nb_scan = self.NumberScan(request.input_t1_folder,request.input_t2_folder)
        
        dentalmodelseg_path = FindDentalModelSeg()

        surf = "None"
        input_csv = "None"
        vtk_folder = "None"
        if os.path.isfile(request.input_t1_folder):
            extension = os.path.splitext(request.input_t1_folder)[1]
            if extension == ".vtk" or extension == ".stl":
              surf = request.input_t1_folder
              
        elif os.path.isdir(request.input_t1_folder):
          input_csv = self.create_csv(request.input_t1_folder,"liste_csv_file")
          vtk_folder = request.input_t1_folder

        seg_ios_folder_path = os.path.join(request.output_folder,"Seg IOS")
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
        seg_process_ios = slicer.modules.crownsegmentationcli
        
        list_process = [
            {
                "Process": seg_process_ios,
                "Parameter": parameter_seg,
                "Module": "CrownSegmentationcli",
                "ReviewId": "ios_segmented",
                "ReviewFolder": seg_ios_folder_path,
                "Display": DisplayCrownSeg(
                    nb_scan, request.log_path,"Segmentation Patient"
                ),
            }]
        
        temp_ali_cbct_folder = slicer.util.tempDirectory()
        cbct_landmarks_folder_path = os.path.join(request.output_folder,"CBCT Landmarks")
        os.makedirs(cbct_landmarks_folder_path, exist_ok=True)

        parameter_ali_cbct = {
            "input": request.input_t2_folder,
            "dir_models": request.model_folder_2,
            "lm_type": "'LL1O','LL3O','LL6O','LR1O','LR3O','LR6O','UL1O','UL3O','UL6O','UR1O','UR3O','UR6O'",
            "output_dir": cbct_landmarks_folder_path,
            "temp_fold": temp_ali_cbct_folder,
            "DCMInput": request.is_dicom_input,
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
        }
        
        logger.info(f"Parameter ALI_CBCT :  {parameter_ali_cbct}")
        ali_process_cbct = slicer.modules.ali_cbct

        list_process.append(
            {
                "Process": ali_process_cbct,
                "Parameter": parameter_ali_cbct,
                "Module": "ALI_CBCT",
                "ReviewId": "cbct_landmarks_registration",
                "ReviewFolder": cbct_landmarks_folder_path,
                "ReviewReferenceFolder": request.input_t2_folder,
                "Display": DisplayALICBCT(
                    12, nb_scan
                ),
            },
        )
        
        temp_ali_ios_folder = os.path.join(slicer.util.tempDirectory(), "process.log")
        ios_landmarks_folder_path = os.path.join(request.output_folder,"IOS Landmarks")
        os.makedirs(ios_landmarks_folder_path, exist_ok=True)

        # Key order matters: the values are passed positionally to the ALI_IOS CLI
        parameter_ali_ios = {
            "input": seg_ios_folder_path,
            "dir_models": request.model_folder_3,
            "lm_type": "'O'",
            "teeth": "LL1 LL3 LL6 LR1 LR3 LR6 UL1 UL3 UL6 UR1 UR3 UR6'",
            "teeth_mg": "None",
            "output_dir": ios_landmarks_folder_path,
            "image_size": "224",
            "blur_radius": "0",
            "faces_per_pixel": "1",
            "log_path": temp_ali_ios_folder
        }

        logger.info(f"Parameter ALI_IOS :  {parameter_ali_ios}")

        ali_process_ios = slicer.modules.ali_ios
    
        list_process.append({
                "Process": ali_process_ios,
                "Parameter": parameter_ali_ios,
                "Module": "ALI_IOS",
                "ReviewId": "ios_landmarks",
                "ReviewFolder": ios_landmarks_folder_path,
                "ReviewReferenceFolder": seg_ios_folder_path,
                "Display": DisplayALIIOS(
                    12, nb_scan
                ),
            })
        
        registered_ios_folder_path = os.path.join(request.output_folder,"Registered IOS")
        os.makedirs(registered_ios_folder_path, exist_ok=True)

        parameter_areg_ioscbct = {
            "IOS_folder": os.path.join(seg_ios_folder_path,"liste_csv_file_Seg"),
            "CBCT_folder": request.input_t2_folder,
            "IOS_lm_folder": ios_landmarks_folder_path,
            "CBCT_lm_folder": cbct_landmarks_folder_path,
            "output": registered_ios_folder_path
        }
        logger.info(f"Parameter reg: {parameter_areg_ioscbct}")

        areg_process = slicer.modules.areg_ioscbct

        list_process.append(
            {
                "Process": areg_process,
                "Parameter": parameter_areg_ioscbct,
                "Module": "AREG IOSCBCT",
                "ReviewId": "ioscbct_registration",
                "ReviewFolder": registered_ios_folder_path,
                "ReviewReferenceFolder": request.input_t2_folder,
                "Display": DisplayAREGIOSCBCT(0),
            }
        )
        return list_process
    

class Reg_IOSCBCT(IOSCBCT):
    def getTestFileList(self):
        return (
            "Registration",
            f"{AREG_IOSCBCT_MODELS}/RegTestFiles.zip",
        )

    def TestScan(self,scan_folder_t1: str,scan_folder_t2: str,mask_folder_t1: str = None) -> str:
        return None

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None
    
    def TestProcess(self, request) -> str:
        out = ""

        if request.input_t1_folder == "":
            out += "Please select an input folder for IOS scans\n"

        if request.input_t2_folder == "":
            out += "Please select an input folder for CBCT scans\n"

        if request.input_t1_mask == "":
            out += "Please select an input folder for IOS Landmarks\n"

        if request.input_t2_landmarks == "":
            out += "Please select an input folder for CBCT Landmarks\n"
            
        if request.output_folder == "":
            out += "Please select an output folder\n"

        if request.add_in_namefile == "":
            out += "Please select a suffix\n"

        if out == "":
            out = None

        return out
    
    def getModelUrl(self):
        return None
    
    def getReferenceList(self):
        return None

    def getReviewSteps(self, request) -> list:
        """Pauses this mode can offer, in the order the run reaches them."""
        return Review.stepsFor([
            "ioscbct_registration",
        ])

    def Process(self, request):

        parameter_areg_ioscbct = {
            "IOS_folder": request.input_t1_folder,
            "CBCT_folder": request.input_t2_folder,
            "IOS_lm_folder": request.input_t1_mask,
            "CBCT_lm_folder": request.input_t2_landmarks,
            "output": request.output_folder
        }
        logger.info(f"Parameter reg: {parameter_areg_ioscbct}")

        areg_process = slicer.modules.areg_ioscbct

        list_process = [
            {
                "Process": areg_process,
                "Parameter": parameter_areg_ioscbct,
                "Module": "AREG IOSCBCT",
                "ReviewId": "ioscbct_registration",
                "ReviewFolder": request.output_folder,
                "Display": DisplayAREGIOSCBCT(0),
            }
        ]
        return list_process


class Auto_IOSCBCT(IOSCBCT):
    def getTestFileList(self):
        return (
            "Fully-Automated-Registration",
            f"{AREG_IOSCBCT_MODELS}/TestFile.zip",
        )

    def TestScan(self,scan_folder_t1: str,scan_folder_t2: str,mask_folder_t1: str = None) -> str:
        return None

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None
    
    def TestProcess(self, request) -> str:
        out = ""

        if request.input_t1_folder == "":
            out += "Please select an input folder for IOS scans\n"

        if request.input_t2_folder == "":
            out += "Please select an input folder for CBCT scans\n"
            
        if request.output_folder == "":
            out += "Please select an output folder\n"

        if request.model_folder_1 == "":
            out += "Please select an Orientation model folder\n"

        if request.model_folder_2 == "":
            out += "Please select a CBCT Landmarks model folder\n"

        if request.model_folder_3 == "":
            out += "Please select an IOS Landmarks model folder\n"

        if request.add_in_namefile == "":
            out += "Please select a suffix\n"

        if out == "":
            out = None

        return out
    
    def getModelUrl(self):
        return {
            "Orientation": {
                "PreASO": f"{ASO_CBCT_PRE}/PreASOModels.zip",
                "Occlusal and Midsagittal Plane": f"{ASO_CBCT_GOLD}/Occlusal_Midsagittal_Plane.zip",
                "Frankfurt Horizontal and Midsagittal Plane": f"{ASO_CBCT_GOLD}/Frankfurt_Horizontal_Midsagittal_Plane.zip",
                "IOS":f"{ASO_IOS_GOLD}/Gold_file.zip"
            },
            "CBCT": {
                "Cranial Base": f"{ADT_MODELS}/Cranial_Base.zip",
                "Lower Bones 1": f"{ADT_MODELS}/Lower_Bones_1.zip",
                "Lower Bones 2": f"{ADT_MODELS}/Lower_Bones_2.zip",
                "Lower Left Teeth": f"{ADT_MODELS}/Lower_Left_Teeth.zip",
                "Lower_Right_Teeth": f"{ADT_MODELS}/Lower_Right_Teeth.zip",
                "Upper Bones v2": f"{ADT_MODELS}/Upper_Bones_v2.zip",
                "Upper Left Teeth v2": f"{ADT_MODELS}/Upper_Left_Teeth_v2.zip",
                "Upper Right Teeth v2": f"{ADT_MODELS}/Upper_Right_Teeth_v2.zip",
        },
            "IOS": f"{ALIDDM}/Models.zip",
        }
    
    def getReferenceList(self):
        return {
            "Occlusal and Midsagittal Plane": f"{ASO_CBCT_GOLD}/Occlusal_Midsagittal_Plane.zip",
            "Frankfurt Horizontal and Midsagittal Plane": f"{ASO_CBCT_GOLD}/Frankfurt_Horizontal_Midsagittal_Plane.zip",
        }

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
        # Written next to the module until now, which needs the extension
        # install to be writable. The basename is what CrownSegmentation names
        # its output subfolder after, so only the folder changes here.
        csv_file = os.path.join(slicer.util.tempDirectory(), f"{name_csv}.csv")
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

    def getReviewSteps(self, request) -> list:
        """Pauses this mode can offer, in the order the run reaches them."""
        return Review.stepsFor([
            "cbct_resampled",
            "cbct_landmarks_orientation",
            "cbct_oriented",
            "ios_segmented",
            "ios_oriented",
            "cbct_landmarks_registration",
            "ios_landmarks",
            "ioscbct_registration",
        ])

    def Process(self, request):

        nb_scan = self.NumberScan(request.input_t1_folder,request.input_t2_folder)

        resample_folder_path = os.path.join(request.output_folder,"CBCT Resampled")
        os.makedirs(resample_folder_path, exist_ok=True)

        parameter_resample_cbct = {
            "input_folder_MRI": "None",
            "input_folder_T2_MRI": "None",
            "input_folder_CBCT": request.input_t2_folder,
            "input_folder_T2_CBCT": "None",
            "input_folder_Seg": "None",
            "input_folder_T2_Seg": "None",
            "output_folder": resample_folder_path,
            "resample_size": "None",
            "spacing": [0.3,0.3,0.3],
            "center": "True"
        }

        logger.info(f"Parameter Resample_CBCT : {parameter_resample_cbct}")
        resample_process_cbct = slicer.modules.mri2cbct_resample_cbct_mri

        
        
        list_process = [
            {
                "Process": resample_process_cbct,
                "Parameter": parameter_resample_cbct,
                "Module": "CBCT Resampling",
                "ReviewId": "cbct_resampled",
                "ReviewFolder": resample_folder_path,
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            }
        ]

        pre_aso_cbct_folder_path = os.path.join(request.output_folder,"PRE ASO CBCT")
        os.makedirs(pre_aso_cbct_folder_path, exist_ok=True)
        temp_pre_aso_folder = slicer.util.tempDirectory()

        parameter_pre_aso_cbct = {
            "input": os.path.join(resample_folder_path,"CBCT"),
            "output_folder": pre_aso_cbct_folder_path,
            "model_folder": os.path.join(request.model_folder_1, "PreASO"),
            "SmallFOV": False,
            "temp_folder": temp_pre_aso_folder,
            "DCMInput": request.is_dicom_input,
        }

        list_lmrk_str, nb_landmark = self.ReferenceLandmarks(request.OrientReference)
        temp_ali_cbct_aso_folder = slicer.util.tempDirectory()

        parameter_ali_cbct = {
            "input": pre_aso_cbct_folder_path,
            "dir_models": request.model_folder_2,
            "lm_type": self.format_lm_string(list_lmrk_str),
            "output_dir": pre_aso_cbct_folder_path,
            "temp_fold": temp_ali_cbct_aso_folder,
            "DCMInput": False,
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
        }

        oriented_cbct_folder_path = os.path.join(request.output_folder,"Oriented CBCT")
        os.makedirs(oriented_cbct_folder_path, exist_ok=True)

        parameter_semi_aso_cbct = {
            "input": pre_aso_cbct_folder_path,
            "gold_folder": os.path.join(request.model_folder_1, request.OrientReference),
            "output_folder": oriented_cbct_folder_path,
            "add_inname": "Or",
            "list_landmark": list_lmrk_str,
        }

        logger.info(f"Parameter PRE_ASO_CBCT :  {parameter_pre_aso_cbct}")
        logger.info(f"Parameter ALI_CBCT :  {parameter_ali_cbct}")
        logger.info(f"Parameter SEMI_ASO_CBCT : {parameter_semi_aso_cbct}")
        
        pre_orient_process_cbct = slicer.modules.pre_aso_cbct
        ali_process_cbct = slicer.modules.ali_cbct
        orient_process_cbct = slicer.modules.semi_aso_cbct
        
        list_process.append(
            {
                "Process": pre_orient_process_cbct,
                "Parameter": parameter_pre_aso_cbct,
                "Module": "PRE_ASO_CBCT",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            })
        list_process.append({
                "Process": ali_process_cbct,
                "Parameter": parameter_ali_cbct,
                "Module": "ALI_CBCT",
                "ReviewId": "cbct_landmarks_orientation",
                "ReviewFolder": pre_aso_cbct_folder_path,
                "ReviewReferenceFolder": pre_aso_cbct_folder_path,
                "Display": DisplayALICBCT(
                    nb_landmark, nb_scan
                ),
            })
        list_process.append({
                "Process": orient_process_cbct,
                "Parameter": parameter_semi_aso_cbct,
                "Module": "SEMI_ASO_CBCT",
                "ReviewId": "cbct_oriented",
                "ReviewFolder": oriented_cbct_folder_path,
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            })
        
        dentalmodelseg_path = FindDentalModelSeg()

        surf = "None"
        input_csv = "None"
        vtk_folder = "None"
        if os.path.isfile(request.input_t1_folder):
            extension = os.path.splitext(request.input_t1_folder)[1]
            if extension == ".vtk" or extension == ".stl":
              surf = request.input_t1_folder
              
        elif os.path.isdir(request.input_t1_folder):
          input_csv = self.create_csv(request.input_t1_folder,"liste_csv_file")
          vtk_folder = request.input_t1_folder

        seg_ios_folder_path = os.path.join(request.output_folder,"Seg IOS")
        os.makedirs(seg_ios_folder_path, exist_ok=True)

        pre_aso_ios_folder_path = os.path.join(request.output_folder,"PRE ASO IOS")
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

        # Giving the teeth of both jaws and leaving the occlusion option off
        # orients each arch on the gold standard of its own jaw.
        #
        # Orienting "in occlusion" fits the upper alone and applies its matrix
        # to the lower unchanged, so where the lower ends up depends on the two
        # arches having been scanned in occlusion -- an assumption nothing here
        # checks. Measured over three patients, the lower lands 4.1 mm from the
        # reference of its own jaw that way against 2.3 mm fitted on its own,
        # which is where its upper lands; and on a pair taken 20 degrees apart
        # it lands 22 mm away against the same 2.3 mm.
        #
        # What that buys downstream is smaller than it looks: ALI_IOS renders
        # each arch from fixed world-axis viewpoints (+Z lower, -Z upper) and
        # does not re-orient for the O model, but it takes a 20 degree rotation
        # with only 0.6 to 0.8 mm of movement in the landmarks it places. So
        # this is robustness, not accuracy: the result stops depending on an
        # unchecked assumption, rather than becoming measurably more precise.
        #
        # Nothing is lost by dropping the link here: AREG_IOSCBCT registers the
        # two arches to the CBCT independently, as it must, since the mandible
        # is not in the same place in the CBCT as in the IOS. The occlusal
        # relationship the link preserves is discarded two steps later anyway.
        parameter_pre_aso_ios = {
            "input": seg_ios_folder_path,
            "gold_folder": os.path.join(request.model_folder_1,"IOS"),
            "output_folder": pre_aso_ios_folder_path,
            "add_inname": "Or",
            "list_teeth": "UR6,UR4,UL4,UL6,LL6,LL4,LR4,LR6",
            "occlusion": "false",
            "jaw": "Upper/Lower",
            "folder_error": path_error,
            "log_path": request.log_path,
        }

        logger.info(f"Parameter CrownSegmentation :  {parameter_seg}")
        logger.info(f"Parameter PRE_ASO_IOS :  {parameter_pre_aso_ios}")

        pre_orient_process_ios = slicer.modules.pre_aso_ios
        seg_process_ios = slicer.modules.crownsegmentationcli
        orient_process_ios = slicer.modules.semi_aso_ios
        
        
        list_process.append(
            {
                "Process": seg_process_ios,
                "Parameter": parameter_seg,
                "Module": "CrownSegmentationcli",
                "ReviewId": "ios_segmented",
                "ReviewFolder": seg_ios_folder_path,
                "Display": DisplayCrownSeg(
                    nb_scan, request.log_path,"Segmentation Patient"
                ),
            })
        list_process.append({
                "Process": pre_orient_process_ios,
                "Parameter": parameter_pre_aso_ios,
                "Module": "PRE_ASO_IOS",
                "ReviewId": "ios_oriented",
                "ReviewFolder": pre_aso_ios_folder_path,
                # Two arches per patient are oriented one by one, and the bar is
                # driven by one log line per arch: counting mouths made it read
                # 200% by the end of the step.
                "Display": DisplayASOIOS(
                    2 * nb_scan, request.log_path,"Orient IOS Patient"
                ),
            })
        
        temp_ali_cbct_folder = slicer.util.tempDirectory()
        cbct_landmarks_folder_path = os.path.join(request.output_folder,"CBCT Landmarks")
        os.makedirs(cbct_landmarks_folder_path, exist_ok=True)

        parameter_ali_cbct_2 = {
            "input": oriented_cbct_folder_path,
            "dir_models": request.model_folder_2,
            "lm_type": "'LL1O','LL3O','LL6O','LR1O','LR3O','LR6O','UL1O','UL3O','UL6O','UR1O','UR3O','UR6O'",
            "output_dir": cbct_landmarks_folder_path,
            "temp_fold": temp_ali_cbct_folder,
            "DCMInput": request.is_dicom_input,
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
        }
        
        logger.info(f"Parameter ALI_CBCT :  {parameter_ali_cbct_2}")

        list_process.append(
            {
                "Process": ali_process_cbct,
                "Parameter": parameter_ali_cbct_2,
                "Module": "ALI_CBCT",
                "ReviewId": "cbct_landmarks_registration",
                "ReviewFolder": cbct_landmarks_folder_path,
                "ReviewReferenceFolder": oriented_cbct_folder_path,
                "Display": DisplayALICBCT(
                    12, nb_scan
                ),
            },
        )
        
        temp_ali_ios_folder = os.path.join(slicer.util.tempDirectory(), "process.log")
        ios_landmarks_folder_path = os.path.join(request.output_folder,"IOS Landmarks")
        os.makedirs(ios_landmarks_folder_path, exist_ok=True)

        # Key order matters: the values are passed positionally to the ALI_IOS CLI
        parameter_ali_ios = {
            "input": pre_aso_ios_folder_path,
            "dir_models": request.model_folder_3,
            "lm_type": "'O'",
            "teeth": "LL1 LL3 LL6 LR1 LR3 LR6 UL1 UL3 UL6 UR1 UR3 UR6'",
            "teeth_mg": "None",
            "output_dir": ios_landmarks_folder_path,
            "image_size": "224",
            "blur_radius": "0",
            "faces_per_pixel": "1",
            "log_path": temp_ali_ios_folder
        }

        logger.info(f"Parameter ALI_IOS :  {parameter_ali_ios}")

        ali_process_ios = slicer.modules.ali_ios
    
        list_process.append({
                "Process": ali_process_ios,
                "Parameter": parameter_ali_ios,
                "Module": "ALI_IOS",
                "ReviewId": "ios_landmarks",
                "ReviewFolder": ios_landmarks_folder_path,
                "ReviewReferenceFolder": pre_aso_ios_folder_path,
                "Display": DisplayALIIOS(
                    12, nb_scan
                ),
            })
        
        registered_ios_folder_path = os.path.join(request.output_folder,"Registered IOS")
        os.makedirs(registered_ios_folder_path, exist_ok=True)

        parameter_areg_ioscbct = {
            "IOS_folder": pre_aso_ios_folder_path,
            "CBCT_folder": oriented_cbct_folder_path,
            "IOS_lm_folder": ios_landmarks_folder_path,
            "CBCT_lm_folder": cbct_landmarks_folder_path,
            "output": registered_ios_folder_path
        }
        logger.info(f"Parameter reg: {parameter_areg_ioscbct}")

        areg_process = slicer.modules.areg_ioscbct

        list_process.append(
            {
                "Process": areg_process,
                "Parameter": parameter_areg_ioscbct,
                "Module": "AREG IOSCBCT",
                "ReviewId": "ioscbct_registration",
                "ReviewFolder": registered_ios_folder_path,
                "ReviewReferenceFolder": oriented_cbct_folder_path,
                "Display": DisplayAREGIOSCBCT(0),
            }
        )
        return list_process
