from ASO_Method.Method import Method
from ASO_Method.Progress import DisplayASOCBCT, DisplayALICBCT
import os
import slicer
import time
import qt
# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.naming import patient_id as read_patient_id, ASO_CBCT_MARKERS, LANDMARK_SUFFIX_MARKERS
from ADTLib.model_registry import ASO_CBCT_GOLD, ASO_CBCT_PRE, ASO_CBCT_TEST_FILES

logger = get_logger("ASO_Method_CBCT")


class CBCT(Method):
    def __init__(self, widget):
        super().__init__(widget)

    def NumberScan(self, scan_folder: str):
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        dic = super().search(scan_folder, scan_extension)
        lenscan = 0
        for key in scan_extension:
            lenscan += len(dic[key])
        return lenscan

    def PatientScanLandmark(self, dic, scan_extension, lm_extension):
        patients = {}

        for extension, files in dic.items():
            for file in files:
                file_name = os.path.basename(file).split(".")[0]
                # TIMEPOINT-SUFFIX: only _T1/_T2 are stripped here, so _T3/_T4 inputs break
                # patient pairing. See the full note above GetPatients in
                # AREG_CBCT/AREG_CBCT_utils/utils.py before changing this.
                patient = (
                    read_patient_id(file_name, ASO_CBCT_MARKERS))

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

        testcheckbox = self.TestCheckbox(request.dic_checkbox)
        if testcheckbox is not None:
            out += testcheckbox

        if request.input_folder == "":
            out += "Please select an input folder\n"

        if request.gold_folder == "":
            out += "Please select a reference folder\n"

        if request.output_folder == "":
            out += "Please select an output folder\n"

        if request.add_in_namefile == "":
            out += "Please select an extension for output files\n"

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
        dic = {
            "Landmark": {
                "Cranial Base": sorted(
                    ["Ba", "S", "N", "RPo", "LPo", "RFZyg", "LFZyg", "C2", "C3", "C4"]
                ),
                "Upper": sorted(
                    [
                        "RInfOr",
                        "LInfOr",
                        "LMZyg",
                        "RPF",
                        "LPF",
                        "PNS",
                        "ANS",
                        "A",
                        "UR3O",
                        "UR1O",
                        "UL3O",
                        "UR6DB",
                        "UR6MB",
                        "UL6MB",
                        "UL6DB",
                        "IF",
                        "ROr",
                        "LOr",
                        "RMZyg",
                        "RNC",
                        "LNC",
                        "UR7O",
                        "UR5O",
                        "UR4O",
                        "UR2O",
                        "UL1O",
                        "UL2O",
                        "UL4O",
                        "UL5O",
                        "UL7O",
                        "UL7R",
                        "UL5R",
                        "UL4R",
                        "UL2R",
                        "UL1R",
                        "UR2R",
                        "UR4R",
                        "UR5R",
                        "UR7R",
                        "UR6MP",
                        "UL6MP",
                        "UL6R",
                        "UR6R",
                        "UR6O",
                        "UL6O",
                        "UL3R",
                        "UR3R",
                        "UR1R",
                    ]
                ),
                "Lower": sorted(
                    [
                        "RCo",
                        "RGo",
                        "Me",
                        "Gn",
                        "Pog",
                        "PogL",
                        "B",
                        "LGo",
                        "LCo",
                        "LR1O",
                        "LL6MB",
                        "LL6DB",
                        "LR6MB",
                        "LR6DB",
                        "LAF",
                        "LAE",
                        "RAF",
                        "RAE",
                        "LMCo",
                        "LLCo",
                        "RMCo",
                        "RLCo",
                        "RMeF",
                        "LMeF",
                        "RSig",
                        "RPRa",
                        "RARa",
                        "LSig",
                        "LARa",
                        "LPRa",
                        "LR7R",
                        "LR5R",
                        "LR4R",
                        "LR3R",
                        "LL3R",
                        "LL4R",
                        "LL5R",
                        "LL7R",
                        "LL7O",
                        "LL5O",
                        "LL4O",
                        "LL3O",
                        "LL2O",
                        "LL1O",
                        "LR2O",
                        "LR3O",
                        "LR4O",
                        "LR5O",
                        "LR7O",
                        "LL6R",
                        "LR6R",
                        "LL6O",
                        "LR6O",
                        "LR1R",
                        "LL1R",
                        "LL2R",
                        "LR2R",
                    ]
                ),
            }
        }

        return dic

    def Suggest(self):
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

    def NumberScanDCM(self, scan_folder: str):
        return len(
            [
                folder
                for folder in os.listdir(scan_folder)
                if os.path.isdir(os.path.join(scan_folder, folder))
                and folder != "NIFTI"
            ]
        )


class Semi_CBCT(CBCT):
    # --- interface description (see `Method`) ---
    stacked_page = 0
    scan_type = "CBCT"
    shows_cbct_input = True
    def getTestFileList(self):
        return (
            "Semi-Automated",
            f"{ASO_CBCT_TEST_FILES}/SemiAuto.zip",
        )

    def getTestFileListDCM(self):
        return (
            "Semi-Automated",
            f"{ASO_CBCT_TEST_FILES}/SemiAuto_DCM.zip",
        )

    def TestScan(self, scan_folder: str):
        out = ""
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        lm_extension = [".json"]

        if self.NumberScan(scan_folder) == 0:
            return "The selected folder must contain scans"

        dic = super().search(scan_folder, scan_extension, lm_extension)

        patients = self.PatientScanLandmark(dic, scan_extension, lm_extension)

        for patient, data in patients.items():
            if "scan" not in data.keys():
                out += "Missing scan for patient : {}\nat {}\n".format(
                    patient, data["dir"]
                )
            if len(data["lmrk"]) == 0:
                out += "Missing landmark for patient : {}\nat {}\n".format(
                    patient, data["dir"]
                )

        if out == "":  # If no errors
            out = None
        return out

    def TestScanDCM(self, scan_folder: str) -> str:
        out = ""
        lm_extension = [".json"]
        lm_patient = [
            read_patient_id(os.path.basename(i), LANDMARK_SUFFIX_MARKERS) for i in self.search(scan_folder, lm_extension)[".json"]
        ]

        if self.NumberScanDCM(scan_folder) == 0:
            return "The selected folder must contain scans"

        patients = [
            folder
            for folder in os.listdir(scan_folder)
            if os.path.isdir(os.path.join(scan_folder, folder)) and folder != "NIFTI"
        ]

        for patient in patients:
            if patient not in lm_patient:
                out += "Missing landmark for patient : {}\n".format(patient)
        for patient in lm_patient:
            if patient not in patients:
                out += "Missing scan for patient : {}\n".format(patient)

        if out == "":  # If no errors
            out = None

        return out

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        out = None
        if input_dir != "" and reference_dir != "":
            input_lm = []
            input_json = super().search(input_dir, "json")["json"]

            gold_json = super().search(reference_dir, "json")["json"]
            gold_lm = self.ListLandmarksJson(gold_json[0])

            available_lm = [lm for lm in gold_lm]  # input_lm if lm in gold_lm]
            available = {key: True for key in available_lm}

            dic = self.DicLandmark()["Landmark"]
            list_lm = []
            for key in dic.keys():
                list_lm.extend(dic[key])

            not_available_lm = [lm for lm in list_lm if lm not in available_lm]
            not_available = {key: False for key in not_available_lm}

            out = {**available, **not_available}

        return out

    def Process(self, request):
        list_lmrk_str = self.CheckboxisChecked(request.dic_checkbox, in_str=True)

        parameter_semi_aso = {
            "input": request.input_folder,
            "gold_folder": request.gold_folder,
            "output_folder": request.output_folder,
            "add_inname": request.add_in_namefile,
            "list_landmark": list_lmrk_str,
        }

        orient_process = slicer.modules.semi_aso_cbct
        
        nb_scan = self.NumberScan(request.input_folder)
        list_process = [
            {
                "Process": orient_process,
                "Parameter": parameter_semi_aso,
                "Module": "SEMI_ASO_CBCT",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            },]

        return list_process


class Auto_CBCT(CBCT):
    # --- interface description (see `Method`) ---
    stacked_page = 1
    scan_type = "CBCT"
    shows_cbct_input = True
    model_label = "Orientation Model Folder"
    def getTestFileList(self):
        return (
            "Fully-Automated",
            f"{ASO_CBCT_TEST_FILES}/FullyAuto.zip",
        )

    def TestScan(self, scan_folder: str) -> str:
        out = ""
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]

        if self.NumberScan(scan_folder) == 0:
            return "The selected folder must contain scans"

        if out == "":
            out = None

        return out

    def getTestFileListDCM(self):
        return (
            "Fully-Automated",
            f"{ASO_CBCT_TEST_FILES}/FullyAuto_DCM.zip",
        )

    def TestScanDCM(self, scan_folder: str) -> str:
        out = ""
        if self.NumberScanDCM(scan_folder) == 0:
            return "The selected folder must contain scans"

        if out == "":
            out = None

        return out

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        out = None

        if reference_dir != "" and model_dir != "":

            gold_json = super().search(reference_dir, "json")["json"]
            gold_lm = self.ListLandmarksJson(gold_json[0])

            list_model_files = super().search(model_dir, "pth")["pth"]
            list_models = [
                os.path.basename(i).split("_Net")[0] for i in list_model_files
            ]

            available_lm = [lm for lm in gold_lm if lm in list_models]
            available = {key: True for key in available_lm}

            dic = self.DicLandmark()["Landmark"]
            list_lm = []
            for key in dic.keys():
                list_lm.extend(dic[key])

            not_available_lm = [lm for lm in list_lm if lm not in available_lm]
            not_available = {key: False for key in not_available_lm}

            out = {**available, **not_available}

        return out
    
    def format_lm_string(self, lm_str: str) -> str:
        """
        Convert a space-separated string of landmarks into a string format like:
        "'Ba', 'LPo', 'N', 'RPo', 'S', 'LOr', 'ROr'"
        """
        lms = lm_str.strip().split()
        return ", ".join(f"'{lm}'" for lm in lms)

    def Process(self, request):

        # PRE ASO CBCT
        temp_folder = slicer.util.tempDirectory()
        time.sleep(0.01)
        temp_preaso_folder = slicer.util.tempDirectory()
        
        # ALI CBCT
        documents_location = qt.QStandardPaths.DocumentsLocation
        documents = qt.QStandardPaths.writableLocation(documents_location)
        temp_ali_folder = os.path.join(
            documents, slicer.app.applicationName + "_temp_ALI"
        )
        
        list_lmrk_str = self.CheckboxisChecked(request.dic_checkbox, in_str=True)
        nb_landmark = len(list_lmrk_str.split(" "))
        
        parameter_pre_aso = {
            "input": request.input_folder,
            "output_folder": temp_folder,
            "model_folder": request.model_folder_segor,
            "SmallFOV": request.smallFOV,
            "temp_folder": temp_preaso_folder,
            "DCMInput": request.is_dicom_input,
        }
        
        parameter_ali = {
            "input": temp_folder,
            "dir_models": request.model_folder_ali,
            "lm_type": self.format_lm_string(list_lmrk_str),
            "output_dir": temp_folder,
            "temp_fold": temp_ali_folder,
            "DCMInput": False,
            "spacing": "[1,0.3]",
            "speed_per_scale": "[1,1]",
            "agent_FOV": "[64,64,64]",
            "spawn_radius": "10",
        }
        
        parameter_semi_aso = {
            "input": temp_folder,
            "gold_folder": request.gold_folder,
            "output_folder": request.output_folder,
            "add_inname": request.add_in_namefile,
            "list_landmark": list_lmrk_str,
        }
        
        nb_scan = (
            self.NumberScan(request.input_folder)
            if not request.is_dicom_input
            else self.NumberScanDCM(request.input_folder)
        )

        logger.info(f"Parameter PRE_ASO :  {parameter_pre_aso}")
        logger.info(f"Parameter ALI :  {parameter_ali}")
        logger.info(f"Parameter SEMI_ASO : {parameter_semi_aso}")
        
        pre_orient_process = slicer.modules.pre_aso_cbct
        ali_process = slicer.modules.ali_cbct
        orient_process = slicer.modules.semi_aso_cbct
        
        list_process = [
            {
                "Process": pre_orient_process,
                "Parameter": parameter_pre_aso,
                "Module": "PRE_ASO_CBCT",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            },
            {
                "Process": ali_process,
                "Parameter": parameter_ali,
                "Module": "ALI_CBCT",
                "Display": DisplayALICBCT(
                    nb_landmark, nb_scan
                ),
            },
            {
                "Process": orient_process,
                "Parameter": parameter_semi_aso,
                "Module": "SEMI_ASO_CBCT",
                "Display": DisplayASOCBCT(
                    nb_scan
                ),
            },
        ]

        return list_process
