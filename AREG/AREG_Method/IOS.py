from AREG_Method.Method import Method
from AREG_Method.Progress import DisplayAREGIOS, DisplayCrownSeg, DisplayASOIOS, DisplayALIIOS
import slicer
import webbrowser
import glob
import os
import vtk
import shutil
import platform
import csv

import logging
import sys
# ===== Logging Configuration =====
logger = logging.getLogger("AREG_Method_CBCT")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# The 13 MG landmarks, in arch order. Must match MG_OUTPUT_NAME in
# ALI_IOS/ALI_IOS_utils/model.py, which is what the CLI translates back.
MGL_TEETH = ("LL6MG LL5MG LL4MG LL3MG LL2MG LL1MG L0MG "
             "LR1MG LR2MG LR3MG LR4MG LR5MG LR6MG")

# The extension segmenting the crowns, and the CLI it brings. Scans with no
# teeth labels cannot be landmarked, so this is a hard requirement of the
# mucogingival flow rather than a convenience.
SEGMENTATION_EXTENSION = "SlicerDentalModelSeg"
SEGMENTATION_MODULE = "CrownSegmentationcli"


def MGLProcess(method, numberscan, areg_mode, **kwargs):
    """Processes registering the lower arches on the mucogingival band.

    ALI predicts the landmarks for both timepoints unless the user points at a
    folder that already holds them, which also lets a run be repeated without
    paying for the prediction again.

    numberscan is how many lower pairs are to be processed, counted by the
    caller on the folders the user selected. Counting it here instead would
    read the segmentation folders, which are still empty while this list is
    being built, and a progress bar dividing by that zero stops the run.
    """
    landmarks_folder = kwargs.get("mgl_landmarks", "").strip()
    predict = not landmarks_folder

    if predict:
        landmarks_folder = os.path.join(slicer.util.tempDirectory(), "MGL_landmarks")
        os.makedirs(landmarks_folder, exist_ok=True)

    list_process = []

    if predict:
        # Key order matters: the values are passed positionally to the ALI_IOS CLI
        for time in ("T1", "T2"):
            parameter_ali = {
                "input": kwargs[f"input_{time.lower()}_folder"],
                "dir_models": kwargs["model_folder_3"],
                "lm_type": "None",
                "teeth": "None",
                "teeth_mg": MGL_TEETH,
                "output_dir": landmarks_folder,
                "image_size": "224",
                "blur_radius": "0",
                "faces_per_pixel": "1",
                "log_path": kwargs["logPath"],
            }
            logger.info(f"Parameter ALI_IOS {time}: {parameter_ali}")
            list_process.append({
                "Process": slicer.modules.ali_ios,
                "Parameter": parameter_ali,
                "Module": f"ALI_IOS {time}",
                "Display": DisplayALIIOS(13, numberscan),
            })

    # Key order matters: the values are passed positionally to the AREG_IOS CLI
    parameter_reg = {
        "T1": kwargs["input_t1_folder"],
        "T2": kwargs["input_t2_folder"],
        "output": kwargs["folder_output"],
        "model": "None",                       # the palatal model is not used
        "suffix": kwargs["add_in_namefile"],
        "log_path": kwargs["logPath"],
        "areg_mode": areg_mode,
        "reg_type": "MGL",
        "patch_radius": kwargs.get("patch_radius", "5.0"),
        "lm_T1": landmarks_folder,
        "lm_T2": landmarks_folder,
    }
    logger.info(f"Parameter AREG_IOS (MGL): {parameter_reg}")
    list_process.append({
        "Process": slicer.modules.areg_ios,
        "Parameter": parameter_reg,
        "Module": "AREG_IOS",
        "Display": DisplayAREGIOS(numberscan, kwargs["logPath"]),
    })

    return list_process


class Auto_IOS(Method):
    def __init__(self, widget):
        super().__init__(widget)

    def NumberScan(self, scan_folder_t1: str, scan_folder_t2: str):
        files_T1 = self.search(scan_folder_t1, ".vtk", ".stl")
        all_files = files_T1[".vtk"] + files_T1[".stl"]
        words_lower = ["Lower", "_L", "L_", "Mandibule", "Md"]
        count = 0
        for file in all_files:
            name = os.path.basename(file)
            if True in [word in name for word in words_lower]:
                count += 1

        return len(all_files) - count

    def NumberScanLower(self, scan_folder_t1: str, scan_folder_t2: str):
        """Count the lower scans, which are the ones MGL registers.

        NumberScan counts the upper arches, since the palatal registration works
        on the maxilla. A folder holding only mandibles would count zero there.
        """
        files = self.search(scan_folder_t1, ".vtk", ".stl")
        all_files = files[".vtk"] + files[".stl"]
        return sum(1 for f in all_files if self.IsLower([f]))

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

    def TestScan(self, scan_folder_t1: str, scan_folder_t2: str, mask_folder_t1: str = None) -> str:
        out = ""
        all_files_t1 = self.search(scan_folder_t1, ".vtk", ".stl")
        all_files_t2 = self.search(scan_folder_t2, ".vtk", ".stl")

        files_t1 = all_files_t1[".vtk"] + all_files_t1[".stl"]
        files_t2 = all_files_t2[".vtk"] + all_files_t2[".stl"]

        if scan_folder_t1 == "" or scan_folder_t2 == "":
            out += "Please select folders with .vtk or .stl files\n"

        if len(files_t1) != len(files_t2):
            out += "T1 and T2 folders must have the same number of surface files\n"

        if len(files_t1) == 0:
            out += "No .vtk or .stl files found in T1 folder\n"

        if out == "":
            out = None
        return out

    def TestModel(self, model_folder: str, lineEditName) -> str:
        out = None
        if model_folder == "":
            out = "Please five folder with one .pht file"
        else:
            if "lineEditModel1" == lineEditName:
                files = self.search(model_folder, ".pth")[".pth"]
                if len(files) != 1:
                    out = "Please give folder with only one .pth file \n"

            elif "lineEditModel3" == lineEditName:
                files = self.search(model_folder, ".ckpt")[".ckpt"]
                if len(files) != 1:
                    out = "Please give folder with only one .ckpt file \n"

        return out
    
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
    
    def windows_to_linux_path(self,windows_path):
        '''
        Convert a windows path to a wsl path
        '''
        windows_path = windows_path.strip()

        path = windows_path.replace('\\', '/')

        if ':' in path:
            drive, path_without_drive = path.split(':', 1)
            path = "/mnt/" + drive.lower() + path_without_drive

        return path

    def TestReference(self, ref_folder: str):

        out = []
        if ref_folder != "":
            dic = self.search(ref_folder, ".vtk", ".json")
            if len(dic[".json"]) == 0:
                out.append("Please choose a folder with json file")
            elif len(dic[".json"]) > 2:
                out.append("Too many json file ")

            if len(dic[".vtk"]) == 0:
                out.append("Please choose a folder with vkt file")

            elif len(dic[".vtk"]) > 2:
                out.append("Too many vkt file in reference folder")

        else:
            out = "Give reference folder with json and vtk file"

        if len(out) == 0:
            out = None

        else:
            out = " ".join(out)
        return out

    def TestCheckbox(self, dic_checkbox) -> str:
        return None

    def getTestFileList(self):
        return (
            "AREG_test_scan",
            "https://github.com/HUTIN1/AREG/releases/download/v1.0.0/AREG_test_scans.zip",
        )

    def getModel(self, path, extension="ckpt"):

        model = self.search(path, f".{extension}")[f".{extension}"][0]

        return model

    def getModelUrl(self):
        return {
            "Registration": "https://github.com/HUTIN1/AREG/releases/download/v1.0.0/AREG_model.zip",
            "Reference": "https://github.com/HUTIN1/ASO/releases/download/v1.0.0/Gold_file.zip",
            "Segmentation": "https://github.com/HUTIN1/ASO/releases/download/v1.0.0/segmentation_model.zip",
            # MGL reads its landmarks from the ALI models, not from a palatal checkpoint
            "ALI": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/ALI_IOS_models/Models.zip",
        }

    def getReferenceList(self):
        return {
            "Gold_Files": "https://github.com/HUTIN1/ASO/releases/download/v1.0.0/Gold_file.zip"
        }

    def getALIModelList(self):
        return super().getALIModelList()


    def TestSegmentationAvailable(self, **kwargs) -> str:
        """Make sure the scans can be segmented, offering the extension if not.

        ALI aims its cameras tooth by tooth and places nothing on a scan with
        no teeth labels, so the run would end on an empty output folder.
        """
        files = []
        for folder in (kwargs.get("input_t1_folder", ""), kwargs.get("input_t2_folder", "")):
            if folder:
                found = self.search(folder, ".vtk", ".stl")
                files += found[".vtk"] + found[".stl"]

        if all(self.__isSegmented__(f) for f in files):
            return ""
        if hasattr(slicer.modules, "crownsegmentationcli"):
            return ""
        if self.InstallSegmentationExtension():
            return ""
        return (f"Some scans carry no teeth segmentation, and {SEGMENTATION_EXTENSION}, "
                "which segments them, was not installed. Install it from the "
                "Extensions Manager, or select scans holding a Universal_ID array\n")

    def InstallSegmentationExtension(self) -> bool:
        """Download the segmentation extension, with the user's go-ahead.

        Being told to go and install an extension costs a trip to the
        Extensions Manager, a restart and the run that was just started, so the
        download is offered here and the module registered on the spot. True
        when the segmentation can run afterwards.
        """
        import qt

        if not slicer.util.confirmYesNoDisplay(
                "These scans carry no teeth segmentation, which the mucogingival "
                f"landmarks need.\n\nThe {SEGMENTATION_EXTENSION} extension "
                "segments them and is not installed yet.\n\n"
                "Download and install it now?"):
            return False

        manager = slicer.app.extensionsManagerModel()
        manager.setInteractive(False)
        logger.info(f"Installing {SEGMENTATION_EXTENSION} from the extensions server")
        if not manager.installExtensionFromServer(SEGMENTATION_EXTENSION, False, False):
            slicer.util.errorDisplay(
                f"{SEGMENTATION_EXTENSION} could not be downloaded. Check the "
                "internet connection, or install it from the Extensions Manager."
            )
            return False

        # Load it right away: a freshly installed extension is otherwise only
        # picked up by the next Slicer session, and the run would be lost.
        factory = slicer.app.moduleManager().factoryManager()
        for path in manager.extensionModulePaths(SEGMENTATION_EXTENSION):
            module = os.path.join(path, f"{SEGMENTATION_MODULE}.py")
            if os.path.isfile(module):
                factory.registerModule(qt.QFileInfo(module))
                factory.loadModules([SEGMENTATION_MODULE])
                break

        if hasattr(slicer.modules, "crownsegmentationcli"):
            logger.info(f"{SEGMENTATION_EXTENSION} installed and loaded")
            return True

        # Loading it in place did not take; a restart always does.
        if slicer.util.confirmYesNoDisplay(
                f"{SEGMENTATION_EXTENSION} is installed, but Slicer has to "
                "restart to use it.\n\nRestart now?"):
            slicer.util.restart()
        return False

    def TestMGLModel(self, **kwargs) -> str:
        """Validate what MGL needs instead of the palatal checkpoint."""
        segmentation = self.TestSegmentationAvailable(**kwargs)
        if segmentation:
            return segmentation

        if kwargs.get("mgl_landmarks", "").strip():
            folder = kwargs["mgl_landmarks"].strip()
            if len(self.search(folder, ".json")[".json"]) == 0:
                return "The MGL landmarks folder holds no json file\n"
            return ""

        if kwargs["model_folder_3"] == "":
            return "Please select the ALI models folder holding the MGL model\n"
        if not [f for f in self.search(kwargs["model_folder_3"], ".pth")[".pth"]
                if os.path.basename(f).split("_")[1:2] == ["MG"]]:
            return ('No MGL model found in the models folder, a file named '
                    '"Lower_MG_*.pth" is expected\n')
        return ""

    def TestProcess(self, **kwargs) -> str:
        out = ""

        scan = self.TestScan(kwargs["input_t1_folder"], kwargs["input_t2_folder"], None)
        if isinstance(scan, str):
            out = out + f"{scan}\n"

        if kwargs["folder_output"] == "":
            out = out + "Please select output folder\n"

        # MGL builds its patch from the landmarks, so it needs neither the
        # palatal orientation reference nor a registration checkpoint.
        if kwargs.get("reg_type") == "MGL":
            out = out + self.TestMGLModel(**kwargs)

        else:
            reference = self.TestReference(kwargs["model_folder_2"])
            if isinstance(reference, str):
                out = out + f"{reference}\n"

            if kwargs["model_folder_1"] == "":
                out = out + "Please select folder for the registration model\n"

            if len(self.search(kwargs["model_folder_3"], ".ckpt")[".ckpt"]) != 1:
                out = (
                    out + "Please select folder with only one model for the registration\n"
                )

            if kwargs["model_folder_3"] == "":
                out = out + "Please select folder for the segmentation model\n"

            if len(self.search(kwargs["model_folder_1"], ".pth")[".pth"]) != 1:
                out = (
                    out + "Please select folder with only one model for the segmentation\n"
                )

        if out != "":
            out = out[:-1]

        else:
            out = None

        return out

    def __BypassCrownseg__(self, folder, folder_toseg, folder_bypass):
        files_vtk = self.search(folder, ".vtk")[".vtk"]
        files_stl = self.search(folder, ".stl")[".stl"]
        files = files_vtk + files_stl
        toseg = 0
        for file in files:
            base_name = os.path.basename(file)
            if self.__isSegmented__(file):
                name, ext = os.path.splitext(base_name)
                new_name = f"{name}_Seg{ext}"
                shutil.copy(file, os.path.join(folder_bypass, new_name))
            else:
                shutil.copy(file, os.path.join(folder_toseg, base_name))
                toseg += 1
        return toseg

    def __isSegmented__(self, path):
        properties = ["PredictedID", "UniversalID", "Universal_ID"]
        extension = os.path.splitext(path)[-1].lower()

        if extension == ".vtk":
            reader = vtk.vtkPolyDataReader()
        elif extension == ".stl":
            reader = vtk.vtkSTLReader()
        else:
            return False

        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()

        if not isinstance(surf, vtk.vtkPolyData):
            return False

        point_data = surf.GetPointData()
        if not point_data:
            return False

        list_label = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
        return any(label in properties for label in list_label if label is not None)

    def SegmentTeeth(self, **kwargs):
        """Steps segmenting the scans of both timepoints, and where they land.

        Scans already carrying Universal_ID are copied straight to the
        segmented folder, so a timepoint that is ready to use produces no step
        at all: an empty list means there is nothing left to segment.

        Returns (processes, path_tmp, path_seg_T1, path_seg_T2).
        """
        path_tmp = slicer.util.tempDirectory()
        path_input = os.path.join(path_tmp, "input_seg")
        path_input_T1 = os.path.join(path_input, "T1")
        path_input_T2 = os.path.join(path_input, "T2")
        path_seg = os.path.join(path_tmp, "seg")
        path_seg_T1 = os.path.join(path_seg, "T1")
        path_seg_T2 = os.path.join(path_seg, "T2")

        for folder in (path_seg, path_seg_T1, path_seg_T2,
                       path_input, path_input_T1, path_input_T2):
            os.makedirs(folder, exist_ok=True)
        os.makedirs(kwargs["folder_output"], exist_ok=True)

        number_scan_toseg_T1 = self.__BypassCrownseg__(
            kwargs["input_t1_folder"], path_input_T1, path_seg_T1
        )
        number_scan_toseg_T2 = self.__BypassCrownseg__(
            kwargs["input_t2_folder"], path_input_T2, path_seg_T2
        )
        slicer_path = slicer.app.applicationDirPath()
        dentalmodelseg_path = os.path.join(slicer_path,"..","lib","Python","bin","dentalmodelseg")

        surf_T1 = "None"
        input_csv_T1 = "None"
        vtk_folder_T1 = "None"
        if os.path.isfile(path_input_T1):
            extension = os.path.splitext(self.input)[1]
            if extension == ".vtk" or extension == ".stl":
              surf_T1 = path_input_T1

        elif os.path.isdir(path_input_T1):
          input_csv_T1 = self.create_csv(path_input_T1,"liste_csv_file_T1")
          vtk_folder_T1 = path_input_T1

        parameter_segteeth_T1 = {
            "surf": surf_T1,
            "input_csv": input_csv_T1,
            "out": path_seg_T1,
            "overwrite": "0",
            "model": "latest",
            "crown_segmentation": "0",
            "array_name": "Universal_ID",
            "fdi": 0,
            "suffix": "Seg",
            "vtk_folder": vtk_folder_T1,
            "dentalmodelseg_path": dentalmodelseg_path
        }

        surf_T2 = "None"
        input_csv_T2 = "None"
        vtk_folder_T2 = "None"
        if os.path.isfile(path_input_T2):
            extension = os.path.splitext(self.input)[1]
            if extension == ".vtk" or extension == ".stl":
              surf_T2 = path_input_T2

        elif os.path.isdir(path_input_T2):
          input_csv_T2 = self.create_csv(path_input_T2,"liste_csv_file_T2")
          vtk_folder_T2 = path_input_T2

        parameter_segteeth_T2 = {
            "surf": surf_T2,
            "input_csv": input_csv_T2,
            "out": path_seg_T2,
            "overwrite": "0",
            "model": "latest",
            "crown_segmentation": "0",
            "array_name": "Universal_ID",
            "fdi": 0,
            "suffix": "Seg",
            "vtk_folder": vtk_folder_T2,
            "dentalmodelseg_path": dentalmodelseg_path
        }

        to_segment = []
        for timepoint, number, parameter in (("T1", number_scan_toseg_T1, parameter_segteeth_T1),
                                             ("T2", number_scan_toseg_T2, parameter_segteeth_T2)):
            if number == 0:
                # Every scan of this timepoint already carries its labels.
                # Running the CLI on the empty folder left behind would only
                # cost minutes and report nothing.
                logger.info(f"{timepoint}: all the scans are already segmented, skipping the segmentation")
                continue
            to_segment.append((timepoint, number, parameter))

        if not to_segment:
            # Nothing to segment, so nothing to ask of SlicerDentalModelSeg:
            # a Slicer without that extension still registers scans that are
            # already labelled.
            return [], path_tmp, path_seg_T1, path_seg_T2

        SegProcess = slicer.modules.crownsegmentationcli
        processes = [{
            "Process": SegProcess,
            "Parameter": parameter,
            "Module": f"CrownSegmentationcli {timepoint}",
            "Display": DisplayCrownSeg(number, kwargs["logPath"], f"{timepoint} Scan"),
        } for timepoint, number, parameter in to_segment]

        return processes, path_tmp, path_seg_T1, path_seg_T2

    def Process(self, **kwargs):

        seg_processes, path_tmp, path_seg_T1, path_seg_T2 = self.SegmentTeeth(**kwargs)

        path_or = os.path.join(path_tmp, "Or")
        path_or_T1 = os.path.join(path_or, "T1")
        path_or_T2 = os.path.join(path_or, "T2")
        for folder in (path_or, path_or_T1, path_or_T2):
            os.makedirs(folder, exist_ok=True)

        path_error = os.path.join(kwargs["folder_output"], "Error")

        numberscan = self.NumberScan(
            kwargs["input_t1_folder"], kwargs["input_t2_folder"]
        )

        if kwargs.get("reg_type") == "MGL":
            # The mucogingival band needs the teeth segmentation, not the palatal
            # orientation: the landmarks carry the pose the patch is built on.
            # MGL works on the mandibles, so the progress counts those, on the
            # folders the user selected rather than on the segmented copies that
            # do not exist yet.
            numberlower = self.NumberScanLower(
                kwargs["input_t1_folder"], kwargs["input_t2_folder"]
            )
            mgl_kwargs = dict(kwargs)
            mgl_kwargs["input_t1_folder"] = path_seg_T1
            mgl_kwargs["input_t2_folder"] = path_seg_T2
            return seg_processes + MGLProcess(self, numberlower, "Auto_IOS", **mgl_kwargs)

        parameter_pre_aso_T1 = {
            "input": path_seg_T1,
            "gold_folder": kwargs["model_folder_2"],
            "output_folder": path_or_T1,
            "add_inname": "Or",
            "list_teeth": "UR6,UR4,UL4,UL6",
            "occlusion": "true" if self.IsLower(kwargs["input_t1_folder"]) else "false",
            "jaw": "Upper",
            "folder_error": path_error,
            "log_path": kwargs["logPath"],
        }

        parameter_pre_aso_T2 = {
            "input": path_seg_T2,
            "gold_folder": kwargs["model_folder_2"],
            "output_folder": path_or_T2,
            "add_inname": "Or",
            "list_teeth": "UR6,UR4,UL4,UL6",
            "occlusion": "true" if self.IsLower(kwargs["input_t2_folder"]) else "false",
            "jaw": "Upper",
            "folder_error": path_error,
            "log_path": kwargs["logPath"],
        }

        parameter_reg = {
            "T1": path_or_T1,
            "T2": path_or_T2,
            "output": kwargs["folder_output"],
            "model": self.getModel(kwargs["model_folder_3"], extension="ckpt"),
            "suffix": kwargs["add_in_namefile"],
            "log_path": kwargs["logPath"],
            "areg_mode": "Auto_IOS",
        }

        logger.info(f"Parameter pre_aso1 : {parameter_pre_aso_T1}")
        logger.info(f"Parameter pre_aso2 : {parameter_pre_aso_T2}")
        logger.info(f"Parameter reg: {parameter_reg}")

        PreOrientProcess = slicer.modules.pre_aso_ios
        RegProcess = slicer.modules.areg_ios

        list_process = seg_processes + [
            {
                "Process": PreOrientProcess,
                "Parameter": parameter_pre_aso_T1,
                "Module": "PRE_ASO_IOS T1",
                "Display": DisplayASOIOS(numberscan, kwargs["logPath"], "T1 Patient"),
            },
            {
                "Process": PreOrientProcess,
                "Parameter": parameter_pre_aso_T2,
                "Module": "PRE_ASO_IOS T2",
                "Display": DisplayASOIOS(numberscan, kwargs["logPath"], "T2 Patient"),
            },
            {
                "Process": RegProcess,
                "Parameter": parameter_reg,
                "Module": "AREG_IOS",
                "Display": DisplayAREGIOS(numberscan, kwargs["logPath"]),
            },
        ]

        #
        return list_process

    def DicLandmark(self):
        pass

    def existsLandmark(self, folderpath, reference_folder, model_folder):
        return None


class Semi_IOS(Auto_IOS):
    def TestProcess(self, **kwargs) -> str:
        out = ""

        scan = self.TestScan(kwargs["input_t1_folder"], kwargs["input_t2_folder"], None)
        if isinstance(scan, str):
            out = out + f"{scan}\n"

        if kwargs["folder_output"] == "":
            out = out + "Please select output folder\n"

        if kwargs.get("reg_type") == "MGL":
            out = out + self.TestMGLModel(**kwargs)

        else:
            if kwargs["model_folder_3"] == "":
                out = out + "Please select folder for the registration model\n"

            if len(self.search(kwargs["model_folder_3"], ".ckpt")[".ckpt"]) != 1:
                out = (
                    out + "Please select folder with only one model for the registration\n"
                )

        if out != "":
            out = out[:-1]
        else:
            out = None

        return out

    def Process(self, **kwargs):

        numberscan = self.NumberScan(
            kwargs["input_t1_folder"], kwargs["input_t2_folder"]
        )

        if kwargs.get("reg_type") == "MGL":
            # ALI aims its cameras tooth by tooth, so it places nothing on a
            # scan carrying no teeth labels: without the segmentation the run
            # ends with no landmark file and an empty output folder. Registering
            # on the mucogingival line therefore segments what needs it here
            # too, and scans already labelled go through untouched.
            numberlower = self.NumberScanLower(
                kwargs["input_t1_folder"], kwargs["input_t2_folder"]
            )
            seg_processes, _path_tmp, path_seg_T1, path_seg_T2 = self.SegmentTeeth(**kwargs)
            mgl_kwargs = dict(kwargs)
            mgl_kwargs["input_t1_folder"] = path_seg_T1
            mgl_kwargs["input_t2_folder"] = path_seg_T2
            return seg_processes + MGLProcess(self, numberlower, "Semi_IOS", **mgl_kwargs)

        parameter_reg = {
            "T1": kwargs["input_t1_folder"],
            "T2": kwargs["input_t2_folder"],
            "output": kwargs["folder_output"],
            "model": self.getModel(kwargs["model_folder_3"], extension="ckpt"),
            "suffix": kwargs["add_in_namefile"],
            "log_path": kwargs["logPath"],
            "areg_mode": "Semi_IOS",
        }

        logger.info(f"Parameter AREG_IOS: {parameter_reg}")
        RegProcess = slicer.modules.areg_ios
        processus = [
            {
                "Process": RegProcess,
                "Parameter": parameter_reg,
                "Module": "AREG_IOS",
                "Display": DisplayAREGIOS(numberscan, kwargs["logPath"]),
            }
        ]
        return processus