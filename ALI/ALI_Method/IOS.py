from ALI_Method.Method import Method
from ALI_Method.Progress import DisplayCrownSeg, DisplayALIIOS
import slicer
import os
import vtk
import shutil
import platform
import csv
from ADTLib.env.conda import windows_to_linux_path as windows_to_linux_path_shared

# --- LOGGING CONFIGURATION ---
from ADTLib.logging_setup import get_logger
from ADTLib.model_registry import ALI_IOS_MODELS, ALIDDM_TEST_FILES, ASO_IOS_GOLD
import re

logger = get_logger("ALI_IOS_Process")


class Auto_IOS(Method):
    def __init__(self, widget):
        super().__init__(widget)

    def NumberScan(self, scan_folder: str):
        if not scan_folder:
            return 0
        if os.path.isfile(scan_folder):
            if scan_folder.endswith(".vtk") or scan_folder.endswith(".stl"):
                return 1
        elif os.path.isdir(scan_folder):
            files = self.search(scan_folder, ".vtk", ".stl")
            all_files = files[".vtk"] + files[".stl"]
            return len(all_files)
        return 0
    
    def NumberLandmark(self, landmarks: str):
        if not landmarks:
            return 0
        cleaned = re.sub(r"[\[\]\"']", "", landmarks)
        teeth_list = re.split(r"[,\s]+", cleaned)
        teeth_list = [t for t in teeth_list if t and t != "None"]
        return len(teeth_list)

    def TestScan(self, scan_folder: str):
        out = ""
        if not scan_folder:
            return "Please select a folder with vtk or stl files"

        if os.path.isfile(scan_folder):
            if os.path.splitext(scan_folder)[1].lower() not in (".vtk", ".stl"):
                out = out + "Please select a vtk or stl file \n"
        elif os.path.isdir(scan_folder):
            files = self.search(scan_folder, ".vtk", ".stl")
            all_files = files[".vtk"] + files[".stl"]
            if len(all_files) == 0:
                out = out + "Please select folder with vkt or stl files \n"

        if out == "":
            out = None
        return out

    def TestModel(self, model_folder: str, line_edit_name) -> str:
        out = None
        if model_folder == "":
            out = "Please five folder with one .pht file"
        
        return out
    
    def is_wsl(self):
        return platform.system() == "Linux" and "microsoft" in platform.release().lower()
    
    def create_csv(self,input_dir,name_csv):
        '''
        create a csv with the complete path of the files in the folder (used for segmentation only)
        '''
        # Written next to the module until now, which needs the extension
        # install to be writable. The basename is what CrownSegmentation names
        # its output subfolder after, so only the folder changes here.
        csv_file = os.path.join(slicer.util.tempDirectory(), f"{name_csv}.csv")
        with open(csv_file, 'w', newline='') as fichier:
            writer = csv.writer(fichier)
            writer.writerow(["surf"])

            # Walk the folder and its subfolders
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".vtk") or file.endswith(".stl"):
                        if platform.system() != "Windows" and not self.is_wsl():
                            writer.writerow([os.path.join(root, file)])
                        else :
                            file_path = os.path.join(root, file)
                            norm_file_path = os.path.normpath(file_path)
                            writer.writerow([self.windows_to_linux_path(norm_file_path)])


        return csv_file
    
    def windows_to_linux_path(self, windows_path):
        """A Windows path as WSL sees it."""
        return windows_to_linux_path_shared(windows_path)

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

    def getTestFileList(self):
        """The test scans: one arch does not make a dataset.

        ALI IOS works on a folder of arches, and the two arches of the same
        patient are published separately -- both are brought down into the
        same folder, which is then usable as it is.
        """
        return (
            "ALI_test_scan",
            {
                "Upper": f"{ALIDDM_TEST_FILES}/T1_01_U_segmented.vtk",
                "Lower": f"{ALIDDM_TEST_FILES}/T1_01_L_segmented.vtk",
            },
        )

    def getModel(self, path, extension="ckpt"):

        model = self.search(path, f".{extension}")[f".{extension}"][0]

        return model
    
    def getModelUrl(self):
        return {
            "Segmentation": f"{ASO_IOS_GOLD}/segmentation_model.zip",
            # Occlusal, Cervical and Mucogingival models. Same content as the
            # historical ALIDDM v1.0.3 archive plus Lower_MG_v6.pth
            "Prediction": f"{ALI_IOS_MODELS}/Models.zip",
        }

    def getReferenceList(self):
        return {}

    def getALIModelList(self):
        return super().getALIModelList()

    def TestProcess(self, request) -> str:
        out = ""

        scan = self.TestScan(request.input_folder)
        if isinstance(scan, str):
            out = out + f"{scan}\n"

        if request.output_folder == "":
            out = out + "Please select output folder\n"

        if request.model_folder == "":
            out = out + "Please select folder for the landmark identification model\n"

        if out != "":
            out = out[:-1]

        else:
            out = None

        return out

    def __BypassCrownseg__(self, folder, folder_toseg, folder_bypass):
        if os.path.isfile(folder):
            all_files = [folder]
        else:
            files = self.search(folder, ".vtk", ".stl")
            all_files = files[".vtk"] + files[".stl"]
        toseg = 0
        for file in all_files:
            base_name  = os.path.basename(file)
            if self.__isSegmented__(file):
                name, ext = os.path.splitext(base_name)
                new_name = f"{name}_Seg{ext}"
                logger.debug(f"Processing file: {new_name}")
                shutil.copy(file, os.path.join(folder_bypass, new_name))

            else:
                shutil.copy(file, os.path.join(folder_toseg, base_name))
                toseg += 1

        return toseg

    def __isSegmented__(self, path):
        properties = ["PredictedID", "UniversalID", "Universal_ID"]
        extension = os.path.splitext(path)[-1].lower()
        if extension == ".stl":
            reader = vtk.vtkSTLReader()
        elif extension == ".vtk":
            reader = vtk.vtkPolyDataReader()
        else:
            return False
        
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()
        list_label = [
            surf.GetPointData().GetArrayName(i)
            for i in range(surf.GetPointData().GetNumberOfArrays())
        ]
        out = False
        if True in [label in properties for label in list_label]:
            out = True

        logger.debug(f"File segmented: {out}, Path: {path}")
        return out

    def Process(self, request):

        path_tmp = slicer.util.tempDirectory()
        path_input = os.path.join(path_tmp, "input_seg")
        path_seg = os.path.join(path_tmp, "seg")
        
        os.makedirs(path_seg, exist_ok=True)
        os.makedirs(path_input, exist_ok=True)
        os.makedirs(request.output_folder, exist_ok=True)

        path_error = os.path.join(request.output_folder, "Error")

        number_scan_toseg = self.__BypassCrownseg__(
            request.input_folder, path_input, path_seg
        )
        slicer_path = slicer.app.applicationDirPath()
        dentalmodelseg_path = os.path.join(slicer_path,"..","lib","Python","bin","dentalmodelseg")

        surf = "None"
        input_csv = "None"
        vtk_folder = "None"
        if os.path.isfile(request.input_folder):
            extension = os.path.splitext(request.input_folder)[1]
            if extension == ".vtk" or extension == ".stl":
              surf = request.input_folder
              
        elif os.path.isdir(request.input_folder):
          input_csv = self.create_csv(path_input,"liste_csv_file")
          vtk_folder = path_input

        parameter_segteeth = {
            "surf": surf,
            "input_csv": input_csv,
            "out": path_seg,
            "overwrite": "0",
            "model": "latest",
            "crown_segmentation": "0",
            "array_name": "Universal_ID",
            "fdi": 0,
            "suffix": "Seg",
            "vtk_folder": vtk_folder,
            "dentalmodelseg_path": dentalmodelseg_path
        }
        
        # Key order matters: values are passed positionally to the ALI_IOS CLI
        parameter_ali = {
            "input": path_seg,
            "dir_models": request.model_folder,
            "lm_type": request.lm_type,
            "teeth": request.teeth,
            "teeth_mg": request.teeth_mg,
            "output_dir": request.output_folder,
            "image_size": "224",
            "blur_radius": "0",
            "faces_per_pixel": "1",
            "log_path": request.log_path,
        }

        logger.debug("=" * 70)
        logger.debug(f"Segmentation parameters: {parameter_segteeth}")
        logger.debug("=" * 70)
        logger.debug(f"Landmark parameters: {parameter_ali}")
        logger.debug("=" * 70)

        landmark_process = slicer.modules.ali_ios

        numberscan = self.NumberScan(
            request.input_folder
        )
        number_lm = self.NumberLandmark(
            request.teeth
        ) + self.NumberLandmark(
            request.teeth_mg
        )

        list_process = []

        # Every scan already carries its teeth segmentation, so there is nothing
        # to segment: skip the step entirely rather than requiring
        # SlicerDentalModelSeg to be installed just to have it do nothing.
        if number_scan_toseg > 0:
            if not hasattr(slicer.modules, "crownsegmentationcli"):
                raise RuntimeError(
                    f"{number_scan_toseg} scan(s) are not segmented and the teeth "
                    "segmentation module is missing.\nPlease install the extension "
                    "SlicerDentalModelSeg, or use scans that already carry a "
                    "'Universal_ID' array."
                )
            list_process.append({
                "Process": slicer.modules.crownsegmentationcli,
                "Parameter": parameter_segteeth,
                "Module": "CrownSegmentationcli",
                "Display": DisplayCrownSeg(
                    number_scan_toseg, request.log_path
                ),
            })
        else:
            logger.info("All scans are already segmented, skipping crown segmentation")

        list_process.append({
            "Process": landmark_process,
            "Parameter": parameter_ali,
            "Module": "ALI_IOS",
            "Display": DisplayALIIOS(
                number_lm, numberscan
            ),
        })

        return list_process

    def DicLandmark(self):
        pass

    def existsLandmark(self, folderpath, reference_folder, model_folder):
        return None