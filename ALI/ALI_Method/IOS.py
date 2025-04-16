from ALI_Method.Method import Method
from ALI_Method.Progress import DisplayCrownSeg, DisplayALIIOS
import slicer
import webbrowser
import glob
import os
import vtk
import shutil
import platform
import csv


class Auto_IOS(Method):
    def __init__(self, widget):
        super().__init__(widget)

    def NumberScan(self, scan_folder: str):
        if os.path.isfile(scan_folder):
            if scan_folder.endswith(".vtk"):
                return 1
        elif os.path.isdir(scan_folder):
            files = self.search(scan_folder, ".vtk")[".vtk"]
            return len(files)
        return 0
    
    def NumberLandmark(self, landmarks: str):
        import re
        cleaned = re.sub(r"[\[\]\"']", "", landmarks)
        teeth_list = re.split(r"[,\s]+", cleaned)
        teeth_list = [t for t in teeth_list if t]
        return len(teeth_list)

    def TestScan(self, scan_folder: str):
        out = ""
        if scan_folder == "":
            out = out + "Please select folder with vtk file \n"
            
        if os.path.isfile(scan_folder):
            if not scan_folder.endswith(".vtk"):
                out = out + "Please select a vtk file \n"
        elif os.path.isdir(scan_folder):
            if len(super().search(scan_folder, ".vtk")[".vtk"]) == 0:
                out = out + "Please select folder with vkt files \n"

        if out == "":
            out = None
        return out

    def TestModel(self, model_folder: str, lineEditName) -> str:
        out = None
        if model_folder == "":
            out = "Please five folder with one .pht file"
        
        return out
    
    def create_csv(self,input_dir,name_csv):
        '''
        create a csv with the complete path of the files in the folder (used for segmentation only)
        '''
        file_path = os.path.abspath(__file__)
        folder_path = os.path.dirname(file_path)
        csv_file = os.path.join(folder_path,f"{name_csv}.csv")
        with open(csv_file, 'w', newline='') as fichier:
            writer = csv.writer(fichier)
            # Écrire l'en-tête du CSV
            writer.writerow(["surf"])

            # Parcourir le dossier et ses sous-dossiers
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".vtk") or file.endswith(".stl"):
                        # Écrire le chemin complet du fichier dans le CSV
                        if platform.system() != "Windows" :    
                            writer.writerow([os.path.join(root, file)])
                        else :
                            file_path = os.path.join(root, file)
                            norm_file_path = os.path.normpath(file_path)
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

    def getTestFileList(self):
        return (
            "ALI_test_scan",
            "https://github.com/baptistebaquero/ALIDDM/releases/tag/v1.0.4/T1_01_U_segmented.vtk",
        )

    def getModel(self, path, extension="ckpt"):

        model = self.search(path, f".{extension}")[f".{extension}"][0]

        return model
    
    def getModelUrl(self):
        return {
            "Segmentation": "https://github.com/HUTIN1/ASO/releases/download/v1.0.0/segmentation_model.zip",
            "Prediction": "https://github.com/baptistebaquero/ALIDDM/releases/download/v1.0.3/Models.zip",
        }

    def getReferenceList(self):
        return {}

    def getALIModelList(self):
        return super().getALIModelList()

    def TestProcess(self, **kwargs) -> str:
        out = ""

        scan = self.TestScan(kwargs["input_folder"])
        if isinstance(scan, str):
            out = out + f"{scan}\n"

        if kwargs["output_dir"] == "":
            out = out + "Please select output folder\n"

        if kwargs["dir_models"] == "":
            out = out + "Please select folder for the landmark identification model\n"

        if out != "":
            out = out[:-1]

        else:
            out = None

        return out

    def __BypassCrownseg__(self, folder, folder_toseg, folder_bypass):
        files = self.search(folder, ".vtk")[".vtk"]
        toseg = 0
        for file in files:
            base_name  = os.path.basename(file)
            if self.__isSegmented__(file):
                name, ext = os.path.splitext(base_name)
                new_name = f"{name}_Seg{ext}"
                print("new_name : ",new_name)
                shutil.copy(file, os.path.join(folder_bypass, new_name))

            else:
                shutil.copy(file, os.path.join(folder_toseg, base_name))
                toseg += 1

        return toseg

    def __isSegmented__(self, path):
        properties = ["PredictedID", "UniversalID", "Universal_ID"]
        reader = vtk.vtkPolyDataReader()
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

        return out

    def Process(self, **kwargs):

        path_tmp = slicer.util.tempDirectory()
        path_input = os.path.join(path_tmp, "input_seg")
        path_seg = os.path.join(path_tmp, "seg")
        
        os.makedirs(path_seg, exist_ok=True)
        os.makedirs(path_input, exist_ok=True)
        os.makedirs(kwargs["output_dir"], exist_ok=True)

        path_error = os.path.join(kwargs["output_dir"], "Error")

        number_scan_toseg = self.__BypassCrownseg__(
            kwargs["input_folder"], path_input, path_seg
        )
        slicer_path = slicer.app.applicationDirPath()
        dentalmodelseg_path = os.path.join(slicer_path,"..","lib","Python","bin","dentalmodelseg")

        surf = "None"
        input_csv = "None"
        vtk_folder = "None"
        if os.path.isfile(kwargs["input_folder"]):
            extension = os.path.splitext(kwargs["input_folder"])[1]
            if extension == ".vtk" or extension == ".stl":
              surf = kwargs["input_folder"]
              
        elif os.path.isdir(kwargs["input_folder"]):
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
        
        parameter_ali = {
            "input": path_seg,
            "dir_models": kwargs["dir_models"],
            "lm_type": kwargs["lm_type"],
            "teeth": kwargs["teeth"],
            "output_dir": kwargs["output_dir"],
            "image_size": "224",
            "blur_radius": "0",
            "faces_per_pixel": "1",
            "log_path": kwargs["logPath"],
        }

        print('-' * 70)
        print("parameter seg", parameter_segteeth)
        print('-' * 70)
        print("parameter ali : ", parameter_ali)
        print('-' * 70)

        SegProcess = slicer.modules.crownsegmentationcli
        LandmarkProcess = slicer.modules.ali_ios

        numberscan = self.NumberScan(
            kwargs["input_folder"]
        )
        number_lm = self.NumberLandmark(
            kwargs["teeth"]
        )
        
        list_process = [
            {
                "Process": SegProcess,
                "Parameter": parameter_segteeth,
                "Module": "CrownSegmentationcli",
                "Display": DisplayCrownSeg(
                    number_scan_toseg, kwargs["logPath"]
                ),
            },
            {
                "Process": LandmarkProcess,
                "Parameter": parameter_ali,
                "Module": "ALI_IOS",
                "Display": DisplayALIIOS(
                    number_lm, numberscan
                ),
            },
        ]

        #
        return list_process

    def DicLandmark(self):
        pass

    def existsLandmark(self, folderpath, reference_folder, model_folder):
        return None