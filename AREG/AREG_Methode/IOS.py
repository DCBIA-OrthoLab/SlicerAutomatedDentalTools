from AREG_Methode.Methode import Methode
from AREG_Methode.Progress import DisplayAREGIOS, DisplayCrownSeg, DisplayASOIOS
import slicer
import webbrowser
import glob
import os
import vtk
import shutil


class Auto_IOS(Methode):
    def __init__(self, widget):
        super().__init__(widget)

    def NumberScan(self, scan_folder_t1: str, scan_folder_t2: str):
        files_T1 = self.search(scan_folder_t1, ".vtk")[".vtk"]
        words_lower = ["Lower", "_L", "L_", "Mandibule", "Md"]
        count = 0
        for file in files_T1:
            name = os.path.basename(file)
            if True in [word in name for word in words_lower]:
                count += 1

        return len(files_T1) - count

    def IsLower(self, list_files):
        words_lower = ["Lower", "_L", "L_", "Mandibule", "Md"]
        bool = False
        for file in list_files:
            name = os.path.basename(file)
            if True in [word in name for word in words_lower]:
                bool = True
        return bool

    def TestScan(self, scan_folder_t1: str, scan_folder_t2: str):
        out = ""
        if scan_folder_t1 == "" or scan_folder_t2 == "":
            out = out + "Please select folder with vtk file \n"
        if len(super().search(scan_folder_t1, ".vtk")[".vtk"]) != len(
            super().search(scan_folder_t1, ".vtk")[".vtk"]
        ):
            out = out + "Please select T1 folder and T2 with the number of vkt files \n"
        if len(super().search(scan_folder_t1, ".vtk")[".vtk"]) == 0:
            out = out + "Please select folder with vkt files \n"

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
        }

    def getReferenceList(self):
        return {
            "Gold_Files": "https://github.com/HUTIN1/ASO/releases/download/v1.0.0/Gold_file.zip"
        }

    def getALIModelList(self):
        return super().getALIModelList()

    def TestProcess(self, **kwargs) -> str:
        out = ""

        scan = self.TestScan(kwargs["input_t1_folder"], kwargs["input_t2_folder"])
        if isinstance(scan, str):
            out = out + f"{scan}\n"

        reference = self.TestReference(kwargs["model_folder_2"])
        if isinstance(reference, str):
            out = out + f"{reference}\n"

        if kwargs["folder_output"] == "":
            out = out + "Please select output folder\n"

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
        files = self.search(folder, ".vtk")[".vtk"]
        toseg = 0
        for file in files:
            name = os.path.basename(file)
            if self.__isSegmented__(file):
                shutil.copy(file, os.path.join(folder_bypass, name))

            else:
                shutil.copy(file, os.path.join(folder_toseg, name))
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
        path_input = os.path.join(path_tmp, "intpu_seg")
        path_input_T1 = os.path.join(path_input, "T1")
        path_input_T2 = os.path.join(path_input, "T2")
        path_seg = os.path.join(path_tmp, "seg")
        path_seg_T1 = os.path.join(path_seg, "T1")
        path_seg_T2 = os.path.join(path_seg, "T2")
        path_or = os.path.join(path_tmp, "Or")
        path_or_T1 = os.path.join(path_or, "T1")
        path_or_T2 = os.path.join(path_or, "T2")

        if not os.path.exists(path_seg):
            os.mkdir(os.path.join(path_seg))

        if not os.path.exists(path_seg_T1):
            os.mkdir(os.path.join(path_seg_T1))

        if not os.path.exists(path_seg_T2):
            os.mkdir(os.path.join(path_seg_T2))

        if not os.path.exists(path_or):
            os.mkdir(path_or)

        if not os.path.exists(path_or_T1):
            os.mkdir(os.path.join(path_or_T1))

        if not os.path.exists(path_or_T2):
            os.mkdir(os.path.join(path_or_T2))

        if not os.path.exists(path_input):
            os.mkdir(path_input)

        if not os.path.exists(path_input_T1):
            os.mkdir(os.path.join(path_input_T1))

        if not os.path.exists(path_input_T2):
            os.mkdir(os.path.join(path_input_T2))

        if not os.path.exists(kwargs["folder_output"]):
            os.mkdir(kwargs["folder_output"])

        path_error = os.path.join(kwargs["folder_output"], "Error")

        number_scan_toseg_T1 = self.__BypassCrownseg__(
            kwargs["input_t1_folder"], path_input_T1, path_seg_T1
        )
        number_scan_toseg_T2 = self.__BypassCrownseg__(
            kwargs["input_t2_folder"], path_input_T2, path_seg_T2
        )

        parameter_segteeth_T1 = {
            "input": path_input_T1,
            "output": path_seg_T1,
            "subdivision_level": 2,
            "resolution": 320,
            "model": self.getModel(kwargs["model_folder_1"], extension="pth"),
            "predictedId": "Universal_ID",
            "sepOutputs": 0,
            "chooseFDI": 0,
            "logPath": kwargs["logPath"],
        }

        parameter_segteeth_T2 = {
            "input": path_input_T2,
            "output": path_seg_T2,
            "subdivision_level": 2,
            "resolution": 320,
            "model": self.getModel(kwargs["model_folder_1"], extension="pth"),
            "predictedId": "Universal_ID",
            "sepOutputs": 0,
            "chooseFDI": 0,
            "logPath": kwargs["logPath"],
        }

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
        }

        print("parameter seg", parameter_segteeth_T1)
        print("parameter aso", parameter_pre_aso_T1)
        print("parameter reg", parameter_reg)

        PreOrientProcess = slicer.modules.pre_aso_ios
        SegProcess = slicer.modules.crownsegmentationcli
        RegProcess = slicer.modules.areg_ios

        numberscan = self.NumberScan(
            kwargs["input_t1_folder"], kwargs["input_t2_folder"]
        )
        list_process = [
            {
                "Process": SegProcess,
                "Parameter": parameter_segteeth_T1,
                "Module": "CrownSegmentationcli T1",
                "Display": DisplayCrownSeg(
                    number_scan_toseg_T1, kwargs["logPath"], "T1 Scan"
                ),
            },
            {
                "Process": SegProcess,
                "Parameter": parameter_segteeth_T2,
                "Module": "CrownSegmentationcli T2",
                "Display": DisplayCrownSeg(
                    number_scan_toseg_T2, kwargs["logPath"], "T2 Scan"
                ),
            },
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

        scan = self.TestScan(kwargs["input_t1_folder"], kwargs["input_t2_folder"])
        if isinstance(scan, str):
            out = out + f"{scan}\n"

        if kwargs["folder_output"] == "":
            out = out + "Please select output folder\n"

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

        parameter_reg = {
            "T1": kwargs["input_t1_folder"],
            "T2": kwargs["input_t2_folder"],
            "output": kwargs["folder_output"],
            "model": self.getModel(kwargs["model_folder_3"], extension="ckpt"),
            "suffix": kwargs["add_in_namefile"],
            "log_path": kwargs["logPath"],
        }

        print("parameter", parameter_reg)
        RegProcess = slicer.modules.areg_ios
        numberscan = self.NumberScan(
            kwargs["input_t1_folder"], kwargs["input_t2_folder"]
        )
        processus = [
            {
                "Process": RegProcess,
                "Parameter": parameter_reg,
                "Module": "AREG_IOS",
                "Display": DisplayAREGIOS(numberscan, kwargs["logPath"]),
            }
        ]
        return processus
