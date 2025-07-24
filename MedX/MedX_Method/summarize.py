from MedX_Method.Method import Method
from MedX_Method.Progress import DisplayMedX
import slicer
import webbrowser
import glob
import os
import vtk
import shutil
import platform
import csv


class MedX_Method(Method):
    def __init__(self, widget):
        super().__init__(widget)

    def TestFile(self, file_folder: str):
        extensions = ['.docx', '.pdf', '.txt']
        if file_folder!="None" :
            found_files = self.search(file_folder, extensions)
            if any(found_files[ext] for ext in extensions):
                return True, ""
            else:
                return False, "No files to run has been found in the "    
        return True,""
    
    def NbScan(self, file_folder: str):
        _, nb_files = self.search(file_folder, ['.docx', '.pdf', '.txt'])
        return nb_files
    
    def TestProcess(self, **kwargs) -> str:
        out = ""
        ok = True
        
        if kwargs["input_notes"] == "":
            out += "Please select an input folder for Clinical Notes\n"
            ok = False

        if kwargs["input_model"] == "":
            out += "Please select an input folder for Model\n"
            ok = False

        if kwargs["output_folder"] == "":
            out += "Please select an output folder\n"
            ok = False

        if out == "":
            out = None

        return ok, out

    def Process(self, **kwargs):
        
        path_tmp = slicer.util.tempDirectory()
        os.makedirs(path_tmp, exist_ok=True)
        os.makedirs(kwargs["output_folder"], exist_ok=True)
        
        parameter_MedX = {
            "input_notes": kwargs["input_notes"],
            "input_model": kwargs["input_model"],
            "output_folder": kwargs["output_folder"],
            "log_path": kwargs["log_path"],
        }
        
        print('-' * 70)
        print("parameter MedX : ", parameter_MedX)
        print('-' * 70)
        
        MedXProcess = slicer.modules.MedX_cli

        nb_files = self.NbScan(
            kwargs["input_notes"]
        )
        
        list_process = [
            {
                "Process": MedXProcess,
                "Parameter": parameter_MedX,
                "Module": "MedX",
                "Display": DisplayMedX(
                    nb_files, kwargs["log_path"], "Patient"
                ),
            },
        ]

        return list_process