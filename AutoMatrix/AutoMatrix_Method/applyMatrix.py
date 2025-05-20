from AutoMatrix_Method.Method import Method
from AutoMatrix_Method.Progress import DisplayAutomatrix
from AutoMatrix_Method.General_tools import GetPatients
import slicer
import webbrowser
import glob
import os
import vtk
import shutil
import platform
import csv


class Automatrix_Method(Method):
    def __init__(self, widget):
        super().__init__(widget)

    def TestScan(self, patient_folder: str, matrix_folder: str):
        _, nb_files = GetPatients(patient_folder, matrix_folder)
        
        if nb_files == 0:
            return "Please select a folder with valid scan files"
        return None
    
    def NbScan(self, input_patient: str, input_matrix: str):
        _, nb_files = GetPatients(input_patient, input_matrix)
        return nb_files
    
    def TestProcess(self, **kwargs) -> str:
        out = ""
        ok = True
        
        if kwargs["input_patient"] == "":
            out += "Please select an input folder for T1 scans\n"
            ok = False

        if kwargs["input_matrix"] == "":
            out += "Please select an input folder for T2 scans\n"
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
        
        parameter_automatrix = {
            "input_patient": kwargs["input_patient"],
            "input_matrix": kwargs["input_matrix"],
            "suffix": kwargs["suffix"],
            "matrix_name": str(kwargs["matrix_name"]),
            "fromAreg": str(kwargs["fromAreg"]),
            "output_folder": kwargs["output_folder"],
            "log_path": kwargs["log_path"],
        }
        
        print('-' * 70)
        print("parameter automatrix : ", parameter_automatrix)
        print('-' * 70)
        
        AutomatrixProcess = slicer.modules.automatrix_cli

        nb_files = self.NbScan(
            kwargs["input_patient"],
            kwargs["input_matrix"]
        )
        
        list_process = [
            {
                "Process": AutomatrixProcess,
                "Parameter": parameter_automatrix,
                "Module": "AutoMatrix",
                "Display": DisplayAutomatrix(
                    nb_files, kwargs["log_path"]
                ),
            },
        ]

        return list_process