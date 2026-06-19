from MedX_Method.Method import Method
from MedX_Method.Progress import DisplayMedX
import slicer
import os

import sys
import logging

# ===== Logging Configuration =====
logger = logging.getLogger("MedX_summarize")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class MedX_Summarize_Method(Method):
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
        files_by_type = self.search(file_folder, ['.docx', '.pdf', '.txt'])
        nb_files = sum(len(v) for v in files_by_type.values())
        return nb_files
    
    def getModelUrl(self):
        return {
            "MedX": {
                "Summarization": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/MedX/BART.zip"
            }
        }
        
    def TestModel(self, model_folder: str) -> str:

        if len(super().search(model_folder, "safetensors")["safetensors"]) == 0:
            return "Folder must have model for "
        else:
            return None

    def TestProcess(self, **kwargs) -> str:
        out = ""
        
        if kwargs["input_notes"] == "":
            out += "Please select an input folder for Clinical Notes\n"

        if kwargs["input_model"] == "":
            out += "Please select an input folder for Model\n"

        if kwargs["output_folder"] == "":
            out += "Please select an output folder\n"

        if out == "":
            out = None

        return out

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

        logger.info(f"Parameter MedX : {parameter_MedX}")
        
        MedXProcess = slicer.modules.medx_summarize

        nb_files = self.NbScan(
            kwargs["input_notes"]
        )
        
        list_process = [
            {
                "Process": MedXProcess,
                "Parameter": parameter_MedX,
                "Module": "MedX Summarize",
                "Display": DisplayMedX(
                    nb_files, kwargs["log_path"], "Patient"
                ),
            },
        ]

        return list_process