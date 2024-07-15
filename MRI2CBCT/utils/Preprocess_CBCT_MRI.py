from utils.Method import Method
from utils.utils_CBCT import GetDictPatients, GetPatients
import os, sys

import SimpleITK as sitk
import numpy as np

from glob import iglob
import slicer
import time
import qt
import platform


class Preprocess_CBCT_MRI(Method):
    def __init__(self, widget):
        super().__init__(widget)
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        documents = qt.QStandardPaths.writableLocation(documentsLocation)
        

    def getGPUUsage(self):
        if platform.system() == "Darwin":
            return 1
        else:
            return 5

    def NumberScan(self, scan_folder_t1: str, scan_folder_t2: str):
        # return len(GetDictPatients(scan_folder_t1, scan_folder_t2))
        return 0

    def getReferenceList(self):
        pass

    def TestReference(self, ref_folder: str):
        out = None
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        lm_extension = [".json"]

        if self.NumberScan(ref_folder) == 0:
            out = "The selected folder must contain scans"

        if self.NumberScan(ref_folder) > 1:
            out = "The selected folder must contain only 1 case"

        return None

    def TestCheckbox(self, dic_checkbox):
        pass

    def TestProcess(self, **kwargs) -> str:
        out = ""

        testcheckbox = self.TestCheckbox(kwargs["dic_checkbox"])
        if testcheckbox is not None:
            out += testcheckbox

        if kwargs["input_folder"] == "":
            out += "Please select an input folder for MRI scans\n"

        if kwargs["direction"] == "":
            out += "Please select a direction for every axe\n"

        if kwargs["output_folder"] == "":
            out += "Please select an output folder\n"

        if out == "":
            out = None

        return out

    def getModelUrl(self):
        pass

    def getALIModelList(self):
        pass
    
    def TestModel(self, model_folder: str, lineEditName):
        pass
    
    def ProperPrint(self, notfound_list):
        pass

    def TestScan(
        self,
        scan_folder_t1: str,
        scan_folder_t2: str,
        liste_keys=["scanT1", "scanT2", "segT1"],
    ):
       pass

    def GetSegmentationLabel(self, seg_folder):
       pass

    def CheckboxisChecked(self, diccheckbox: dict, in_str=False):
        pass

    def DicLandmark(self):
       pass

    def TranslateModels(self, listeModels, mask=False):
        pass

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None

    def getTestFileList(self):
        pass

    def Process(self, **kwargs):
        list_process=[]
        # MRI2CBCT_ORIENT_CENTER_MRI
        MRI2CBCT_RESAMPLE_CBCT_MRI = slicer.modules.mri2cbct_resample_cbct_mri
        parameter_mri2cbct_resample_cbct_mri = {
            "input_folder_MRI": kwargs["input_folder_MRI"],
            "input_folder_CBCT": kwargs["input_folder_CBCT"],
            "output_folder": kwargs["output_folder"],
            "resample_size": kwargs["resample_size"],
            "spacing" : kwargs["spacing"]
        }
        
        list_process.append(
            {
                "Process": MRI2CBCT_RESAMPLE_CBCT_MRI,
                "Parameter": parameter_mri2cbct_resample_cbct_mri,
                "Module": "Resample files",
            }
        )
           
        return list_process
    
  
