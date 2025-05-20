from abc import ABC, abstractmethod
import os
import glob
import json


class Method(ABC):
    def __init__(self, widget):
        self.widget = widget
        self.diccheckbox = {}
        self.diccheckbox2 = {}
    

    @abstractmethod
    def TestScan(self, patient_folder: str, matrix_folder: str) -> str:
        """Verify if the input folder seems good (have everything required to run the mode selected), if something is wrong the function return string with error message

        This function is called when the user want to import scan

        Args:
            patient_folder (str): path of folder with scans
            matrix_folder (str): path of folder with matrices

        Returns:
            str or None: Return str with error message if something is wrong, else return None
        pass
        """
        
    @abstractmethod
    def NbScan(self, input_patient: str, input_matrix: str) -> int:
        """Count the number of scan in the folder

        Args:
            input_patient (str): path of folder with scans
            input_matrix (str): path of folder with matrices

        Returns:
            int: number of scan in the folder
        """
        pass

    @abstractmethod
    def TestProcess(self, **kwargs) -> str:
        """Check if everything is OK before launching the process, if something is wrong return string with all error



        Returns:
            str or None: return None if there no problem with input of the process, else return str with all error
        """
        pass

    @abstractmethod
    def Process(self, **kwargs):
        """Launch extension"""

        pass
