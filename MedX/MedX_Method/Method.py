from abc import ABC, abstractmethod
import os
import glob
import json


class Method(ABC):
    def __init__(self, widget):
        self.widget = widget

    @abstractmethod
    def NbScan(self, file_folder: str):
        """
            Count the number of patient in folder
        Args:
            file_folder (str): folder path with Clinical Notes

        Return:
            int : return the number of patient.
        """
        pass

    @abstractmethod
    def TestFile(self, file_folder: str) -> str:
        """Verify if the input folder seems good (have everything required to run the mode selected), if something is wrong the function return string with error message

        This function is called when the user want to import scan

        Args:
            file_folder (str): path of folder with Clinical Notes

        Returns:
            str or None: Return str with error message if something is wrong, else return None
        pass
        """

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
    
    @abstractmethod
    def TestModel(self, model_folder: str) -> str:
        """Verify whether the model folder contains the right models used for MedX

        Args:
            model_folder (str): folder path with the model

        Return :
            str or None : display str to user like warning
        """

        pass

    @abstractmethod
    def getModelUrl(self):
        """
        Return dictionnary contains the url for each model

        dict = {'name':{'type1':'url1','type2':'url2'},...}
        or
        dict = {'name':'url'}

        """
        pass

    def search(self, path, *args):
        """
        Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

        Example:
        args = ('json',['.nii.gz','.nrrd'])
        return:
            {
                'json' : ['path/a.json', 'path/b.json','path/c.json'],
                '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
                '.nrrd.gz' : ['path/c.nrrd']
            }
        """
        arguments = []
        for arg in args:
            if type(arg) == list:
                arguments.extend(arg)
            else:
                arguments.append(arg)
        return {
            key: [
                i
                for i in glob.iglob(
                    os.path.normpath("/".join([path, "**", "*"])), recursive=True
                )
                if i.endswith(key)
            ]
            for key in arguments
        }