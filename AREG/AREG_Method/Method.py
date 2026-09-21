from abc import ABC, abstractmethod
import os
import glob
import json
import re
import shutil

import logging
import sys
# ===== Logging Configuration =====
logger = logging.getLogger("AREG_Method")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Name of the conda environment SlicerDentalModelSeg installs its tools into.
SEGMENTATION_ENV = "shapeaxi"
SEGMENTATION_EXECUTABLE = "dentalmodelseg"


def FindDentalModelSeg():
    """Where dentalmodelseg actually is on this machine.

    The path built here used to be `<Slicer>/lib/Python/bin/dentalmodelseg`,
    which holds only when the tool was pip-installed into Slicer's own
    interpreter. That is not how SlicerDentalModelSeg installs it: it creates a
    conda environment, and its own module asks conda where the executable
    landed rather than assuming. Where the file is absent, CrownSegmentationcli
    runs it anyway and dies on FileNotFoundError -- which takes the whole
    registration with it, since the mucogingival flow segments the scans first.

    The historical path is still tried first, so an install that has it keeps
    behaving exactly as before, and it is what gets returned when nothing is
    found, so the failure a user sees does not change either.
    """
    import slicer

    historical = os.path.join(slicer.app.applicationDirPath(), "..",
                              "lib", "Python", "bin", SEGMENTATION_EXECUTABLE)
    if os.path.isfile(historical):
        return historical

    try:
        from CondaSetUp import CondaSetUpCall, CondaSetUpCallWsl
        import platform
        conda = (CondaSetUpCallWsl() if platform.system() == "Windows"
                 else CondaSetUpCall())
        answer = conda.condaRunCommand(["which", SEGMENTATION_EXECUTABLE],
                                       SEGMENTATION_ENV)
        found = re.search(r"Result: (.+)", answer or "")
        if found:
            path = found.group(1).strip().replace("\\n", "")
            if path:
                logger.info(f"Found {SEGMENTATION_EXECUTABLE} in the "
                            f"{SEGMENTATION_ENV} environment")
                return path
    except Exception as error:
        logger.warning(f"Could not ask conda for {SEGMENTATION_EXECUTABLE}: {error}")

    # Asking conda needs a live Slicer session; when there is none, look for
    # the environment on disk instead. This only ever returns a file that is
    # there, so it is a search rather than another assumption.
    beside = os.path.join(slicer.app.applicationDirPath(), "*conda*", "envs",
                          SEGMENTATION_ENV, "bin", SEGMENTATION_EXECUTABLE)
    for candidate in sorted(glob.glob(beside)):
        if os.path.isfile(candidate):
            logger.info(f"Found {SEGMENTATION_EXECUTABLE} at {candidate}")
            return candidate

    on_path = shutil.which(SEGMENTATION_EXECUTABLE)
    if on_path:
        return on_path

    logger.warning(f"{SEGMENTATION_EXECUTABLE} was not found; the segmentation "
                   "will report that its executable is missing")
    return historical


class Method(ABC):
    def __init__(self, widget):
        self.widget = widget
        self.diccheckbox = {}
        self.diccheckbox2 = {}

    @abstractmethod
    def NumberScan(self, scan_folder_t1: str, scan_folder_t2: str):
        """
            Count the number of patient in folder
        Args:
            scan_folder_t1 (str): folder path with Scan for T1
            scan_folder_t2 (str): folder path with Scan for T2

        Return:
            int : return the number of patient.
        """
        pass

    @abstractmethod
    def TestScan(self, scan_folder_t1: str, scan_folder_t2) -> str:
        """Verify if the input folder seems good (have everything required to run the mode selected), if something is wrong the function return string with error message

        This function is called when the user want to import scan

        Args:
            scan_folder (str): path of folder with scan

        Returns:
            str or None: Return str with error message if something is wrong, else return None
        pass
        """

    @abstractmethod
    def TestReference(self, ref_folder: str) -> str:
        """Verify if the reference folder contains reference gold files with landmarks and scans, if True return None and if False return str with error message to user

        Args:
            ref_folder (str): folder path with gold landmark

        Return :
            str or None : display str to user like warning
        """

        pass

    @abstractmethod
    def TestModel(self, model_folder: str, lineEditName) -> str:
        """Verify whether the model folder contains the right models used for ALI and other AI tool

        Args:
            model_folder (str): folder path with different models

        Return :
            str or None : display str to user like warning
        """

        pass

    @abstractmethod
    def TestCheckbox(self) -> str:
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

    @abstractmethod
    def DicLandmark(self):
        """
        return dic landmark like this:
        dic = {'teeth':{
                        'Lower':['LR6','LR5',...],
                        'Upper':['UR6',...]
                        },
                'Landmark':{
                        'Occlusual':['O',...],
                        'Cervical':['R',...]
                        }
                }
        """

        pass

    @abstractmethod
    def existsLandmark(self, pathfile: str, pathref: str, pathmodel: str):
        """return dictionnary. when the value of the landmark in dictionnary is true, the landmark is in input folder and in gold folder
        Args:
            pathfile (str): path

        Return :
        dict : exemple dic = {'O':True,'UL6':False,'UR1':False,...}
        """
        pass

    @abstractmethod
    def getTestFileList(self):
        """Return a tuple with both the name and the Download link of the test files

        tuple = ('name','link')
        """
        pass

    @abstractmethod
    def getReferenceList(self):
        """
        Return a dictionnary with both the name and the Download link of the references

        dict = {'name1':'link1','name2':'link2',...}

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

    @abstractmethod
    def getALIModelList(self):
        """
                Return a tuple with both the name and the Download link for ALI model
        else:
                    name, url = self.ActualMeth.getTestFileList()

                tuple = ('name','link')

        """
        pass

    def getReviewSteps(self, **kwargs) -> list:
        """Pauses this mode can offer, in the order the run reaches them.

        Declared without running Process(): the widget needs the list to build
        its checkboxes long before anything is computed, and Process() creates
        folders on the way.

        Returns:
            list: catalogue entries, each carrying its id
        """
        return []

    def getcheckbox(self):
        return self.diccheckbox

    def setcheckbox(self, dicccheckbox):
        self.diccheckbox = dicccheckbox

    def getcheckbox2(self):
        return self.diccheckbox2

    def setcheckbox2(self, dicccheckbox):
        self.diccheckbox2 = dicccheckbox

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
        # An empty path makes the pattern "/**/*", which walks the WHOLE
        # filesystem recursively: minutes at 100% of a core, silent, with
        # the panel frozen. Measured on AREG IOS, where the field-is-empty
        # message is only produced after this call -- so the user waited a
        # quarter of an hour to be told to pick a folder.
        if not isinstance(path, str) or not path.strip():
            return {key: [] for key in arguments}
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

    def ListLandmarksJson(self, json_file):
        with open(json_file) as f:
            data = json.load(f)

        return [
            data["markups"][0]["controlPoints"][i]["label"]
            for i in range(len(data["markups"][0]["controlPoints"]))
        ]

    def getTestFileListDCM(self):
        """Return a tuple with both the name and the Download link of the test files but only for DCM files (AREG CBCT)
        tuple = ('name','link')
        """
        pass

    def TestScanDCM(self, scan_folder_t1: str, scan_folder_t2) -> str:
        """Verify if the input folder seems good (have everything required to run the mode selected), if something is wrong the function return string with error message for DCM as input

        This function is called when the user want to import scan

        Args:
            scan_folder (str): path of folder with scan

        Returns:
            str or None: Return str with error message if something is wrong, else return None
        """
        pass

    def NumberScanDCM(self, scan_folder_t1: str, scan_folder_t2: str):
        """
            Count the number of patient in folder for DCM as input
        Args:
            scan_folder_t1 (str): folder path with Scan for T1
            scan_folder_t2 (str): folder path with Scan for T2

        Return:
            int : return the number of patient.
        """
        pass
