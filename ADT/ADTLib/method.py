"""The contract the modules share, and the capabilities they compose.

Six modules declared their own `Method(ABC)`: 934 lines of which the greater
part was not code but repeated contracts. Measured side by side, they do not
overlap at random -- three clusters stand out, sharply:

    core       ASO AREG ALI MRI2CBCT MedX AutoMatrix   Process, TestProcess,
                                                        NumberScan, search
    landmarks  ASO AREG ALI                            7 methods
    checkboxes ASO AREG                                5 methods
    DICOM      ASO AREG ALI MRI2CBCT                   3 methods

(MedX appears in this measurement because it was there when it was taken; it
has since been archived in the `archive/medx` branch and removed from the tree.)

Hence a core plus mixins, rather than a single class where MedX would have
inherited `getcheckbox` and `DicLandmark`. A module composes what it really
offers; what it declares stays true.

What is NOT here, and why: `TestScan` and `TestModel` have plainly different
arities from one tool to the next (up to four arguments in AREG), and
`getModelUrl` returns a selection specific to each tool. Lifting them would
take a decision about behaviour, not a move of code.
"""
from abc import ABC, abstractmethod

from ADTLib.io.fs import search as search_files
from ADTLib.io.landmarks import ListLandmarksJson as list_landmarks_json


class ADTMethod(ABC):
    """What the six modules have in common, and nothing more."""

    def __init__(self, widget):
        self.widget = widget
        self.diccheckbox = {}
        self.diccheckbox2 = {}

    @abstractmethod
    def Process(self, **kwargs):
        """Launch extension"""

        pass

    @abstractmethod
    def TestProcess(self, **kwargs) -> str:
        """Check if everything is OK before launching the process, if something is wrong return string with all error



        Returns:
            str or None: return None if there no problem with input of the process, else return str with all error
        """
        pass

    @abstractmethod
    def NumberScan(self, *scan_folders):
        """
            Count the number of patient in folder
        Args:
            scan_folder_t1 (str): folder path with Scan for T1
            scan_folder_t2 (str): folder path with Scan for T2

        Return:
            int : return the number of patient.
        """
        pass

    def search(self, path, *args):
        """Delegated to ADTLib; the signature is kept for the callers."""
        return search_files(path, *args)


class LandmarkMethod(ABC):
    """The tools that handle landmarks: ASO, AREG, ALI."""

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
    def getALIModelList(self):
        """
                Return a tuple with both the name and the Download link for ALI model
        else:
                    name, url = self.ActualMeth.getTestFileList()

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
    def getTestFileList(self):
        """Return a tuple with both the name and the Download link of the test files

        tuple = ('name','link')

        A mode whose input is a folder may need several files -- ALI IOS wants
        an upper and a lower arch -- and then gives a dict instead of a link:
        tuple = ('name', {'part': 'link', ...}), every part landing in the one
        folder named by 'name'.
        """
        pass

    @abstractmethod
    def TestReference(self, ref_folder: str) -> str:
        """Verify if the reference folder contains reference gold files with landmarks and scans, if True return None and if False return str with error message to user

        Args:
            ref_folder (str): folder path with gold landmark

        Return :
            str or None : display str to user like warning
        """

        pass

    def ListLandmarksJson(self, json_file):
        """Delegated to ADTLib."""
        return list_landmarks_json(json_file)


class CheckboxMethod(ABC):
    """The tools whose interface carries checkboxes: ASO, AREG."""

    @abstractmethod
    def TestCheckbox(self) -> str:
        pass

    def getcheckbox(self):
        return self.diccheckbox

    def setcheckbox(self, checkboxes):
        self.diccheckbox = checkboxes

    def getcheckbox2(self):
        return self.diccheckbox2

    def setcheckbox2(self, checkboxes):
        self.diccheckbox2 = checkboxes


class DicomMethod(ABC):
    """The tools that accept DICOM as input."""

    def NumberScanDCM(self, *scan_folders):
        """
            Count the number of patient in folder for DCM as input
        Args:
            scan_folder_t1 (str): folder path with Scan for T1
            scan_folder_t2 (str): folder path with Scan for T2

        Return:
            int : return the number of patient.
        """
        pass

    def TestScanDCM(self, *scan_folders) -> str:
        """Verify if the input folder seems good (have everything required to run the mode selected), if something is wrong the function return string with error message for DCM as input

        This function is called when the user want to import scan

        Args:
            scan_folder (str): path of folder with scan

        Returns:
            str or None: Return str with error message if something is wrong, else return None
        """
        pass

    def getTestFileListDCM(self):
        """Return a tuple with both the name and the Download link of the test files but only for DCM files (AREG CBCT)
        tuple = ('name','link')
        """
        pass
