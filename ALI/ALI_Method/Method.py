from abc import abstractmethod
from ADTLib.method import ADTMethod, LandmarkMethod, DicomMethod


class Method(ADTMethod, LandmarkMethod, DicomMethod):
    @abstractmethod
    def NumberLandmark(self, landmarks: str):
        """
            Count the number of landmarks in to check
        Args:
            landmarks (str): string with landmarks to check

        Return:
            int : return the number of landmarks.
        """
        pass
    

    # The input folders depend on the tool: a single one for ASO and ALI, two
    # timepoints for AREG and MRI2CBCT, patients and matrices for AutoMatrix.
    # The variadic form says that without lying about the arity -- the MRI2CBCT
    # ABC announced two where its six subclasses take one. Each implementation
    # declares the arity it really expects.
    @abstractmethod
    def TestScan(self, *scan_folders) -> str:
        """Verify if the input folder seems good (have everything required to run the mode selected), if something is wrong the function return string with error message

        This function is called when the user want to import scan

        Args:
            scan_folder (str): path of folder with scan

        Returns:
            str or None: Return str with error message if something is wrong, else return None
        pass
        """

    @abstractmethod
    def TestModel(self, model_folder: str, line_edit_name) -> str:
        """Verify whether the model folder contains the right models used for ALI and other AI tool

        Args:
            model_folder (str): folder path with different models

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
