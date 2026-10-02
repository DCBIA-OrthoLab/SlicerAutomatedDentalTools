from abc import abstractmethod

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.method import ADTMethod, DicomMethod

logger = get_logger("MRI2CBCT")


class Method(ADTMethod, DicomMethod):
    # The input folders depend on the tool: a single one for ASO and ALI, two
    # timepoints for AREG and MRI2CBCT, patients and matrices for AutoMatrix.
    # The variadic form says that without lying about the arity -- the ABC of
    # MRI2CBCT announced two where its six subclasses take one. Each
    # implementation declares the arity it really expects.
    @abstractmethod
    def TestScan(self, *scan_folders):
        """Verify if the input folder seems good (have everything required to run the mode selected), if something is wrong the function return string with error message

        This function is called when the user want to import scan

        Args:
            scan_folder (str): path of folder with scan

        Returns:
            str and bool: Return str with error message if something is wrong and a boolean to indicate if there is a message
        pass
        """


