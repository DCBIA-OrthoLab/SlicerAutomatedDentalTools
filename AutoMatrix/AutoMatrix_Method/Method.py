from abc import abstractmethod
from ADTLib.method import ADTMethod


class Method(ADTMethod):
    # The input folders depend on the tool: a single one for ASO and ALI, two
    # timepoints for AREG and MRI2CBCT, patients and matrices for AutoMatrix.
    # The variadic form says that without lying about the arity -- the ABC of
    # MRI2CBCT announced two where its six subclasses take one. Each
    # implementation declares the arity it really expects.
    @abstractmethod
    def TestScan(self, *scan_folders) -> str:
        """Verify if the input folder seems good (have everything required to run the mode selected), if something is wrong the function return string with error message

        This function is called when the user want to import scan

        Args:
            patient_folder (str): path of folder with scans
            matrix_folder (str): path of folder with matrices

        Returns:
            str or None: Return str with error message if something is wrong, else return None
        pass
        """
        