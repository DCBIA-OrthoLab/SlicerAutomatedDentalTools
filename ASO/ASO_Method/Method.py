from abc import abstractmethod

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.method import ADTMethod, LandmarkMethod, CheckboxMethod, DicomMethod

logger = get_logger("ASO_Method")


class Method(ADTMethod, LandmarkMethod, CheckboxMethod, DicomMethod):
    """The contract of an ASO method, and the description of its interface.

    The four attributes below say what the interface must show when this
    method is picked. They used to live in the widget, as a chain of `if/elif`
    on combo box indices plus an `isinstance`: the abstraction existed, and
    the callers short-circuited it. Adding a fifth method meant going and
    editing the widget.

    They are data, not calls into Qt: the method describes, the widget
    applies. Nothing here imports `qt`, and the methods stay usable outside
    the interface.
    """

    #: page of the `stackedWidget` to show
    stacked_page = 0
    #: what the widget stores in `self.type`
    scan_type = "CBCT"
    #: is the CBCT input type combo box shown
    shows_cbct_input = True
    #: text of `labelModelFolder`, or None to leave whatever is already there
    model_label = None
    #: does the method work from a segmentation model
    #: (what `isinstance(meth, (Auto_IOS, Semi_IOS))` used to test)
    uses_segmentation_model = False

    # The input folders depend on the tool: one for ASO and ALI, two timepoints
    # for AREG and MRI2CBCT, patients and matrices for AutoMatrix. The variadic
    # form says so without lying about the arity -- the MRI2CBCT ABC announced
    # two where its six subclasses take one. Each implementation declares the
    # arity it really expects.
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
    def Suggest(self):
        pass

    @abstractmethod
    def getSegOrModelList(self):
        """
        Return a tuple with both the name and the Download link of the Seg or Or model

        tuple = ('name','link')

        """
        pass
