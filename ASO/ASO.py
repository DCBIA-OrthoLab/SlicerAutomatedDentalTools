import os, sys, re, time
import vtk, qt, slicer
from qt import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QTabWidget,
    QCheckBox,
    QPixmap,
    QLabel,
    QGridLayout,
)
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin,pip_install
from functools import partial
import subprocess
import threading
import textwrap
import platform
import signal


def _get_installed_version(lib_name):
    try:
        return importlib_metadata.version(lib_name)
    except importlib_metadata.PackageNotFoundError:
        raise importlib_metadata.PackageNotFoundError

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
# This has to run before the first import of anything local, not just before
# the ADTLib ones: ALI reaches ADTLib through ALI_Method.IOS.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

from ADTLib.logging_setup import get_logger

logger = get_logger("ASO")

from ASO_Method.IOS import Auto_IOS, Semi_IOS
from ASO_Method.CBCT import Semi_CBCT, Auto_CBCT
from ASO_Method.Method import Method
from ASO_Method.Progress import Display

from ADTLib.format import format_elapsed, elapsed_since
from ADTLib.theming import update_line_edit_and_combo_box
from ADTLib.env.deps import check_lib_installed as lib_satisfies, requirement
from ADTLib.env.conda import (
    check_pythonpath, conda_quote, give_pythonpath,
    init_conda as init_conda_call, check_lib_wsl as wsl_libraries_present,
    windows_to_linux_path as windows_to_linux_path_shared)
from ADTLib.format import format_timer
from ADTLib.requests import ASORequest
from ADTLib.model_registry import SLICER_TESTING_DATA
from ADTLib.testdata import ensure_with_progress, TestDataError

def check_lib_installed(lib_name, required_version=None):
    """Whether the library is installed and satisfies the constraint."""
    return lib_satisfies(lib_name, required_version)

# import csv
    
def install_function(self):
    """Check and install required libraries with comprehensive error handling."""
    try:
        logger.info("Checking required libraries")
        
        # ===== BUILD LIBRARY LIST =====
        try:
            libs = [('itk', None), ('torch','2.2.0'),('pytorch_lightning',None),('dicom2nifti', '>=2.6.2'),('pydicom', '3.0.2')]
            monai_version = '1.3.2' if sys.version_info >= (3, 10) else '0.7.0'
            libs.append(('monai', monai_version))
            logger.debug(f"Library list created with {len(libs)} libraries")
        except Exception as e:
            logger.error(f"Error building library list: {e}")
            raise

        # ===== CHECK INSTALLED LIBRARIES =====
        try:
            libs_to_install = []
            for lib, version in libs:
                try:
                    if not check_lib_installed(lib, version):
                        libs_to_install.append((lib, version))
                        logger.warning(f"Missing or outdated library: {lib} (version: {version})")
                    else:
                        logger.debug(f"Library {lib} installed correctly")
                except Exception as e:
                    logger.warning(f"Error checking library {lib}: {e}")
                    libs_to_install.append((lib, version))
            
            logger.info(f"Need to install/update {len(libs_to_install)} library/libraries")
        except Exception as e:
            logger.error(f"Error checking installed libraries: {e}")
            raise

        # ===== USER CONFIRMATION =====
        if libs_to_install:
            try:
                message = "The following libraries are not installed or need updating:\n"
                message += "\n".join([requirement(lib, version) for lib, version in libs_to_install])
                message += "\n\nDo you want to install/update these libraries?\n Doing it could break other modules"
                
                logger.debug("Showing user confirmation dialog")
                user_choice = slicer.util.confirmYesNoDisplay(message)
                logger.info(f"User choice: {'install' if user_choice else 'skip'}")
            except Exception as e:
                logger.error(f"Error showing confirmation dialog: {e}")
                raise

            # ===== INSTALL LIBRARIES =====
            if user_choice:
                try:
                    self.ui.label_LibsInstallation.setVisible(True)
                    logger.info(f"Starting installation of {len(libs_to_install)} library/libraries")
                    
                    for lib, version in libs_to_install:
                        try:
                            lib_version = requirement(lib, version)
                            logger.debug(f"Installing library: {lib_version}")
                            pip_install(lib_version)
                            logger.info(f"Successfully installed: {lib_version}")
                        except Exception as e:
                            logger.error(f"Error installing {lib}: {e}")
                            # Continue with next library instead of failing completely
                            continue
                    
                    logger.info("Library installation completed")
                except Exception as e:
                    logger.error(f"Error during library installation: {e}")
                    raise
            else:
                logger.warning("User declined library installation")
                return False
        else:
            logger.info("All required libraries are already installed")
        
        return True
    
    except Exception as e:
        logger.error(f"Fatal error in install_function: {e}")
        return False

def condaQuote(conda, value):
    """Delegated to ADTLib; kept as a module function for the call sites."""
    return conda_quote(conda, value)


class ASO(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = (
            "ASO"
        )
        self.parent.categories = [
            "Automated Dental Tools"
        ]  # set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = (
            []
        )
        self.parent.contributors = [
            "Nathan Hutin (UoM), Luc Anchling (UoM)"
        ]
        
        self.parent.helpText = """
        This is an example of scripted loadable module bundled in an extension.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools#ASO">module documentation</a>.
        """
        
        self.parent.acknowledgementText = """
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.registerSampleData)

        #
        # Register sample data sets in Sample Data module
        #

    def registerSampleData(self):
        """
        Add data sets to Sample Data module.
        """
        # It is always recommended to provide sample data for users to make it easy to try the module,
        # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

        import SampleData

        icons_path = os.path.join(os.path.dirname(__file__), "Resources/Icons")

        # To ensure that the source code repository remains small (can be downloaded and installed quickly)
        # it is recommended to store data sets that are larger than a few MB in a Github release.

        # ALI1
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            # Category and sample name displayed in Sample Data module
            category="ASO",
            sampleName="ASO1",
            # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
            # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
            thumbnailFileName=os.path.join(icons_path, "ASO1.png"),
            # Download URL and target file name
            uris=f"{SLICER_TESTING_DATA}/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
            fileNames="ASO1.nrrd",
            # Checksum to ensure file integrity. Can be computed by this command:
            #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
            checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
            # This node name will be used when the data set is loaded
            nodeNames="ASO1",
        )

        # ASO2
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            # Category and sample name displayed in Sample Data module
            category="ASO",
            sampleName="ASO2",
            thumbnailFileName=os.path.join(icons_path, "ASO2.png"),
            # Download URL and target file name
            uris=f"{SLICER_TESTING_DATA}/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
            fileNames="ASO2.nrrd",
            checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
            # This node name will be used when the data set is loaded
            nodeNames="ASO2",
        )

class PopUpWindow(qt.QDialog):
    """PopUpWindow class
    This class is used to create a pop-up window with a list of buttons
    """

    def __init__(
        self,
        title="Title",
        text=None,
        listename=["1", "2", "3"],
        type=None,
        tocheck=None,
    ):
        QWidget.__init__(self)
        self.setWindowTitle(title)
        layout = QGridLayout()
        self.setLayout(layout)
        self.ListButtons = []
        self.listename = listename
        self.type = type

        if self.type == "radio":
            self.radiobutton(layout)

        elif self.type == "checkbox":
            self.checkbox(layout)
            if tocheck is not None:
                self.toCheck(tocheck)

        elif text is not None:
            label = qt.QLabel(text)
            layout.addWidget(label)
            # add ok button to close the window
            button = qt.QPushButton("OK")
            button.connect("clicked()", self.onClickedOK)
            layout.addWidget(button)
        
        # Apply dark mode styling
        self.applyPopUpDarkMode()

    def checkbox(self, layout):
        j = 0
        # Check if dark mode
        app = qt.QApplication.instance()
        palette = app.palette()
        bg_color = palette.color(qt.QPalette.Window)
        is_dark_mode = bg_color.lightness() < 128
        
        checkbox_stylesheet = """
          QCheckBox {
            color: #ffffff;
            background-color: transparent;
            font-weight: 500;
          }
          QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555555;
            border-radius: 3px;
            background-color: #3c3c3c;
          }
          QCheckBox::indicator:hover {
            border: 1px solid #5dade2;
          }
          QCheckBox::indicator:checked {
            border: 1px solid #5dade2;
            background-color: #5dade2;
          }
          QCheckBox::indicator:checked:hover {
            border: 1px solid #7bbcef;
            background-color: #7bbcef;
          }
        """
        
        for i in range(len(self.listename)):
            button = qt.QCheckBox(self.listename[i])
            # Apply dark mode style directly if needed
            if is_dark_mode:
                button.setStyleSheet(checkbox_stylesheet)
            self.ListButtons.append(button)
            if i % 20 == 0:
                j += 1
            layout.addWidget(button, i % 20, j)
        # Add a button to select and deselect all
        button = qt.QPushButton("Select All")
        button.connect("clicked()", self.onClickedSelectAll)
        layout.addWidget(button, len(self.listename) + 1, j - 2)
        button = qt.QPushButton("Deselect All")
        button.connect("clicked()", self.onClickedDeselectAll)
        layout.addWidget(button, len(self.listename) + 1, j - 1)

        # Add a button to close the dialog
        button = qt.QPushButton("OK")
        button.connect("clicked()", self.onClickedCheckbox)
        layout.addWidget(button, len(self.listename) + 1, j)

    def toCheck(self, tocheck):
        for i in range(len(self.listename)):
            if self.listename[i] in tocheck:
                self.ListButtons[i].setChecked(True)

    def onClickedSelectAll(self):
        for button in self.ListButtons:
            button.setChecked(True)

    def onClickedDeselectAll(self):
        for button in self.ListButtons:
            button.setChecked(False)

    def onClickedCheckbox(self):
        true_false = [button.isChecked() for button in self.ListButtons]
        self.checked = [
            self.listename[i] for i in range(len(self.listename)) if true_false[i]
        ]
        self.accept()

    def radiobutton(self, layout):
        for i in range(len(self.listename)):
            radiobutton = qt.QRadioButton(self.listename[i])
            self.ListButtons.append(radiobutton)
            radiobutton.connect("clicked(bool)", self.onClickedRadio)
            layout.addWidget(radiobutton, i, 0)

    def onClickedRadio(self):
        self.checked = self.listename[
            [button.isChecked() for button in self.ListButtons].index(True)
        ]
        self.accept()

    def onClickedOK(self):
        self.accept()

    def applyPopUpDarkMode(self):
        """Apply dark mode styling to the popup window and all its widgets."""
        app = qt.QApplication.instance()
        palette = app.palette()
        bg_color = palette.color(qt.QPalette.Window)
        if bg_color.lightness() < 128:
            # Dark mode stylesheet for popup
            dark_stylesheet = """
QDialog {
  background-color: #2b2b2b;
  color: #ffffff;
}
QLabel {
  color: #ffffff;
  background-color: transparent;
}
QPushButton {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5dade2, stop:1 #3498db);
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 600;
  padding: 8px;
}
QPushButton:hover:!pressed {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7bbcef, stop:1 #5dade2);
}
QPushButton:pressed {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1e638d);
}
QCheckBox {
  color: #ffffff;
  background-color: transparent;
  font-weight: 500;
}
QCheckBox::indicator {
  width: 18px;
  height: 18px;
  border: 1px solid #555555;
  border-radius: 3px;
  background-color: #3c3c3c;
}
QCheckBox::indicator:hover {
  border: 1px solid #5dade2;
}
QCheckBox::indicator:checked {
  border: 1px solid #5dade2;
  background-color: #5dade2;
}
QCheckBox::indicator:checked:hover {
  border: 1px solid #7bbcef;
  background-color: #7bbcef;
}
QRadioButton {
  color: #ffffff;
  background-color: transparent;
  font-weight: 500;
}
QRadioButton::indicator {
  width: 18px;
  height: 18px;
  border: 1px solid #555555;
  border-radius: 9px;
  background-color: #3c3c3c;
}
QRadioButton::indicator:hover {
  border: 1px solid #5dade2;
}
QRadioButton::indicator:checked {
  border: 1px solid #5dade2;
  background-color: #5dade2;
}
            """
            self.setStyleSheet(dark_stylesheet)
            
            # Style checkboxes from ListButtons directly
            for button in self.ListButtons:
                self._styleCheckboxWidget(button)
            
            # Recursively style all checkboxes and radiobuttons
            self._stylePopUpWidgets(self)
    
    def _styleCheckboxWidget(self, widget):
        """Style a single checkbox or radiobutton widget."""
        checkbox_stylesheet = """
          QCheckBox {
            color: #ffffff;
            background-color: transparent;
            font-weight: 500;
          }
          QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555555;
            border-radius: 3px;
            background-color: #3c3c3c;
          }
          QCheckBox::indicator:hover {
            border: 1px solid #5dade2;
          }
          QCheckBox::indicator:checked {
            border: 1px solid #5dade2;
            background-color: #5dade2;
          }
          QCheckBox::indicator:checked:hover {
            border: 1px solid #7bbcef;
            background-color: #7bbcef;
          }
        """
        
        radio_stylesheet = """
          QRadioButton {
            color: #ffffff;
            background-color: transparent;
            font-weight: 500;
          }
          QRadioButton::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555555;
            border-radius: 9px;
            background-color: #3c3c3c;
          }
          QRadioButton::indicator:hover {
            border: 1px solid #5dade2;
          }
          QRadioButton::indicator:checked {
            border: 1px solid #5dade2;
            background-color: #5dade2;
          }
        """
        
        try:
            if isinstance(widget, qt.QCheckBox):
                widget.setStyleSheet(checkbox_stylesheet)
            elif isinstance(widget, qt.QRadioButton):
                widget.setStyleSheet(radio_stylesheet)
        except (AttributeError, RuntimeError):
            # A widget with no such method, or whose C++ object is already gone.
            pass
    
    def _stylePopUpWidgets(self, parent):
        """Recursively style all checkboxes and radiobuttons in the popup."""
        checkbox_stylesheet = """
          QCheckBox {
            color: #ffffff;
            background-color: transparent;
            font-weight: 500;
          }
          QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555555;
            border-radius: 3px;
            background-color: #3c3c3c;
          }
          QCheckBox::indicator:hover {
            border: 1px solid #5dade2;
          }
          QCheckBox::indicator:checked {
            border: 1px solid #5dade2;
            background-color: #5dade2;
          }
          QCheckBox::indicator:checked:hover {
            border: 1px solid #7bbcef;
            background-color: #7bbcef;
          }
        """
        
        radio_stylesheet = """
          QRadioButton {
            color: #ffffff;
            background-color: transparent;
            font-weight: 500;
          }
          QRadioButton::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555555;
            border-radius: 9px;
            background-color: #3c3c3c;
          }
          QRadioButton::indicator:hover {
            border: 1px solid #5dade2;
          }
          QRadioButton::indicator:checked {
            border: 1px solid #5dade2;
            background-color: #5dade2;
          }
        """
        
        if isinstance(parent, qt.QCheckBox):
            try:
                parent.setStyleSheet(checkbox_stylesheet)
            except (AttributeError, RuntimeError):
                pass
        
        if isinstance(parent, qt.QRadioButton):
            try:
                parent.setStyleSheet(radio_stylesheet)
            except (AttributeError, RuntimeError):
                pass
        
        # Recursively process all children
        if hasattr(parent, 'children'):
            for child in parent.children():
                self._stylePopUpWidgets(child)


#
# ASOWidget
#


class ASOWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initiASOzed.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.nb_patient = 0  # number of scans in the input folder
        self.time_log = 0  # time of the last log update

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initiASOzed.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/ASO.ui"))
        self.layout.addWidget(ui_widget)
        self.uiWidget = ui_widget  # Store reference for styling

        self.ui = slicer.util.childWidgetVariables(ui_widget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        ui_widget.setMRMLScene(slicer.mrmlScene)

        # Apply dark mode styling if needed
        self.applyDarkModeStyles()

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ASOLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).

        """
            888     888        d8888 8888888b.  8888888        d8888 888888b.   888      8888888888  .d8888b.
            888     888       d88888 888   Y88b   888         d88888 888  "88b  888      888        d88P  Y88b
            888     888      d88P888 888    888   888        d88P888 888  .88P  888      888        Y88b.
            Y88b   d88P     d88P 888 888   d88P   888       d88P 888 8888888K.  888      8888888     "Y888b.
             Y88b d88P     d88P  888 8888888P"    888      d88P  888 888  "Y88b 888      888            "Y88b.
              Y88o88P     d88P   888 888 T88b     888     d88P   888 888    888 888      888              "888
               Y888P     d8888888888 888  T88b    888    d8888888888 888   d88P 888      888        Y88b  d88P
                Y8P     d88P     888 888   T88b 8888888 d88P     888 8888888P"  88888888 8888888888  "Y8888P"
        """

        self.MethodDic = {
            "Semi_IOS": Semi_IOS(self),
            "Auto_IOS": Auto_IOS(self),
            "Semi_CBCT": Semi_CBCT(self),
            "Auto_CBCT": Auto_CBCT(self),
        }
        self.reference_lm = []
        self.ActualMeth = Method
        self.ActualMeth = self.MethodDic["Auto_CBCT"]
        self.type = "CBCT"
        self.display = Display
        self.nb_scan = 0
        self.startprocess = 0
        self.patient_process = 0
        self.checkboxes = {}
        self.checkboxes2 = {}
        self.isDCMInput = False
        """
        example dic = {'teeth'=['A,....],'Type'=['O',...]}
        """

        self.log_path = os.path.join(slicer.util.tempDirectory(), "process.log")
        self.time = 0

        # use messletter to add big comment with univers as police

        documents_location = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documents_location)
        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
            "ASO",
            "ASO_" + self.type,
        )

        if not os.path.exists(self.SlicerDownloadPath):
            os.makedirs(self.SlicerDownloadPath)

        """

                                        8888888 888b    888 8888888 88888888888
                                          888   8888b   888   888       888
                                          888   88888b  888   888       888
                                          888   888Y88b 888   888       888
                                          888   888 Y88b888   888       888
                                          888   888  Y88888   888       888
                                          888   888   Y8888   888       888
                                        8888888 888    Y888 8888888     888

        """
        self.initCheckboxIOS(
            self.MethodDic["Auto_IOS"],
            self.ui.LayoutAutoIOS_tooth,
            self.ui.tohideAutoIOS_tooth,
            self.ui.LayoutLandmarkAutoIOS,
            self.ui.checkBoxOcclusionAutoIOS,
        )
        self.initCheckboxIOS(
            self.MethodDic["Semi_IOS"],
            self.ui.LayoutSemiIOS_tooth,
            self.ui.tohideSemiIOS_tooth,
            self.ui.LayoutLandmarkSemiIOS,
            self.ui.checkBoxOcclusionSemiIOS,
        )

        self.initCheckbox(
            self.MethodDic["Semi_CBCT"],
            self.ui.LayoutLandmarkSemiCBCT,
            self.ui.tohideCBCT,
        )  # a decommmente
        self.initCheckbox(
            self.MethodDic["Auto_CBCT"],
            self.ui.LayoutLandmarkAutoCBCT,
            self.ui.tohideCBCT,
        )
        self.HideComputeItems()
        self.SwitchType()

        """

                     .d8888b.   .d88888b.  888b    888 888b    888 8888888888  .d8888b.  88888888888
                    d88P  Y88b d88P" "Y88b 8888b   888 8888b   888 888        d88P  Y88b     888
                    888    888 888     888 88888b  888 88888b  888 888        888    888     888
                    888        888     888 888Y88b 888 888Y88b 888 8888888    888            888
                    888        888     888 888 Y88b888 888 Y88b888 888        888            888
                    888    888 888     888 888  Y88888 888  Y88888 888        888    888     888
                    Y88b  d88P Y88b. .d88P 888   Y8888 888   Y8888 888        Y88b  d88P     888
                     "Y8888P"   "Y88888P"  888    Y888 888    Y888 8888888888  "Y8888P"      888

        """

        self.ui.ButtonSearchScanLmFolder.connect("clicked(bool)", self.SearchScanLm)
        self.ui.ButtonSearchReference.connect("clicked(bool)", self.SearchReference)
        self.ui.ButtonSearchModelSegOr.connect("clicked(bool)", self.SearchModelSegOr)
        self.ui.ButtonSearchModelAli.connect("clicked(bool)", self.SearchModelALI)
        self.ui.ButtonOriented.connect("clicked(bool)", self.onPredictButton)
        self.ui.ButtonOutput.connect("clicked(bool)", self.ChosePathOutput)
        self.ui.ButtonCancel.connect("clicked(bool)", self.onCancel)
        self.ui.ButtonSuggestLmIOS.clicked.connect(self.SelectSuggestLandmark)
        self.ui.ButtonSuggestLmIOSSemi.clicked.connect(self.SelectSuggestLandmark)
        self.ui.CbInputType.currentIndexChanged.connect(self.SwitchType)
        self.ui.CbModeType.currentIndexChanged.connect(self.SwitchType)
        self.ui.CbCBCTInputType.currentIndexChanged.connect(self.SwitchCBCTInputType)
        self.ui.ButtonTestFiles.clicked.connect(self.TestFiles)
        self.ui.checkBoxOcclusionAutoIOS.toggled.connect(
            partial(
                self.logic.OcclusionCheckbox,
                self.MethodDic["Auto_IOS"].getcheckbox()["Jaw"]["Upper"],
                self.MethodDic["Auto_IOS"].getcheckbox()["Jaw"]["Lower"],
                self.MethodDic["Semi_IOS"].getcheckbox()["Teeth"],
            )
        )

    """


                888888b.   888     888 88888888888 88888888888  .d88888b.  888b    888  .d8888b.
                888  "88b  888     888     888         888     d88P" "Y88b 8888b   888 d88P  Y88b
                888  .88P  888     888     888         888     888     888 88888b  888 Y88b.
                8888888K.  888     888     888         888     888     888 888Y88b 888  "Y888b.
                888  "Y88b 888     888     888         888     888     888 888 Y88b888     "Y88b.
                888    888 888     888     888         888     888     888 888  Y88888       "888
                888   d88P Y88b. .d88P     888         888     Y88b. .d88P 888   Y8888 Y88b  d88P
                8888888P"   "Y88888P"      888         888      "Y88888P"  888    Y888  "Y8888P"



    """

    def SwitchCBCTInputType(self, index):
        if index == 0:  # NIFTI, NRRD, GIPL as input
            self.isDCMInput = False
        if index == 1:  # DICOM as input
            self.isDCMInput = True

    def SwitchMode(self, index):
        """Function to change the UI depending on the mode selected (Semi or Fully Automated)"""
        if index == 1:  # Semi-Automated
            self.ui.label_3.setText("Scan / Landmark Folder")
            self.ui.label_6.setVisible(False)
            self.ui.labelModelFolder.setVisible(False)
            self.ui.lineEditModelAli.setVisible(False)
            self.ui.lineEditModelAli.setText(" ")
            self.ui.lineEditModelSegOr.setVisible(False)
            self.ui.lineEditModelSegOr.setText(" ")
            self.ui.ButtonSearchModelAli.setVisible(False)
            self.ui.ButtonSearchModelSegOr.setVisible(False)
            self.ui.checkBoxSmallFOV.setVisible(False)

        if index == 0:  # Fully Automated
            self.ui.label_3.setText("Scan Folder")
            self.ui.checkBoxSmallFOV.setVisible(False)
            self.ui.lineEditModelAli.setVisible(False)
            self.ui.ButtonSearchModelAli.setVisible(False)
            self.ui.label_6.setVisible(False)
            if self.ActualMeth.uses_segmentation_model:
                self.ui.labelModelFolder.setVisible(True)
                self.ui.lineEditModelSegOr.setVisible(True)
                self.ui.ButtonSearchModelSegOr.setVisible(True)
                self.ui.label_CBCTInputType.setVisible(False)
            else:
                self.ui.labelModelFolder.setVisible(False)
                self.ui.lineEditModelSegOr.setVisible(False)
                self.ui.ButtonSearchModelSegOr.setVisible(False)
                self.ui.label_CBCTInputType.setVisible(True)

    #: Which method answers which (input type, mode) pair. The table replaces
    #: a chain of four `if/elif` on indices: adding a method is done here and
    #: in `MethodDic`, and nowhere else.
    METHOD_FOR_COMBO = {
        (0, 1): "Semi_CBCT",
        (0, 0): "Auto_CBCT",
        (1, 1): "Semi_IOS",
        (1, 0): "Auto_IOS",
    }

    def SwitchType(self):
        """Pick the method, then apply the description it gives of itself.

        What the interface must show is no longer decided here but read off
        the method -- `stacked_page`, `scan_type`, `shows_cbct_input`,
        `model_label`. See `ASO_Method.Method`.
        """
        key = (self.ui.CbInputType.currentIndex, self.ui.CbModeType.currentIndex)
        self.ActualMeth = self.MethodDic[self.METHOD_FOR_COMBO[key]]

        self.ui.stackedWidget.setCurrentIndex(self.ActualMeth.stacked_page)
        self.ui.CbCBCTInputType.setVisible(self.ActualMeth.shows_cbct_input)
        self.ui.label_LibsInstallation.setVisible(False)
        self.type = self.ActualMeth.scan_type
        if self.ActualMeth.model_label is not None:
            self.ui.labelModelFolder.setText(self.ActualMeth.model_label)

        # UI Changes
        self.SwitchMode(self.ui.CbModeType.currentIndex)

        self.checkboxes = self.ActualMeth.getcheckbox()
        self.checkboxes2 = self.ActualMeth.getcheckbox2()

        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
            "ASO",
            "ASO_" + self.type,
        )

        self.ClearAllLineEdits()

        self.enableCheckbox()

        self.HideComputeItems()

        if self.type == "IOS":
            self.isDCMInput = False

    def ClearAllLineEdits(self):
        """Function to clear all the line edits"""
        self.ui.lineEditScanLmPath.setText("")
        self.ui.lineEditRefFolder.setText("")
        self.ui.lineEditModelAli.setText("")
        self.ui.lineEditModelSegOr.setText("")
        self.ui.lineEditOutputPath.setText("")

    def DownloadUnzip(
        self, url, directory, folder_name=None, num_downl=1, total_downloads=1
    ):
        """The folder holding this dataset, downloaded only when it is missing.

        The work belongs to `ADTLib.testdata`, which this module shares with the
        six others that carried the same copy. What the copy here got wrong, and
        the shared one does not: it created the destination folder *before*
        downloading, so a cancelled or failed download left an empty folder that
        every later call read as "already there". And nothing checked what the
        server actually sent -- a mistyped release link answers 200 with a web
        page, which then failed as "not a zip file".
        """
        return ensure_with_progress(
            url,
            directory,
            folder_name,
            parent=self.parent,
            title="Downloading {} (File {}/{})".format(
                folder_name.split(os.sep)[0], num_downl, total_downloads
            ),
        )

    def testFileListForMode(self):
        """The (name, url) of the test set for the mode and input type in use.

        `getTestFileListDCM` is only defined by the modes that publish a DICOM
        set; the base class answers `None`, which unpacked as a `TypeError` with
        nothing in it for the user. The modes without one are reachable only
        while `isDCMInput` stays False, so the mistake never showed -- say it
        instead of relying on that.
        """
        method_name = type(self.ActualMeth).__name__
        if self.isDCMInput:
            files = self.ActualMeth.getTestFileListDCM()
            if not files:
                raise TestDataError(
                    "%s publishes no DICOM test set. Switch the CBCT input type "
                    "back to NIfTI to use its test files." % method_name)
            return files
        files = self.ActualMeth.getTestFileList()
        if not files:
            raise TestDataError("%s publishes no test set." % method_name)
        return files

    def TestFiles(self):
        """Fill every field of the selected mode from its published test set.

        Same entry point, and same reporting, as AREG's button: the download is
        a chain -- scans, reference, then models -- and any link of it can fail
        on a bad address or on the network. Reported as a message rather than as
        a traceback in the Python console, which is where it went until now.
        """
        try:
            self.SearchScanLm(test=True)
        except TestDataError as error:
            qt.QMessageBox.warning(self.parent, "Test Files", str(error))
        except OSError as error:
            qt.QMessageBox.warning(
                self.parent, "Test Files",
                "The test files could not be downloaded: %s" % error)

    def SearchScanLm(self, test=False):
        """Function to search the scan folder and to check if the scans are valid"""
        if not test:
            scan_folder = qt.QFileDialog.getExistingDirectory(
                self.parent, "Select a scan folder for Input"
            )
        else:
            name, url = self.testFileListForMode()
            scan_folder = self.DownloadUnzip(
                url=url,
                directory=os.path.join(self.SlicerDownloadPath),
                folder_name=os.path.join("Test_Files", name)
                if not self.isDCMInput
                else os.path.join("Test_Files", "DCM", name),
            )
            self.SearchReference(test=True)
            self.SearchModelSegOr()
            if self.type == "CBCT":
                self.SearchModelALI(test=True)

        if not scan_folder == "":
            if self.isDCMInput:
                nb_scans = self.ActualMeth.NumberScanDCM(scan_folder)
                error = self.ActualMeth.TestScanDCM(scan_folder)
            else:
                nb_scans = self.ActualMeth.NumberScan(scan_folder)
                error = self.ActualMeth.TestScan(scan_folder)

            if isinstance(error, str):
                qt.QMessageBox.warning(self.parent, "Warning", error)
            else:
                self.nb_patient = nb_scans
                self.ui.lineEditScanLmPath.setText(scan_folder)
                self.ui.LabelInfoPreProc.setText(
                    "Number of scans to process : " + str(nb_scans)
                )
                self.ui.LabelProgressPatient.setText(
                    "Patient process : 0 /" + str(nb_scans)
                )
                self.enableCheckbox()

                if self.ui.lineEditOutputPath.text == "":
                    dir, spl = os.path.split(scan_folder)
                    self.ui.lineEditOutputPath.setText(os.path.join(dir, spl + "Or"))

    def SearchReference(self, test=False):
        """Function to search the reference folder and to check if the reference is valid"""
        reference_list = self.ActualMeth.getReferenceList()
        ref_list = list(reference_list.keys())
        ref_list.append("Select your own folder")

        if test:
            ret = ref_list[0]

        else:
            s = PopUpWindow(
                title="Choice of Reference Files", listename=ref_list, type="radio"
            )
            s.exec_()
            ret = s.checked

        if ret == "Select your own folder":
            # TODO: Change UI to show Orientation Model Folder and Model Folder ALI
            ref_folder = qt.QFileDialog.getExistingDirectory(
                self.parent, "Select a scan folder for Reference"
            )

        else:  # Automatically Download the reference, unzip it and set the path
            ref_folder = self.DownloadUnzip(
                url=reference_list[ret],
                directory=os.path.join(self.SlicerDownloadPath),
                folder_name=os.path.join("Reference", ret),
            )

        if not ref_folder == "":
            error = self.ActualMeth.TestReference(ref_folder)

            if isinstance(error, str):
                qt.QMessageBox.warning(self.parent, "Warning", error)

            else:
                self.ui.lineEditRefFolder.setText(ref_folder)
                self.enableCheckbox()
                self.reference_lm = self.ActualMeth.ListLandmarksJson(
                    self.ActualMeth.search(ref_folder, "json")["json"][0]
                )
                if self.type == "CBCT":
                    if ret != "Select your own folder":
                        self.SearchModelSegOr()
                        self.SearchModelALI(test=True)
                    else:
                        self.ui.lineEditModelAli.setVisible(True)
                        self.ui.ButtonSearchModelAli.setVisible(True)
                        self.ui.label_6.setVisible(True)
                        self.ui.labelModelFolder.setVisible(True)
                        self.ui.lineEditModelSegOr.setVisible(True)
                        self.ui.ButtonSearchModelSegOr.setVisible(True)

    def SearchModelSegOr(self):
        """Function to search the model folder of either the segmentation or the orientation model and to check if the model is valid"""

        name, url = self.ActualMeth.getSegOrModelList()

        model_folder = self.DownloadUnzip(
            url=url,
            directory=os.path.join(self.SlicerDownloadPath),
            folder_name=os.path.join("Models", name),
        )

        if not model_folder == "":
            error = self.ActualMeth.TestModel(
                model_folder, self.ui.lineEditModelSegOr.name
            )

            if isinstance(error, str):
                qt.QMessageBox.warning(self.parent, "Warning", error)

            else:
                self.ui.lineEditModelSegOr.setText(model_folder)
                self.enableCheckbox()

    def SearchModelALI(self, test=False):
        """Function to search the model folder of the ALI model and to check if the model is valid"""
        liste_landmark = []
        for key, data in self.ActualMeth.DicLandmark()["Landmark"].items():
            liste_landmark += data

        if test:
            ret = self.reference_lm

        else:

            s = PopUpWindow(
                title="Chose ALI Models to Download",
                listename=sorted(liste_landmark),
                type="checkbox",
                tocheck=self.reference_lm,
            )
            s.exec_()
            ret = s.checked

        name, url = self.ActualMeth.getALIModelList()

        for i, model in enumerate(ret):
            _ = self.DownloadUnzip(
                url=os.path.join(url, "{}.zip".format(model)),
                directory=os.path.join(self.SlicerDownloadPath.replace("ASO", "ALI")),
                folder_name=model,
                num_downl=i + 1,
                total_downloads=len(ret),
            )

        model_folder = os.path.join(self.SlicerDownloadPath.replace("ASO", "ALI"))

        if not model_folder == "":
            error = self.ActualMeth.TestModel(
                model_folder, self.ui.lineEditModelAli.name
            )

            if isinstance(error, str):
                qt.QMessageBox.warning(self.parent, "Warning", error)

            else:
                self.ui.lineEditModelAli.setText(model_folder)
                self.enableCheckbox()

    def ChosePathOutput(self):
        out_folder = qt.QFileDialog.getExistingDirectory(
            self.parent, "Select a scan folder"
        )
        if not out_folder == "":
            self.ui.lineEditOutputPath.setText(out_folder)

    def SelectSuggestLandmark(self):
        best = self.ActualMeth.Suggest()
        for checkbox in self.logic.iterillimeted(self.checkboxes):
            if checkbox.text in best and checkbox.isEnabled():
                checkbox.setCheckState(True)

    def onJawToggled(self, all_checkbox, jaw, boolean):
        """One jaw was ticked: let the Logic enable its landmarks, then refresh.

        The two steps sit on either side of the widget boundary. Enabling the
        checkboxes needs nothing but the checkboxes, so it stays on the Logic;
        deciding which of them the data actually allows needs this widget's
        fields, so it stays here. Wiring the signal straight to the Logic put
        both on the wrong side of it and the refresh raised on every toggle.
        """
        self.logic.UpperLowerCheckbox(all_checkbox, jaw, boolean)
        self.enableCheckbox()

    def enableCheckbox(self):
        """Function to enable the checkbox depending on the presence of landmarks"""
        status = self.ActualMeth.existsLandmark(
            self.ui.lineEditScanLmPath.text,
            self.ui.lineEditRefFolder.text,
            self.ui.lineEditModelAli.text,
        )

        if status is None:
            return

        if self.type == "IOS":
            for checkbox, checkbox2 in zip(
                self.logic.iterillimeted(self.checkboxes),
                self.logic.iterillimeted(self.checkboxes),
            ):
                try:
                    checkbox.setCheckable(status[checkbox.text])
                    checkbox2.setCheckable(status[checkbox2.text])

                except KeyError:
                    # status is keyed by the checkbox label: a box missing from the
                    # dictionary is left as it is, which deserves at least a trace.
                    logger.debug("No status for %s nor %s", checkbox.text, checkbox2.text)

        if self.type == "CBCT":
            for checkboxs, checkboxs2 in zip(
                self.checkboxes.values(), self.checkboxes2.values()
            ):
                for checkbox, checkbox2 in zip(checkboxs, checkboxs2):
                    checkbox.setVisible(status[checkbox.text])
                    checkbox2.setVisible(status[checkbox2.text])
                    if status[checkbox.text]:
                        checkbox.setChecked(True)
                        checkbox2.setChecked(True)

    """

                    8888888b.  8888888b.   .d88888b.   .d8888b.  8888888888  .d8888b.   .d8888b.
                    888   Y88b 888   Y88b d88P" "Y88b d88P  Y88b 888        d88P  Y88b d88P  Y88b
                    888    888 888    888 888     888 888    888 888        Y88b.      Y88b.
                    888   d88P 888   d88P 888     888 888        8888888     "Y888b.    "Y888b.
                    8888888P"  8888888P"  888     888 888        888            "Y88b.     "Y88b.
                    888        888 T88b   888     888 888    888 888              "888       "888
                    888        888  T88b  Y88b. .d88P Y88b  d88P 888        Y88b  d88P Y88b  d88P
                    888        888   T88b  "Y88888P"   "Y8888P"  8888888888  "Y8888P"   "Y8888P"


    """

    def onPredictButton(self):
        """Function to launch the prediction"""
        
        is_installed = False
        if self.type == "IOS":
            check_env = self.onCheckRequirements()
            if not check_env:
                return
            self.logic.check_cli_script()
            
        is_installed = install_function(self)

        if not is_installed:
            qt.QMessageBox.warning(self.parent, 'Warning', 'The module will not work properly without the required libraries.\nPlease install them and try again.')
            return
        
        self.ui.label_LibsInstallation.setVisible(False)
        error = self.ActualMeth.TestProcess(
            ASORequest(input_folder=self.ui.lineEditScanLmPath.text,
            gold_folder=self.ui.lineEditRefFolder.text,
            output_folder=self.ui.lineEditOutputPath.text,
            model_folder_ali=self.ui.lineEditModelAli.text,
            model_folder_segor=self.ui.lineEditModelSegOr.text,
            add_in_namefile=self.ui.lineEditAddName.text,
            dic_checkbox=self.checkboxes,
            smallFOV=str(self.ui.checkBoxSmallFOV.isChecked()),
            is_dicom_input=self.isDCMInput,
        ))
        if isinstance(error, str):
            qt.QMessageBox.warning(self.parent, "Warning", error.replace(",", "\n"))

        else:
            self.list_Processes_Parameters = self.ActualMeth.Process(
                ASORequest(input_folder=self.ui.lineEditScanLmPath.text,
                gold_folder=self.ui.lineEditRefFolder.text,
                output_folder=self.ui.lineEditOutputPath.text,
                model_folder_ali=self.ui.lineEditModelAli.text,
                model_folder_segor=self.ui.lineEditModelSegOr.text,
                add_in_namefile=self.ui.lineEditAddName.text,
                dic_checkbox=self.checkboxes,
                log_path=self.log_path,
                smallFOV=str(self.ui.checkBoxSmallFOV.isChecked()),
                is_dicom_input=self.isDCMInput,
            ))

            self.nb_extension_launch = len(self.list_Processes_Parameters)
            self.onProcessStarted()
            
            module = self.list_Processes_Parameters[0]["Module"]
            logger.info(f"Module name: {module}")
            if module == "CrownSegmentationcli":
                self.nb_extension_did += 1
                self.run_conda_tool()

            # /!\ Launch of the first process /!\
            self.process = slicer.cli.run(
                self.list_Processes_Parameters[0]["Process"],
                None,
                self.list_Processes_Parameters[0]["Parameter"],
            )
            self.module_name = self.list_Processes_Parameters[0]["Module"]
            self.displayModule = self.list_Processes_Parameters[0]["Display"]
            self.processObserver = self.process.AddObserver(
                "ModifiedEvent", self.onProcessUpdate
            )

            del self.list_Processes_Parameters[0]

    def onProcessStarted(self):
        self.ui.label_LibsInstallation.setHidden(True)
        self.startTime = time.time()

        self.ui.progressBar.setValue(0)

        self.ui.LabelProgressPatient.setText(f"Patient : 0 / {self.nb_patient}")
        self.ui.LabelProgressExtension.setText(
            f"Extension : 1 / {self.nb_extension_launch}"
        )
        self.nb_extension_did = 0

        self.module_name_before = 0
        self.nb_change_bystep = 0

        self.RunningUI(True)
    
    def onCondaProcessUpdate(self):
        if os.path.isfile(self.log_path):
            self.ui.LabelProgressExtension.setText(
                f"Extension : {self.nb_extension_did} / {self.nb_extension_launch}"
            )
            time_progress = os.path.getmtime(self.log_path)
            line = self.logic.read_log_path(self.log_path)
            if (time_progress != self.time_log) and line:
                progress = line.strip()
            
                self.progress = int(progress)
                self.ui.LabelProgressPatient.setText(f"Patient : {self.progress}/{self.nb_patient}")
                
                progress_bar_value = round((self.progress) / self.nb_patient * 100,2)
                self.time_log = time_progress
                
                self.ui.progressBar.setValue(progress_bar_value)
                self.ui.progressBar.setFormat(f"{progress_bar_value:.2f}%")

    def onProcessUpdate(self, caller, event):
        currentTime = time.time() - self.startTime
        timer = format_timer(currentTime)

        self.ui.LabelTimer.setText(timer)
        progress = caller.GetProgress()
        self.module_name = caller.GetModuleTitle()
        self.ui.LabelNameExtension.setText(self.module_name)

        if self.module_name_before != self.module_name:
            self.ui.LabelProgressPatient.setText(f"Patient : 0 / {self.nb_patient}")
            self.nb_extension_did += 1
            self.ui.LabelProgressExtension.setText(
                f"Extension : {self.nb_extension_did} / {self.nb_extension_launch}"
            )
            self.ui.progressBar.setValue(0)

            self.module_name_before = self.module_name
            self.nb_change_bystep = 0

        if progress == 0:
            self.updateProgressBar = False

        if self.displayModule.isProgress(
            progress=progress, updateProgressBar=self.updateProgressBar
        ):
            progress_bar, message = self.displayModule()
            self.ui.progressBar.setValue(progress_bar)
            self.ui.LabelProgressPatient.setText(message)
            self.nb_change_bystep += 1

        if caller.GetStatus() & caller.Completed:
            if caller.GetStatus() & caller.ErrorsMask:
                # error
                logger.info("========= PROCESS COMPLETED WITH ERRORS =========")
                logger.info(self.process.GetOutputText())
                logger.error("========= ERROR DETAILS =========")
                error_text = self.process.GetErrorText()
                logger.error(f"CLI execution failed: \n{error_text}")
                self.onCancel()

            else:
                logger.info("========= PROCESS COMPLETED SUCCESSFULLY =========")
                logger.info(self.process.GetOutputText())
                try:
                    self.process = slicer.cli.run(
                        self.list_Processes_Parameters[0]["Process"],
                        None,
                        self.list_Processes_Parameters[0]["Parameter"],
                    )
                    self.module_name = self.list_Processes_Parameters[0]["Module"]
                    self.displayModule = self.list_Processes_Parameters[0]["Display"]
                    self.processObserver = self.process.AddObserver(
                        "ModifiedEvent", self.onProcessUpdate
                    )
                    del self.list_Processes_Parameters[0]
                except IndexError:
                    self.OnEndProcess()

    def OnEndProcess(self):
        """Function called when the process is finished."""
        self.ui.LabelProgressPatient.setText(f"Patient : 0 / {self.nb_patient}")
        self.ui.LabelProgressExtension.setText(
            f"Extension : {self.nb_extension_did} / {self.nb_extension_launch}"
        )
        self.ui.progressBar.setValue(0)

        self.module_name_before = self.module_name
        self.nb_change_bystep = 0
        total_time = time.time() - self.startTime
        average_time = total_time / self.nb_patient
        logger.info("PROCESS DONE.")
        logger.info(
            "Done in {} min and {} sec".format(
                int(total_time / 60), int(total_time % 60)
            )
        )
        logger.info(
            "Average time per patient : {} min and {} sec".format(
                int(average_time / 60), int(average_time % 60)
            )
        )
        self.RunningUI(False)
        self.RunningUI(False)

        stop_time = time.time()

        logger.info(f"Processing completed in {stop_time-self.startTime:.2f} seconds")

        s = PopUpWindow(
            title="Process Done",
            text="Successfully done in {} min and {} sec \nAverage time per Patient: {} min and {} sec".format(
                int(total_time / 60),
                int(total_time % 60),
                int(average_time / 60),
                int(average_time % 60),
            ),
        )
        s.exec_()

    def onCancel(self):
        try:
            self.process.Cancel()
        except Exception as e:
            self.logic.cancel_process()
            
        logger.warning("========= PROCESS CANCELED =========")

        self.RunningUI(False)

    def RunningUI(self, run=False):
        self.ui.ButtonOriented.setVisible(not run)

        self.ui.progressBar.setVisible(run)
        self.ui.LabelTimer.setVisible(run)

        self.HideComputeItems(run)
        
    def run_conda_tool(self):
        output_command = self.logic.conda.condaRunCommand(["which","dentalmodelseg"],self.logic.name_env).strip()
        clean_output = re.search(r"Result: (.+)", output_command)
        if clean_output:
            dentalmodelseg_path = clean_output.group(1).strip()
            dentalmodelseg_path_clean = dentalmodelseg_path.replace("\\n","")
        else:
            logger.error("Error: Unable to find dentalmodelseg path.")
            return
        
        args = self.list_Processes_Parameters[0]["Parameter"]
        logger.debug(f"Arguments: {args}")
        conda_exe = self.logic.conda.getCondaExecutable()
        command = [conda_exe, "run", "-n", self.logic.name_env, "python" ,"-m", f"CrownSegmentationcli"]
        for key, value in args.items():
            if key in ["out","input_csv","vtk_folder","dentalmodelseg_path"]:
                value = self.logic.windows_to_linux_path(value)
            if key == "dentalmodelseg_path":
                value = dentalmodelseg_path_clean
            command.append(f"\"{value}\"")
        logger.debug("="*50)
        logger.debug(f"Command: {command}")

        # running in // to not block Slicer
        process = threading.Thread(target=self.logic.condaRunCommand, args=(command,))
        process.start()
        self.ui.LabelTimer.setHidden(False)
        self.ui.LabelTimer.setText(f"Time : 0.00s")
        previous_time = self.startTime
        while process.is_alive():
            self.ui.ButtonCancel.setVisible(True)
            self.onCondaProcessUpdate()
            slicer.app.processEvents()
            current_time = time.time()
            gap=current_time-previous_time
            if gap>0.3:
                currentTime = time.time() - self.startTime
                previous_time = currentTime
                timer = format_timer(currentTime)
                
                self.ui.LabelTimer.setText(timer)

        del self.list_Processes_Parameters[0]

    """

            8888888888 888     888 888b    888  .d8888b.      8888888 888b    888 8888888 88888888888
            888        888     888 8888b   888 d88P  Y88b       888   8888b   888   888       888
            888        888     888 88888b  888 888    888       888   88888b  888   888       888
            8888888    888     888 888Y88b 888 888              888   888Y88b 888   888       888
            888        888     888 888 Y88b888 888              888   888 Y88b888   888       888
            888        888     888 888  Y88888 888    888       888   888  Y88888   888       888
            888        Y88b. .d88P 888   Y8888 Y88b  d88P       888   888   Y8888   888       888
            888         "Y88888P"  888    Y888  "Y8888P"      8888888 888    Y888 8888888     888




    """

    def initCheckbox(self, method, layout, tohide: qt.QLabel):
        """Function to create the checkbox at the beginning of the program"""
        if not tohide is None:
            tohide.setHidden(True)
        
        # Check if dark mode
        app = qt.QApplication.instance()
        palette = app.palette()
        bg_color = palette.color(qt.QPalette.Window)
        is_dark_mode = bg_color.lightness() < 128
        
        checkbox_stylesheet = """
          QCheckBox {
            color: #ffffff;
            background-color: transparent;
            font-weight: 500;
          }
          QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555555;
            border-radius: 3px;
            background-color: #3c3c3c;
          }
          QCheckBox::indicator:hover {
            border: 1px solid #5dade2;
          }
          QCheckBox::indicator:checked {
            border: 1px solid #5dade2;
            background-color: #5dade2;
          }
          QCheckBox::indicator:checked:hover {
            border: 1px solid #7bbcef;
            background-color: #7bbcef;
          }
        """
        
        dic = method.DicLandmark()
        dicchebox = {}
        dicchebox2 = {}
        for type, tab in dic.items():
            Tab = QTabWidget()
            layout.addWidget(Tab)
            listcheckboxlandmark = []
            listcheckboxlandmark2 = []

            all_checkboxtab = self.CreateMiniTab(Tab, "All", 0)
            for i, (name, listlandmark) in enumerate(tab.items()):
                widget = self.CreateMiniTab(Tab, name, i + 1)
                for landmark in listlandmark:
                    checkbox = QCheckBox()
                    checkbox2 = QCheckBox()
                    checkbox.setText(landmark)
                    checkbox2.setText(landmark)
                    if is_dark_mode:
                        checkbox.setStyleSheet(checkbox_stylesheet)
                        checkbox2.setStyleSheet(checkbox_stylesheet)
                    checkbox2.toggled.connect(checkbox.setChecked)
                    checkbox.toggled.connect(checkbox2.setChecked)
                    widget.addWidget(checkbox)
                    all_checkboxtab.addWidget(checkbox2)

                    listcheckboxlandmark.append(checkbox)
                    listcheckboxlandmark2.append(checkbox2)

            dicchebox[type] = listcheckboxlandmark
            dicchebox2[type] = listcheckboxlandmark2

        method.setcheckbox(dicchebox)
        method.setcheckbox2(dicchebox2)

        return dicchebox, dicchebox2

    def CreateMiniTab(self, tab_widget: QTabWidget, name: str, index: int):
        """Function to create a new tab in the tabWidget"""
        new_widget = QWidget()
        new_widget.resize(tab_widget.size)

        layout = QGridLayout(new_widget)

        scr_box = QScrollArea(new_widget)
        scr_box.resize(tab_widget.size)

        layout.addWidget(scr_box, 0, 0)

        new_widget2 = QWidget(scr_box)
        layout2 = QVBoxLayout(new_widget2)

        scr_box.setWidgetResizable(True)
        scr_box.setWidget(new_widget2)

        tab_widget.insertTab(index, new_widget, name)

        return layout2

    def HideComputeItems(self, run=False):
        self.ui.ButtonOriented.setVisible(not run)

        self.ui.ButtonCancel.setVisible(run)

        self.ui.LabelProgressPatient.setVisible(run)
        self.ui.LabelProgressExtension.setVisible(run)
        self.ui.LabelNameExtension.setVisible(run)
        self.ui.progressBar.setVisible(run)

        self.ui.LabelTimer.setVisible(run)

    def format_time(self, seconds):
        """Seconds as HH:MM:SS."""
        return format_elapsed(seconds)

    def update_ui_time(self, start_time, previous_time):
        """Elapsed time since `start_time`, formatted for the installation label.

        `previous_time` is kept for signature parity with the call sites, which
        pass it but never update their own copy. It used to throttle this to one
        update every 0.3s and return None in between, which is what wrote
        "time: None" into the label. Formatting unconditionally is both simpler
        and correct.
        """
        self.elapsed_time = elapsed_since(start_time)
        return self.format_time(self.elapsed_time)

    def initCheckboxIOS(
        self,
        method: Auto_IOS,
        layout: QGridLayout,
        tohide: QLabel,
        layout2: QVBoxLayout,
        occlusion: QCheckBox,
    ):
        """Function to create the checkbox at the beginning of the program for IOS"""
        diccheckbox = {"Adult": {}, "Child": {}}
        tohide.setHidden(True)
        
        # Check if dark mode
        app = qt.QApplication.instance()
        palette = app.palette()
        bg_color = palette.color(qt.QPalette.Window)
        is_dark_mode = bg_color.lightness() < 128
        
        # Create stylesheet based on mode
        if is_dark_mode:
            checkbox_stylesheet = """
              QCheckBox {
                color: #ffffff;
                background-color: transparent;
                font-weight: 500;
              }
              QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #3c3c3c;
              }
              QCheckBox::indicator:hover {
                border: 1px solid #5dade2;
              }
              QCheckBox::indicator:checked {
                border: 1px solid #5dade2;
                background-color: #5dade2;
              }
              QCheckBox::indicator:checked:hover {
                border: 1px solid #7bbcef;
                background-color: #7bbcef;
              }
            """
        else:
            checkbox_stylesheet = """
              QCheckBox {
                color: #000000;
                background-color: transparent;
                font-weight: 500;
              }
              QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #ffffff;
              }
              QCheckBox::indicator:hover {
                border: 1px solid #0099cc;
              }
              QCheckBox::indicator:checked {
                border: 1px solid #0099cc;
                background-color: #0099cc;
              }
              QCheckBox::indicator:checked:hover {
                border: 1px solid #0077aa;
                background-color: #0077aa;
              }
            """
        
        dic_teeth = {
            1: "A",
            2: "B",
            3: "C",
            4: "D",
            5: "E",
            6: "F",
            7: "G",
            8: "H",
            9: "I",
            10: "J",
            11: "T",
            12: "S",
            13: "R",
            14: "Q",
            15: "P",
            16: "O",
            17: "N",
            18: "M",
            19: "L",
            20: "K",
        }
        upper = []
        lower = []

        list = []
        for i in range(1, 11):
            label = QLabel()
            pixmap = QPixmap(self.resourcePath(f"Image/{i}_resize_child.png"))
            label.setPixmap(pixmap)
            check = QCheckBox()
            check.setText(dic_teeth[i])
            check.setEnabled(False)
            check.setStyleSheet(checkbox_stylesheet)

            layout.addWidget(label, 0, i + 3)
            layout.addWidget(check, 1, i + 3)
            list.append(check)
        diccheckbox["Child"]["Upper"] = list
        upper += list

        dic = {
            1: "UR8",
            2: "UR7",
            3: "UR6",
            4: "UR5",
            5: "UR4",
            6: "UR3",
            7: "UR2",
            8: "UR1",
            9: "UL1",
            10: "UL2",
            11: "UL3",
            12: "UL4",
            13: "UL5",
            14: "UL6",
            15: "UL7",
            16: "UL8",
            17: "LL8",
            18: "LL7",
            19: "LL6",
            20: "LL5",
            21: "LL4",
            22: "LL3",
            23: "LL2",
            24: "LL1",
            25: "LR1",
            26: "LR2",
            27: "LR3",
            28: "LR4",
            29: "LR5",
            30: "LR6",
            31: "LR7",
            32: "LR8",
        }

        list = []
        for i in range(1, 17):
            label = QLabel()
            pixmap = QPixmap(self.resourcePath(f"Image/{i}_resize.png"))
            label.setPixmap(pixmap)
            check = QCheckBox()
            check.setText(dic[i])
            check.setEnabled(False)
            check.setStyleSheet(checkbox_stylesheet)

            layout.addWidget(label, 2, i)
            layout.addWidget(check, 3, i)

            list.append(check)

        diccheckbox["Adult"]["Upper"] = list
        upper += list

        list = []
        for i in range(1, 17):
            label = QLabel()
            pixmap = QPixmap(self.resourcePath(f"Image/{i+16}_resize.png"))
            label.setPixmap(pixmap)
            check = QCheckBox()
            check.setText(dic[i + 16])
            check.setEnabled(False)
            check.setStyleSheet(checkbox_stylesheet)

            layout.addWidget(check, 4, 17 - i)
            layout.addWidget(label, 5, 17 - i)

            list.append(check)

        diccheckbox["Adult"]["Lower"] = list
        lower += list

        list = []
        for i in range(1, 11):
            label = QLabel()
            pixmap = QPixmap(self.resourcePath(f"Image/{i+10}_resize_child.png"))
            label.setPixmap(pixmap)
            check = QCheckBox()
            check.setText(dic_teeth[i + 10])
            check.setEnabled(False)
            check.setStyleSheet(checkbox_stylesheet)

            layout.addWidget(check, 6, i + 3)
            layout.addWidget(label, 7, i + 3)

            list.append(check)

        diccheckbox["Child"]["Lower"] = list
        lower += list

        upper_checbox = QCheckBox()
        upper_checbox.setText("Upper")
        upper_checbox.toggled.connect(
            partial(self.onJawToggled, {"Upper": upper, "Lower": lower}, "Upper")
        )
        layout.addWidget(upper_checbox, 3, 0)
        lower_checkbox = QCheckBox()
        lower_checkbox.setText("Lower")
        lower_checkbox.toggled.connect(
            partial(self.onJawToggled, {"Upper": upper, "Lower": lower}, "Lower")
        )
        layout.addWidget(lower_checkbox, 4, 0)

        upper_checbox.toggled.connect(
            partial(self.logic.UpperLowerChooseOcclusion, lower_checkbox, occlusion)
        )
        lower_checkbox.toggled.connect(
            partial(self.logic.UpperLowerChooseOcclusion, upper_checbox, occlusion)
        )

        if isinstance(method, Semi_IOS):
            dic1, dic2 = self.initCheckbox(method, layout2, None)

            method.setcheckbox(
                {
                    "Teeth": diccheckbox,
                    "Landmark": dic1,
                    "Jaw": {"Upper": upper_checbox, "Lower": lower_checkbox},
                    "Occlusion": occlusion,
                }
            )
            method.setcheckbox2(
                {
                    "Teeth": diccheckbox,
                    "Landmark": dic2,
                    "Jaw": {"Upper": upper_checbox, "Lower": lower_checkbox},
                    "Occlusion": occlusion,
                }
            )
        else:

            method.setcheckbox(
                {
                    "Teeth": diccheckbox,
                    "Jaw": {"Upper": upper_checbox, "Lower": lower_checkbox},
                    "Occlusion": occlusion,
                }
            )
            method.setcheckbox2(
                {
                    "Teeth": diccheckbox,
                    "Jaw": {"Upper": upper_checbox, "Lower": lower_checkbox},
                    "Occlusion": occlusion,
                }
            )

    """
                          .d88888b.  88888888888 888    888 8888888888 8888888b.   .d8888b.
                         d88P" "Y88b     888     888    888 888        888   Y88b d88P  Y88b
                         888     888     888     888    888 888        888    888 Y88b.
                         888     888     888     8888888888 8888888    888   d88P  "Y888b.
                         888     888     888     888    888 888        8888888P"      "Y88b.
                         888     888     888     888    888 888        888 T88b         "888
                         Y88b. .d88P     888     888    888 888        888  T88b  Y88b  d88P
                          "Y88888P"      888     888    888 8888888888 888   T88b  "Y8888P"
    """

    def onCheckRequirements(self):
        if not self.logic.isCondaSetUp:
            message_box = qt.QMessageBox()
            text = textwrap.dedent("""
            SlicerConda is not set up, please click
            <a href=\"https://github.com/DCBIA-OrthoLab/SlicerConda/\">here</a> for installation.
            """).strip()
            message_box.information(None, "Information", text)
            return False
        
        if platform.system() == "Windows":
            self.ui.label_LibsInstallation.setHidden(False)
            self.ui.label_LibsInstallation.setText(f"Checking if wsl is installed, this task may take a moments")
            
            if self.logic.testWslAvailable():
                self.ui.label_LibsInstallation.setText(f"WSL installed")
                if not self.logic.check_lib_wsl():
                    self.ui.label_LibsInstallation.setText(f"Checking if the required librairies are installed, this task may take a moments")
                    message_box = qt.QMessageBox()
                    text = textwrap.dedent("""
                        WSL doesn't have all the necessary libraries, please download the installer
                        and follow the instructions
                        <a href=\"https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_WSL2.zip\">here</a>
                        for installation. The link may be blocked by Chrome, just authorize it.""").strip()

                    message_box.information(None, "Information", text)
                    return False
                
            else : # if wsl not install, ask user to install it ans stop process
                message_box = qt.QMessageBox()
                text = textwrap.dedent("""
                    WSL is not installed, please download the installer and follow the instructions
                    <a href=\"https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_WSL2.zip\">here</a>
                    for installation. The link may be blocked by Chrome, just authorize it.""").strip()

                message_box.information(None, "Information", text)
                return False
            
        
        ## MiniConda
        
        
        self.ui.label_LibsInstallation.setText(f"Checking if miniconda is installed")
        if "no setup" in self.logic.conda.condaRunCommand([self.logic.conda.getCondaExecutable(),"--version"]):
            message_box = qt.QMessageBox()
            text = textwrap.dedent("""
            Code can't be launch. \nConda is not setup.
            Please go the extension CondaSetUp in SlicerConda to do it.""").strip()
            message_box.information(None, "Information", text)
            return False
        
        
        ## shapeAXI


        self.ui.label_LibsInstallation.setText(f"Checking if environnement exists")
        if not self.logic.conda.condaTestEnv(self.logic.name_env) : # check is environnement exist, if not ask user the permission to do it
            user_response = slicer.util.confirmYesNoDisplay("The environnement to run the classification doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
            if user_response :
                start_time = time.time()
                previous_time = start_time
                formatted_time = self.format_time(0)
                self.ui.label_LibsInstallation.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
                process = self.logic.install_shapeaxi()
                
                while self.logic.process.is_alive():
                    slicer.app.processEvents()
                    formatted_time = self.update_ui_time(start_time, previous_time)
                    self.ui.label_LibsInstallation.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
            
                start_time = time.time()
                previous_time = start_time
                formatted_time = self.format_time(0)
                text = textwrap.dedent(f"""
                Installation of librairies into the new environnement.
                This task may take a few minutes.\ntime: {formatted_time}""").strip()
                self.ui.label_LibsInstallation.setText(text)
            else:
                return False
        else:
            self.ui.label_LibsInstallation.setText(f"Ennvironnement already exists")
            
        
        ## pytorch3d


        self.ui.label_LibsInstallation.setText(f"Checking if pytorch3d is installed")
        if "Error" in self.logic.check_if_pytorch3d() : # pytorch3d not installed or badly installed
            process = self.logic.install_pytorch3d()
            start_time = time.time()
            previous_time = start_time
            
            while self.logic.process.is_alive():
                slicer.app.processEvents()
                formatted_time = self.update_ui_time(start_time, previous_time)
                text = textwrap.dedent(f"""
                Installation of pytorch into the new environnement.
                This task may take a few minutes.\ntime: {formatted_time}
                """).strip()
                self.ui.label_LibsInstallation.setText(text)
        else:
            self.ui.label_LibsInstallation.setText(f"pytorch3d is already installed")
            logger.info("pytorch3d already installed")

        self.all_installed = True
        return True
    
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        if self.logic.cliNode is not None:
            # if self.logic.cliNode.GetStatus() & self.logic.cliNode.Running:
            self.logic.cliNode.Cancel()

        # Apply dark mode if enabled
        self.applyDarkModeStyles()

        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(
            self._parameterNode,
            vtk.vtkCommand.ModifiedEvent,
            self.updateGUIFromParameterNode,
        )

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

    def setParameterNode(self, input_parameter_node):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        # if inputParameterNode:
        self.setParameterNode(self.logic.getParameterNode())

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        self._parameterNode = input_parameter_node
        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(
            self._parameterNode.GetNodeReference("InputVolume")
        )
        self.ui.outputSelector.setCurrentNode(
            self._parameterNode.GetNodeReference("OutputVolume")
        )
        self.ui.invertedOutputSelector.setCurrentNode(
            self._parameterNode.GetNodeReference("OutputVolumeInverse")
        )
        # self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        self.ui.invertOutputCheckBox.checked = (
            self._parameterNode.GetParameter("Invert") == "true"
        )

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        was_modified = (
            self._parameterNode.StartModify()
        )  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID(
            "InputVolume", self.ui.inputSelector.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "OutputVolume", self.ui.outputSelector.currentNodeID
        )
        self._parameterNode.SetParameter(
            "Invert", "true" if self.ui.invertOutputCheckBox.checked else "false"
        )
        self._parameterNode.SetNodeReferenceID(
            "OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID
        )

        self._parameterNode.EndModify(was_modified)

    def applyDarkModeStyles(self):
        app = qt.QApplication.instance()
        palette = app.palette()
        bg_color = palette.color(qt.QPalette.Window)
        if bg_color.lightness() < 128:
            # Complete dark mode stylesheet
            dark_stylesheet = """
ctkCollapsibleButton {
  background-color: #3c3c3c;
  border: 1px solid #555555;
  border-radius: 6px;
  margin-bottom: 8px;
  font-weight: 600;
  padding: 6px 10px;
  color: #ffffff;
}
ctkCollapsibleButton:hover {
  border: 1px solid #5dade2;
  background-color: #454545;
}
QLineEdit, QTextEdit {
  background-color: #3c3c3c;
  border: 1px solid #555555;
  border-radius: 4px;
  padding: 6px;
  color: #ffffff;
  selection-background-color: #5dade2;
}
QLineEdit:focus, QTextEdit:focus {
  border: 2px solid #5dade2;
}
QComboBox {
  background-color: #3c3c3c;
  border: 1px solid #555555;
  border-radius: 4px;
  padding: 4px 6px;
  color: #ffffff;
}
QComboBox:focus {
  border: 2px solid #5dade2;
}
QComboBox::drop-down {
  width: 20px;
  border: none;
}
QComboBox QAbstractItemView {
  background-color: #3c3c3c;
  color: #ffffff;
  selection-background-color: #5dade2;
}
QLabel {
  color: #ffffff;
  font-weight: 500;
  background-color: transparent;
}
QPushButton {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5dade2, stop:1 #3498db);
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 600;
  font-size: 10pt;
  padding: 8px;
  margin-top: 4px;
}
QPushButton:hover:!pressed {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7bbcef, stop:1 #5dade2);
}
QPushButton:pressed {
  background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1e638d);
}
QPushButton:disabled {
  background-color: #555555;
  color: #888888;
}
QCheckBox {
  color: #ffffff;
  font-weight: 500;
  spacing: 6px;
  background-color: transparent;
}
QCheckBox::indicator {
  width: 18px;
  height: 18px;
  border: 1px solid #555555;
  border-radius: 3px;
  background-color: #3c3c3c;
}
QCheckBox::indicator:hover {
  border: 1px solid #5dade2;
}
QCheckBox::indicator:checked {
  width: 18px;
  height: 18px;
  border: 1px solid #5dade2;
  border-radius: 3px;
  background-color: #5dade2;
}
QCheckBox::indicator:checked:hover {
  border: 1px solid #7bbcef;
  background-color: #7bbcef;
}
QProgressBar {
  border: 1px solid #555555;
  border-radius: 4px;
  background-color: #3c3c3c;
  padding: 2px;
  color: #ffffff;
}
QProgressBar::chunk {
  background-color: #5dade2;
  border-radius: 3px;
}
QSpinBox, QDoubleSpinBox {
  background-color: #3c3c3c;
  border: 1px solid #555555;
  border-radius: 4px;
  padding: 4px 6px;
  color: #ffffff;
}
QSpinBox:focus, QDoubleSpinBox:focus {
  border: 2px solid #5dade2;
}
QSlider::groove:horizontal {
  background-color: #555555;
  border-radius: 4px;
}
QSlider::handle:horizontal {
  background-color: #5dade2;
  width: 12px;
  margin: -4px 0;
  border-radius: 6px;
}
QSlider::handle:horizontal:hover {
  background-color: #7bbcef;
}
qMRMLNodeComboBox {
  background-color: #3c3c3c;
  border: 1px solid #555555;
  border-radius: 4px;
  padding: 4px 6px;
  color: #ffffff;
}
qMRMLNodeComboBox:focus {
  border: 2px solid #5dade2;
}
            """
            self.uiWidget.setStyleSheet(dark_stylesheet)
            
            # Update QLineEdit, QComboBox, and QLabel for dark mode
            self._updateLineEditAndComboBoxDarkMode(self.uiWidget)
            
            # Update dynamically created checkboxes
            self._updateDynamicCheckboxesDarkMode()

    def _updateLineEditAndComboBoxDarkMode(self, parent):
        """Shared recursive pass, kept as a method for the existing call sites."""
        update_line_edit_and_combo_box(parent)
    
    def _updateDynamicCheckboxesDarkMode(self):
      """
      Apply dark mode styles to all dynamically created checkboxes.
      """
      checkbox_stylesheet = """
        QCheckBox {
          color: #ffffff;
          font-weight: 500;
          spacing: 6px;
        }
        QCheckBox::indicator {
          width: 18px;
          height: 18px;
          border: 1px solid #555555;
          border-radius: 3px;
          background-color: #3c3c3c;
        }
        QCheckBox::indicator:hover {
          border: 1px solid #5dade2;
        }
        QCheckBox::indicator:checked {
          width: 18px;
          height: 18px;
          border: 1px solid #5dade2;
          border-radius: 3px;
          background-color: #5dade2;
        }
        QCheckBox::indicator:checked:hover {
          border: 1px solid #7bbcef;
          background-color: #7bbcef;
        }
      """
      
      # Style all QCheckBox widgets recursively from uiWidget
      self._styleAllCheckboxes(self.uiWidget, checkbox_stylesheet)
    
    def _styleAllCheckboxes(self, parent, stylesheet):
      """
      Recursively find and style all QCheckBox widgets in the widget tree.
      """
      if isinstance(parent, qt.QCheckBox):
        try:
          parent.setStyleSheet(stylesheet)
        except (AttributeError, RuntimeError):
            pass
      
      # Recursively process all children
      if hasattr(parent, 'children'):
        for child in parent.children():
          self._styleAllCheckboxes(child, stylesheet)

"""
                d8888  .d8888b.   .d88888b.      888       .d88888b.   .d8888b.  8888888  .d8888b.
               d88888 d88P  Y88b d88P" "Y88b     888      d88P" "Y88b d88P  Y88b   888   d88P  Y88b
              d88P888 Y88b.      888     888     888      888     888 888    888   888   888    888
             d88P 888  "Y888b.   888     888     888      888     888 888          888   888
            d88P  888     "Y88b. 888     888     888      888     888 888  88888   888   888
           d88P   888       "888 888     888     888      888     888 888    888   888   888    888
          d8888888888 Y88b  d88P Y88b. .d88P     888      Y88b. .d88P Y88b  d88P   888   Y88b  d88P
         d88P     888  "Y8888P"   "Y88888P"      88888888  "Y88888P"   "Y8888P88 8888888  "Y8888P"
"""


class ASOLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.isCondaSetUp = False
        self.conda = self.init_conda()
        self.name_env = "shapeaxi"
        self.cliNode = None
        self.python_version = "3.12"
        
    def init_conda(self):
        """The SlicerConda entry point for this platform, or False without it."""
        call = init_conda_call()
        self.isCondaSetUp = bool(call)
        return call
        
    def run_conda_command(self, target, command):
        self.process = threading.Thread(target=target, args=command) #run in parallel to not block slicer
        self.process.start()
        
    def install_shapeaxi(self):
        # Only SimpleITK here. Everything that depends on torch - torch itself,
        # torchvision, ocnn, pytorch3d, shapeaxi - is installed afterwards by
        # install_pytorch, which is the only place that can pass the
        # --index-url selecting a CUDA build. Asked for here, pip took PyPI's
        # default variant (2.12.1+cu130) and no pytorch3d wheel is published
        # for it; shapeaxi on top of that fails outright, since it declares
        # pytorch3d and PyPI carries no distribution for it at all.
        self.run_conda_command(target=self.conda.condaCreateEnv, command=(self.name_env,self.python_version,["SimpleITK"],)) #run in parallel to not block slicer
        
    def check_if_pytorch3d(self):
        conda_exe = self.conda.getCondaExecutable()
        # Unquoted where nothing strips the quotes: kept, they turn the body
        # into a single string literal that Python evaluates and exits 0 on,
        # so the check reported pytorch3d present in an env without it.
        command = [conda_exe, "run", "-n", self.name_env, "python" ,"-c", condaQuote(self.conda, "import pytorch3d;import pytorch3d.renderer;import shapeaxi.dental_model_seg as d;d.saxi_nets_lightning.DentalModelSeg")]
        return self.conda.condaRunCommand(command)
    
    def install_pytorch3d(self):
        result_pythonpath = self.check_pythonpath_windows("ADTLib.env.install_pytorch")
        if not result_pythonpath :
            self.give_pythonpath_windows()
            result_pythonpath = self.check_pythonpath_windows("ADTLib.env.install_pytorch")
        
        if result_pythonpath :
            conda_exe = self.conda.getCondaExecutable()
            path_pip = self.conda.getCondaPath()+f"/envs/{self.name_env}/bin/pip"
            command = [conda_exe, "run", "-n", self.name_env, "python" ,"-m", f"ADTLib.env.install_pytorch",path_pip]

        self.run_conda_command(target=self.conda.condaRunCommand, command=(command,))
        
    def check_lib_wsl(self) -> bool:
        """Whether WSL carries the system libraries the tools need."""
        return wsl_libraries_present()
    
    def check_pythonpath_windows(self, file):
        """Whether `file` is importable by the Python of this module's environment."""
        return check_pythonpath(self.conda, self.name_env, file)
    
    def give_pythonpath_windows(self):
        """Publish Slicer's module search paths into this module's environment."""
        give_pythonpath(self.conda, self.name_env)
        
    def windows_to_linux_path(self, windows_path):
        """A Windows path as WSL sees it."""
        return windows_to_linux_path_shared(windows_path)
    
    def cancel_process(self):
        if platform.system() == 'Windows':
            self.subpro.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(self.subpro.pid), signal.SIGTERM)
        logger.warning("Cancellation requested. Terminating process...")

        self.subpro.wait() ## important
        self.cancel = True
    
    def check_cli_script(self):
        if not self.check_pythonpath_windows("PRE_ASO_IOS"):
            self.give_pythonpath_windows()
            results = self.check_pythonpath_windows("PRE_ASO_IOS")
            
        if not self.check_pythonpath_windows("SEMI_ASO_IOS"):
            self.give_pythonpath_windows()
            results = self.check_pythonpath_windows("SEMI_ASO_IOS")
            
        if not self.check_pythonpath_windows("CrownSegmentationcli"):
            self.give_pythonpath_windows()
            results = self.check_pythonpath_windows("CrownSegmentationcli")
            
    def condaRunCommand(self, command: list[str]):
        '''
        Runs a command in a specified Conda environment, handling different operating systems.
        
        copy paste from SlicerConda and change the process line to be able to get the stderr/stdout
        and cancel the process without blocking slicer
        '''
        path_activate = self.conda.getActivateExecutable()

        if path_activate=="None":
            return "Path to conda no setup"

        if platform.system() == "Windows":
            command_execute = f"source {path_activate} {self.name_env} &&"
            for com in command :
                command_execute = command_execute+ " "+com

            user = self.conda.getUser()
            command_to_execute = ["wsl", "--user", user,"--","bash","-c", command_execute]
            logger.debug(f"Command to execute in condaRunCommand: {command_to_execute}")

            self.subpro = subprocess.Popen(command_to_execute, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding='utf-8', errors='replace', env=slicer.util.startupEnvironment(),
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # For Windows
                              )
        else:
            path_conda_exe = self.conda.getCondaExecutable()
            command_execute = f"{path_conda_exe} run -n {self.name_env}"
            for com in command :
                command_execute = command_execute+ " "+com

            logger.debug(f"Command to execute in conda run: {command_execute}")
            self.subpro = subprocess.Popen(command_execute, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', env=slicer.util.startupEnvironment(), executable="/bin/bash", preexec_fn=os.setsid)
    
        self.stdout, self.stderr = self.subpro.communicate()

    def iterillimeted(self, iter):
        out = []
        if isinstance(iter, dict):
            iter = list(iter.values())

        for thing in iter:
            if isinstance(thing, (dict, list, set)):
                out += self.iterillimeted(thing)
            else:
                out.append(thing)

        return out
    def UpperLowerCheckbox(self, all_checkbox: dict, jaw, boolean):
        """Enable one jaw's landmark checkboxes, and clear what it disables.

        The refresh that used to follow -- `self.enableCheckbox()` -- belongs to
        the widget: it reads ActualMeth, three line edits and self.type, none of
        which exist here. Called on the Logic it raised
        `AttributeError: 'ASOLogic' object has no attribute 'enableCheckbox'`
        on every single toggle, so the refresh never happened. The widget calls
        it now, in `onJawToggled`.
        """
        for checkbox in all_checkbox[jaw]:
            checkbox.setEnabled(boolean)
            if (not boolean) and checkbox.isChecked():
                checkbox.setChecked(False)

    def OcclusionCheckbox(
        self, Upper: QCheckBox, Lower: QCheckBox, all_checkbox: dict, boolean: bool
    ):
        if boolean:
            if Upper.isChecked() and Lower.isChecked():
                Lower.setChecked(False)
                Lower.setEnabled(True)

    def UpperLowerChooseOcclusion(
        self, opposit_jaw: QCheckBox, occlusion_checkbox: QCheckBox, booleean: bool
    ):
        if booleean and occlusion_checkbox.isChecked() and opposit_jaw.isChecked():
            opposit_jaw.setChecked(False)
    def read_log_path(self, log_path):
      with open(log_path, 'r') as f:
          line = f.readline()
          if line != '':
              return line
  


