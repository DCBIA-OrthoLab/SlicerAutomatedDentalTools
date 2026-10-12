import os
import shutil
import zipfile
import urllib.request
import vtk
import slicer
import sys
import ctypes
import qt
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
import importlib

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

from ADTLib.logging_setup import get_logger



# ===== Logging Configuration =====
logger = get_logger("SurgMovPred")


# Library dependency management
def check_lib_installed(import_name: str) -> bool:
    """
    Silently checks if a Python library is installed and accessible.
    'import_name' is the name used in the code (e.g., 'llama_cpp').
    """
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False
    
def install_function(list_libs: list) -> None:
    """
    Installs a list of packages via pip in the 3D Slicer environment.
    Assumes the user has already given permission.
    """

    original_cc = os.environ.get("CC")
    original_cxx = os.environ.get("CXX")

    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"

    for lib in list_libs:
        slicer.util.showStatusMessage(f"Installing {lib}... Please wait.")
        slicer.app.processEvents()

        try:
            if lib == "llama-cpp-python":
                slicer.util.pip_install("llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
            else:
                slicer.util.pip_install(lib)

            slicer.util.showStatusMessage(f"{lib} successfully installed!", 3000)
            logger.info(f"Successfully installed {lib}")

        except Exception as e:
            logger.error(f"Failed to install {lib}: {str(e)}")
            slicer.util.errorDisplay(f"Failed to install {lib}.\nError: {str(e)}")

    if original_cc is not None:
        os.environ["CC"] = original_cc
    else:
        del os.environ["CC"]

    if original_cxx is not None:
        os.environ["CXX"] = original_cxx
    else:
        del os.environ["CXX"]


def ensure_mac_openmp():
    """
    Downloads libomp.dylib and places it directly in Slicer's CLI modules folder
    so that all subprocesses (SurgMovPred_CLI) find it natively.
    """
    if sys.platform != "darwin":
        return

    # 1. Locate the CLI modules folder of the current Slicer application
    # Based on Slicer's standard directory layout
    slicer_bin_dir = os.path.dirname(sys.executable) # Contains /Applications/Slicer.app/Contents/bin
    slicer_contents_dir = os.path.dirname(slicer_bin_dir) # /Applications/Slicer.app/Contents

    # Look for the dynamic lib/Slicer-X.XX/cli-modules folder
    lib_dir = os.path.join(slicer_contents_dir, "lib")
    cli_modules_dir = None
    
    if os.path.exists(lib_dir):
        for item in os.listdir(lib_dir):
            if item.startswith("Slicer-"):
                potential_cli_dir = os.path.join(lib_dir, item, "cli-modules")
                if os.path.exists(potential_cli_dir):
                    cli_modules_dir = potential_cli_dir
                    break

    # Fall back to the current module's folder if the specific one isn't found
    if not cli_modules_dir:
        cli_modules_dir = os.path.dirname(os.path.realpath(__file__))

    target_libomp_path = os.path.join(cli_modules_dir, "libomp.dylib")

    # 2. Download it if it isn't already there
    if not os.path.exists(target_libomp_path):
        slicer.util.showStatusMessage("Installing macOS compatibility layer for CLI...")
        url = "https://mac.r-project.org/openmp/openmp-14.0.6-darwin20-Release.tar.gz"
        
        module_dir = os.path.dirname(os.path.realpath(__file__))
        tar_path = os.path.join(module_dir, "openmp.tar.gz")
        
        try:
            import urllib.request
            import tarfile
            
            urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path, "r:gz") as tar:
                member = tar.getmember("usr/local/lib/libomp.dylib")
                f = tar.extractfile(member)
                if f:
                    with open(target_libomp_path, "wb") as dest:
                        dest.write(f.read())
                        
            if os.path.exists(tar_path):
                os.remove(tar_path)
            logger.info(f"libomp.dylib successfully deployed to CLI directory: {target_libomp_path}")
        except Exception as e:
            logger.error(f"Failed to deploy libomp.dylib to CLI directory: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            return

    # 3. Also load the dylib in the parent process (Slicer UI)
    try:
        ctypes.CDLL(target_libomp_path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        logger.error(f"Failed to load libomp.dylib in main process: {e}")

def check_dependencies() -> bool:
    """
    Checks dependencies when the Apply button is clicked.
    Returns True if everything is ready, False if it should be cancelled.
    Attempts up to 2 verifications with proper logging.
    """

    if sys.platform == "darwin":
        ensure_mac_openmp()

    # Maps the import name (key) to the pip package name (value)
    DEPENDENCIES = {
        "pandas": "pandas",
        "joblib": "joblib",
        "openpyxl": "openpyxl",
        "sklearn": "scikit-learn",
        "lightgbm":"lightgbm"
    }
    slicer.util.pip_install("numpy==2.4.0")

    max_retries = 1
    for attempt in range(max_retries + 1):
        missing_import_names = []

        # 1. Check via the import name (e.g. "sklearn")
        for import_name in DEPENDENCIES.keys():
            if not check_lib_installed(import_name):
                missing_import_names.append(import_name)

        if not missing_import_names:
            logger.info("All dependencies verified and available.")
            return True

        if attempt < max_retries:
            # 2. Get the actual pip package names (e.g. "scikit-learn")
            libs_to_install = [DEPENDENCIES[imp] for imp in missing_import_names]
            
            libs_str = "\n".join([f"- {lib}" for lib in libs_to_install])

            msg = (
                "The SurgMovPred module requires the following libraries to function:\n\n"
                f"{libs_str}\n\n"
                "Do you agree to modify Slicer's environment to install them? "
                "This may take a few minutes."
            )

            if slicer.util.confirmOkCancelDisplay(msg):
                logger.info(f"Installing missing dependencies: {libs_to_install}")
                install_function(libs_to_install)
                slicer.app.processEvents()
            else:
                logger.warning("Installation cancelled by user.")
                slicer.util.warningDisplay("Installation cancelled. Extraction has been stopped.")
                return False
        else:
            libs_to_install = [DEPENDENCIES[imp] for imp in missing_import_names]
            libs_str = "\n".join([f"- {lib}" for lib in libs_to_install])
            
            logger.error(f"Failed to install required dependencies: {libs_to_install}")
            error_msg = (
                "Failed to install required dependencies:\n\n"
                f"{libs_str}\n\n"
                "Please restart Slicer and try again."
            )
            slicer.util.errorDisplay(error_msg)
            return False

    return False

class SurgMovPred(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SurgMovPred")
        self.parent.categories = ["Automated Dental Tools" ]
        self.parent.dependencies = []
        self.parent.contributors = ["Paul Dumont, University of North Carolina, Chapell Hill"]
        self.parent.helpText = _("""
        This tool helps to create summaries of clinical notes.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """)


#
# SurgMovPredParameterNode
#
@parameterNodeWrapper
class SurgMovPredParameterNode:
    """
    Parameters for Clinical Notes Extraction UI.

    notesFolder_input - Folder containing clinical notes (.docx/.pdf/.txt).
    modelType - Model type selection: 'Mini' (Light/Fast) or 'Max' (Heavy/Precise).
    notesType - Notes type selection: 'TMJ' or 'Ortho'.
    notesFolder_output - Folder for summary output.
    """

    inputFolder: str = ""
    modelPath: str = ""
    outputFolder: str = ""

#
# SurgMovPredWidget
#
class SurgMovPredWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._updatingGUIFromParameterNode = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)


        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SurgMovPred.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Create logic class.
        self.logic = SurgMovPredLogic()

        isDarkMode = self._isDarkMode()
        styleSheet = self._getStyleSheet(isDarkMode)
        uiWidget.setStyleSheet(styleSheet)
        
        # Also apply label-specific stylesheet
        self._applyLabelStyleSheets(isDarkMode)
        self._applyButtonStyleSheets(isDarkMode)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.cancelButton.connect("clicked(bool)", self.onCancelCliButton)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.downloadTestFilesButton.connect("clicked(bool)", self.onRunTestFilesButton)
        self.ui.DefaultModelButton.connect("clicked(bool)", self.onDownloadDefaultModel)

        self.ui.inputFolderLineEdit.currentPathChanged.connect(self._checkCanApply)
        self.ui.modelFolderLineEdit.currentPathChanged.connect(self._checkCanApply)
        self.ui.outputFolderLineEdit.currentPathChanged.connect(self._checkCanApply)

        self.ui.cancelButton.setVisible(False)

        documentsLocation = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documentsLocation)

        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
        )

        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        if not self._parameterNode.inputFolder:
            self._parameterNode.inputFolder = ""

        if not self._parameterNode.modelPath:
            self._parameterNode.modelPath = ""
        
        if not self._parameterNode.outputFolder:
            self._parameterNode.outputFolder = ""

    def setParameterNode(self, inputParameterNode: SurgMovPredParameterNode | None) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        """Enable/disable the apply button based on required fields."""
        if not self._parameterNode:
            self.ui.applyButton.enabled = False
            return
        
        self._parameterNode.inputFolder = self.ui.inputFolderLineEdit.currentPath
        self._parameterNode.modelPath = self.ui.modelFolderLineEdit.currentPath
        self._parameterNode.outputFolder = self.ui.outputFolderLineEdit.currentPath

        if self._parameterNode.inputFolder != "" and self._parameterNode.outputFolder != "" and self._parameterNode.modelPath != "":
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.enabled = False

    def onRunTestFilesButton(self) -> None:
        """Run test files when user clicks 'Run Test Files' button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to download test files."), waitCursor=True):
            
            self.DownloadTestFiles()

    def onApplyButton(self) -> None:
        """Run processing when user clicks Apply button."""

        if not check_dependencies():
            return

        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):

            inputFolder = self._parameterNode.inputFolder
            modelPath = self._parameterNode.modelPath
            outputFolder = self._parameterNode.outputFolder

            logger.info(f"Input folder   : {inputFolder}")
            logger.info(f"Output folder  : {outputFolder}")
            logger.info(f"Selected model : {modelPath}")


            cliNode = self.logic.process(inputFolder,modelPath,outputFolder)

            if cliNode:
                self.ui.applyButton.setVisible(False)
                self.ui.cancelButton.setVisible(True)
                self.addObserver(cliNode, slicer.vtkMRMLCommandLineModuleNode.StatusModifiedEvent, self.onCliFinished)


    def onCancelCliButton(self) -> None:
        """Cancel the running CLI process."""
        if self.logic and hasattr(self.logic, 'cliNode') and self.logic.cliNode:
            self.logic.cliNode.Cancel()
            self.ui.applyButton.setVisible(True)
            self.ui.cancelButton.setVisible(False)
            slicer.util.warningDisplay("Processing cancelled by user.")

    def onCliFinished(self, caller, event) -> None:
        """Restore the apply button and hide the cancel button when CLI finishes."""
        status = caller.GetStatus()
        if status & (slicer.vtkMRMLCommandLineModuleNode.Completed | slicer.vtkMRMLCommandLineModuleNode.Cancelled):
            self.ui.applyButton.setVisible(True)
            self.ui.cancelButton.setVisible(False)

    def _isDarkMode(self) -> bool:
        """Check if the application is in dark mode."""
        try:
            palette = slicer.app.palette()
            bgColor = palette.color(qt.QPalette.Window)
            luminance = (0.299 * bgColor.red() + 0.587 * bgColor.green() + 0.114 * bgColor.blue()) / 255.0
            return luminance < 0.5
        except Exception:
            return False
        
    def _getStyleSheet(self, isDarkMode: bool) -> str:
        """Generate stylesheet based on theme."""
        if isDarkMode:
            return """
            qMRMLWidget {
              background-color: #2b2b2b;
            }
            ctkCollapsibleButton {
              background-color: #383838;
              border: 1px solid #454545;
              border-radius: 6px;
              margin-bottom: 8px;
              font-weight: 600;
              padding: 6px 10px;
              color: #e0e0e0;
            }
            ctkCollapsibleButton:hover {
              border: 1px solid #3498db;
              background-color: #414141;
            }
            QLineEdit, QTextEdit {
              background-color: #353535;
              border: 1px solid #454545;
              border-radius: 4px;
              padding: 6px;
              color: #e0e0e0;
              selection-background-color: #3498db;
            }
            QLineEdit:focus, QTextEdit:focus {
              border: 2px solid #3498db;
              background-color: #383838;
            }
            QComboBox {
              background-color: #353535;
              border: 1px solid #454545;
              border-radius: 4px;
              padding: 4px 6px;
              color: #e0e0e0;
            }
            QComboBox:focus {
              border: 2px solid #3498db;
            }
            QComboBox::drop-down {
              width: 20px;
              border: none;
            }
            QComboBox QAbstractItemView {
              background-color: #353535;
              color: #e0e0e0;
              selection-background-color: #3498db;
              border: 1px solid #454545;
            }
            QProgressBar {
              border: 1px solid #454545;
              border-radius: 4px;
              background-color: #353535;
              padding: 2px;
              color: #e0e0e0;
            }
            QProgressBar::chunk {
              background-color: #3498db;
              border-radius: 3px;
            }
            """
        else:
            return """
            qMRMLWidget {
              background-color: #f8f9fa;
            }
            ctkCollapsibleButton {
              background-color: #ffffff;
              border: 1px solid #e0e6ed;
              border-radius: 6px;
              margin-bottom: 8px;
              font-weight: 600;
              padding: 6px 10px;
              color: #2c3e50;
            }
            ctkCollapsibleButton:hover {
              border: 1px solid #3498db;
              background-color: #fbfcfd;
            }
            QLineEdit, QTextEdit {
              background-color: #ffffff;
              border: 1px solid #e0e6ed;
              border-radius: 4px;
              padding: 6px;
              color: #2c3e50;
              selection-background-color: #3498db;
            }
            QLineEdit:focus, QTextEdit:focus {
              border: 2px solid #3498db;
            }
            QComboBox {
              background-color: #ffffff;
              border: 1px solid #e0e6ed;
              border-radius: 4px;
              padding: 4px 6px;
              color: #2c3e50;
            }
            QComboBox:focus {
              border: 2px solid #3498db;
            }
            QComboBox::drop-down {
              width: 20px;
              border: none;
            }
            QComboBox QAbstractItemView {
              background-color: #ffffff;
              color: #2c3e50;
              selection-background-color: #3498db;
              border: 1px solid #e0e6ed;
            }
            QProgressBar {
              border: 1px solid #e0e6ed;
              border-radius: 4px;
              background-color: #ffffff;
              padding: 2px;
              color: #2c3e50;
            }
            QProgressBar::chunk {
              background-color: #3498db;
              border-radius: 3px;
            }
            """

    def _applyLabelStyleSheets(self, isDarkMode: bool) -> None:
        """Apply label-specific stylesheets."""
        if isDarkMode:
            labelStyle = "color: #b0b0b0; font-weight: 600;"
        else:
            labelStyle = "color: #34495e; font-weight: 600;"
        
        # List of labels to style
        labels = [
            'label_5', 'label_4', 'label_2', 'label_6', 'label_7', 'label_3', 'label', 'modeLabel', 't2label', 'excellabel'
        ]
        
        for labelName in labels:
            if hasattr(self.ui, labelName):
                label = getattr(self.ui, labelName)
                label.setStyleSheet(labelStyle)

    def _applyButtonStyleSheets(self, isDarkMode: bool) -> None:
        """Apply button-specific stylesheets."""
        if isDarkMode:
            # Dark mode button styles
            standardButtonStyle = """
            QPushButton {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4ba3ff, stop:1 #3498db);
              color: white;
              border: none;
              border-radius: 6px;
              font-weight: 600;
              font-size: 10pt;
              padding: 8px;
              margin-top: 4px;
            }
            QPushButton:hover:!pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5cb3ff, stop:1 #2980b9);
            }
            QPushButton:pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1f618d);
            }
            QPushButton:disabled {
              background-color: #555555;
              color: #888888;
            }
            """
            
            cancelButtonStyle = """
            QPushButton {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
              color: white;
              border: none;
              border-radius: 6px;
              font-weight: 600;
              font-size: 10pt;
              padding: 8px;
              margin-top: 4px;
            }
            QPushButton:hover:!pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ec7063, stop:1 #a93226);
            }
            QPushButton:pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #a93226, stop:1 #922b21);
            }
            QPushButton:disabled {
              background-color: #555555;
              color: #888888;
            }
            """
        else:
            # Light mode button styles
            standardButtonStyle = """
            QPushButton {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4ba3ff, stop:1 #3498db);
              color: white;
              border: none;
              border-radius: 6px;
              font-weight: 600;
              font-size: 10pt;
              padding: 8px;
              margin-top: 4px;
            }
            QPushButton:hover:!pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5cb3ff, stop:1 #2980b9);
            }
            QPushButton:pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1f618d);
            }
            QPushButton:disabled {
              background-color: #bdc3c7;
              color: #95a5a6;
            }
            """
            
            cancelButtonStyle = """
            QPushButton {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
              color: white;
              border: none;
              border-radius: 6px;
              font-weight: 600;
              font-size: 10pt;
              padding: 8px;
              margin-top: 4px;
            }
            QPushButton:hover:!pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ec7063, stop:1 #a93226);
            }
            QPushButton:pressed {
              background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #a93226, stop:1 #922b21);
            }
            QPushButton:disabled {
              background-color: #bdc3c7;
              color: #95a5a6;
            }
            """
        
        # Apply standard style to most buttons
        for buttonName in ['applyButton','DefaultModelButton','downloadTestFilesButton']:
            if hasattr(self.ui, buttonName):
                button = getattr(self.ui, buttonName)
                button.setStyleSheet(standardButtonStyle)
        
        # Apply cancel style to cancel button
        if hasattr(self.ui, 'cancelButton'):
            self.ui.cancelButton.setStyleSheet(cancelButtonStyle)
    
    def DownloadUnzip(self, url, directory, folder_name=None, num_downl=1, total_downloads=1):

        out_path = os.path.join(directory, folder_name)
        if not os.path.exists(out_path):
            logger.info("Downloading {}...".format(folder_name.split(os.sep)[-1]))
            os.makedirs(out_path)

            temp_path = os.path.join(directory, "temp.zip")

            # Download the zip file from the url
            with urllib.request.urlopen(url) as response, open(
                temp_path, "wb"
            ) as out_file:
                # Pop up a progress bar with a QProgressDialog
                progress = qt.QProgressDialog(
                    "Downloading {} (File {}/{})".format(
                        folder_name.split(os.sep)[0], num_downl, total_downloads
                    ),
                    "Cancel",
                    0,
                    100,
                    self.parent,
                )
                progress.setCancelButton(None)
                progress.setWindowModality(qt.Qt.WindowModal)
                progress.setWindowTitle(
                    "Downloading {}...".format(folder_name.split(os.sep)[0])
                )
                # progress.setWindowFlags(qt.Qt.WindowStaysOnTopHint)
                progress.show()
                length = response.info().get("Content-Length")
                if length:
                    length = int(length)
                    blocksize = max(4096, length // 100)
                    read = 0
                    while True:
                        buffer = response.read(blocksize)
                        if not buffer:
                            break
                        read += len(buffer)
                        out_file.write(buffer)
                        progress.setValue(read * 100.0 / length)
                        qt.QApplication.processEvents()
                shutil.copyfileobj(response, out_file)

            # Unzip the file
            with zipfile.ZipFile(temp_path, "r") as zip:
                zip.extractall(out_path)

            # Delete the zip file
            os.remove(temp_path)

            logger.info(f"{folder_name} has been successfully installed")
    
    def DownloadTestFiles(self):

        if not os.path.exists(self.SlicerDownloadPath):
                os.makedirs(self.SlicerDownloadPath)

        self.DownloadUnzip(
            url="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/SurgMovPred/TestFiles.zip",
            directory=self.SlicerDownloadPath,
            folder_name="SurgMovPred",
        )
        if os.path.exists(os.path.join(self.SlicerDownloadPath,"V_FACE/DefaultList")):
            self.ui.inputFolderLineEdit.setCurrentPath(os.path.join(self.SlicerDownloadPath,"SurgMovPred/TestFiles"))

            if not os.path.exists(os.path.join(self.SlicerDownloadPath,"SurgMovPred/Output")):
                os.makedirs(os.path.join(self.SlicerDownloadPath,"SurgMovPred/Output"))
            self.ui.outputFolderLineEdit.setCurrentPath(os.path.join(self.SlicerDownloadPath,"SurgMovPred/Output"))

        self.onDownloadDefaultModel()

    def onDownloadDefaultModel(self):
        if not os.path.exists(self.SlicerDownloadPath):
                os.makedirs(self.SlicerDownloadPath)

        self.DownloadUnzip(
            url="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/SurgMovPred/all_models.zip",
            directory=self.SlicerDownloadPath,
            folder_name="SurgMovPred/Models")

        if os.path.exists(os.path.join(self.SlicerDownloadPath,"SurgMovPred/Models/all_models")):
            self.ui.modelFolderLineEdit.setCurrentPath(os.path.join(self.SlicerDownloadPath,"SurgMovPred/Models/all_models"))


#
# SurgMovPredLogic
#
class SurgMovPredLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self,):
        return SurgMovPredParameterNode(super().getParameterNode())

    def process(self, notesFolder_input: str, modelPath: str, notesFolder_output: str) -> bool:
        """Process clinical notes using the selected model and parameters."""

        if not notesFolder_input or not modelPath or not notesFolder_output:
            missing = []
            if not notesFolder_input:
                missing.append("Input folder")
            if not modelPath:
                missing.append("Model Path")
            if not notesFolder_output:
                missing.append("Output folder")

            error_msg = f"Process cancelled: Missing required parameters: {', '.join(missing)}"
            logger.error(error_msg)
            slicer.util.errorDisplay(error_msg)
            return None

        os.makedirs(notesFolder_output, exist_ok=True)

        CLI_module = slicer.modules.surgmovpred_cli
        parameters = {
            "inputFolder": notesFolder_input,
            "modelPath": modelPath,
            "outputFolder": notesFolder_output,
        }

        logger.info(f"Launching CLI")
        self.cliNode = slicer.cli.run(CLI_module, None, parameters)
        self.cliNode.AddObserver(slicer.vtkMRMLCommandLineModuleNode.StatusModifiedEvent, self.onCliModified)

        return self.cliNode

    def onCliProgress(self, caller, event):
        """Callback triggered on CLI progress updates."""
        progress = caller.GetProgress()

    def onCliModified(self, caller, event):
        """Callback triggered when CLI status changes (completed, cancelled, etc.)."""
        status = caller.GetStatus()

        if status & (slicer.vtkMRMLCommandLineModuleNode.Completed | slicer.vtkMRMLCommandLineModuleNode.Cancelled):
            logger.info("Background process finished (CLI)")

            if status == slicer.vtkMRMLCommandLineModuleNode.Completed:
                logger.info("SurgMovPred- COMPLETE")
            elif status == slicer.vtkMRMLCommandLineModuleNode.Cancelled:
                logger.info("PROCESS CANCELLED BY USER")

            output_text = caller.GetOutputText()
            if output_text:
                logger.info("--- Detailed CLI Logs ---")
                logger.info(output_text.strip())
                logger.info("---------------------------\n")

            error_text = caller.GetErrorText()
            if error_text:
                logger.error("--- CLI ERRORS ---")
                logger.error(error_text.strip())
                logger.error("---------------------\n")
