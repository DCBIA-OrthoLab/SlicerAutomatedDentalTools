import os
import shutil
import vtk
import slicer
import qt
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
import importlib

import sys
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
logger = get_logger("CNE")


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

def check_dependencies() -> bool:
    """
    Checks dependencies when the Apply button is clicked.
    Returns True if everything is ready, False if it should be cancelled.
    Attempts up to 2 verifications with proper logging.
    """
    max_retries = 1
    for attempt in range(max_retries + 1):
        missing_libs = []

        if not check_lib_installed("llama_cpp"):
            missing_libs.append("llama-cpp-python")

        if not check_lib_installed("fitz"):
            missing_libs.append("pymupdf")

        if not check_lib_installed("docx"):
            missing_libs.append("python-docx")

        if not missing_libs:
            logger.info("All dependencies verified and available.")
            return True

        if attempt < max_retries:
            libs_str = "\n".join([f"- {lib}" for lib in missing_libs])

            msg = (
                "The CNE module requires the following libraries to function:\n\n"
                f"{libs_str}\n\n"
                "Do you agree to modify Slicer's environment to install them? "
                "This may take a few minutes."
            )

            if slicer.util.confirmOkCancelDisplay(msg):
                logger.info(f"Installing missing dependencies: {missing_libs}")
                install_function(missing_libs)
                slicer.app.processEvents()
            else:
                logger.warning("Installation cancelled by user.")
                slicer.util.warningDisplay("Installation cancelled. Extraction has been stopped.")
                return False
        else:
            logger.error(f"Failed to install required dependencies: {missing_libs}")
            error_msg = (
                "Failed to install required dependencies:\n\n"
                f"{libs_str}\n\n"
                "Please restart Slicer and try again."
            )
            slicer.util.errorDisplay(error_msg)
            return False

    return False

class CNE(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("CNE")
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
# CNEParameterNode
#
@parameterNodeWrapper
class CNEParameterNode:
    """
    Parameters for Clinical Notes Extraction UI.

    notesFolder_input - Folder containing clinical notes (.docx/.pdf/.txt).
    notesType - Notes type selection: 'TMJ' or 'Ortho'.
    notesFolder_output - Folder for summary output.
    """

    notesFolder_input: str = ""
    notesType: str = "TMJ"
    notesFolder_output: str = ""

#
# CNEWidget
#
class CNEWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/CNE.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)

        # Create logic class.
        self.logic = CNELogic()

        # Create QButtonGroup for notes type selection
        self.notesTypeButtonGroup = qt.QButtonGroup()
        self.notesTypeButtonGroup.addButton(self.ui.notesTypeTMJRadioButton)
        self.notesTypeButtonGroup.addButton(self.ui.notesTypeOrthoRadioButton)
        self.notesTypeButtonGroup.setExclusive(True)

        self.cliProgressBar = slicer.qSlicerCLIProgressBar()
        self.cliProgressBar.visible = False
        self.layout.addWidget(self.cliProgressBar)

        self.cliCancelButton = qt.QPushButton("Cancel Processing")
        self.cliCancelButton.visible = False
        self.cliCancelButton.connect("clicked(bool)", self.onCancelCliButton)
        self.layout.addWidget(self.cliCancelButton)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.downloadTestFilesButton.connect("clicked(bool)", self.onRunTestFilesButton)

        self.ui.notesTypeTMJRadioButton.connect("toggled(bool)", self._updateParameterNodeFromGUI)
        self.ui.notesTypeOrthoRadioButton.connect("toggled(bool)", self._updateParameterNodeFromGUI)

        self.initializeParameterNode()
        self._syncNotesTypeRadioWithParameterNode()

    def _syncNotesTypeRadioWithParameterNode(self):
        """Synchronize UI radio buttons with parameter node values."""
        if not self._parameterNode:
            return

        self._updatingGUIFromParameterNode = True

        if self._parameterNode.notesType == "TMJ":
            self.ui.notesTypeTMJRadioButton.checked = True
        elif self._parameterNode.notesType == "Ortho":
            self.ui.notesTypeOrthoRadioButton.checked = True

        self._updatingGUIFromParameterNode = False

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

    def setParameterNode(self, input_parameter_node: CNEParameterNode | None) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = input_parameter_node
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()
            self._syncNotesTypeRadioWithParameterNode()

    def _checkCanApply(self, caller=None, event=None) -> None:
        """Enable/disable the apply button based on required fields."""
        if self._parameterNode:
            self.ui.applyButton.toolTip = _("Extract clinical notes using selected model")
            self.ui.applyButton.enabled = True


    def onRunTestFilesButton(self) -> None:
        """Run test files when user clicks 'Run Test Files' button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to prepare the test files."), waitCursor=True):
            # Get the selected notes type from parameter node
            self._updateParameterNodeFromGUI()
            notes_type = self._parameterNode.notesType
            
            # Copy test files and get the paths
            input_path, output_path = self.logic.copyTestFiles(notes_type)
            
            # Update the folder paths in the UI. An output already chosen is
            # the user's own: we do not replace it with ours.
            self.ui.notesFolderLineEdit_input.currentPath = input_path
            if not self.ui.notesFolderLineEdit_output.currentPath:
                self.ui.notesFolderLineEdit_output.currentPath = output_path
            self._updateParameterNodeFromGUI()


    def onApplyButton(self) -> None:
        """Run processing when user clicks Apply button."""

        if not check_dependencies():
            return

        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):

            logger.info("CNE (Clinical Notes Extraction)")
            self._updateParameterNodeFromGUI()

            notes_folder_input = self._parameterNode.notesFolder_input
            notes_type = self._parameterNode.notesType
            notes_folder_output = self._parameterNode.notesFolder_output

            logger.info(f"Input folder   : {notes_folder_input}")
            logger.info(f"Output folder  : {notes_folder_output}")
            logger.info(f"Notes type     : {notes_type}")

            cli_node = self.logic.process(
                notes_folder_input,
                notes_type, notes_folder_output
            )

            if cli_node:
                self.cliProgressBar.setCommandLineModuleNode(cli_node)
                self.cliProgressBar.visible = True
                self.cliCancelButton.visible = True
                self.addObserver(cli_node, slicer.vtkMRMLCommandLineModuleNode.StatusModifiedEvent, self.onCliFinished)


    def onCancelCliButton(self) -> None:
        """Cancel the running CLI process."""
        if self.logic and hasattr(self.logic, 'cliNode') and self.logic.cliNode:
            self.logic.cliNode.Cancel()
            self.cliProgressBar.visible = False
            self.cliCancelButton.visible = False
            slicer.util.warningDisplay("Processing cancelled by user.")

    def onCliFinished(self, caller, event) -> None:
        """Hide progress bar and cancel button when CLI finishes."""
        status = caller.GetStatus()
        if status & (slicer.vtkMRMLCommandLineModuleNode.Completed | slicer.vtkMRMLCommandLineModuleNode.Cancelled):
            self.cliProgressBar.visible = False
            self.cliCancelButton.visible = False

    def _updateParameterNodeFromGUI(self) -> None:
        """Update parameter node from GUI values."""
        if not self._parameterNode or self._updatingGUIFromParameterNode:
            return

        was_modified = self._parameterNode.StartModify()

        self._parameterNode.notesFolder_input = self.ui.notesFolderLineEdit_input.currentPath
        self._parameterNode.notesFolder_output = self.ui.notesFolderLineEdit_output.currentPath

        if self.ui.notesTypeTMJRadioButton.checked:
            self._parameterNode.notesType = "TMJ"
        elif self.ui.notesTypeOrthoRadioButton.checked:
            self._parameterNode.notesType = "Ortho"
        else:
            self._parameterNode.notesType = ""

        self._parameterNode.EndModify(was_modified)




#
# CNELogic
#
class CNELogic(ScriptedLoadableModuleLogic):
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
        return CNEParameterNode(super().getParameterNode())
    
    def copyTestFiles(self, notesType: str) -> tuple:
        """The example notes of this mode, and the folder to write their summaries.

        The notes live in the repository: this is a copy, not a download. It
        is redone on every call, which makes the button idempotent -- a note
        erased by an earlier try comes back.

        The output folder, on the other hand, is not in the repository: it is
        created here, and nothing is copied into it. The previous version
        looked for it next to the notes, did not find it (`logger.warning`
        then `continue`), and still put its non-existent path in the field --
        which the user saw in red without knowing why.

        Args:
            notesType: Either 'TMJ' or 'Ortho' to specify which test files to copy

        Returns:
            tuple: (input_folder_path, output_folder_path)
        """
        folders = {
            "Ortho": ("input_Ortho", "output_Ortho"),
            "TMJ": ("input_TMJ", "output_TMJ"),
        }
        if notesType not in folders:
            raise ValueError(f"Unknown notes type: {notesType}")
        input_folder, output_folder = folders[notesType]

        # Get the path to the testfiles directory (relative to this module)
        module_dir = os.path.dirname(__file__)
        source_folder = os.path.join(module_dir, "Resources", "testfiles", input_folder)

        if not os.path.isdir(source_folder):
            raise FileNotFoundError(
                f"The {notesType} example notes are missing from the module: "
                f"{source_folder}")

        # Define destination path in SlicerDownloads/CNE/testfiles/{notesType}
        documents = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation)
        dest_base_path = os.path.join(
            documents,
            slicer.app.applicationName + "Downloads",
            "CNE",
            "testfiles",
            notesType
        )

        input_path = os.path.join(dest_base_path, input_folder)
        output_path = os.path.join(dest_base_path, output_folder)

        # `dirs_exist_ok` puts the notes back without throwing the folder
        # away: the earlier `rmtree` also erased what the user had put there.
        shutil.copytree(source_folder, input_path, dirs_exist_ok=True)
        logger.info(f"Test notes copied from {source_folder} to {input_path}")

        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Test output folder ready: {output_path}")

        return input_path, output_path
    
    def getModelPath(self, notesType: str):
        """Returns the local path to the model, downloading it if necessary with a progress popup."""

        # 1. Configuration of the model based on UI selection
        if notesType == "Ortho":
                repo_id = "dcbia/Meta-Llama-3.1-8B-Instruct-Ortho"
                file_name = "model-q4_0.gguf"
                local_model_name = "Meta-Llama-3.1-8B-Ortho.gguf"
                dialog_text = "Downloading Max Ortho AI model (approx. 4.7 GB)..."

        # 1. Configuration of the model based on UI selection
        elif notesType == "TMJ":
                repo_id = "dcbia/Qwen-2.5-7B-Instruct-TMJ"
                file_name = "qwen-ft-q4_k_m.gguf"
                local_model_name = "Qwen-2.5-7B-TMJ.gguf"
                dialog_text = "Downloading Max TMJ AI model (approx. 4.4 GB)..."


        model_url = f"https://huggingface.co/{repo_id}/resolve/main/{file_name}"
        
        # 2. Directory structure
        documents = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation)
        slicer_download_path = os.path.join(
            documents,
            slicer.app.applicationName + "Downloads",
            "CNE",
            "model"
        )
        
        if not os.path.exists(slicer_download_path):
            os.makedirs(slicer_download_path)
            
        dest_path = os.path.join(slicer_download_path, local_model_name)

        # 3. Check and download
        if not os.path.exists(dest_path):
            logger.info(f"Downloading  model to: {dest_path}")
            
            # --- Create the popup (QProgressDialog) ---
            progress_dialog = qt.QProgressDialog(dialog_text, "Cancel", 0, 100)
            progress_dialog.setWindowTitle(f"CNE - Preparing AI Model")
            progress_dialog.setWindowModality(qt.Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()

            # --- Callback function to update the popup ---
            def download_progress(count, block_size, total_size):
                if progress_dialog.wasCanceled:
                    raise Exception("Download cancelled by user.")
                
                if total_size > 0:
                    percent = min(int((count * block_size * 100) / total_size), 100)
                    progress_dialog.setValue(percent)
                
                # Forces Slicer to refresh the UI (prevents freezing)
                slicer.app.processEvents()
            
            # --- Start the download ---
            import urllib.request
            try:
                urllib.request.urlretrieve(model_url, dest_path, reporthook=download_progress)
                progress_dialog.setValue(100)
                slicer.util.showStatusMessage(f"AI model download completed!", 3000)
                
            except Exception as e:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                slicer.util.errorDisplay(f"Download failed or was cancelled: {e}")
                progress_dialog.close()
                raise e
                
            finally:
                progress_dialog.close()
            
        return dest_path

    def process(self, notesFolder_input: str,
                notesType: str, notesFolder_output: str) -> bool:
        """Process clinical notes using the selected model and parameters."""

        if not notesFolder_input or not notesFolder_output:
            missing = []
            if not notesFolder_input:
                missing.append("Input folder")
            if not notesType:
                missing.append("Notes type")
            if not notesFolder_output:
                missing.append("Output folder")

            error_msg = f"Process cancelled: Missing required parameters: {', '.join(missing)}"
            logger.error(error_msg)
            slicer.util.errorDisplay(error_msg)
            return None

        try:
            model_path = self.getModelPath(notesType)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load model: {e}")
            return None

        os.makedirs(notesFolder_output, exist_ok=True)

        cli_module = slicer.modules.cne_cli
        parameters = {
            "notesFolder_input": notesFolder_input,
            "notesType": notesType,
            "notesFolder_output": notesFolder_output,
            "modelPath": model_path,
        }

        logger.info(f"Launching CLI with model: {model_path}")
        self.cliNode = slicer.cli.run(cli_module, None, parameters)
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
                logger.info("CNE (Clinical Notes Extraction) - COMPLETE")
                slicer.util.messageBox("Notes extraction is complete!")
            elif status == slicer.vtkMRMLCommandLineModuleNode.Cancelled:
                logger.info("PROCESS CANCELLED BY USER")

            output_text = caller.GetOutputText()
            if output_text:
                logger.info("\n--- Detailed CLI Logs ---")
                logger.info(output_text.strip())
                logger.info("---------------------------\n")

            error_text = caller.GetErrorText()
            if error_text:
                logger.error("\n--- CLI ERRORS ---")
                logger.error(error_text.strip())
                logger.error("---------------------\n")
