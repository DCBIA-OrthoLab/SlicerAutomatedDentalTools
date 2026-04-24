import logging
import os
from typing import Annotated
import urllib.request
import shutil
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import importlib
try:
    from VFACE_utils import Progress
    importlib.reload(Progress)
    from VFACE_utils.Progress import DisplayALICBCT,DisplayAMASSS,DisplayASOCBCT,Display
    
    from VFACE_utils import createlistprocess
    importlib.reload(createlistprocess)
    from VFACE_utils.createlistprocess import CreateListProcess

except Exception as e:
    logger.error(f"Error loading VFACE utilities: {e}")
    from VFACE_utils.Progress import DisplayALICBCT,DisplayAMASSS,DisplayASOCBCT,Display
    from VFACE_utils.createlistprocess import CreateListProcess

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import qt


#
# VFACE Module
#


class VFACE(ScriptedLoadableModule):
    """
    VFACE (Vertical Facial Asymmetry Classification Engine)
    
    Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("V FACE")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.contributors = ["Alexandre Buisson (University of North Carolina at Chapel Hill)"] 
        self.parent.helpText = _("""
        VFACE - Vertical Facial Asymmetry Classification Engine
        
        This extension helps classify facial asymmetry in dental patients.
        It provides automated measurement extraction and machine learning-based classification
        of facial structures including mandible and maxilla.
        """)
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """
    Add sample data sets to the Sample Data module for easy demonstration.
    
    This function registers test datasets that can be downloaded and used
    to quickly test the module functionality without requiring real patient data.
    """
    try:
        import SampleData

        iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

        # To ensure that the source code repository remains small (can be downloaded and installed quickly)
        # it is recommended to store data sets that are larger than a few MB in a Github release.

        # VFACE1 - First test dataset
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            # Category and sample name displayed in Sample Data module
            category="VFACE",
            sampleName="VFACE1",
            # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
            # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
            thumbnailFileName=os.path.join(iconsPath, "VFACE1.png"),
            # Download URL and target file name
            uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
            fileNames="VFACE1.nrrd",
            # Checksum to ensure file integrity. Can be computed by this command:
            #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
            checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
            # This node name will be used when the data set is loaded
            nodeNames="VFACE1",
        )

        # VFACE2 - Second test dataset
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            # Category and sample name displayed in Sample Data module
            category="VFACE",
            sampleName="VFACE2",
            thumbnailFileName=os.path.join(iconsPath, "VFACE2.png"),
            # Download URL and target file name
            uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
            fileNames="VFACE2.nrrd",
            checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
            # This node name will be used when the data set is loaded
            nodeNames="VFACE2",
        )
    except Exception as e:
        logger.error(f"Error registering sample data: {e}")


#
# VFACEParameterNode
#


@parameterNodeWrapper
class VFACEParameterNode:
    """
    Parameter node for VFACE module.
    
    Contains the key parameters required by the module:
    - InputFolder: Path to the input directory containing scan files
    - OutputFolder: Path where processed outputs will be saved
    - MeasurementsFolder: Path containing measurement reference files
    """

    InputFolder: str
    OutputFolder: str
    MeasurementsFolder: str


#
# VFACE Widget
#

class PopUpWindow(qt.QDialog):
    """
    Custom dialog window for displaying messages and interactive controls.
    
    Supports radio buttons, checkboxes, or simple message display.
    Provides user-friendly interface for multi-selection or single-choice operations.
    """

    def __init__(
        self,
        title="Title",
        text=None,
        listename=["1", "2", "3"],
        type=None,
        tocheck=None,
    ):
        """
        Initialize the popup window.
        
        Args:
            title: Window title
            text: Text message to display
            listename: List of item names for selection
            type: Dialog type - 'radio' for single selection, 'checkbox' for multiple
            tocheck: Items to pre-check (for checkbox mode)
        """
        qt.QWidget.__init__(self)
        self.setWindowTitle(title)
        layout = qt.QGridLayout()
        self.setLayout(layout)
        self.ListButtons = []
        self.listename = listename
        self.type = type

        if self.type == "radio":
            self._setup_radio_buttons(layout)

        elif self.type == "checkbox":
            self._setup_checkboxes(layout)
            if tocheck is not None:
                self._check_items(tocheck)

        elif text is not None:
            label = qt.QLabel(text)
            layout.addWidget(label)
            # Add OK button to close the window
            button = qt.QPushButton("OK")
            button.connect("clicked()", self.onClickedOK)
            layout.addWidget(button)

    def _setup_checkboxes(self, layout):
        """Create and arrange checkbox controls."""
        j = 0
        for i in range(len(self.listename)):
            button = qt.QCheckBox(self.listename[i])
            self.ListButtons.append(button)
            if i % 20 == 0:
                j += 1
            layout.addWidget(button, i % 20, j)
        
        # Add a button to select all items
        button = qt.QPushButton("Select All")
        button.connect("clicked()", self.onClickedSelectAll)
        layout.addWidget(button, len(self.listename) + 1, j - 2)
        
        # Add a button to deselect all items
        button = qt.QPushButton("Deselect All")
        button.connect("clicked()", self.onClickedDeselectAll)
        layout.addWidget(button, len(self.listename) + 1, j - 1)

        # Add a button to confirm selection
        button = qt.QPushButton("OK")
        button.connect("clicked()", self.onClickedCheckbox)
        layout.addWidget(button, len(self.listename) + 1, j)

    def _check_items(self, tocheck):
        """Pre-check specified items."""
        for i in range(len(self.listename)):
            if self.listename[i] in tocheck:
                self.ListButtons[i].setChecked(True)

    def onClickedSelectAll(self):
        """Handle select all button click."""
        for button in self.ListButtons:
            button.setChecked(True)

    def onClickedDeselectAll(self):
        """Handle deselect all button click."""
        for button in self.ListButtons:
            button.setChecked(False)

    def onClickedCheckbox(self):
        """Handle checkbox confirmation."""
        TrueFalse = [button.isChecked() for button in self.ListButtons]
        self.checked = [
            self.listename[i] for i in range(len(self.listename)) if TrueFalse[i]
        ]
        self.accept()

    def _setup_radio_buttons(self, layout):
        """Create and arrange radio button controls."""
        for i in range(len(self.listename)):
            radiobutton = qt.QRadioButton(self.listename[i])
            self.ListButtons.append(radiobutton)
            radiobutton.connect("clicked(bool)", self.onClickedRadio)
            layout.addWidget(radiobutton, i, 0)

    def onClickedRadio(self):
        """Handle radio button selection."""
        self.checked = self.listename[
            [button.isChecked() for button in self.ListButtons].index(True)
        ]
        self.accept()

    def onClickedOK(self):
        """Handle OK button click."""
        self.accept()



class VFACEWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    VFACE Widget - Main user interface for the module.
    
    Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Initialize the widget when the user opens the module.
        
        Args:
            parent: Parent widget
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # Needed for parameter node observation
        
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.CliStartTime = 0
        self.CliStepTime = 0
        self.NumberProcess = 0
        self.ActualProcess = 1
        self.cliNode = None
        self.list_process = []
        self.paused_for_visualization = False
        self.current_output_to_load = None
        self.current_process_info = None

    def reloadCustomModules(self) -> None:
        """
        Reload custom utility modules to ensure latest changes are loaded.
        
        This is useful during development and when modules have been updated.
        """
        try:
            import importlib
            import sys

            modules_to_reload = ['createlistprocess', 'Progress', 'functionaq3dc']
            
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
            logger.info("All utility modules reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading custom modules: {e}")

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        self.reloadCustomModules()

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/VFACE.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Detect dark mode and apply stylesheet
        isDarkMode = self._isDarkMode()
        styleSheet = self._getStyleSheet(isDarkMode)
        uiWidget.setStyleSheet(styleSheet)
        
        # Also apply label-specific stylesheet
        self._applyLabelStyleSheets(isDarkMode)
        self._applyButtonStyleSheets(isDarkMode)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = VFACELogic()

        # Connections

        self.ui.PathLineEdit.currentPathChanged.connect(self._checkCanApply)
        self.ui.PathLineEdit_2.currentPathChanged.connect(self._checkCanApply)
        self.ui.PathLineEdit_3.currentPathChanged.connect(self._checkCanApply)
        self.ui.PathLineEdit_4.currentPathChanged.connect(self._checkCanApply)
        
        # ComboBox connections
        self.ui.comboBox.currentTextChanged.connect(self.onComboBoxChanged)
        
        self.ui.comboBox2.currentTextChanged.connect(self.onComboBox2Changed)
        self.ui.comboBox3.currentTextChanged.connect(self.onComboBox3Changed)
        self.ui.comboBox4.currentTextChanged.connect(self.onComboBox4Changed)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.CheckDependencyButton.connect("clicked(bool)", self.CheckDependency)
        self.ui.cancelButton.connect("clicked(bool)", self.onCancelButton)

        self.ui.continueButton.setVisible(False)
        self.ui.continueButton.connect("clicked(bool)", self.onContinueButton)

        documentsLocation = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documentsLocation)

        self.display = Display
        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
        )

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.ui.label_3.setVisible(False)
        self.ui.progressBar.setVisible(False)

    def _isDarkMode(self) -> bool:
        """Check if the application is in dark mode."""
        try:
            palette = slicer.app.palette()
            bgColor = palette.color(qt.QPalette.Window)
            luminance = (0.299 * bgColor.red() + 0.587 * bgColor.green() + 0.114 * bgColor.blue()) / 255.0
            return luminance < 0.5
        except:
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
        for buttonName in ['applyButton', 'CheckDependencyButton', 'continueButton']:
            if hasattr(self.ui, buttonName):
                button = getattr(self.ui, buttonName)
                button.setStyleSheet(standardButtonStyle)
        
        # Apply cancel style to cancel button
        if hasattr(self.ui, 'cancelButton'):
            self.ui.cancelButton.setStyleSheet(cancelButtonStyle)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.reloadCustomModules()
        
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

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.InputFolder:
            self._parameterNode.InputFolder = "test"

        if not self._parameterNode.OutputFolder:
            self._parameterNode.OutputFolder = "testt"
        
        if not self._parameterNode.MeasurementsFolder:
            self._parameterNode.MeasurementsFolder = ""

    def setParameterNode(self, inputParameterNode: VFACEParameterNode | None) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

        self._parameterNode = inputParameterNode

        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        """
        Validate if the apply button should be enabled based on selected options and folder paths.
        
        Updates button state and tooltip based on current parameter configuration.
        """
        if not self._parameterNode:
            self.ui.applyButton.enabled = False
            return

        # Update parameter node with current values
        self._parameterNode.InputFolder = self.ui.PathLineEdit.currentPath
        self._parameterNode.OutputFolder = self.ui.PathLineEdit_2.currentPath
        self._parameterNode.MeasurementsFolder = self.ui.PathLineEdit_3.currentPath

        # Determine required folders based on selected options
        viz_mode = self.ui.comboBox2.currentText
        file_mode = self.ui.comboBox3.currentText
        t2_path = self.ui.PathLineEdit_4.currentPath
        
        # Case 1: Visualization with pre-registered files
        if viz_mode == "Visualization (Heatmaps)" and file_mode == "File already Registered":
            if (self._parameterNode.InputFolder != "" and 
                self._parameterNode.OutputFolder != "" and 
                t2_path != ""):
                self.ui.applyButton.toolTip = _("Click to classify patient facial asymmetry")
                self.ui.applyButton.enabled = True
            else:
                self.ui.applyButton.toolTip = _("Please fill in all required folder paths")
                self.ui.applyButton.enabled = False

        # Case 2: Pre-registered files with measurements
        elif file_mode == "File already Registered":
            if (self._parameterNode.InputFolder != "" and 
                self._parameterNode.OutputFolder != "" and 
                self._parameterNode.MeasurementsFolder != "" and 
                t2_path != ""):
                self.ui.applyButton.toolTip = _("Click to classify patient facial asymmetry")
                self.ui.applyButton.enabled = True
            else:
                self.ui.applyButton.toolTip = _("Please fill in all required folder paths")
                self.ui.applyButton.enabled = False

        # Case 3: Visualization only
        elif viz_mode == "Visualization (Heatmaps)":
            if (self._parameterNode.InputFolder != "" and 
                self._parameterNode.OutputFolder != ""):
                self.ui.applyButton.toolTip = _("Click to classify patient facial asymmetry")
                self.ui.applyButton.enabled = True
            else:
                self.ui.applyButton.toolTip = _("Please fill in all required folder paths")
                self.ui.applyButton.enabled = False

        # Case 4: Full processing pipeline
        else:
            if (self._parameterNode.InputFolder != "" and 
                self._parameterNode.OutputFolder != "" and 
                self._parameterNode.MeasurementsFolder != ""):
                self.ui.applyButton.toolTip = _("Click to classify patient facial asymmetry")
                self.ui.applyButton.enabled = True
            else:
                self.ui.applyButton.toolTip = _("Please fill in all required folder paths")
                self.ui.applyButton.enabled = False

    def DownloadAllFiles(self) -> None:

        dic_url = {   
            "Mirror_matrix": "https://github.com/GaelleLeroux/DCBIA_Apply_matrix/releases/download/AutoMatrixMirror/Mirror.zip",

            "ASO/ASO_CBCT/Reference": {
                "Occlusal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Occlusal_Midsagittal_Plane.zip",
                "Frankfurt Horizontal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Frankfurt_Horizontal_Midsagittal_Plane.zip"},

            "AREG/AREG_CBCT/Models/Segmentation": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/AMASSS_CBCT/AMASSS_Models.zip",

            
            "ALI/ALI_CBCT/Models/Landmark": {
                "Cranial Base": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Cranial_Base.zip",
                "Lower Bones 1": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Bones_1.zip",
                "Lower Bones 2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Bones_2.zip",
                "Lower Left Teeth": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Left_Teeth.zip",
                "Lower_Right_Teeth": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Lower_Right_Teeth.zip",
                "Upper Bones v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Bones_v2.zip",
                "Upper Left Teeth v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Left_Teeth_v2.zip",
                "Upper Right Teeth v2": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/v0.1-v2.0_models/Upper_Right_Teeth_v2.zip",
            },
            "V_FACE": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/VFACE/V_FACE_Models.zip",
        }

        if not os.path.exists(self.SlicerDownloadPath):
            os.makedirs(self.SlicerDownloadPath)

        for name, url_or_dict in dic_url.items():
            if isinstance(url_or_dict, str):
                self.DownloadUnzip(
                    url=url_or_dict,
                    directory=self.SlicerDownloadPath,
                    folder_name=name,
                )
            elif isinstance(url_or_dict, dict):
                for subfolder_name, url in url_or_dict.items():
                    self.DownloadUnzip(
                        url=url,
                        directory=self.SlicerDownloadPath,
                        folder_name=os.path.join(name, subfolder_name),
                    )
            else:
                print(f"Warning: Unknown type for {name}: {type(url_or_dict)}")
            
    def DownloadUnzip(self, url, directory, folder_name=None, num_downl=1, total_downloads=1):

        out_path = os.path.join(directory, folder_name)
        if not os.path.exists(out_path):
            print("Downloading {}...".format(folder_name.split(os.sep)[-1]))
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

            print(f"{folder_name} has been successfully installed")

    def CheckDependency(self) -> None:
        """
        Check and install required Python dependencies for VFACE module.
        
        Verifies installation of joblib and lightgbm, installs if missing,
        and downloads required model files.
        """
        try:
            logger.info("=== Checking and installing Python dependencies ===")
            
            # Check and install joblib
            try:
                import joblib
                logger.info(f"✓ joblib is already installed (version: {joblib.__version__})")
            except ImportError:
                logger.warning("✗ joblib not found, installing...")
                try:
                    logger.info("Installing joblib...")
                    slicer.util.pip_install('joblib')
                    import joblib
                    logger.info(f"✓ joblib successfully installed (version: {joblib.__version__})")
                except Exception as e:
                    logger.error(f"✗ Failed to install joblib: {str(e)}")
                    raise
            
            # Check and install lightgbm
            try:
                import lightgbm
                logger.info(f"✓ lightgbm is already installed (version: {lightgbm.__version__})")
            except ImportError:
                logger.warning("✗ lightgbm not found, installing...")
                try:
                    logger.info("Installing lightgbm... (this may take a while)")
                    slicer.util.pip_install('lightgbm')
                    import lightgbm
                    logger.info(f"✓ lightgbm successfully installed (version: {lightgbm.__version__})")
                except Exception as e:
                    logger.error(f"✗ Failed to install lightgbm: {str(e)}")
                    raise
            
            logger.info("=== Python dependencies check completed ===")
            logger.info("--- Downloading model files ---")
            self.DownloadAllFiles()
            logger.info("✓ All dependencies have been successfully installed")
            
        except Exception as e:
            logger.error(f"Error during dependency check: {e}")
            raise

    def onComboBoxChanged(self, text):
        """Called when the main comboBox value changes"""
        print(f"ComboBox changed to: {text}")

    def onComboBox2Changed(self, text):
        
        if text == "Visualization (Heatmaps)":
            self.ui.excellabel.setVisible(False)
            self.ui.PathLineEdit_3.setVisible(False)
        else:
            self.ui.excellabel.setVisible(True)
            self.ui.PathLineEdit_3.setVisible(True)

        self._checkCanApply()

    def onComboBox3Changed(self, text):
        if text == "File already Registered":
            self.ui.t2label.setText("Registered T2 Folder")
            self.ui.t2label.setVisible(True)
            self.ui.PathLineEdit_4.setVisible(True)
            self.ui.label_2.setVisible(False)
            self.ui.comboBox.setVisible(False)
        elif self.ui.comboBox4.currentText != "Longitudinal studies":
            self.ui.t2label.setVisible(False)
            self.ui.PathLineEdit_4.setVisible(False)
            self.ui.label_2.setVisible(True)
            self.ui.comboBox.setVisible(True)
        else:
            self.ui.t2label.setText("T2 Folder")
        if text != "Full pipeline":
            self.ui.modeLabel.setText("Oriented T1 Folder")
        else:
            self.ui.modeLabel.setText("T1 Folder")
        self._checkCanApply()

    def onComboBox4Changed(self, text):
        if text == "Longitudinal studies":
            self.ui.t2label.setVisible(True)
            self.ui.PathLineEdit_4.setVisible(True)
            self.ui.excellabel.setText("List of measurements folder")
        else:
            self.ui.t2label.setVisible(False)
            self.ui.PathLineEdit_4.setVisible(False)
            self.ui.excellabel.setText("List of measurements + ML readable result folder")

        if self.ui.comboBox3.currentText == "File already Registered":
                self.ui.t2label.setText("Registered T2 Folder")
            
        self._checkCanApply()

    def onApplyButton(self) -> None:
        import time
        self.CliStartTime = time.time()
        slicer.app.processEvents()

        self.list_process = CreateListProcess(InputFolder = self._parameterNode.InputFolder
                               ,OutputFolder = self._parameterNode.OutputFolder
                               ,model_folder = os.path.join(self.SlicerDownloadPath,"AREG/AREG_CBCT/Models/Segmentation"),
                               model_folder_ali = os.path.join(self.SlicerDownloadPath,"ALI/ALI_CBCT/Models/Landmark"),
                               reg_type = self.ui.comboBox.currentText,
                               gold_folder = os.path.join(self.SlicerDownloadPath,"ASO/ASO_CBCT/Reference"),
                               mirror_matrix = os.path.join(self.SlicerDownloadPath,"Mirror_matrix/Mirror/Matrix_mirror.tfm"),
                                bool_visualization = True if "Visualization" in self.ui.comboBox2.currentText else False,
                                bool_quantification = True if "Quantitative" in self.ui.comboBox2.currentText else False,
                                measurements_folder = self._parameterNode.MeasurementsFolder,
                                mode = self.ui.comboBox3.currentText,
                                t2_folder = self.ui.PathLineEdit_4.currentPath,
                                mode2 = self.ui.comboBox4.currentText,
                                model_vface = os.path.join(self.SlicerDownloadPath,"V_FACE"))

        self.ui.applyButton.enabled = False
        self.ui.CheckDependencyButton.enabled = False
        self.ui.cancelButton.setVisible(True)
        self.ui.label_3.setVisible(True)
        self.ui.progressBar.setVisible(True)

        self.NumberProcess = len(self.list_process)
        self.executeProcess(self.list_process[0])
        del self.list_process[0]

    def onContinueButton(self) -> None:
        """
        Handle continue button click to resume processing after visualization.
        
        Called after user has reviewed visualization and is ready to proceed with next steps.
        """
        logger.info("Continuing process after visualization...")
        self.ui.continueButton.setVisible(False)
        self.ui.cancelButton.setVisible(True)
        self.paused_for_visualization = False
        self.ui.label_3.setVisible(True)
        self.ui.progressBar.setVisible(True)
        
        if self.list_process:
            self.ActualProcess += 1
            self.executeProcess(self.list_process[0])
            del self.list_process[0]
        else:
            self.OnEndProcess()

    def onCancelButton(self) -> None:
        """
        Handle cancel button click with user confirmation.
        
        Displays confirmation dialog before canceling the current process.
        """
        msgBox = qt.QMessageBox()
        msgBox.setWindowTitle("Confirm Cancellation")
        msgBox.setText("Are you sure you want to cancel the current process?")
        msgBox.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        msgBox.setDefaultButton(qt.QMessageBox.No)
        
        result = msgBox.exec_()
        
        if result == qt.QMessageBox.Yes:
            self.cancelProcess()
    
    def cancelProcess(self) -> None:
        """
        Cancel the currently running process (CLI or Python).
        
        Handles cleanup of resources and resets UI after cancellation.
        """
        logger.info("Canceling process...")
        
        # Stop CLI process if running
        if hasattr(self, 'cliNode') and self.cliNode:
            try:
                self.cliNode.Cancel()
                logger.info("CLI process canceled")
            except Exception as e:
                logger.error(f"Error canceling CLI process: {e}")
        
        # Stop Python process if running
        if hasattr(self, 'python_process') and self.python_process:
            try:
                # Special handling for segmentation processes
                if hasattr(self.python_process, '__name__') and 'bds' in self.python_process.__name__:
                    try:
                        import sys
                        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        if parent_dir not in sys.path:
                            sys.path.insert(0, parent_dir)
                        
                        from VFACE_utils.segmentation_logic import SegmentationLogic
                        # Create temporary instance to stop all processes
                        temp_logic = SegmentationLogic()
                        temp_logic.stop()
                        logger.info("Segmentation process canceled")
                    except Exception as e:
                        logger.error(f"Error stopping segmentation: {e}")
                
                self.python_process_completed = True
                logger.info("Python process canceled")
            except Exception as e:
                logger.error(f"Error canceling Python process: {e}")
        
        # Reset interface state
        self.list_process = []
        self.ui.applyButton.enabled = True
        self.ui.CheckDependencyButton.enabled = True
        self.ui.cancelButton.setVisible(False)
        self.ui.label_3.setVisible(False)
        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setValue(0)
        
        if hasattr(self, 'continueButton'):
            self.ui.continueButton.setVisible(False)
        
        logger.info("Process cancellation completed")
        self.cancelCurrentProcess()

    def cancelCurrentProcess(self) -> None:
        """
        Cancel the currently running process with cleanup.
        
        Removes observers and clears process list, then displays confirmation dialog.
        """
        logger.info("Cancelling current process...")
        
        if hasattr(self, 'cliNode') and self.cliNode is not None:
            try:
                self.cliNode.Cancel()
                self.removeObserver(self.cliNode, vtk.vtkCommand.ModifiedEvent, self.onCliUpdated)
                logger.info("Process cancelled")
            except Exception as e:
                logger.error(f"Error cancelling CLI process: {e}")
        
        self.list_process.clear()
        self.resetUIAfterCancel()

        confirmation = PopUpWindow(title="Process Cancelled", text="The process has been successfully cancelled.")
        confirmation.exec_()

    def resetUIAfterCancel(self) -> None:
        """
        Reset all UI elements to their initial state after process cancellation.
        
        Hides progress indicators, re-enables buttons, and clears internal state.
        """
        self.ui.label_3.setVisible(False)
        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setValue(0)
        self.ui.continueButton.setVisible(False)
        self.ui.cancelButton.setVisible(False)
        
        self.ui.applyButton.enabled = True
        self.ui.CheckDependencyButton.enabled = True

        self.paused_for_visualization = False
        self.current_output_to_load = None
        self.current_process_info = None
        self.cliNode = None
        self.ActualProcess = 1
        self.NumberProcess = 0
        
        logger.info("Interface reset after cancellation")

    def loadOutputInSlicer(self, output_path: str) -> bool:
        """
        Load processed output into Slicer viewer for visualization.
        
        Supports volume formats (.nrrd, .nii, .nii.gz) and model format (.vtk).
        Automatically configures appropriate layout and view settings.
        
        Args:
            output_path: Path to the output file to load
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            if not output_path or not os.path.exists(output_path):
                logger.warning(f"Output path does not exist: {output_path}")
                return False

            if output_path.endswith(('.nrrd', '.nii', '.nii.gz')):
                # Load volume file
                volume_node = slicer.util.loadVolume(output_path)
                if volume_node:
                    slicer.util.setSliceViewerLayers(background=volume_node)
                    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
                    slicer.util.resetSliceViews()
                    logger.info(f"Volume loaded: {output_path}")
                    return True
                    
            elif output_path.endswith('.vtk'):
                # Load model file
                model_node = slicer.util.loadModel(output_path)
                if model_node:
                    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
                    threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
                    threeDView.resetFocalPoint()
                    logger.info(f"Model loaded: {output_path}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error loading output file {output_path}: {e}")
        
        return False

    def shouldPauseAfterProcess(self, process_info: dict) -> bool:
        """
        Determine if process execution should pause for visualization review.
        
        Args:
            process_info: Process information dictionary
            
        Returns:
            bool: True if pause is requested, False otherwise
        """
        return process_info.get("pause_for_visualization", False)

    def getOutputPathForModule(self, module_name: str) -> str:
        output_paths = {
            "Orient T1 (CB)": os.path.join(self._parameterNode.OutputFolder, "Oriented Scans", "Oriented relative CB"),
            "Orient T1 (MAX)": os.path.join(self._parameterNode.OutputFolder, "Oriented Scans", "Oriented relative MAX"),
            "Masks Generation for T1 (MAX)": os.path.join(self._parameterNode.OutputFolder, "Masks"),
            "Mirroring Masks": os.path.join(self._parameterNode.OutputFolder, "Mirrored_Masks"),
            "Mirroring MAX Oriented Scan": os.path.join(self._parameterNode.OutputFolder, "Mirrored_Scan", "CB"),
            "AREG - Registering Scan (MAND)": os.path.join(self._parameterNode.OutputFolder, "Registered Scan", "Mandible"),
            "BDS - Segmentation T2 MAX": os.path.join(self._parameterNode.OutputFolder, "VTK Files","T2 MAX"),
        }
        
        base_path = output_paths.get(module_name)
        if base_path and os.path.exists(base_path):
            for ext in ['.nrrd', '.nii.gz', '.nii', '.vtk']:
                for file in os.listdir(base_path):
                    if file.endswith(ext):
                        return os.path.join(base_path, file)
            
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith(('.nrrd', '.nii.gz', '.nii', '.vtk')):
                        return os.path.join(root, file)
        return None

    def executeProcess(self, process_info):
        import time
        self.CliStepTime = time.time()
        self.module_name = process_info["Module"]
        self.displayModule = process_info["Display"]
        self.current_process_info = process_info  # Stocker les infos du processus actuel
        
        process = process_info["Process"]
        parameters = process_info["Parameter"]
        
        test1 = hasattr(process, '__module__') and 'slicer' in str(process.__module__) if hasattr(process, '__module__') else False
        test2 = str(type(process)).find('vtkMRML') != -1
        test3 = str(process).startswith('<vtkMRMLCommandLineModuleNodePython')
        test4 = 'slicer.modules' in str(process)
        test5 = hasattr(process, 'GetModuleTitle')
        test6 = hasattr(process, 'GetID') and hasattr(process, 'GetModuleTitle')
        test7 = str(type(process)).find('vtkMRMLCommandLineModuleNode') != -1
        test8 = str(type(process)).find('qSlicerCLIModule') != -1
        test9 = 'PythonQt.qSlicerBaseQTCLI' in str(process.__module__) if hasattr(process, '__module__') else False

        is_slicer_module = (
            test1 or test2 or test3 or test4 or test5 or test6 or test7 or test8 or test9
        )
        
        if is_slicer_module:
            print(f"{self.module_name} is executed.")
            self.cliNode = slicer.cli.run(process, None, parameters)
            self.addObserver(self.cliNode, vtk.vtkCommand.ModifiedEvent, self.onCliUpdated)
        else:
            print(f"{self.module_name} is executed.")
            self.ui.label_3.setText(f"Process : {self.module_name} ({self.ActualProcess}/{self.NumberProcess})")
            
            # Pour les processus Python longs, utiliser un timer pour maintenir la réactivité
            self.python_process = process
            self.python_parameters = parameters
            self.python_process_completed = False
            self.python_process_error = None
            
            # Démarrer le processus dans un timer pour permettre les processEvents
            self.startPythonProcess()

    def startPythonProcess(self):
        """Démarre un processus Python avec gestion de la réactivité"""
        try:
            if callable(self.python_process):
                # Exécuter le processus
                result = self.python_process(**self.python_parameters)
                print(f"Result of {self.module_name}: {result}")
                self.python_process_completed = True
            else:
                print(f"Error: {self.python_process} is not a callable function")
                self.python_process_error = "Process is not callable"
                self.python_process_completed = True
                
        except Exception as e:
            print(f"Error during the execution of {self.module_name}: {e}")
            import traceback
            traceback.print_exc()
            self.python_process_error = str(e)
            self.python_process_completed = True
        
        # Programmer la vérification de fin de processus
        import qt
        qt.QTimer.singleShot(100, self.checkPythonProcessStatus)

    def checkPythonProcessStatus(self):
        """Vérifie le statut du processus Python"""
        if self.python_process_completed:
            self.onProcessCompleted()
        else:
            # Continuer à vérifier le statut
            import qt
            qt.QTimer.singleShot(100, self.checkPythonProcessStatus)

    def onProcessCompleted(self):
        print("\n\n ========= PROCESSED ========= \n")

        if self.shouldPauseAfterProcess(self.current_process_info) and self.ui.checkBox_2.isChecked():
            output_path = self.getOutputPathForModule(self.module_name)
            if output_path:
                if self.loadOutputInSlicer(output_path):
                    self.paused_for_visualization = True
                    self.current_output_to_load = output_path
                    self.ui.continueButton.setVisible(True)
                    self.ui.label_3.setText(f"Result of {self.module_name} loaded - Click on Continue to start next steps")
                    print(f"Process on pause after {self.module_name}. Result loaded.")
                    return

        try:
            self.executeProcess(self.list_process[0])
            self.ActualProcess += 1
            del self.list_process[0]
        except IndexError:
            self.OnEndProcess()

    def onCliUpdated(self, caller, event):
        import time
        import json
        import subprocess
        cliNode = caller

        status = cliNode.GetStatus()

        if status & slicer.vtkMRMLCommandLineModuleNode.Completed or \
           status & slicer.vtkMRMLCommandLineModuleNode.Cancelled:

            self.removeObserver(cliNode, vtk.vtkCommand.ModifiedEvent, self.onCliUpdated)

            print("\n\n ========= PROCESSED ========= \n")
            print(caller.GetOutputText())
            
            if self.shouldPauseAfterProcess(self.current_process_info) and self.ui.checkBox_2.isChecked():
                output_path = self.getOutputPathForModule(self.module_name)
                if output_path:
                    if self.loadOutputInSlicer(output_path):
                        self.paused_for_visualization = True
                        self.current_output_to_load = output_path
                        self.ui.continueButton.setVisible(True)
                        self.ui.label_3.setText(f"Result of {self.module_name} loaded - Click on Continue to start next steps")
                        print(f"Process on pause after {self.module_name}. Result loaded.")
                        self.ui.progressBar.setValue(0)
                        return
            
            try:
                # Exécuter le processus suivant
                self.ui.progressBar.setValue(0)
                self.executeProcess(self.list_process[0])
                self.ActualProcess += 1
                
                del self.list_process[0]
            except IndexError:
                self.OnEndProcess()

        progress = caller.GetProgress()
        if progress == 0:
            self.updateProgessBar = False

        if self.displayModule.isProgress(progress=progress, updateProgessBar=self.updateProgessBar):
            progress_bar, message = self.displayModule()
            self.ui.progressBar.setValue(progress_bar)

        act_time = time.time()
        intermediary_time = act_time-self.CliStepTime
        total_time = act_time-self.CliStartTime

        if intermediary_time < 60:
            intermediary_timer = f"Time : {int(intermediary_time)}s"
        elif intermediary_time < 3600:
            intermediary_timer = f"Time : {int(intermediary_time/60)}min and {int(intermediary_time%60)}s"
        else:
            intermediary_timer = f"Time : {int(intermediary_time/3600)}h, {int(intermediary_time%3600/60)}min and {int(intermediary_time%60)}s"

        if total_time < 60:
            timer = f"Total : {int(total_time)}s"
        elif total_time < 3600:
            timer = f"Total : {int(total_time/60)}min and {int(total_time%60)}s"
        else:
            timer = f"Total : {int(total_time/3600)}h, {int(total_time%3600/60)}min and {int(total_time%60)}s"

        self.ui.label_3.setText(f"Process : {self.module_name} ({self.ActualProcess}/{self.NumberProcess})({intermediary_timer},{timer})")

    def OnEndProcess(self):
        from pathlib import Path
        import time
        act_time = time.time()
        total_time = act_time-self.CliStartTime

        if total_time < 60:
            timer = f"{int(total_time)}s"
        elif total_time < 3600:
            timer = f"{int(total_time/60)}min and {int(total_time%60)}s"
        else:
            timer = f"{int(total_time/3600)}h, {int(total_time%3600/60)}min and {int(total_time%60)}s"

        logger.info(f"PROCESS COMPLETED in {timer}")

        self.ui.label_3.setVisible(False)
        self.ui.progressBar.setVisible(False)
        self.ui.continueButton.setVisible(False)
        self.ui.cancelButton.setVisible(False)
        self.ui.CheckDependencyButton.enabled = True
        self.paused_for_visualization = False
        self.ActualProcess = 1
        self.NumberProcess = 0
        self.cliNode = None
        self.list_process = []
        self.current_output_to_load = None
        self.current_process_info = None

        completion_dialog = PopUpWindow(title="Process Complete", text="Processing completed successfully!")
        completion_dialog.exec_()
        self._checkCanApply()

        # Clean up temporary files if requested
        if not self.ui.checkBox.isChecked():
            files_to_keep = []
            if "Visualization" in self.ui.comboBox2.currentText:
                files_to_keep.append("Heatmaps")
                files_to_keep.append("VTK Files")
            if "Quantification" in self.ui.comboBox2.currentText:
                files_to_keep.append("Measurements")
                files_to_keep.append("Classification")

            try:
                output_path = Path(self._parameterNode.OutputFolder)
                for item in output_path.iterdir():
                    if item.is_dir() and item.name not in files_to_keep:
                        shutil.rmtree(item)
                        logger.info(f"Cleaned temporary folder: {item.name}")
            except Exception as e:
                logger.error(f"Error cleaning temporary files: {e}")
            
            
            

class VFACELogic(ScriptedLoadableModuleLogic):
    """
    Logic class for VFACE module.
    
    This class implements all computations and should be designed such that
    other python code can import it and use its functionality without requiring
    an instance of the Widget.
    
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Initialize the logic class.
        
        Called when the logic instance is created. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self) -> VFACEParameterNode:
        """
        Get the VFACE parameter node.
        
        Returns:
            VFACEParameterNode: The parameter node for this module
        """
        return VFACEParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# VFACETest
#


class VFACETest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_VFACE1()

    def test_VFACE1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("VFACE1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = VFACELogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
