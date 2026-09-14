import contextlib
import logging
import glob
import os
import re
import stat
import tempfile
from typing import Annotated
import urllib.request
import shutil
import zipfile
import sys

# ===== Logging Configuration =====
logger = logging.getLogger("VFACE")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

import importlib
try:
    from VFACE_utils import Progress
    importlib.reload(Progress)
    from VFACE_utils.Progress import DisplayALICBCT,DisplayAMASSS,DisplayASOCBCT,Display
    
    from VFACE_utils import createlistprocess
    importlib.reload(createlistprocess)
    from VFACE_utils.createlistprocess import CreateListProcess, NumberScan, patientIdFromFileName

    from VFACE_utils import review_steps
    importlib.reload(review_steps)

except Exception as e:
    logger.error(f"Error loading VFACE utilities: {e}")
    from VFACE_utils.Progress import DisplayALICBCT,DisplayAMASSS,DisplayASOCBCT,Display
    from VFACE_utils.createlistprocess import CreateListProcess, NumberScan, patientIdFromFileName
    from VFACE_utils import review_steps

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
        # Without a parent the window manager attaches the dialog to a 1x1
        # dummy window and never maps it. A modal one then takes every click
        # with nothing on screen to dismiss - the run looks frozen at the end.
        qt.QWidget.__init__(self, slicer.util.mainWindow())
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
        self.current_process_info = None
        # One review item per patient for the step being reviewed, plus the
        # nodes it put in the scene and the landmark positions as loaded, so
        # Continue can tell an edited file from an untouched one.
        self.pause_queue = []
        self.pause_index = 0
        self.pause_nodes = []
        self.pause_markups_nodes = []
        self.pause_markups_start = {}
        self.pause_transform = None
        self.pause_transform_item = None
        self.pause_pending = False
        # Batch review: the patients marked for rework, the steps already run
        # so a rollback knows what lies behind, the patients a rollback is
        # replaying, and the temporary folders a narrowed replay leaves behind.
        self.review_flagged = set()
        # {landmark: [scans it was missed on, scans tried]}, filled from the
        # summary ALI prints when a run ends.
        self.missing_landmarks = {}
        # (display node, its colour legend) so one follows the other's visibility
        self.legend_pairs = []
        self.executed_steps = []
        self.review_flagged_carry = []
        self.review_temp_folders = []
        # onCliUpdated only assigns this when progress is 0, so a first event
        # carrying a non-zero progress would read it before it exists.
        self.updateProgessBar = False

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

    @staticmethod
    def widenCapturedPipes() -> None:
        """Give Slicer's captured output more room than the default 64 KB.

        Precaution, not a proven cure. What is established: long VFACE runs have
        frozen several times with the main thread in `pipe_write` on the pipe
        Slicer captures its own stdout into, zero CPU, never recovering. What is
        not established: what fills it. Writing 200 KB from inside a VTK observer
        callback - the shape the freeze was blamed on - does NOT deadlock, with
        or without a main window, so that explanation is wrong or incomplete.

        Widening suppresses no output and changes no behaviour; it only raises
        the ceiling from 64 KB to whatever the kernel allows, typically 1 MB,
        against a whole run's console output of roughly 100 KB. If the freeze
        really is an accumulation, this removes it; if it is something else, this
        costs nothing. Do not read it as the fix until a run confirms it.

        Fails quietly wherever it does not apply - Windows, output redirected to
        a file rather than a pipe, a kernel that refuses - because a smaller pipe
        is not worth failing over.
        """
        try:
            import fcntl
        except ImportError:
            return  # not a POSIX platform; nothing to widen
        F_SETPIPE_SZ, F_GETPIPE_SZ = 1031, 1032
        try:
            with open("/proc/sys/fs/pipe-max-size") as fh:
                target = int(fh.read().strip())
        except (OSError, ValueError):
            target = 1024 * 1024
        for fd in (1, 2):
            try:
                if not stat.S_ISFIFO(os.fstat(fd).st_mode):
                    continue
                before = fcntl.fcntl(fd, F_GETPIPE_SZ)
                if before >= target:
                    continue
                fcntl.fcntl(fd, F_SETPIPE_SZ, target)
                logger.info(
                    f"Captured output fd{fd} widened {before} -> "
                    f"{fcntl.fcntl(fd, F_GETPIPE_SZ)} bytes"
                )
            except (OSError, ValueError) as e:
                logger.warning(f"Could not widen fd{fd}, leaving it as it is: {e}")

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Before anything else writes: the deadlock this avoids takes the whole
        # application down, and an undersized pipe is only a problem once it is
        # already full.
        self.widenCapturedPipes()

        self.reloadCustomModules()

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/VFACE.ui"))
        self.uiWidget = uiWidget
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Detect dark mode and apply stylesheet
        isDarkMode = self._isDarkMode()
        styleSheet = self._getStyleSheet(isDarkMode)
        uiWidget.setStyleSheet(styleSheet)
        
        # Also apply label-specific stylesheet
        self._applyLabelStyleSheets(isDarkMode)
        self._applyButtonStyleSheets(isDarkMode)
        self._applyCheckboxStyleSheets(isDarkMode)

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
        self.ui.DefaultListButton.connect("clicked(bool)", self.onDefaultButton)
        self.ui.TestFilesButton.connect("clicked(bool)", self.onTestFilesButton)

        self.ui.continueButton.setVisible(False)
        self.ui.continueButton.connect("clicked(bool)", self.onContinueButton)
        self.ui.reviewPrevPatientButton.connect("clicked(bool)", self.onReviewPreviousPatient)
        self.ui.reviewNextPatientButton.connect("clicked(bool)", self.onReviewNextPatient)
        self.ui.reviewFlagButton.connect("clicked(bool)", self.onReviewToggleFlag)
        self.ui.reviewGoBackButton.connect("clicked(bool)", self.onReviewGoBack)

        documentsLocation = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documentsLocation)

        self.display = Display
        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
        )

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.setupReviewUi()
        self.ui.label_3.setVisible(False)
        self.ui.reviewLabel.setVisible(False)
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
            'label_5', 'label_4', 'label_2', 'label_6', 'label_3', 'label', 'modeLabel', 't2label', 'excellabel'
        ]
        
        for labelName in labels:
            if hasattr(self.ui, labelName):
                label = getattr(self.ui, labelName)
                label.setStyleSheet(labelStyle)

        if hasattr(self.ui, "reviewLabel"):
            if isDarkMode:
                reviewStyle = (
                    "color: #e8e8e8; background-color: #2f3b47;"
                    " border: 1px solid #4ba3ff; border-radius: 4px; padding: 8px;"
                )
            else:
                reviewStyle = (
                    "color: #1f2d3a; background-color: #eaf3fb;"
                    " border: 1px solid #3498db; border-radius: 4px; padding: 8px;"
                )
            self.ui.reviewLabel.setStyleSheet(reviewStyle)

    def _applyCheckboxStyleSheets(self, isDarkMode: bool) -> None:
        """
        Style every checkbox of the module, the review list included.

        Slicer's default indicator all but disappears against a dark panel, so
        it is redrawn - the same treatment AREG gives its own. In light mode
        the default is already right, and the style is cleared rather than
        left behind when the theme changes back.

        Args:
            isDarkMode: Whether the application is in dark mode
        """
        if isDarkMode:
            stylesheet = """
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
              image: url(:/Icons/SmallCheckMark.png);
            }
            QCheckBox::indicator:checked:hover {
              border: 1px solid #7bbcef;
              background-color: #7bbcef;
            }
            """
        else:
            # Same shape, light palette. AREG only redraws its checkboxes in
            # dark mode, which left this module's own look inconsistent: its
            # buttons are styled in both themes, so its boxes have to be too.
            stylesheet = """
            QCheckBox {
              color: #34495e;
              font-weight: 500;
              spacing: 6px;
            }
            QCheckBox::indicator {
              width: 18px;
              height: 18px;
              border: 1px solid #b0bec5;
              border-radius: 3px;
              background-color: #ffffff;
            }
            QCheckBox::indicator:hover {
              border: 1px solid #3498db;
            }
            QCheckBox::indicator:checked {
              width: 18px;
              height: 18px;
              border: 1px solid #3498db;
              border-radius: 3px;
              background-color: #3498db;
              image: url(:/Icons/SmallCheckMark.png);
            }
            QCheckBox::indicator:checked:hover {
              border: 1px solid #5cb3ff;
              background-color: #5cb3ff;
            }
            QCheckBox:disabled {
              color: #9aa5ab;
            }
            QCheckBox::indicator:disabled {
              border: 1px solid #d5dbdd;
              background-color: #eceff1;
            }
            """

        self._styleAllCheckboxes(getattr(self, "uiWidget", None), stylesheet)

    def _styleAllCheckboxes(self, parent, stylesheet: str) -> None:
        """
        Walk the widget tree and style every checkbox it holds.

        The review steps are built after setup and rebuilt on every change of
        mode, so they cannot be styled by name from a fixed list.

        Args:
            parent: Widget to start from
            stylesheet: Style to apply, empty to restore the default
        """
        if parent is None:
            return
        if isinstance(parent, qt.QCheckBox):
            try:
                parent.setStyleSheet(stylesheet)
            except Exception as e:
                logger.warning(f"Could not style a checkbox: {e}")
        if hasattr(parent, "children"):
            for child in parent.children():
                self._styleAllCheckboxes(child, stylesheet)

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
        for buttonName in ['applyButton', 'CheckDependencyButton', 'continueButton',
                           'DefaultListButton', 'TestFilesButton',
                           'reviewSelectAllButton', 'reviewSelectNoneButton',
                           'reviewSelectRecommendedButton',
                           'reviewPrevPatientButton', 'reviewNextPatientButton',
                           'reviewFlagButton', 'reviewGoBackButton']:
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

        # A file each archive is known to contain. Without it the folder alone is
        # used to decide whether the download already happened, and the V_FACE
        # folder is created by the Default List button before the models exist.
        check_files = {
            "V_FACE": "sym_asymm.txt",
        }

        if not os.path.exists(self.SlicerDownloadPath):
            os.makedirs(self.SlicerDownloadPath)

        for name, url_or_dict in dic_url.items():
            if isinstance(url_or_dict, str):
                self.DownloadUnzip(
                    url=url_or_dict,
                    directory=self.SlicerDownloadPath,
                    folder_name=name,
                    check_file=check_files.get(name),
                )
            elif isinstance(url_or_dict, dict):
                for subfolder_name, url in url_or_dict.items():
                    self.DownloadUnzip(
                        url=url,
                        directory=self.SlicerDownloadPath,
                        folder_name=os.path.join(name, subfolder_name),
                    )
            else:
                logger.warning(f"Warning: Unknown type for {name}: {type(url_or_dict)}")
            
    def DownloadUnzip(self, url, directory, folder_name=None, num_downl=1, total_downloads=1, check_file=None):

        out_path = os.path.join(directory, folder_name)
        # The folder alone is a poor "already installed" test: another feature may
        # have created it (Default List creates V_FACE/DefaultList, hence V_FACE),
        # and a download that fails leaves it behind empty. Either way this skipped
        # the download for ever. check_file names something the archive contains.
        installed = os.path.join(out_path, check_file) if check_file else out_path
        if not os.path.exists(installed):
            logger.info("Downloading {}...".format(folder_name.split(os.sep)[-1]))
            os.makedirs(out_path, exist_ok=True)

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
                logger.info(f"joblib is already installed (version: {joblib.__version__})")
            except ImportError:
                logger.warning("joblib not found, installing...")
                try:
                    logger.info("Installing joblib...")
                    slicer.util.pip_install('joblib')
                    import joblib
                    logger.info(f"joblib successfully installed (version: {joblib.__version__})")
                except Exception as e:
                    logger.error(f"Failed to install joblib: {str(e)}")
                    raise
            
            # Check and install lightgbm
            try:
                import lightgbm
                logger.info(f"lightgbm is already installed (version: {lightgbm.__version__})")
            except ImportError:
                logger.warning("lightgbm not found, installing...")
                try:
                    logger.info("Installing lightgbm... (this may take a while)")
                    slicer.util.pip_install('lightgbm')
                    import lightgbm
                    logger.info(f"lightgbm successfully installed (version: {lightgbm.__version__})")
                except Exception as e:
                    logger.error(f"Failed to install lightgbm: {str(e)}")
                    raise
            
            logger.info("=== Python dependencies check completed ===")
            logger.info("--- Downloading model files ---")
            self.DownloadAllFiles()
            logger.info("All dependencies have been successfully installed")
            
        except Exception as e:
            logger.error(f"Error during dependency check: {e}")
            raise

    def onComboBoxChanged(self, text):
        """Called when the main comboBox value changes"""
        logger.info(f"ComboBox changed to: {text}")
        self.rebuildReviewSteps()

    def onComboBox2Changed(self, text):
        
        if text == "Visualization (Heatmaps)":
            self.ui.excellabel.setVisible(False)
            self.ui.PathLineEdit_3.setVisible(False)
            self.ui.DefaultListButton.setVisible(False)
        else:
            self.ui.excellabel.setVisible(True)
            self.ui.PathLineEdit_3.setVisible(True)
            self.ui.DefaultListButton.setVisible(True)

        self.rebuildReviewSteps()
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
        self.rebuildReviewSteps()
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
            
        self.rebuildReviewSteps()
        self._checkCanApply()

    def onApplyButton(self) -> None:
        import time

        # Every step downstream reports "0 file" on an input folder holding nothing
        # it can read, and the run walks its whole plan producing nothing. Say so
        # here instead, while the user can still act on it.
        if not self.checkInputFolder():
            return

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

        if self.list_process:
            self.applyReviewSelection()

        if not self.list_process:
            PopUpWindow(
                title="Nothing to run",
                text="No processing step could be built for the selected options.\n"
                     "Check the log for the reason.",
            ).exec_()
            return

        self.ui.applyButton.enabled = False
        self.ui.CheckDependencyButton.enabled = False
        self.ui.cancelButton.setVisible(True)
        self.ui.label_3.setVisible(True)
        self.ui.reviewLabel.setVisible(False)
        self.ui.progressBar.setVisible(True)

        self.NumberProcess = len(self.list_process)
        # A second run in the same session must not see the first one's steps,
        # nor report the landmarks the previous one missed.
        self.executed_steps = []
        self.missing_landmarks = {}
        self.review_flagged = set()
        self.review_flagged_carry = []
        self.clearReviewTempFolders()
        self.executeProcess(self.list_process[0])
        del self.list_process[0]

    def checkInputFolder(self) -> bool:
        """Warn and refuse to start when the input folder holds no readable scan."""
        input_folder = self._parameterNode.InputFolder

        if not input_folder or not os.path.isdir(input_folder):
            PopUpWindow(
                title="Input folder not found",
                text=f"This input folder does not exist:\n\n{input_folder}",
            ).exec_()
            return False

        nb_scan = NumberScan(input_folder)
        if nb_scan == 0:
            PopUpWindow(
                title="No scan found",
                text=(
                    f"No scan found in:\n\n{input_folder}\n\n"
                    "Expected a CBCT volume per patient, as .nii, .nii.gz, .nrrd,\n"
                    ".nrrd.gz, .gipl or .gipl.gz. Surface meshes (.vtk, .stl) are\n"
                    "produced by this module, they are not an input for it."
                ),
            ).exec_()
            logger.error(f"No scan found in the input folder: {input_folder}")
            return False

        logger.info(f"{nb_scan} patient(s) found in the input folder")
        return True

    def onContinueButton(self) -> None:
        """
        Handle continue button click to resume processing after visualization.
        
        Called after user has reviewed visualization and is ready to proceed with next steps.
        """
        # Moving between patients is what the navigation buttons are for, so
        # this one does the single thing its label promises: it ends the pause.
        # What the user changed on the patient in front of them is saved first,
        # and the patients they never opened keep the results they already have.
        if self.pause_queue:
            self.savePauseEdits()
            seen = self.pause_index + 1
            total = len(self.pause_queue)
            if seen < total:
                logger.info(f"{total - seen} patient(s) accepted without being opened")

        self.resetPauseState()

        logger.info("Continuing process after visualization...")
        self.ui.continueButton.setVisible(False)
        self.ui.cancelButton.setVisible(True)
        self.ui.label_3.setVisible(True)
        self.ui.progressBar.setVisible(True)
        
        if self.list_process:
            self.ActualProcess += 1
            self.executeProcess(self.list_process[0])
            del self.list_process[0]
        else:
            self.OnEndProcess()
    
    # VFACE has no test archive of its own, and its T1 input is a plain CBCT:
    # AREG's orientation set holds exactly that, unoriented, which is what the
    # full pipeline expects. Replace with a VFACE release asset if one is ever
    # published.
    TEST_FILES_NAME = "Oriented-Automated"
    TEST_FILES_URL = (
        "https://github.com/lucanchling/Areg_CBCT/releases/download/TestFiles/"
        "Or_FullyAuto.zip"
    )

    def onTestFilesButton(self) -> None:
        """
        Download a sample CBCT and point the T1 input at it.

        The archive carries a T1 and a T2; only the T1 is of any use here, so
        that is the folder the input is set to.
        """
        if not os.path.exists(self.SlicerDownloadPath):
            os.makedirs(self.SlicerDownloadPath)

        # Under V_FACE, next to DefaultList: the archive is borrowed from AREG
        # but the copy belongs to this module, and uninstalling one must not
        # take the other's sample away.
        folder_name = os.path.join("V_FACE", "Test_Files", self.TEST_FILES_NAME)
        try:
            self.DownloadUnzip(
                url=self.TEST_FILES_URL,
                directory=self.SlicerDownloadPath,
                folder_name=folder_name,
                check_file="T1",
            )
        except Exception as e:
            logger.error(f"Could not download the test files: {e}")
            PopUpWindow(
                title="Download failed",
                text=f"The sample scan could not be downloaded:\n\n{e}",
            ).exec_()
            return

        # The archive is AREG's, so it carries a second timepoint. VFACE makes
        # its own T2 by mirroring the T1 and never reads one from disk, so that
        # half is dropped rather than left to take up room for nothing.
        t2_folder = os.path.join(self.SlicerDownloadPath, folder_name, "T2")
        if os.path.isdir(t2_folder):
            try:
                shutil.rmtree(t2_folder)
                logger.info("Test files: dropped the T2 this module has no use for")
            except OSError as e:
                logger.warning(f"Could not remove the unused T2 folder: {e}")

        t1_folder = os.path.join(self.SlicerDownloadPath, folder_name, "T1")
        if not os.path.isdir(t1_folder):
            logger.error(f"No T1 folder in the test files: {t1_folder}")
            PopUpWindow(
                title="Test files incomplete",
                text=f"The archive holds no T1 folder:\n\n{t1_folder}",
            ).exec_()
            return

        nb_scan = NumberScan(t1_folder)
        if nb_scan == 0:
            logger.error(f"No scan found in the test files: {t1_folder}")
            PopUpWindow(
                title="No scan found",
                text=f"No readable scan in the downloaded folder:\n\n{t1_folder}",
            ).exec_()
            return

        self.ui.PathLineEdit.setCurrentPath(t1_folder)
        logger.info(f"Test files ready: {nb_scan} patient(s) in {t1_folder}")

        # Somewhere to write, beside the scans it will read, so the sample runs
        # on a single click. A folder the user already chose is left alone.
        if not self._parameterNode.OutputFolder:
            output = os.path.join(self.SlicerDownloadPath, folder_name, "Output")
            self.ui.PathLineEdit_2.setCurrentPath(output)
            logger.info(f"Test files: output set to {output}")

        # The sample is only runnable with the measurement lists beside it, so
        # fetch those too rather than leaving the user one unexplained click
        # short. A folder they already chose is left alone.
        if not self.ui.PathLineEdit_3.currentPath:
            self.onDefaultButton()

    def onDefaultButton(self):
        if not os.path.exists(self.SlicerDownloadPath):
            os.makedirs(self.SlicerDownloadPath)

        self.DownloadUnzip(
            url="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/VFACE/DefaultList.zip",
            directory=self.SlicerDownloadPath,
            folder_name="V_FACE/DefaultList",
        )
        if os.path.exists(os.path.join(self.SlicerDownloadPath,"V_FACE/DefaultList")):
            self.ui.PathLineEdit_3.setCurrentPath(os.path.join(self.SlicerDownloadPath,"V_FACE/DefaultList"))

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
                        
                        from VFACE_utils.segmentation_logic import stop_active_segmentation
                        # Stop the logic that is actually running: building a fresh
                        # SegmentationLogic here only killed a brand new idle process.
                        if stop_active_segmentation():
                            logger.info("Segmentation process canceled")
                        else:
                            logger.info("No segmentation currently running")
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

        # Deferred for the same reason as the completion dialog above: cancelling
        # can be reached while a CLI observer is still dispatching.
        qt.QTimer.singleShot(
            0,
            lambda: PopUpWindow(
                title="Process Cancelled", text="The process has been successfully cancelled."
            ).exec_(),
        )

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

        self.resetPauseState()
        self.current_process_info = None
        self.cliNode = None
        self.ActualProcess = 1
        self.NumberProcess = 0
        
        logger.info("Interface reset after cancellation")

    def abortOnCliError(self, module_name: str, error_text: str) -> None:
        """
        Stop the run where a CLI failed.

        Every later step reads what the previous one wrote, so carrying on past a
        failure only produces a chain of steps finding nothing and a run that
        claims to have completed. Stop here and say which step broke.

        Args:
            module_name: Step whose CLI reported errors
            error_text: Tail of that CLI's stderr, already trimmed
        """
        self.list_process = []
        self.resetUIAfterCancel()

        details = error_text.strip() if error_text else ""
        if not details:
            details = "The step produced no error output; see the Python console."

        # Deferred for the same reason as the completion dialog: this is reached
        # from inside a VTK observer callback, and an application-modal dialog
        # opened there runs a nested event loop while the CLI node is still
        # dispatching events.
        qt.QTimer.singleShot(
            0,
            lambda: PopUpWindow(
                title="Process failed",
                text=(
                    f"'{module_name}' failed, so the run was stopped.\n\n{details}"
                ),
            ).exec_(),
        )

    # ===== Manual review pauses =====
    #
    # A step marked "pause_for_visualization" stops the run once it finishes, so
    # the user can look at what it produced and, for landmarks, correct it before
    # the rest of the pipeline consumes it. The step itself says where its output
    # went ("ReviewFolder"), which is what keeps this in step with the folders
    # createlistprocess actually writes to.

    VOLUME_EXT = (".nrrd", ".nii.gz", ".nii", ".nrrd.gz", ".gipl.gz", ".gipl")
    MODEL_EXT = (".vtk", ".vtp", ".stl")
    CONTINUE_BUTTON_TEXT = "Next step"

    # ------------------------------------------------------- choosing the pauses

    REVIEW_SETTINGS_KEY = "VFACE/ReviewSteps"

    def setupReviewUi(self) -> None:
        """Wire the review section up. Called once, from setup()."""
        self.ui.checkBox_2.connect("toggled(bool)", self.onReviewEnableToggled)
        self.ui.reviewSelectAllButton.connect("clicked(bool)", lambda: self.setAllReviewSteps(True))
        self.ui.reviewSelectNoneButton.connect("clicked(bool)", lambda: self.setAllReviewSteps(False))
        self.ui.reviewSelectRecommendedButton.connect("clicked(bool)", self.setRecommendedReviewSteps)
        self.onReviewEnableToggled(self.ui.checkBox_2.isChecked())
        self.rebuildReviewSteps()

    def onReviewEnableToggled(self, enabled: bool) -> None:
        """Grey the list out when pausing is off, rather than hiding it."""
        for widget in (self.ui.reviewScrollArea, self.ui.reviewSelectAllButton,
                       self.ui.reviewSelectNoneButton, self.ui.reviewHelpLabel):
            widget.setEnabled(enabled)

    def rebuildReviewSteps(self) -> None:
        """Rebuild the checkboxes for the settings now chosen.

        A run that skips a step must not offer it: ticking a pause the run
        never reaches reads as a broken feature the first time it goes
        straight past. What the user ticked before is restored wherever the
        same step still exists.
        """
        contents = getattr(self.ui, "reviewScrollContents", None)
        layout = contents.layout() if contents is not None else None
        if layout is None:
            logger.warning("Review step list not found in the UI, no pause offered")
            return

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.review_checkboxes = {}

        steps = review_steps.availableSteps(
            mode=self.ui.comboBox3.currentText,
            mode2=self.ui.comboBox4.currentText,
            reg_type=self.ui.comboBox.currentText,
            visualization="Visualization" in self.ui.comboBox2.currentText,
            quantification="Quantitative" in self.ui.comboBox2.currentText,
        )

        remembered = self.loadReviewSelection()
        group = None
        for step in steps:
            if step["group"] != group:
                group = step["group"]
                layout.addWidget(qt.QLabel(f"<b>{group}</b>"))

            label = step["label"]
            if step["kind"] != review_steps.VIEW:
                label += "  (editable)"
            box = qt.QCheckBox(label)
            box.setToolTip(
                "You can correct this one." if step["kind"] != review_steps.VIEW
                else "Look only."
            )
            box.setChecked(step["id"] in remembered)
            box.connect("toggled(bool)", lambda _checked: self.saveReviewSelection())
            layout.addWidget(box)
            self.review_checkboxes[step["id"]] = box

        layout.addStretch(1)
        # These boxes did not exist when the theme was applied.
        self._applyCheckboxStyleSheets(self._isDarkMode())

    # What a clinician checks on a normal run: the landmarks they can actually
    # correct, each of the three registrations, and the surfaces the measurements
    # rest on. Everything else is worth a look while debugging, not routinely.
    RECOMMENDED_REVIEW_IDS = (
        "t1_landmarks",
        "registration_cb",
        "registration_max",
        "registration_mand",
        "bone_surfaces",
    )

    def setRecommendedReviewSteps(self) -> None:
        """Tick the steps worth pausing on for a routine run.

        Only among the steps this mode actually offers: ticking an id the run
        will never reach reads as a broken feature the first time it goes
        straight past it.
        """
        wanted = set(self.RECOMMENDED_REVIEW_IDS)
        for review_id, box in self.review_checkboxes.items():
            box.setChecked(review_id in wanted)
        offered = wanted & set(self.review_checkboxes)
        logger.info(f"{len(offered)} recommended step(s) ticked: {sorted(offered)}")
        if not offered:
            logger.warning("None of the recommended steps exist in this mode")

    def setAllReviewSteps(self, checked: bool) -> None:
        """Tick or untick every step currently offered."""
        for box in self.review_checkboxes.values():
            box.setChecked(checked)

    def selectedReviewIds(self) -> set:
        """Ids the user ticked, empty when pausing is switched off."""
        if not self.ui.checkBox_2.isChecked():
            return set()
        return {i for i, box in self.review_checkboxes.items() if box.isChecked()}

    def saveReviewSelection(self) -> None:
        """Keep the ticked steps between sessions."""
        chosen = [i for i, box in self.review_checkboxes.items() if box.isChecked()]
        qt.QSettings().setValue(self.REVIEW_SETTINGS_KEY, ",".join(sorted(chosen)))

    def loadReviewSelection(self) -> set:
        """Steps ticked the last time, as a set of ids."""
        stored = qt.QSettings().value(self.REVIEW_SETTINGS_KEY, "")
        if not stored:
            return set()
        if isinstance(stored, (list, tuple)):
            return set(stored)
        return {i for i in str(stored).split(",") if i}

    def applyReviewSelection(self) -> int:
        """Drop the pause from the steps the user did not ask to stop at.

        The pipeline marks every step it could pause on; this is what turns
        that into the handful the user actually wants.
        """
        chosen = self.selectedReviewIds()
        kept = 0
        for step in self.list_process:
            if not step.get("pause_for_visualization"):
                continue
            if step.get("ReviewId") in chosen:
                kept += 1
            else:
                step["pause_for_visualization"] = False
        logger.info(f"{kept} step(s) will pause for review")
        return kept

    def shouldPauseAfterProcess(self, process_info: dict) -> bool:
        """
        Determine if process execution should pause for visualization review.

        Args:
            process_info: Process information dictionary

        Returns:
            bool: True if pause is requested, False otherwise
        """
        return process_info.get("pause_for_visualization", False)

    def runPatientIds(self) -> set:
        """The patients this run is about, read from the folder it was given.

        The scans on the input side are what defines a run. An output folder may
        hold months of earlier work, and a step that skips a patient it has
        already done leaves that patient's file untouched and old - so the file
        date says nothing useful about who is being processed. Only the input
        list does.

        Returns:
            set: Patient ids, empty if the folder cannot be read
        """
        node = getattr(self, "_parameterNode", None)
        folder = getattr(node, "InputFolder", None) if node is not None else None
        if not folder or not os.path.isdir(folder):
            return set()

        wanted = self.VOLUME_EXT + self.MODEL_EXT + (".json",)
        ids = set()
        for root, _, files in os.walk(folder):
            for name in files:
                if name.endswith(wanted):
                    ids.add(patientIdFromFileName(name))
        return ids

    @staticmethod
    def belongsToRun(patient: str, wanted_ids: set) -> bool:
        """Whether a produced file's id names one of the run's patients.

        Usually the ids match outright. Some steps append a marker that
        patientIdFromFileName does not know how to strip - the heatmaps come out
        as "C_0001_Mandible_ModelDistance" - which leaves a longer id built on
        the patient's own. Those still belong to the run, so accept an id that
        extends an expected one at a separator. "C_0001" must not swallow
        "C_00011", hence the boundary rather than a bare startswith.
        """
        if patient in wanted_ids:
            return True
        return any(patient.startswith(w + "_") for w in wanted_ids)

    def buildPauseQueue(self, process_info: dict) -> list:
        """
        Build one review item per patient for the step that just finished.

        Args:
            process_info: Process information dictionary of the finished step

        Returns:
            list: Review items, each holding a patient's files and its scan
        """
        review_folder = process_info.get("ReviewFolder")
        if not review_folder or not os.path.isdir(review_folder):
            logger.warning(f"Nothing to review: {review_folder} not found")
            return []

        editable = process_info.get("ReviewEditable", False)
        adjustable = process_info.get("ReviewAdjustable", False)
        if editable:
            wanted = (".json",)
        elif adjustable:
            wanted = self.VOLUME_EXT
        else:
            wanted = self.VOLUME_EXT + self.MODEL_EXT

        # Output folders get reused between runs. Without this the review walks
        # patients this run never processed - "patient 1 of 5" on a batch of
        # three - and asks the user to check results that are not theirs.
        # Only what the step just produced is filtered: the scans and matrices
        # it is judged against are often copied in from earlier, and dropping
        # those would leave the landmarks with nothing underneath.
        #
        # Identity, not file date: a step that skips a patient whose output is
        # already there leaves that file weeks old, and dating it drops a
        # patient the run really is processing.
        # After a rollback only the marked patients were replayed, so they are
        # the only ones this step is about - the others still hold the results
        # they were accepted with.
        expected = self.review_flagged_carry or self.runPatientIds()
        self.review_flagged_carry = []
        wanted_ids = set(expected) or None

        # Some steps write several files per scan - BDS writes one surface per
        # anatomical structure plus a merged one holding them all - and loading
        # every one buries the view. A step can name what is worth looking at.
        keep = process_info.get("ReviewNameContains")

        by_patient = {}
        held_back = {}
        skipped = set()
        for root, _, files in os.walk(review_folder):
            for name in sorted(files):
                if not name.endswith(wanted):
                    continue
                if keep and keep not in name:
                    continue
                patient = patientIdFromFileName(name)
                path = os.path.join(root, name)
                if wanted_ids is not None and not self.belongsToRun(patient, wanted_ids):
                    skipped.add(patient)
                    held_back.setdefault(patient, []).append(path)
                    continue
                by_patient.setdefault(patient, []).append(path)

        # A filter that empties the review is worse than one that lets an extra
        # patient through: the user would be told there is nothing to check and
        # the run would carry on past a step they asked to inspect. If nothing
        # survived but files were found, the ids simply do not line up with the
        # input folder - show them all rather than nothing.
        if not by_patient and held_back:
            logger.warning(
                "None of the files in this step matched the run's patients "
                f"{sorted(wanted_ids)}; showing all {len(held_back)} found "
                "instead of skipping the review"
            )
            by_patient = held_back
        elif skipped:
            logger.info(
                f"{len(skipped)} patient(s) left out of the review, not part of "
                f"this run: {sorted(skipped)}"
            )

        # Landmarks are only editable against the scan they were placed on, so
        # pair each patient with its oriented volume when the step names one.
        scans = {}
        volume_folder = process_info.get("ReviewVolumeFolder")
        if volume_folder and os.path.isdir(volume_folder):
            for root, _, files in os.walk(volume_folder):
                for name in sorted(files):
                    if name.endswith(self.VOLUME_EXT):
                        scans.setdefault(
                            patientIdFromFileName(name), os.path.join(root, name)
                        )

        # An adjusted registration is only worth anything if its matrix can be
        # updated too: the landmarks downstream follow the matrix, not the voxels.
        matrices = {}
        if adjustable:
            for root, _, files in os.walk(review_folder):
                for name in sorted(files):
                    if name.endswith(".tfm"):
                        matrices.setdefault(
                            patientIdFromFileName(name), os.path.join(root, name)
                        )

        queue = []
        for patient in sorted(by_patient):
            if editable and patient not in scans:
                logger.warning(
                    f"No scan found for {patient}, its landmarks cannot be reviewed"
                )
            if adjustable and patient not in matrices:
                logger.warning(
                    f"No registration matrix found for {patient}, its registration "
                    f"can be looked at but not adjusted"
                )
            queue.append(
                {
                    "patient": patient,
                    "files": by_patient[patient],
                    "volume": scans.get(patient),
                    "editable": editable,
                    "adjustable": adjustable and patient in matrices,
                    "matrix": matrices.get(patient),
                }
            )

        logger.info(f"{len(queue)} patient(s) to review after {self.module_name}")
        return queue

    def enterPauseForReview(self, process_info: dict) -> bool:
        """
        Decide whether to pause; the loading itself is left to the event loop.

        Args:
            process_info: Process information dictionary of the finished step

        Returns:
            bool: True if the run is now paused and must not advance
        """
        if not self.shouldPauseAfterProcess(process_info):
            return False
        if not self.ui.checkBox_2.isChecked():
            return False

        # Reading a volume makes Slicer write to the stdout pipe it only drains
        # from the Qt event loop. This runs inside a VTK observer callback,
        # where that loop cannot turn, so doing it here fills the pipe with no
        # reader and hangs the main thread in write() - the same window that
        # never comes back as the CLI output above. Hand the work to the event
        # loop and return at once, exactly as that fix does.
        self.pause_pending = True
        qt.QTimer.singleShot(0, self.beginReview)
        return True

    def beginReview(self) -> None:
        """Load the finished step's results, once the event loop is turning."""
        if not self.pause_pending:
            # cancelled between the callback returning and this firing
            return
        self.pause_pending = False

        process_info = self.current_process_info or {}
        self.pause_queue = self.buildPauseQueue(process_info)
        self.pause_index = 0

        if not self.pause_queue or not self.showPauseItem():
            logger.warning(
                f"Nothing could be loaded to review after {self.module_name}, continuing"
            )
            self.onContinueButton()
            return

        self.paused_for_visualization = True
        self.ui.continueButton.setVisible(True)
        self.ui.progressBar.setValue(0)
        logger.info(f"Process on pause after {self.module_name}.")

    def showPauseItem(self) -> bool:
        """
        Load the patient at the current queue position, skipping unloadable ones.

        Returns:
            bool: True if a patient is now on screen
        """
        while self.pause_index < len(self.pause_queue):
            item = self.pause_queue[self.pause_index]
            if self.loadPauseItem(item):
                self.showReviewMessage(item)
                self.updateReviewButtons()
                return True
            logger.warning(f"Nothing to load for {item['patient']}, skipping it")
            self.pause_index += 1
        return False

    def showReviewMessage(self, item: dict) -> None:
        """
        Say where the run is and what the user may change, in the module panel.

        The console log is not where a clinician looks, and the step counter must
        stay readable: the pause used to overwrite it with its own text, leaving
        no sign of how far along the run was.

        Args:
            item: Review item currently on screen
        """
        info = self.current_process_info or {}
        title = info.get("ReviewTitle") or self.module_name
        hint = info.get("ReviewHint", "")

        self.ui.label_3.setText(
            f"Paused at step {self.ActualProcess}/{self.NumberProcess} "
            f"- {self.module_name}"
        )

        patient = item["patient"]
        position = ""
        if len(self.pause_queue) > 1:
            position = f" - patient {self.pause_index + 1} of {len(self.pause_queue)}"

        # A registration is adjustable, not editable: testing only the one flag
        # labelled a step "Review only" while its own hint told the user to drag
        # it, and the checkbox that armed it said it could be changed.
        if item.get("editable"):
            action = "You can move the points"
        elif item.get("adjustable"):
            action = "You can drag it into place"
        else:
            action = "Review only"
        # Name the modules: a clinician who does not already know Slicer has no
        # way to guess that the points are editable in Markups, or that the 3D
        # view they are looking at is driven by Volume Rendering.
        if item.get("editable"):
            tools = ("Open <b>Markups</b> to pick a point from the list, and "
                     "<b>Volume Rendering</b> to change how the bone is shown.")
        elif item.get("adjustable"):
            tools = ("Drag the scan in a slice view. <b>Transforms</b> shows the "
                     "displacement you are applying, in numbers; "
                     "<b>Volume Rendering</b> changes how the bone is shown.")
        else:
            tools = "<b>Models</b> and <b>Volume Rendering</b> change how this is shown."

        self.ui.reviewLabel.setText(
            f"<b>{title}</b><br/>"
            f"{patient}{position} &nbsp;·&nbsp; <i>{action}</i><br/>{hint}"
            f"<br/><span style='color:#7f8c8d'>{tools}</span>"
        )
        self.ui.reviewLabel.setVisible(True)

        # Moving between patients is what the navigation buttons are for, so
        # this one says the single thing it does: it ends the pause.
        self.ui.continueButton.setText(self.CONTINUE_BUTTON_TEXT)

        logger.info(f"Review - {title} - {patient}{position}")

    def showVolumeRendering(self, volume) -> None:
        """Turn on volume rendering for the scan a landmark review sits on.

        Points are placed on anatomy, and anatomy reads far better in 3D than on
        three grey slices. Slicer can do this in one call but nobody thinks to
        ask for it mid-run, so the pause sets it up and the clinician only has to
        look.

        Best effort throughout: a machine without the Volume Rendering module, or
        one that refuses the GPU, still gets a perfectly usable review on the
        slices. Failing the pause over a convenience would be the wrong trade.

        Args:
            volume: The loaded scalar volume node
        """
        if volume is None or not hasattr(slicer.modules, "volumerendering"):
            return
        try:
            vr = slicer.modules.volumerendering.logic()
            display = vr.CreateDefaultVolumeRenderingNodes(volume)
            if display is None:
                return
            # CT-Bone reads the skeleton, which is what every landmark here sits
            # on. If this build does not ship it, the default transfer function
            # still shows something rather than nothing.
            preset = vr.GetPresetByName("CT-Bone") or vr.GetPresetByName("CT-AAA")
            if preset is not None and display.GetVolumePropertyNode():
                display.GetVolumePropertyNode().Copy(preset)
            display.SetVisibility(True)
            self.pause_nodes.append(display)
            logger.info(f"Volume rendering on for {volume.GetName()}")
        except Exception as e:
            logger.warning(f"Could not turn on volume rendering: {e}")

    def loadPauseItem(self, item: dict) -> bool:
        """
        Load one patient's result into the scene for review.

        Args:
            item: Review item built by buildPauseQueue

        Returns:
            bool: True if anything could be loaded
        """
        self.clearPauseNodes()
        loaded = False
        reference = None

        if item.get("volume"):
            reference = self.loadPauseFile(item["volume"])
            if reference:
                slicer.util.setSliceViewerLayers(background=reference)
                loaded = True

        moving = None
        for path in item["files"]:
            node = self.loadPauseFile(path)
            if node:
                moving = node
                if not item.get("volume") and path.endswith(self.VOLUME_EXT):
                    slicer.util.setSliceViewerLayers(background=node)
                loaded = True

        if loaded and item.get("adjustable") and moving is not None:
            self.setUpAdjustment(item, reference, moving)

        # On the reference only, and on every kind of pause. Rendering both scans
        # of a registration would put two opaque blocks inside each other and
        # show nothing - but rendering one costs nothing, because the rendering
        # lives in the 3D view while the overlap is judged on the slices. An
        # earlier version skipped registrations entirely, on the mistaken idea
        # that a rendering would cover that overlap; it cannot.
        if loaded and reference is not None:
            self.showVolumeRendering(reference)

        if loaded:
            self.applyPauseLayout(item)
        return loaded

    def setUpAdjustment(self, item: dict, reference, moving) -> None:
        """
        Show the registered scan over the original and let the user move it.

        The user drags the scan into place; the matrix that displacement amounts
        to is what the landmarks need, and it is read back on Continue. Nothing
        is asked of the user beyond moving the image.

        Args:
            item: Review item currently on screen
            reference: The T1 volume node the registration is judged against
            moving: The registered volume node the user may move
        """
        transform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", f"{item['patient']} manual adjustment"
        )
        moving.SetAndObserveTransformNodeID(transform.GetID())
        self.pause_nodes.append(transform)
        self.pause_transform = transform
        self.pause_transform_item = item

        # Half-opaque over the original is what makes a misalignment visible.
        if reference is not None:
            slicer.util.setSliceViewerLayers(
                background=reference, foreground=moving, foregroundOpacity=0.5
            )
        try:
            display = transform.GetDisplayNode()
            if display is None:
                transform.CreateDefaultDisplayNodes()
                display = transform.GetDisplayNode()
            if display is not None:
                display.SetEditorVisibility(True)
        except Exception as e:
            logger.warning(f"Could not show the interaction handles: {e}")

    def loadPauseFile(self, path: str):
        """
        Load a single review file, dispatching on its type.

        Args:
            path: File to load

        Returns:
            The loaded node, or None if it could not be loaded
        """
        try:
            if path.endswith(".json"):
                node = self.loadEditableMarkups(path)
            elif path.endswith(self.MODEL_EXT):
                node = slicer.util.loadModel(path)
            else:
                node = slicer.util.loadVolume(path)
        except Exception as e:
            logger.error(f"Could not load {path}: {e}")
            return None

        if node is None:
            logger.warning(f"Nothing loaded from {path}")
            return None

        self.pause_nodes.append(node)
        return node

    def loadEditableMarkups(self, path: str):
        """
        Load a landmark file so its points can actually be moved.

        ALI writes every control point with "locked": true and the display
        hidden, so loading one as-is shows an empty view holding points that
        cannot be dragged. Both are forced here, on the node only - the file
        keeps its own flags unless the user edits a position.

        Args:
            path: Landmark .mrk.json file

        Returns:
            The loaded markups node, or None
        """
        node = slicer.util.loadMarkups(path)
        if node is None:
            return None

        node.SetLocked(False)
        for i in range(node.GetNumberOfControlPoints()):
            node.SetNthControlPointLocked(i, False)

        display = node.GetDisplayNode()
        if display:
            display.SetVisibility(True)
            display.SetPointLabelsVisibility(True)

        self.pause_markups_nodes.append(node)
        self.pause_markups_start[node.GetID()] = self.markupsPositions(node)
        return node

    @staticmethod
    def markupsPositions(node) -> list:
        """Control point positions of a markups node, in order."""
        positions = []
        for i in range(node.GetNumberOfControlPoints()):
            position = [0.0, 0.0, 0.0]
            node.GetNthControlPointPosition(i, position)
            positions.append(tuple(position))
        return positions

    def savePauseEdits(self) -> None:
        """
        Write back the landmark files whose points the user moved.

        Files left untouched are not rewritten, so a run where the user only
        looked leaves ALI's output exactly as it was.
        """
        for node in self.pause_markups_nodes:
            storage = node.GetStorageNode()
            path = storage.GetFileName() if storage else None
            if not path:
                logger.warning(f"No file to save {node.GetName()} back to")
                continue

            before = self.pause_markups_start.get(node.GetID())
            after = self.markupsPositions(node)
            if before == after:
                logger.info(f"{os.path.basename(path)} unchanged, not rewritten")
                continue

            moved = sum(1 for a, b in zip(before, after) if a != b)
            try:
                if slicer.util.saveNode(node, path):
                    logger.info(f"{moved} landmark(s) adjusted, saved to {path}")
                else:
                    logger.error(f"Could not save adjusted landmarks to {path}")
            except Exception as e:
                logger.error(f"Could not save adjusted landmarks to {path}: {e}")

        self.saveAdjustedRegistration()

    def saveAdjustedRegistration(self) -> None:
        """
        Fold the user's displacement into the registration it corrects.

        AutoMatrix applies one matrix per patient and per structure downstream,
        and it cannot chain two. Composing here keeps that contract: the file it
        already reads now carries the registration and the correction together,
        so nothing after this step has to know an adjustment happened.
        """
        transform = self.pause_transform
        item = self.pause_transform_item
        if transform is None or not item or not item.get("matrix"):
            return

        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(matrix)
        if self.isIdentityMatrix(matrix):
            logger.info(f"{item['patient']}: registration left as AREG produced it")
            return

        import numpy as np
        import SimpleITK as sitk

        path = item["matrix"]
        try:
            areg = sitk.ReadTransform(path)
        except Exception as e:
            logger.error(f"Could not read {os.path.basename(path)}: {e}")
            return

        ras = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        # Slicer holds the displacement in RAS; a .tfm is LPS, and the two differ
        # by a flip of the first two axes.
        flip = np.diag([-1.0, -1.0, 1.0, 1.0])
        lps = flip @ ras @ flip

        nudge = sitk.AffineTransform(3)
        nudge.SetMatrix(lps[:3, :3].flatten().tolist())
        nudge.SetTranslation(lps[:3, 3].tolist())

        # Verified against applying the two in sequence: AutoMatrix inverts the
        # matrix before moving a point, and this is the order that survives it.
        # CompositeTransform([A, B]) applies A(B(p)) - the last added acts first.
        composed = sitk.CompositeTransform([areg, nudge.GetInverse()])

        # Written as one matrix whenever both are affine, which is the normal
        # case. A composite .tfm is valid and AutoMatrix does read it, but it
        # leaves two matrices in a file every other tool expects to hold one, and
        # a second manual correction would stack a third. The product is exactly
        # equivalent - checked point by point - so there is nothing to lose.
        composed = self.flattenIfAffine(composed, areg, nudge.GetInverse())

        try:
            sitk.WriteTransform(composed, path)
            logger.info(f"{item['patient']}: adjustment folded into {os.path.basename(path)}")
        except Exception as e:
            logger.error(f"Could not write the adjusted matrix to {path}: {e}")
            return

        self.saveAdjustedVolume(item, transform)

    @staticmethod
    def flattenIfAffine(composite, first, second):
        """One matrix instead of two, when both parts are affine.

        CompositeTransform([A, B]) moves a point as A(B(p)), which is the product
        of their matrices. Collapsing keeps the file to a single transform, so
        nothing downstream has to know an adjustment happened - which is what the
        pipeline assumed all along.

        Returns the composite untouched if either part is not affine: correctness
        first, tidiness second.

        Args:
            composite: The composed transform, returned as-is on any doubt
            first: Transform applied second to a point
            second: Transform applied first to a point

        Returns:
            A single AffineTransform, or the composite unchanged
        """
        import numpy as np
        import SimpleITK as sitk

        def as_matrix(t):
            affine = sitk.AffineTransform(t)      # raises unless truly affine
            m = np.eye(4)
            m[:3, :3] = np.array(affine.GetMatrix()).reshape(3, 3)
            m[:3, 3] = affine.GetTranslation()
            return m

        try:
            product = as_matrix(first) @ as_matrix(second)
        except Exception as e:
            logger.info(f"Keeping a composite transform, not both parts are affine: {e}")
            return composite

        flat = sitk.AffineTransform(3)
        flat.SetMatrix(product[:3, :3].flatten().tolist())
        flat.SetTranslation(product[:3, 3].tolist())
        return flat

    def saveAdjustedVolume(self, item: dict, transform) -> None:
        """
        Write the moved scan back, so the surfaces and heatmaps match the matrix.

        Args:
            item: Review item currently on screen
            transform: The transform node holding the user's displacement
        """
        for node in self.pause_nodes:
            if not node.IsA("vtkMRMLScalarVolumeNode"):
                continue
            if node.GetTransformNodeID() != transform.GetID():
                continue
            storage = node.GetStorageNode()
            path = storage.GetFileName() if storage else None
            if not path:
                continue
            try:
                node.HardenTransform()
                if slicer.util.saveNode(node, path):
                    logger.info(f"{item['patient']}: adjusted scan saved to {os.path.basename(path)}")
                else:
                    logger.error(f"Could not save the adjusted scan to {path}")
            except Exception as e:
                logger.error(f"Could not save the adjusted scan to {path}: {e}")
            return

    @staticmethod
    def isIdentityMatrix(matrix, tolerance: float = 1e-9) -> bool:
        """
        Tell whether a 4x4 holds no displacement at all.

        Args:
            matrix: vtkMatrix4x4 to test
            tolerance: Largest deviation still counted as identity

        Returns:
            bool: True if the matrix is the identity within tolerance
        """
        for row in range(4):
            for col in range(4):
                expected = 1.0 if row == col else 0.0
                if abs(matrix.GetElement(row, col) - expected) > tolerance:
                    return False
        return True

    def clearPauseNodes(self) -> None:
        """Remove the nodes the previous review item put in the scene."""
        for node in self.pause_nodes:
            try:
                slicer.mrmlScene.RemoveNode(node)
            except Exception as e:
                logger.warning(f"Could not remove a reviewed node from the scene: {e}")
        self.pause_nodes = []
        self.pause_markups_nodes = []
        self.pause_markups_start = {}
        self.pause_transform = None
        self.pause_transform_item = None

    # ALI says which landmarks its agents could not place, then moves on. Those
    # lines used to scroll past in the console: the first sign of trouble was an
    # empty column in the measurements, or a KeyError three steps later.
    MISSING_LANDMARK_RE = re.compile(r"Landmark '([^']+)': (\d+)/(\d+) failures")

    def collectMissingLandmarks(self, output_text: str) -> None:
        """Note the landmarks a finished ALI run reported it could not find.

        Args:
            output_text: The CLI's standard output, untrimmed
        """
        for name, failed, total in self.MISSING_LANDMARK_RE.findall(output_text or ""):
            entry = self.missing_landmarks.setdefault(name, [0, 0])
            entry[0] += int(failed)
            entry[1] += int(total)

    def missingLandmarkReport(self) -> str:
        """One line per landmark the run never placed, or an empty string."""
        if not self.missing_landmarks:
            return ""
        lines = [
            f"- {name}: not found on {failed} of {total} scan(s)"
            for name, (failed, total) in sorted(self.missing_landmarks.items())
        ]
        return "\n".join(lines)

    def showHeatmaps(self) -> int:
        """Put the distance maps on screen once the run is over.

        The heatmaps are the point of a visualization run, and they were being
        left as files nobody opened. A model carrying a "Distance" array shows
        nothing until that array is made active with a colour range, so this does
        the three things that turn a grey surface into a readable map.

        The range is made symmetric around zero on purpose: the distances are
        signed, and a range like [-2.7, 2.4] would paint zero slightly off the
        middle colour, so untouched anatomy would read as a small displacement.

        Only the merged map per patient is loaded. The per-structure files say
        the same thing over a smaller area, and one of these surfaces runs to
        120 MB - loading all of them would cost a gigabyte to show the same
        thing three times. They stay in the folder for anyone who wants them.

        Returns:
            int: how many maps were put on screen
        """
        folder = os.path.join(self._parameterNode.OutputFolder or "", "Heatmaps")
        if not os.path.isdir(folder):
            return 0

        files = sorted(glob.glob(os.path.join(folder, "*.vtk")))
        if not files:
            logger.info("No heatmap to show")
            return 0
        # All of them are loaded and coloured, so a tick in Models is enough to
        # see one. Only the merged map starts visible: the per-structure maps
        # cover the same anatomy, and showing them at once would stack surfaces
        # on top of each other until none is readable.
        merged = [f for f in files if "merged" in os.path.basename(f).lower()]

        # Rainbow reads as a map; if this build ships the cold-to-hot variant,
        # its ends are clearer for signed data.
        table = None
        for name in ("ColdToHotRainbow", "Rainbow"):
            table = slicer.mrmlScene.GetFirstNodeByName(name)
            if table is not None:
                break

        shown, loaded, spans = 0, 0, []
        for path in files:
            try:
                model = slicer.util.loadModel(path)
                if model is None:
                    continue
                data = model.GetPolyData()
                array = data.GetPointData().GetArray("Distance") if data else None
                display = model.GetDisplayNode()
                if array is not None and display is not None:
                    low, high = array.GetRange()
                    edge = max(abs(low), abs(high)) or 1.0
                    display.SetActiveScalarName("Distance")
                    display.SetScalarRangeFlag(display.UseManualScalarRange)
                    display.SetScalarRange(-edge, edge)
                    if table is not None:
                        display.SetAndObserveColorNodeID(table.GetID())
                    display.SetScalarVisibility(True)
                    on = (not merged) or (path in merged)
                    display.SetVisibility(on)
                    # A legend on every map, not only the visible one: ticking a
                    # hidden map in Models would otherwise show a coloured
                    # surface with no scale, and its range can be ten times
                    # smaller than the merged map's.
                    self.addColorLegend(display)
                    spans.append((os.path.basename(path).replace("_ModelDistance.vtk", ""), edge))
                    if on:
                        shown += 1
                loaded += 1
            except Exception as e:
                logger.warning(f"Could not show {os.path.basename(path)}: {e}")

        if shown:
            try:
                # The 3D view, because a surface map is unreadable on slices -
                # but the module panel stays on VFACE. Switching to Models here
                # would move the clinician away at the exact moment the run ends,
                # hiding the end-of-run message and the buttons. The message
                # points at Models for anyone who wants to go further.
                slicer.app.layoutManager().setLayout(
                    slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
                slicer.util.resetThreeDViews()
            except Exception as e:
                logger.warning(f"Heatmaps loaded but the view could not be set: {e}")
            # In the panel, not only in the end-of-run dialog: that dialog is
            # dismissed and leaves nothing behind, so a clinician looking at a
            # coloured skull a minute later has no clue what to do with it. The
            # explanatory box is where every other instruction has appeared.
            # The same numbers in the panel, out of the 3D view: with several maps
            # shown the bars compete for the same corner, and a written range is
            # readable whatever is on screen.
            ranges = "<br/>".join(
                f"&nbsp;&nbsp;{name} &nbsp;<b>&plusmn;{edge:.1f} mm</b>"
                for name, edge in sorted(spans)
            )
            self.ui.reviewLabel.setText(
                f"<b>Distance maps</b><br/>"
                f"{shown} map(s) in the 3D view, coloured by signed distance "
                "around zero; the scale beside them is in millimetres."
                f"<br/>{ranges}"
                f"<br/><span style='color:#7f8c8d'>{loaded} map(s) loaded in all: "
                "tick one in <b>Models</b> to show it, where you can also change "
                "the colours or the range.</span>"
            )
            self.layoutLegends()
            self.ui.reviewLabel.setVisible(True)
            logger.info(f"{shown} heatmap(s) on screen, coloured by signed distance")
        return shown

    def layoutLegends(self) -> None:
        """Spread the visible scales across the view instead of stacking them.

        Every colour legend is created at the same spot, (0.95, 0.5), so showing
        three maps at once puts three bars exactly on top of each other. Each
        visible one gets its own column, right to left, in the order the maps
        were loaded.
        """
        visible = [legend for display, legend in self.legend_pairs
                   if display.GetVisibility()]
        for i, legend in enumerate(visible):
            try:
                legend.SetSize(0.13, 0.45)
                # 0.16 apart for a bar 0.13 wide: they sit side by side without
                # touching, and three still fit inside the view.
                legend.SetPosition(max(0.05, 0.95 - 0.16 * i), 0.5)
            except Exception as e:
                logger.warning(f"Could not place a colour scale: {e}")

    def onHeatmapVisibilityChanged(self, caller, event) -> None:
        """Keep each scale with the surface it describes.

        Args:
            caller: The model display node that changed
            event: Unused, required by the observer signature
        """
        for display, legend in self.legend_pairs:
            if display is caller:
                legend.SetVisibility(bool(display.GetVisibility()))
                self.layoutLegends()
                return

    def addColorLegend(self, display) -> None:
        """Put a scale next to the map, so the colours mean a distance.

        Without it a gradient says nothing: the merged map of a case runs to
        +/-32 mm while its mandible alone stays under +/-3 mm, and the two look
        identical. Best effort - an older Slicer without colour legends still
        shows the map, just without its scale.

        Args:
            display: The model display node whose scalars are already visible
        """
        try:
            logic = slicer.modules.colors.logic()
            legend = logic.AddDefaultColorLegendDisplayNode(display)
            if legend is None:
                return
            legend.SetTitleText("Distance (mm)")
            legend.SetVisibility(bool(display.GetVisibility()))
            # A legend does not follow its surface on its own - measured: hiding
            # the model leaves its scale on screen. Without this, loading three
            # maps would stack three bars over each other for good.
            self.legend_pairs.append((display, legend))
            self.addObserver(display, vtk.vtkCommand.ModifiedEvent,
                             self.onHeatmapVisibilityChanged)
        except Exception as e:
            logger.warning(f"No colour legend on this build: {e}")

    def showDoneMessage(self) -> None:
        """Say the run is over without taking the application hostage.

        Nothing waits on this answer, so it is shown rather than executed: a
        modal dialog that fails to appear leaves the user with no way to click
        anything, and that is exactly what a finished run must not do.
        """
        # A measurement is only as good as the points it rests on: a run that
        # finished with landmarks missing produced empty columns, and saying so
        # here is the difference between a known gap and a silent one.
        shown = self.showHeatmaps()
        report = self.missingLandmarkReport()
        if report:
            logger.warning(f"Landmarks never placed during this run:\n{report}")
            text = (
                "Processing completed, but some landmarks were never found:\n\n"
                f"{report}\n\n"
                "Measurements that rest on them are empty. A landmark is usually "
                "missed because it falls outside the scan's field of view."
            )
        else:
            text = "Processing completed successfully!"
        if shown:
            text += (
                f"\n\n{shown} distance map(s) are on screen in the 3D view, "
                "coloured by signed distance around zero.\n"
                "Open the Models module to look closer - colour scale, range, "
                "or hiding a surface."
            )

        self.done_popup = PopUpWindow(title="Process Complete", text=text)
        self.done_popup.setModal(False)
        self.done_popup.show()
        self.done_popup.raise_()

    def previousCorrectableStep(self):
        """The nearest step behind this one the user can actually change.

        Looking at a bad orientation is useless without a way back to the
        landmarks that caused it. Steps that can only ever be looked at are
        skipped over, so the button lands where something can be done.

        Returns:
            tuple: (step, steps to replay after it), or (None, []) when there is
                nothing correctable behind the current one
        """
        current = self.current_process_info
        history = self.executed_steps
        # By identity, not equality: two runs of the same step carry equal
        # dictionaries, and the one meant here is the one on screen.
        here = None
        for i in range(len(history) - 1, -1, -1):
            if history[i] is current:
                here = i
                break
        if here is None:
            return None, []

        for i in range(here - 1, -1, -1):
            kind = review_steps.describe(history[i].get("ReviewId", "")).get("kind")
            if kind in (review_steps.LANDMARKS, review_steps.REGISTRATION):
                return history[i], history[i + 1:here + 1]
        return None, []

    def flaggedPatients(self) -> list:
        """The marked patients, in the order the queue shows them."""
        return [i["patient"] for i in self.pause_queue
                if i["patient"] in self.review_flagged]

    def updateReviewButtons(self) -> None:
        """Show the actions this patient, and this step, actually allow.

        Each button does one thing and says so: moving between patients never
        advances the run, and going back never hides behind a forward label.
        """
        total = len(self.pause_queue)
        index = self.pause_index
        patient = self.pause_queue[index]["patient"] if total else ""

        self.ui.reviewPatientLabel.setVisible(bool(patient))
        self.ui.reviewPatientLabel.setText(
            f"<b>{patient}</b>"
            + (f" &nbsp;({index + 1} / {total})" if total > 1 else "")
        )

        self.ui.reviewPrevPatientButton.setVisible(total > 1)
        self.ui.reviewPrevPatientButton.setEnabled(index > 0)
        self.ui.reviewNextPatientButton.setVisible(total > 1)
        self.ui.reviewNextPatientButton.setEnabled(index < total - 1)

        # Marking is only worth offering when there is somewhere to go back to.
        target, _ = self.previousCorrectableStep()
        self.ui.reviewFlagButton.setVisible(target is not None)
        if patient in self.review_flagged:
            self.ui.reviewFlagButton.setText("Cancel - this patient is fine")
        else:
            self.ui.reviewFlagButton.setText("Go back and edit this patient")

        flagged = self.flaggedPatients()
        self.ui.reviewGoBackButton.setVisible(bool(flagged) and target is not None)
        if flagged and target is not None:
            name = review_steps.describe(target.get("ReviewId", "")).get(
                "label", "the previous step")
            self.ui.reviewGoBackButton.setText(
                f"Go back to {name} for {len(flagged)} patient(s)"
            )

    def onReviewPreviousPatient(self) -> None:
        """Show the patient before this one, keeping any edit made here."""
        if self.pause_index > 0:
            self.savePauseEdits()
            self.pause_index -= 1
            self.showPauseItem()

    def onReviewNextPatient(self) -> None:
        """Show the next patient of this step, keeping any edit made here."""
        if self.pause_index < len(self.pause_queue) - 1:
            self.savePauseEdits()
            self.pause_index += 1
            self.showPauseItem()

    def onReviewToggleFlag(self) -> None:
        """Mark this patient for rework, or take the mark back."""
        if not self.pause_queue:
            return
        patient = self.pause_queue[self.pause_index]["patient"]
        if patient in self.review_flagged:
            self.review_flagged.discard(patient)
            logger.info(f"{patient}: mark removed")
        else:
            self.review_flagged.add(patient)
            logger.info(f"{patient}: marked for rework")
        self.updateReviewButtons()

    def onReviewGoBack(self) -> None:
        """Return to the last correctable step, for the patients marked there.

        Correcting the landmarks changes nothing on its own - the orientation
        was computed from the old ones. So the steps in between are queued to
        run again, narrowed to the marked patients: the rest of the batch keeps
        the results it already has, and a run of fifty does not start over
        because one case was wrong.
        """
        target, replay = self.previousCorrectableStep()
        if target is None:
            logger.warning("Nothing correctable behind this step")
            return

        flagged = self.flaggedPatients()
        if not flagged:
            logger.warning("No patient marked for rework")
            return

        name = review_steps.describe(target.get("ReviewId", "")).get(
            "label", target.get("Module"))
        logger.info(
            f"Going back to '{name}' for {flagged}; "
            f"{len(replay)} step(s) will run again for them"
        )

        narrowed = []
        for step in replay:
            restricted, folders = review_steps.restrictStepToPatients(step, flagged)
            self.review_temp_folders.extend(folders)
            narrowed.append(restricted)

        self.resetPauseState()

        # The steps between the two run again, ahead of whatever was left, and
        # the user lands back on the step they can actually fix.
        self.list_process[0:0] = narrowed
        self.NumberProcess += len(narrowed)
        self.review_flagged_carry = list(flagged)
        self.current_process_info = target
        self.module_name = target.get("Module", self.module_name)
        # beginReview guards against a pause cancelled between the callback and
        # the event loop; this one comes from a button, so it is armed here.
        self.pause_pending = True
        self.beginReview()

    def clearReviewTempFolders(self) -> None:
        """Drop the temporary folders a narrowed replay left behind."""
        for folder in self.review_temp_folders:
            try:
                shutil.rmtree(folder, ignore_errors=True)
            except OSError as e:
                logger.warning(f"Could not remove {folder}: {e}")
        self.review_temp_folders = []

    def resetPauseState(self) -> None:
        """Drop everything held for the review in progress."""
        self.pause_pending = False
        self.clearPauseNodes()
        self.ui.reviewLabel.setVisible(False)
        self.ui.reviewLabel.setText("")
        self.ui.continueButton.setText(self.CONTINUE_BUTTON_TEXT)
        for name in ("reviewPatientLabel", "reviewPrevPatientButton",
                     "reviewNextPatientButton", "reviewFlagButton",
                     "reviewGoBackButton"):
            getattr(self.ui, name).setVisible(False)
        self.pause_queue = []
        self.pause_index = 0
        self.review_flagged = set()
        self.paused_for_visualization = False

    def applyPauseLayout(self, item: dict) -> None:
        """
        Set a layout suited to what is being reviewed.

        Args:
            item: Review item currently on screen
        """
        layoutManager = slicer.app.layoutManager()
        surfaces_only = not item.get("volume") and all(
            f.endswith(self.MODEL_EXT) for f in item["files"]
        )

        if surfaces_only:
            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
            widget = layoutManager.threeDWidget(0)
            if widget:
                widget.threeDView().resetFocalPoint()
        else:
            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            slicer.util.resetSliceViews()

    def executeProcess(self, process_info):
        import time
        self.CliStepTime = time.time()
        self.module_name = process_info["Module"]
        self.displayModule = process_info["Display"]
        self.current_process_info = process_info  # Stocker les infos du processus actuel
        # Kept in order: going back means finding what ran before this step.
        self.executed_steps.append(process_info)
        
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
            logger.info(f"{self.module_name} is executed.")
            self.cliNode = slicer.cli.run(process, None, parameters)
            self.addObserver(self.cliNode, vtk.vtkCommand.ModifiedEvent, self.onCliUpdated)
        else:
            logger.info(f"{self.module_name} is executed.")
            # A python step holds the event loop for its whole duration, so this
            # is the last thing drawn until it returns and the window stops
            # repainting meanwhile. Say so, or a segmentation that legitimately
            # takes minutes is indistinguishable from a freeze.
            self.ui.label_3.setText(
                f"Process : {self.module_name} "
                f"({self.ActualProcess}/{self.NumberProcess}) - running, "
                f"the window stays still until it finishes"
            )
            slicer.app.processEvents()

            # No CLI is running during a Python step: forget the previous node so
            # a late event from it cannot advance the chain from under our feet.
            self.cliNode = None

            # For long Python process, use a timer to maintain reactivity
            self.python_process = process
            self.python_parameters = parameters
            self.python_process_completed = False
            self.python_process_error = None
            
            #Start process with a timer
            self.startPythonProcess()

    def startPythonProcess(self):
        """Start a Python process"""
        import time

        started = time.time()
        log_path = None
        try:
            if callable(self.python_process):
                with self.outputToFile() as log_path:
                    result = self.python_process(**self.python_parameters)
                logger.info(
                    f"Result of {self.module_name}: {result} "
                    f"(in {self.readableDuration(time.time() - started)})"
                )
                self.python_process_completed = True
            else:
                logger.error(f"Error: {self.python_process} is not a callable function")
                self.python_process_error = "Process is not callable"
                self.python_process_completed = True

        except Exception as e:
            logger.error(f"Error during the execution of {self.module_name}: {e}")
            import traceback
            traceback.print_exc()
            self.python_process_error = str(e)
            self.python_process_completed = True

        finally:
            # A step that raised still wrote to the file, and its last lines are
            # exactly the ones worth seeing; reporting here also stops the
            # temporary file leaking on that path.
            if log_path:
                self._reportStepOutput(log_path)

        import qt
        qt.QTimer.singleShot(100, self.checkPythonProcessStatus)

    @contextlib.contextmanager
    def outputToFile(self):
        """
        Send everything a step prints to a file instead of Slicer's own pipe.

        Slicer captures its standard output into a pipe it drains from the Qt
        event loop. A python step runs with that loop stopped, so nothing drains
        the pipe while it prints: a talkative one - the segmentation prints per
        epoch - fills it and the main thread blocks in write() for ever, with a
        window that never comes back. Writing to a file cannot block, which is
        why the heatmap workers already do this.

        The redirection is at file-descriptor level on purpose: torch and the
        segmentation write from C, straight to fd 1, where swapping sys.stdout
        would not catch them.

        Yields:
            str: path of the file the step's output was written to
        """
        handle = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".log", prefix="vface_step_", delete=False
        )
        saved_out, saved_err = None, None
        saved_sys_out, saved_sys_err = sys.stdout, sys.stderr
        try:
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            saved_out = os.dup(1)
            saved_err = os.dup(2)
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)
            sys.stdout, sys.stderr = handle, handle
            yield handle.name
        finally:
            sys.stdout, sys.stderr = saved_sys_out, saved_sys_err
            try:
                handle.flush()
            except Exception:
                pass
            if saved_out is not None:
                os.dup2(saved_out, 1)
                os.close(saved_out)
            if saved_err is not None:
                os.dup2(saved_err, 2)
                os.close(saved_err)
            try:
                handle.close()
            except Exception:
                pass

    def _reportStepOutput(self, log_path: str, keep_lines: int = 12) -> None:
        """
        Put the tail of a step's own output back in the log, and drop the file.

        Args:
            log_path: File the step's output was redirected to
            keep_lines: How many trailing lines to bring back
        """
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                lines = [line.rstrip() for line in f if line.strip()]
        except OSError:
            return
        finally:
            try:
                os.remove(log_path)
            except OSError:
                pass

        if not lines:
            return
        shown = lines[-keep_lines:]
        skipped = len(lines) - len(shown)
        prefix = f"[{skipped} earlier line(s) not shown]\n" if skipped else ""
        logger.info(f"{self.module_name} said:\n{prefix}" + "\n".join(shown))

    @staticmethod
    def readableDuration(seconds: float) -> str:
        """
        Spell out a duration the way the CLI steps already report theirs.

        Args:
            seconds: Elapsed seconds

        Returns:
            str: Human readable duration
        """
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}min and {seconds % 60}s"
        return f"{seconds // 3600}h, {seconds % 3600 // 60}min and {seconds % 60}s"

    def checkPythonProcessStatus(self):
        """Check Python process status"""
        if self.python_process_completed:
            self.onProcessCompleted()
        else:
            import qt
            qt.QTimer.singleShot(100, self.checkPythonProcessStatus)

    def onProcessCompleted(self):
        logger.info("\n\n ========= PROCESSED ========= \n")

        if self.enterPauseForReview(self.current_process_info):
            return

        try:
            self.executeProcess(self.list_process[0])
            self.ActualProcess += 1
            del self.list_process[0]
        except IndexError:
            self.OnEndProcess()

    # onCliUpdated runs inside a VTK observer callback, where the Qt event loop is
    # not running. With --disable-terminal-outputs Slicer captures its own stdout
    # into a pipe that it drains from that same loop, so a single oversized write
    # fills the pipe and blocks the main thread for good - Slicer goes black and
    # never comes back. Slicer already logs each CLI's full standard output
    # itself, so echoing a bounded tail here is enough.
    MAX_CLI_OUTPUT_CHARS = 8000

    @classmethod
    def _briefCliOutput(cls, text) -> str:
        """The tail of a CLI's output, small enough to never fill the stdout pipe."""
        text = text or ""
        if len(text) <= cls.MAX_CLI_OUTPUT_CHARS:
            return text
        kept = text[-cls.MAX_CLI_OUTPUT_CHARS:]
        return (
            f"[... {len(text) - len(kept)} characters omitted, "
            f"full output in the Slicer log ...]\n{kept}"
        )

    def onCliUpdated(self, caller, event):
        import time
        import json
        import subprocess

        # Only the node the pipeline is currently waiting on may advance it.
        # Observers can outlive their step, so a stale callback would start the
        # next step while the current CLI was still writing its results - the
        # measurements then read a folder that AutoMatrix had not filled yet -
        # and would fire again after OnEndProcess had cleared the state, raising
        # AttributeError on current_process_info.
        if self.cliNode is None or self.current_process_info is None:
            return
        if caller.GetID() != self.cliNode.GetID():
            return

        cliNode = caller

        status = cliNode.GetStatus()

        if status & slicer.vtkMRMLCommandLineModuleNode.Completed or \
           status & slicer.vtkMRMLCommandLineModuleNode.Cancelled:

            self.removeObserver(cliNode, vtk.vtkCommand.ModifiedEvent, self.onCliUpdated)

            # Deferred on purpose: Slicer captures its own stdout into a pipe
            # that it drains from the Qt event loop, and this method runs inside
            # a VTK observer callback where that loop cannot run. Writing here
            # is what lets the pipe fill until the main thread blocks in write()
            # with no reader left - the black window that never comes back.
            full_output = caller.GetOutputText() or ""
            # From the untrimmed text: the summary sits at the very end of ALI's
            # output, but _briefCliOutput keeps only a tail and a longer run
            # could push it out.
            self.collectMissingLandmarks(full_output)
            cli_output = self._briefCliOutput(full_output)

            # CompletedWithErrors is Completed | ErrorsMask, so the test above is
            # also true of a CLI that died. Without this branch the failure was
            # invisible - a Python traceback goes to stderr, which GetOutputText()
            # does not carry - and the chain went on running every later step on
            # the empty folders the dead one never filled, reporting success.
            if status & slicer.vtkMRMLCommandLineModuleNode.ErrorsMask:
                failed_module = self.module_name
                cli_error = self._briefCliOutput(caller.GetErrorText())
                qt.QTimer.singleShot(0, lambda: logger.error(
                    f"\n\n ========= {failed_module} FAILED ========= \n{cli_output}"
                    f"\n ========= ERROR DETAILS ========= \n{cli_error}"
                ))
                self.abortOnCliError(failed_module, cli_error)
                return

            qt.QTimer.singleShot(
                0, lambda: logger.info(f"\n\n ========= PROCESSED ========= \n{cli_output}")
            )
            
            if self.enterPauseForReview(self.current_process_info):
                return
            
            try:
                # Run next process
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

        # The narrowed replays are only links into the real output; once the run
        # is over they say nothing and would pile up run after run.
        self.clearReviewTempFolders()

        self.ui.label_3.setVisible(False)
        self.ui.progressBar.setVisible(False)
        self.ui.continueButton.setVisible(False)
        self.ui.cancelButton.setVisible(False)
        self.ui.CheckDependencyButton.enabled = True
        self.resetPauseState()
        self.ActualProcess = 1
        self.NumberProcess = 0
        self.cliNode = None
        self.list_process = []
        self.current_process_info = None

        self._checkCanApply()

        # Clean up temporary files if requested. This used to sit after the dialog
        # below, so a dialog left unanswered also left the output folder half done.
        if not self.ui.checkBox.isChecked():
            files_to_keep = []
            if "Visualization" in self.ui.comboBox2.currentText:
                files_to_keep.append("Heatmaps")
                files_to_keep.append("VTK Files")
            # The menu says "Quantitative", which is also what onApplyButton tests
            # to enable the quantification steps. Testing "Quantification" here
            # never matched, so the run deleted the results it had just produced.
            if "Quantitative" in self.ui.comboBox2.currentText:
                files_to_keep.append("Measurements")
                files_to_keep.append("Classification")

            try:
                # Path("") is Path("."), and iterdir() would then walk whatever
                # directory Slicer happens to be running from, deleting every
                # sub-folder in it and leaving the files - which is how a run
                # with no output folder set can empty a source tree. Nothing is
                # cleaned unless the folder is an absolute path that exists.
                output = (self._parameterNode.OutputFolder or "").strip()
                if not output or not os.path.isabs(output) or not os.path.isdir(output):
                    raise ValueError(
                        f"output folder {output!r} is not an absolute existing "
                        "directory; nothing cleaned"
                    )
                output_path = Path(output)
                for item in output_path.iterdir():
                    if item.is_dir() and item.name not in files_to_keep:
                        shutil.rmtree(item)
                        logger.info(f"Cleaned temporary folder: {item.name}")
            except Exception as e:
                logger.error(f"Error cleaning temporary files: {e}")

        # OnEndProcess is reached from onCliUpdated, i.e. from inside a VTK
        # observer callback. Opening an application-modal dialog there starts a
        # nested event loop while the CLI node is still dispatching events, and
        # the modal grab can leave the whole desktop session unresponsive, not
        # just Slicer. Same fix as AREG: defer it so the callback returns first.
        qt.QTimer.singleShot(0, self.showDoneMessage)
            
            
            

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
        logger.info("Processing started")

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
        logger.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


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
