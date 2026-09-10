import os, sys, platform, shutil, zipfile, urllib, textwrap, time, threading, re, io, tempfile
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata
import qt

from qt import (
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QCheckBox,
    QLabel,
    QLineEdit,
    QStackedWidget,
    QComboBox,
    QPushButton,
    QSlider,
    QFileDialog,
    QSpinBox,
    QWidget,
    QTimer,
    QApplication,
    QStandardPaths,
    QDialog,
    QSizePolicy,
    QSpacerItem,
    QProgressDialog,
    Qt,
    QStandardPaths
)

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import subprocess

from functools import partial
from pathlib import Path

import logging

# ===== Logging Configuration =====
logger = logging.getLogger("FlexReg")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def _get_installed_version(lib_name):
    try:
        return importlib_metadata.version(lib_name)
    except importlib_metadata.PackageNotFoundError:
        raise importlib_metadata.PackageNotFoundError

from FlexReg_utils.util import ToothNoExist, NoSegmentationSurf
from FlexReg_utils.orientation import orientation_f
from FlexReg_utils.butterfly_preview import ButterflyPreview, ADJUST_SIGN
from FlexReg_utils.mgl_patch import (
    MGLPatchBuilder, MGL_ARRAY_NAME, MGL_PREVIEW_ARRAY_NAME, MGL_ORDER,
    DEFAULT_HEIGHT, MIN_HEIGHT, MAX_HEIGHT, ReadLandmarks, WriteLandmarks,
    DoubtfulLandmarks,
)

# Travel of the joystick pads along the antero-posterior axis, in mm. Typing a
# larger value in the line edit still works, the knob just saturates.
ADJUST_RANGE = 5.0

# Travel of the translation pad, in mm, on both axes. It moves the four
# centroids together, so it is a rigid shift of the whole patch.
SHIFT_RANGE = 15.0

# Travel of the MGL joystick, in mm on both axes. It moves the checked
# landmarks off the line ALI drew, sideways towards the cheek or the tongue and
# up or down towards the crown or the vestibule.
MGL_OFFSET_RANGE = 6.0

# Side of the joystick pads, in pixels. The whole 0-1 ratio range is spread
# across the pad, so this is what sets how much a single pixel of mouse travel
# is worth : roughly 0.016 of ratio at 96 px, 0.009 at 128, 0.006 at 192.
# Raise it for a coarser hand, and remember it costs panel width four times
# over per scan. Ctrl while dragging is the finer tool -- it divides the
# sensitivity by five without costing a pixel.
PAD_SIZE = 128

# The translation pad drives one patch instead of one corner, and sits on a row
# of its own, so it does not need to be as large.
SHIFT_PAD_SIZE = 128



def check_lib_installed(lib_name, required_version=None):
    '''
    Check if the library is installed and meets the required version constraint (if any).
    - lib_name: "torch"
    - required_version: ">=1.10.0", "==0.7.0", "<2.0.0", etc.
    '''
    try:
        installed_version = _get_installed_version(lib_name)
        if required_version:
            # Simple version check - for minimal change, assume it's satisfied if installed
            # In future, could use packaging to parse required_version
            pass
        return True
    except importlib_metadata.PackageNotFoundError:
        return False

# import csv

def install_function(self, list_libs: list):
    '''
    Test the necessary libraries and install them with the specific version if needed.
    '''
    libs_to_install = []
    libs_to_update = []
    installation_errors = []

    for lib, version_constraint, url in list_libs:
        if not check_lib_installed(lib, version_constraint):
            try:
                if _get_installed_version(lib):
                    libs_to_update.append((lib, version_constraint))
            except:
                libs_to_install.append((lib, version_constraint))

    if libs_to_install or libs_to_update:
        message = "The following changes are required for the libraries:\n"

        if libs_to_update:
            message += "\n --- Libraries to update (version mismatch): \n"
            message += "\n".join([
                f"{lib} (current: {_get_installed_version(lib)}) -> {version_constraint.replace('==', '').replace('<=', '').replace('>=', '').replace('<', '').replace('>', '')}"
                for lib, version_constraint in libs_to_update
            ])
            message += "\n"

        if libs_to_install:
            message += "\n --- Libraries to install:  \n"
            message += "\n".join([
                f"{lib}{version_constraint}" if version_constraint else lib
                for lib, version_constraint in libs_to_install
            ])

        message += "\n\nDo you agree to modify these libraries? Doing so could cause conflicts with other installed Extensions."
        user_choice = slicer.util.confirmYesNoDisplay(message)

        if user_choice:
            for lib, version_constraint in libs_to_install + libs_to_update:
                try:
                    if not version_constraint:
                        pip_install(lib)
                    elif "https:/" in version_constraint:
                        pip_install(version_constraint)
                    else:
                        # Correctly format the library and version constraint
                        lib_version = f"{lib}{version_constraint}" if version_constraint.startswith(("==", ">=", "<=", ">", "<")) else f"{lib}=={version_constraint}"
                        pip_install(lib_version)
                except Exception as e:
                    installation_errors.append((lib, str(e)))

            if installation_errors:
                error_message = "The following errors occurred during installation:\n"
                error_message += "\n".join([f"{lib}: {error}" for lib, error in installation_errors])
                slicer.util.errorDisplay(error_message)
                return False
        else:
            return False
    return True


def ensureBooted(widget):
    '''
    Check the environment, once, when something is about to need it.

    The probes below shell out to conda and take seconds -- measured at 7.6 s
    on a warm machine: 2.2 s for `conda --version`, 0.6 s for the environment
    test and 4.8 s to import the install module inside it. None of that is
    needed to read a .vtk and put it on screen, so the check belongs to the
    actions that run a CLI or a conda command, not to viewScan.

    widget is a WidgetParameter: the check reports through its label_time and
    its logic. Returns True when the CLIs can be run.
    '''
    if FlexRegBootManager.booted:
        return True

    # The probes give no sign of life while they run, and the label set inside
    # onCheckRequirements is only painted once they are over. Say what is
    # happening first, and pump the event loop so it actually reaches the
    # screen, rather than leaving a panel that looks hung.
    widget.label_time.setVisible(True)
    widget.label_time.setText('Checking the environment, this takes a few seconds '
                              'the first time...')
    slicer.app.processEvents()

    if not widget.onCheckRequirements():
        return False

    # Only torch is missing from a stock Slicer: vtk, numpy, scipy and
    # SimpleITK all ship with it. numpy in particular must be left exactly as
    # Slicer installed it: its scipy is built for that numpy, and swapping
    # numpy on disk under a running session leaves the next scipy import
    # reading a mix of the two ("No module named 'numpy.strings'"), which is
    # what used to kill the MGL patch preview.
    if not install_function(widget, [('torch', None, None)]):  # (lib, version, url)
        qt.QMessageBox.warning(
            widget.parent, 'Warning',
            'The module will not work properly without the required libraries.\n'
            'Please install them and try again.')
        return False

    FlexRegBootManager.booted = True
    widget.label_time.setHidden(True)
    return True

#
# FlexReg
#

class FlexReg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "FlexReg"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Automated Dental Tools"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#FlexReg">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # FlexReg1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='FlexReg',
        sampleName='FlexReg1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'FlexReg1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='FlexReg1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='FlexReg1'
    )

    # FlexReg2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='FlexReg',
        sampleName='FlexReg2',
        thumbnailFileName=os.path.join(iconsPath, 'FlexReg2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='FlexReg2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='FlexReg2'
    )


#
# FlexRegWidget
#

class FlexRegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.reg = Reg() #Creation of an object reg for the registration

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/FlexReg.ui'))
        self.layout.addWidget(uiWidget)
        self.uiWidget = uiWidget  # Store reference for styling
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = FlexRegLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.ui.spinBoxnumberscan.valueChanged.connect(self.manageNumberWidgetScan)
        self.ui.spinBoxnumberscan.setVisible(False)
        self.ui.label.setVisible(False)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).


        # Make sure parameter node is initialized (needed for module reload)

        
        self.initializeParameterNode()

        # One arch for the whole registration : both scans are the same jaw,
        # so the choice is made once here instead of once per scan.
        row_arch = QHBoxLayout()
        row_arch.addWidget(QLabel('Arch :'))
        self.combobox_arch = QComboBox()
        self.combobox_arch.addItems(['Upper arch (palate)', 'Lower arch (MGL)'])
        self.combobox_arch.activated.connect(self.onArchChanged)
        row_arch.addWidget(self.combobox_arch)
        self.ui.verticalLayout_2.insertLayout(0, row_arch)

        self.number_widget_scan = 0
        self.list_widget_scan = []
        self.manageNumberWidgetScan(2)
        self.ui.applyButton.enabled = True
        self.ui.seeButton.enabled = True
        self.ui.buttonSelectOutput.connect("clicked(bool)",partial(self.openFinder,"Output"))
        self.ui.ButtonLowerArch.connect("clicked(bool)",partial(self.openFinder,"LowerArch"))
        self.ui.applyButton.connect("clicked(bool)",self.on_apply_button_clicked)
        self.ui.seeButton.connect("clicked(bool)",self.on_see_button_clicked)
        
        # Apply dark mode styling
        self.applyDarkModeStyles()

# Creation of the custom layout with 3 windows
        customLayout = """
<layout type="horizontal">
  <item>
    <view class="vtkMRMLViewNode" singletontag="1">
      <property name="viewlabel" action="default">1</property>
    </view>
  </item>
  <item>
    <view class="vtkMRMLViewNode" singletontag="2">
      <property name="viewlabel" action="default">2</property>
    </view>
  </item>
  <item>
    <view class="vtkMRMLViewNode" singletontag="3">
      <property name="viewlabel" action="default">3</property>
    </view>
  </item>
</layout>
"""

        customLayoutId=501

        layoutManager = slicer.app.layoutManager()
        layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(customLayoutId, customLayout)

        # Switch to the new custom layout
        layoutManager.setLayout(customLayoutId)

    def on_apply_button_clicked(self)->None:
        '''
        Launch the registration
        '''
        output_text = self.ui.lineEditOutput.text
        suffix_text = self.ui.lineEditSuffix.text
        lower_arch = self.ui.lineEditLowerArch.text
        
        if Path(lower_arch).is_file():
            self.reg.run(output_text, suffix_text, lower_arch)
        else :
            self.reg.run(output_text, suffix_text, "None")

    def on_see_button_clicked(self)->None:
        '''
        Same registration, shown in the third view, but nothing is kept : the
        CLI writes into a temporary folder that is removed once the result has
        been loaded into the scene.
        '''
        self.reg.run(None, self.ui.lineEditSuffix.text, "None", preview=True)

    def manageNumberWidgetScan(self,number)->None:
        '''
        Manage the number of widgets, all the widgets are the same and they're stock in list_widget_scan
        '''
        for i in  self.list_widget_scan:
            if i.getName()=="WidgetGo":
                self.removeWidgetScan()

        while self.number_widget_scan != number :
            if number >= self.number_widget_scan :
                self.addWidgetScan(self.number_widget_scan+1)
                self.number_widget_scan += 1
            elif number <= self.number_widget_scan :
                self.removeWidgetScan()
                self.number_widget_scan -= 1

        self.reg.setT1T2(self.list_widget_scan[0],self.list_widget_scan[1])

    def isLowerArch(self):
        return self.combobox_arch.currentIndex == 1

    def onArchChanged(self, _index=None):
        '''The arch selector at the top drives every scan panel at once.'''
        for widget in self.list_widget_scan:
            widget.setArch(self.isLowerArch())


    def removeWidgetScan(self):
        '''
        remove one widget of list_widget_scan
        '''
        mainwidgetscan = self.list_widget_scan.pop(-1).getMainWidget()
        mainwidgetscan.deleteLater()
        mainwidgetscan = None

        

    def addWidgetScan(self,title:int):
        '''
        add one widget of list_widget_scan
        '''
        self.list_widget_scan.append(
            WidgetParameter(self.ui.verticalLayout_2,self.parent,title,self.list_widget_scan))
        self.list_widget_scan[-1].setArch(self.isLowerArch())

    def openFinder(self,nom : str,_) -> None : 
        """
         Open finder to let the user choose is folder
        """ 


        if nom=="Output":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditOutput.setText(surface_folder)

        if nom=="LowerArch":
            path_file = QFileDialog.getOpenFileName(self.parent,'Open a file','', 'Surfaces (*.vtk *.stl)')
            self.ui.lineEditLowerArch.setText(path_file)

    def applyDarkModeStyles(self):
        """Apply dark mode styling to the widget if needed"""
        app = qt.QApplication.instance()
        palette = app.palette()
        bg_color = palette.color(qt.QPalette.Window)
        if bg_color.lightness() < 128:
            # Complete dark mode stylesheet
            dark_stylesheet = """
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
  image: url(:/Icons/SmallCheckMark.png);
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
            """
            self.uiWidget.setStyleSheet(dark_stylesheet)
            
            # Update QLineEdit, QComboBox, and QLabel for dark mode
            self._updateLineEditAndComboBoxDarkMode(self.uiWidget)

    def _updateLineEditAndComboBoxDarkMode(self, parent):
        """
        Recursively apply dark mode styles to QLineEdit, QComboBox, and QLabel widgets.
        """
        # Update QLabel
        if isinstance(parent, qt.QLabel):
            try:
                parent.setStyleSheet("""
                    QLabel {
                      color: #ffffff;
                      font-weight: 500;
                    }
                """)
            except:
                pass
        
        # Update QLineEdit
        if isinstance(parent, qt.QLineEdit):
            try:
                parent.setStyleSheet("""
                    QLineEdit {
                      background-color: #3c3c3c;
                      border: 1px solid #555555;
                      border-radius: 4px;
                      padding: 6px;
                      color: #ffffff;
                    }
                    QLineEdit:focus {
                      border: 2px solid #5dade2;
                    }
                """)
            except:
                pass
        
        # Update QComboBox
        if isinstance(parent, qt.QComboBox):
            try:
                parent.setStyleSheet("""
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
                """)
            except:
                pass
        
        # Recursively update all children
        if hasattr(parent, 'children'):
            for child in parent.children():
                self._updateLineEditAndComboBoxDarkMode(child)

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
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
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

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

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

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

        self._updatingGUIFromParameterNode = False

        

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)



#
# FlexRegLogic
#

class FlexRegLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self,lineedit=None,
                 lineedit_teeth_left_top=None,
                 lineedit_teeth_right_top=None,
                 lineedit_teeth_left_bot=None,
                 lineedit_teeth_right_bot=None,
                 lineedit_ratio_left_top=None,
                 lineedit_ratio_right_top=None,
                 lineedit_ratio_left_bot=None,
                 lineedit_ratio_right_bot=None,
                 lineedit_adjust_left_top=None,
                 lineedit_adjust_right_top=None,
                 lineedit_adjust_left_bot=None,
                 lineedit_adjust_right_bot=None,
                 curve="",
                 middle_point="",
                 type=None,
                 path_reg="",
                 path_output="",
                 suffix="",
                 index_patch=0,
                 lower_arch="None",
                 shift_lr=0.0,
                 shift_ap=0.0):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.lineedit=lineedit
        self.lineedit_teeth_left_top=lineedit_teeth_left_top
        self.lineedit_teeth_right_top=lineedit_teeth_right_top
        self.lineedit_teeth_left_bot=lineedit_teeth_left_bot
        self.lineedit_teeth_right_bot=lineedit_teeth_right_bot

        self.lineedit_ratio_left_top=lineedit_ratio_left_top
        self.lineedit_ratio_right_top=lineedit_ratio_right_top
        self.lineedit_ratio_left_bot=lineedit_ratio_left_bot
        self.lineedit_ratio_right_bot=lineedit_ratio_right_bot

        self.lineedit_adjust_left_top=lineedit_adjust_left_top
        self.lineedit_adjust_right_top=lineedit_adjust_right_top
        self.lineedit_adjust_left_bot=lineedit_adjust_left_bot
        self.lineedit_adjust_right_bot=lineedit_adjust_right_bot

        # Translation of the whole patch, in mm, in the oriented frame.
        self.shift_lr=shift_lr
        self.shift_ap=shift_ap

        self.curve=curve
        self.middle_point=middle_point

        self.type=type

        self.path_reg=path_reg
        self.path_output=path_output
        self.suffix=suffix
        
        self.index_patch=index_patch
        
        self.lower_arch=lower_arch
        
        self.isCondaSetUp = False
        self.conda = self.init_conda()
        self.name_env = "shapeaxi"
        # ALI, AREG, ASO and DOCShapeAXI share this environment, so the Python
        # version has to move in all of them at once. 3.9 was forced by
        # shapeaxi 1.0.10, which pinned grpcio==1.51.1 (no cp312 wheel, and a
        # source build that fails on modern setuptools). shapeaxi >= 2.0.2
        # dropped that pin, and prebuilt pytorch3d wheels only exist for
        # cp310-cp313, so 3.12 is now both possible and required.
        self.python_version = "3.12"

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self)->None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        """

        parameters = {}
        
        parameters ["lineedit"] = self.lineedit

        parameters ["lineedit_teeth_left_top"] = self.lineedit_teeth_left_top
        parameters ["lineedit_teeth_right_top"] = self.lineedit_teeth_right_top
        parameters ["lineedit_teeth_left_bot"] = self.lineedit_teeth_left_bot
        parameters ["lineedit_teeth_right_bot"] = self.lineedit_teeth_right_bot

        parameters ["lineedit_ratio_left_top"] = self.lineedit_ratio_left_top
        parameters ["lineedit_ratio_right_top"] = self.lineedit_ratio_right_top
        parameters ["lineedit_ratio_left_bot"] = self.lineedit_ratio_left_bot
        parameters ["lineedit_ratio_right_bot"] = self.lineedit_ratio_right_bot

        parameters ["lineedit_adjust_left_top"] = self.lineedit_adjust_left_top
        parameters ["lineedit_adjust_right_top"] = self.lineedit_adjust_right_top
        parameters ["lineedit_adjust_left_bot"] = self.lineedit_adjust_left_bot
        parameters ["lineedit_adjust_right_bot"] = self.lineedit_adjust_right_bot

        parameters ["shift_lr"] = self.shift_lr
        parameters ["shift_ap"] = self.shift_ap

        parameters ["curve"] = self.curve
        parameters ["middle_point"] = self.middle_point

        parameters ["type"] = self.type

        parameters ["path_reg"] = self.path_reg
        parameters["path_output"] = self.path_output
        parameters["suffix"] = self.suffix

        parameters["index_patch"] = self.index_patch
        
        parameters["lower_arch"] = self.lower_arch

        logger.info(f"Running FlexReg_CLI with parameters: {parameters}")

        flybyProcess = slicer.modules.flexreg_cli
        self.cliNode = slicer.cli.run(flybyProcess,None, parameters)
        self.cliNode.AddObserver(slicer.vtkMRMLCommandLineModuleNode.StatusModifiedEvent, self.onCliModified)  
        return flybyProcess
    
    def onCliModified(self, caller, event):
        """Callback triggered when CLI status changes (completed, cancelled, etc.)."""
        status = caller.GetStatus()

        if status & (slicer.vtkMRMLCommandLineModuleNode.Completed | slicer.vtkMRMLCommandLineModuleNode.Cancelled):
            logger.info("Background process finished (CLI)")

            if status == slicer.vtkMRMLCommandLineModuleNode.Completed:
                logger.info("FlexReg - COMPLETE")
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
    
    def init_conda(self):
        # check if CondaSetUp exists
        try:
            import CondaSetUp
        except:
            return False
        self.isCondaSetUp = True
        
        # set up conda on windows with WSL
        if platform.system() == "Windows":
            from CondaSetUp import CondaSetUpCallWsl
            return CondaSetUpCallWsl()
        else:
            from CondaSetUp import CondaSetUpCall
            return CondaSetUpCall()
        
    def run_conda_command(self, target, command):
        self.process = threading.Thread(target=target, args=command) #run in parallel to not block slicer
        self.process.start()
        
    def install_shapeaxi(self):
        # shapeaxi is installed later, by install_pytorch: it declares pytorch3d,
        # which PyPI does not carry, so pip fails here and the env is never created.
        # torch has to come first, install_pytorch reads its version to pick a wheel.
        self.run_conda_command(target=self.conda.condaCreateEnv, command=(self.name_env,self.python_version,["torch>=2.8,<2.13","ocnn==2.2.1","SimpleITK"],)) #run in parallel to not block slicer
        
    def check_if_pytorch3d(self):
        conda_exe = self.conda.getCondaExecutable()
        command = [conda_exe, "run", "-n", self.name_env, "python" ,"-c", f"\"import pytorch3d;import pytorch3d.renderer;import shapeaxi.dental_model_seg as d;d.saxi_nets_lightning.DentalModelSeg\""]
        return self.conda.condaRunCommand(command)
    
    def install_pytorch3d(self):
        '''
        Start the pytorch3d installation in the conda environment.
        Returns whether it could be started : the caller waits on
        self.process, so it must not wait on a thread that never ran.
        '''
        result_pythonpath = self.check_pythonpath_windows("FlexReg_utils.install_pytorch")
        if not result_pythonpath :
            self.give_pythonpath_windows()
            result_pythonpath = self.check_pythonpath_windows("FlexReg_utils.install_pytorch")

        if not result_pythonpath :
            # Nothing to run. Falling through to run_conda_command here used to
            # raise an UnboundLocalError on 'command', which hid the real
            # failure -- usually that the environment cannot import the module.
            logger.error(
                f"The conda environment '{self.name_env}' cannot import "
                "FlexReg_utils.install_pytorch. Run that import by hand in the "
                "environment to see why."
            )
            return False

        conda_exe = self.conda.getCondaExecutable()
        path_pip = self.conda.getCondaPath()+f"/envs/{self.name_env}/bin/pip"
        command = [conda_exe, "run", "-n", self.name_env, "python" ,"-m", f"FlexReg_utils.install_pytorch",path_pip]

        self.run_conda_command(target=self.conda.condaRunCommand, command=(command,))
        return True
        
    def setup_cli_command(self):
        args = self.find_cli_parameters()
        # condaRunCommand already wraps everything in `conda run -n <env>`
        command = ["python", "-m", "FlexReg_CLI"]
        for arg in args :
            command.append("\""+arg+"\"")

        self.run_conda_command(target=self.condaRunCommand, command=(command,))
        
    def check_lib_wsl(self) -> bool:
        # Ubuntu versions < 24.04
        required_libs_old = ["libxrender1", "libgl1-mesa-glx"]
        # Ubuntu versions >= 24.04
        required_libs_new = ["libxrender1", "libgl1", "libglx-mesa0"]


        all_installed = lambda libs: all(
            subprocess.run(
                f"wsl -- bash -c \"dpkg -l | grep {lib}\"", capture_output=True, text=True
            ).stdout.encode("utf-16-le").decode("utf-8").replace("\x00", "").find(lib) >= 0
            for lib in libs
        )

        return all_installed(required_libs_old) or all_installed(required_libs_new)

        return "libxrender1" in clean_output1 and "libgl1-mesa-glx" in clean_output2
    
    def check_pythonpath_windows(self,file):
        '''
        Check if the environment env_name in wsl know the path to a specific file (ex : Crownsegmentationcli.py)
        return : bool
        '''
        conda_exe = self.conda.getCondaExecutable()
        command = [conda_exe, "run", "-n", self.name_env, "python" ,"-c", f"\"import {file} as check;import os; print(os.path.isfile(check.__file__))\""]
        result = self.conda.condaRunCommand(command)
        if "True" in result :
            return True
        return False
    
    def give_pythonpath_windows(self):
        '''
        take the pythonpath of Slicer and give it to the environment name_env in wsl.
        '''
        paths = slicer.app.moduleManager().factoryManager().searchPaths
        mnt_paths = []
        for path in paths :
            mnt_paths.append(f"\"{self.windows_to_linux_path(path)}\"")
        pythonpath_arg = 'PYTHONPATH=' + ':'.join(mnt_paths)
        conda_exe = self.conda.getCondaExecutable()
        argument = [conda_exe, 'env', 'config', 'vars', 'set', '-n', self.name_env, pythonpath_arg]
        results = self.conda.condaRunCommand(argument)
        
    def windows_to_linux_path(self,windows_path):
        '''
        convert a windows path to a wsl path
        '''
        windows_path = windows_path.strip()

        path = windows_path.replace('\\', '/')

        if ':' in path:
            drive, path_without_drive = path.split(':', 1)
            path = "/mnt/" + drive.lower() + path_without_drive

        return path
    
    def check_cli_script(self):
        if not self.check_pythonpath_windows("FlexReg_CLI"): 
            self.give_pythonpath_windows()
            results = self.check_pythonpath_windows("FlexReg_CLI")
            
        if not self.check_pythonpath_windows("CrownSegmentationcli"):
            self.give_pythonpath_windows()
            results = self.check_pythonpath_windows("CrownSegmentationcli")
            
    def runALI(self, arguments):
        '''
        Predict the MG landmarks of one scan, in the conda environment where
        ALI_IOS has its dependencies. Blocks until the prediction is over.
        '''
        if not self.check_pythonpath_windows("ALI_IOS"):
            self.give_pythonpath_windows()

        # condaRunCommand already wraps everything in `conda run -n <env>`
        command = ["python", "-m", "ALI_IOS"]
        for argument in arguments:
            value = argument
            if isinstance(value, str) and ("\\" in value or (len(value) > 1 and value[1] == ":")):
                value = self.windows_to_linux_path(value)
            command.append(f'"{value}"')

        logger.info(f"Running ALI_IOS: {' '.join(command)}")
        self.condaRunCommand(command)

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
            logger.info(f"command_to_execute in condaRunCommand : {command_to_execute}")

            self.subpro = subprocess.Popen(command_to_execute, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                    text=True, encoding='utf-8', errors='replace', env=slicer.util.startupEnvironment(),
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # For Windows
                                    )
        else:
            path_conda_exe = self.conda.getCondaExecutable()
            command_execute = f"{path_conda_exe} run -n {self.name_env}"
            for com in command :
                command_execute = command_execute+ " "+com

            logger.info(f"command_to_execute in conda run : {command_execute}")
            self.subpro = subprocess.Popen(command_execute, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', env=slicer.util.startupEnvironment(), executable="/bin/bash", preexec_fn=os.setsid)
    
        self.stdout, self.stderr = self.subpro.communicate()


#
# FlexRegTest
#

class FlexRegTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_FlexReg1()

    def test_FlexReg1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
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
        inputVolume = SampleData.downloadSample('FlexReg1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = FlexRegLogic()

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

        self.delayDisplay('Test passed')

# Class that create a pop up which display the time since the begenning
class TimerDialog(QDialog):
    def __init__(self, parent=None):
        super(TimerDialog, self).__init__(parent)
        
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Registration")
        
        self.timeLabel = QLabel("Starting timer...", self)
        self.layout().addWidget(self.timeLabel)

        self.closeButton = QPushButton("Close", self)
        self.closeButton.setEnabled(False)  # Disable it initially
        self.closeButton.clicked.connect(lambda _: self.accept())
        self.layout().addWidget(self.closeButton)

        self.start_time = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTime)
        
    def startTimer(self):
        self.start_time = time.time()
        self.timer.start(1000)  # Update every second

    def updateTime(self):
        elapsed_time = time.time() - self.start_time
        self.timeLabel.setText(f"Registration in process \n time : {round(float(elapsed_time), 2)}s")
        
    def endTimer(self):
        elapsed_time = time.time() - self.start_time
        self.timer.stop()
        self.timeLabel.setText(f"End of the registration ! \n time : {round(float(elapsed_time), 2)}s")
        self.closeButton.setEnabled(True)


# Class doing the registration
class Reg:
    def __init__(self,T1=None,T2=None) -> None:
        self.T1 = T1
        self.T2 = T2
        self.surfT1=None
        self.surfT2=None
        self.start_time=0
        self.output_folder=None
        self.suffix=None
        self.lower_arch=None
        self.preview=False
        self.temp_folder=None
        self.timer = QTimer()

    def run(self,output_folder:str,suffix:str, lower_arch:str, preview:bool=False)->None:
        '''
        call the cli for the registration with icp method and launch onProcessUpdateICP

        preview : run exactly the same registration but into a throwaway
        folder, removed as soon as the result is in the scene. Looking at a
        result then no longer leaves a .vtk and a .tfm behind every time.
        '''
        if self.T1.getSurf()!=None and  self.T2.getSurf()!=None :
            array_name = self.patchArrayName()
            if (self.isButterflyPatchAvailable(self.T1.getSurf(), array_name)
                    and self.isButterflyPatchAvailable(self.T2.getSurf(), array_name)):
                # The registration itself is a CLI, so the environment is
                # checked here rather than when the scans were displayed.
                if not ensureBooted(self.T1):
                    return
                self.preview = preview
                self.temp_folder = None
                if preview:
                    # The CLI only reports back through files, so it still
                    # writes -- just somewhere that gets deleted.
                    self.temp_folder = tempfile.mkdtemp(prefix="FlexReg_see_")
                    output_folder = self.temp_folder
                    lower_arch = "None"
                self.output_folder=output_folder
                self.suffix=suffix
                self.lower_arch=lower_arch
                self._processed = False # To allow onProcessUpdateICP to display the time and launch endProcess
                # CLI 
                self.logic = FlexRegLogic(self.T2.getPath(),
                                int(0),
                            int(0),
                            int(0),
                            int(0),
                            float(0),
                            float(0),
                            float(0),
                            float(0),
                            float(0),
                            float(0),
                            float(0),
                            float(0),
                            "None",
                            "None",
                            "icp_mgl" if self.isLowerArch() else "icp",
                            self.T1.getPath(),
                            output_folder,
                            suffix,
                            0,
                            lower_arch)
                self.logic.process()

                self.start_time = time.time()
                self.timer.timeout.connect(self.onProcessUpdateICP)
                self.timer.start(500)

            else:
                slicer.util.infoDisplay(
                    f"Create the {array_name} patch on T1 and T2 before registration")
        else :
            slicer.util.infoDisplay(f"Load a vtk file in window number : 1 and 2 \nTo do this, enter the path to a vtk file and click on view.")

    def isButterflyPatchAvailable(self, model_node, array_name="Butterfly")->bool:
        """
        Check if the patch of the current arch is available for the model node.
        """
        polyData = model_node.GetPolyData()
        if polyData:
            scalars = (polyData.GetPointData().GetScalars(array_name)
                       or polyData.GetPointData().GetArray(array_name))
            return scalars is not None
        return False

    def isLowerArch(self)->bool:
        """True when both scans are set to the lower arch."""
        return self.T1.isLowerArch() and self.T2.isLowerArch()

    def patchArrayName(self)->str:
        return MGL_ARRAY_NAME if self.isLowerArch() else "Butterfly"


    def onProcessUpdateICP(self)->None:
        '''
        Called at the same time of the cli, update every 500ms to update the time since the begenning.
        Launch the display of the registration after the end of the cli
        '''
        # To make sure you don't launch the display twice.
        if hasattr(self, "_processed") and self._processed:
            return

        # Launch pop up with time
        if not hasattr(self, "timerDialog"):
            # Parented to the main window so it stays on top of Slicer and is
            # repainted with it, instead of floating as a separate window.
            self.timerDialog = TimerDialog(slicer.util.mainWindow())
            self.timerDialog.show()
            self.timerDialog.startTimer()

        # If end cli launch display and end timer
        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
            self._processed = True
            self.timer.stop()
            self.timerDialog.endTimer()
            del self.timerDialog
            try:
                self.endProcess()
            finally:
                self.discardTempFolder()

    def discardTempFolder(self)->None:
        '''
        Drop the throwaway folder used by See. endProcess has already read the
        meshes into the scene, so the files are no longer needed.
        '''
        if not self.temp_folder:
            return
        shutil.rmtree(self.temp_folder, ignore_errors=True)
        logger.info(f"Preview folder discarded : {self.temp_folder}")
        self.temp_folder = None



    def endProcess(self)->None:
        '''
        Display the registration in the third windows with 2 different color for T1 and T2
        '''
        self.cleanView()
        # Load the result of the registration and T1 model
        outpath = self.T2.getPath().replace(os.path.dirname(self.T2.getPath()),self.output_folder)
        path_newT2 = outpath.split('.vtk')[0].split('vtp')[0]+self.suffix+'.vtk'
        self.surfT1 = slicer.util.loadModel(self.T1.getPath())
        self.surfT2 = slicer.util.loadModel(path_newT2)

        if self.preview:
            # The file behind this node is about to disappear, so keep the node
            # out of any scene save and say plainly that nothing was written.
            self.surfT2.SetName(f"{os.path.basename(self.T2.getPath())} registered (preview, not saved)")
            self.surfT2.SetSaveWithScene(False)
            storage = self.surfT2.GetStorageNode()
            if storage:
                storage.SetSaveWithScene(False)

        # Get data model
        displayNodeT1 = self.surfT1.GetDisplayNode()
        displayNodeT2 = self.surfT2.GetDisplayNode()
        
        # Get all vtkMRMLViewNodes of the scene
        viewNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLViewNode')
        viewNodes.UnRegister(None) # De-register to avoid memory leaks
        
        # Access to our custom layout
        customLayoutId=501
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(customLayoutId)

        # Access layout 2
        viewNode = viewNodes.GetItemAsObject(2) if viewNodes.GetNumberOfItems() >= 2 else None

        # Set colors of the model
        colors = [[255/256,51/256,200/256], [102/256,102/256,255/256]]
        displayNodeT1.SetColor(colors[0])
        displayNodeT2.SetColor(colors[1])
        
        if viewNode:
            # Display model in windows
            displayNodeT1.SetViewNodeIDs([viewNode.GetID()])
            displayNodeT2.SetViewNodeIDs([viewNode.GetID()])

        else:
            slicer.util.errorDisplay(f"There is 3D windows available with the index : {2}.")

        # T1 model was not modify during the register process. Get his matrix to center and apply to the oth
        matrix = self.T1.getMatrix()

        transform_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
        transform_node.SetMatrixTransformToParent(matrix)
        model = self.surfT1
        model.SetAndObserveTransformNodeID(transform_node.GetID())
        model.HardenTransform()

        model = self.surfT2
        model.SetAndObserveTransformNodeID(transform_node.GetID())
        model.HardenTransform()

  

    def cleanView(self)->None:
        '''
        Delete all model load in windows 2
        '''
        viewNode1 = slicer.mrmlScene.GetSingletonNode("3", "vtkMRMLViewNode")
        modelNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
        modelNodes.InitTraversal()
        modelsToDelete = []
        for i in range(modelNodes.GetNumberOfItems()):
            modelNode = modelNodes.GetNextItemAsObject()
            modelDisplayNode = modelNode.GetDisplayNode()

            if modelDisplayNode and modelDisplayNode.GetViewNodeIDs() and viewNode1.GetID() in modelDisplayNode.GetViewNodeIDs():
                modelsToDelete.append(modelNode)
        
        for model in modelsToDelete:
            slicer.mrmlScene.RemoveNode(model)


    
    def getName(self)->str:
        '''
        Return the name of the class
        '''
        return "Reg"
    
    def setT1T2(self,T1,T2)->None:
        '''
        Set the widget using for T1 and T2
        '''
        self.T1 = T1
        self.T2 = T2


def _eventPosition(event):
    '''
    Read a mouse position out of a Qt event. PythonQt exposes some getters as
    plain attributes and others as methods depending on the build, so try both
    rather than betting on one.
    '''
    position = event.pos
    if callable(position):
        position = position()
    x = position.x() if callable(position.x) else position.x
    y = position.y() if callable(position.y) else position.y
    return float(x), float(y)


def _wheelSteps(event):
    '''Number of notches scrolled, positive upwards.'''
    try:
        delta = event.angleDelta
        delta = delta() if callable(delta) else delta
        value = delta.y() if callable(delta.y) else delta.y
    except AttributeError:
        value = event.delta
        if callable(value):
            value = value()
    return float(value) / 120.0


class JoystickPad(QWidget):
    '''
    Absolute 2D pad driving one corner of the butterfly patch. The knob
    position *is* the state : horizontal is the medio-lateral ratio, vertical
    the antero-posterior adjust in mm.

    Both axes are read spatially rather than numerically, so the knob always
    sits where the point sits. A ratio of 1 lands on the tooth itself, at the
    outer edge of the arch, and 0 at mid-arch, so the horizontal axis is
    mirrored for the corners on the other side (outward_right). The CLI negates
    the adjust of the posterior corners, so their vertical axis is mirrored too
    (adjust_sign) -- push up on any of the four pads and the point moves
    anteriorly.

    The same pad also drives the whole-patch translation, with ratio_range
    turning the horizontal axis into millimetres instead of a 0-1 ratio and
    side_labels naming both ends of it. Either way the two values are called
    ratio and adjust here : what they mean is the caller's business.

    Dragging, the wheel, the arrow keys and the line edits all funnel into
    setValues(), and any change calls onChanged.
    '''

    GUTTER = 11      # room above and below the pad, for the ANT / POST labels
    OUT_GUTTER = 15  # room for the OUT marker, on the outward side only
    KNOB = 7
    FINE = 0.2       # sensitivity multiplier while Ctrl is held

    def __init__(self, outward_right=True, adjust_range=5.0, size=None, adjust_sign=1.0,
                 ratio_range=(0.0, 1.0), side_labels=None, spring_back=False, parent=None):
        QWidget.__init__(self, parent)
        # A spring_back pad is relative : each drag deals out a displacement
        # and the knob returns to the centre on release, so repeated pushes
        # never saturate against the ends of the travel.
        self.spring_back = bool(spring_back)
        self.onReleased = None
        self.outward_right = outward_right
        self.adjust_range = adjust_range
        self.adjust_sign = float(adjust_sign)
        self.ratio_min = float(ratio_range[0])
        self.ratio_max = float(ratio_range[1])
        # One notch of the wheel, or one arrow key, walks a hundredth of the
        # horizontal range -- 0.01 of ratio on a corner pad, 0.1 mm on a range
        # of 10 mm.
        self.ratio_step = (self.ratio_max - self.ratio_min) / 100.0
        self.side_labels = side_labels
        self.SIDE = int(size or PAD_SIZE)
        self.ratio = 0.0
        self.adjust = 0.0
        self.default_ratio = 0.0
        self.default_adjust = 0.0
        self.onWheel = None
        self.onChanged = None
        self.enabled_preview = True
        self._dragging = False
        self._anchor = None

        self.setFixedSize(self.SIDE, self.SIDE)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip(
            'Drag to move this corner of the patch.\n'
            'Horizontal : ratio (outwards / inwards on the arch)\n'
            'Vertical : adjust (anterior / posterior, in mm)\n'
            'Ctrl+drag : five times finer\n'
            'Wheel : fine adjust step, Shift+wheel : fine ratio step\n'
            'Arrow keys : one fine step, double-click : back to the default'
        )

    # ---- values ---------------------------------------------------------

    def setValues(self, ratio, adjust, notify=False):
        ratio = min(max(float(ratio), self.ratio_min), self.ratio_max)
        adjust = min(max(float(adjust), -self.adjust_range), self.adjust_range)
        if ratio == self.ratio and adjust == self.adjust:
            return
        self.ratio = ratio
        self.adjust = adjust
        self.update()
        if notify and self.onChanged:
            self.onChanged(self)

    def setOutwardRight(self, outward_right):
        if bool(outward_right) != self.outward_right:
            self.outward_right = bool(outward_right)
            self.update()

    def setPreviewEnabled(self, enabled):
        '''Greys the pad out when the scan cannot be previewed.'''
        self.enabled_preview = bool(enabled)
        self.update()

    # ---- geometry -------------------------------------------------------

    def _box(self):
        '''The pad itself. Labels sit outside it, the knob never leaves it.'''
        if self.side_labels:
            # Both ends of the axis are named, so both gutters are taken.
            return (self.OUT_GUTTER, self.GUTTER,
                    self.SIDE - 2 * self.OUT_GUTTER, self.SIDE - 2 * self.GUTTER)
        left = 0 if self.outward_right else self.OUT_GUTTER
        return left, self.GUTTER, self.SIDE - self.OUT_GUTTER, self.SIDE - 2 * self.GUTTER

    def _area(self):
        '''Where the centre of the knob is allowed to travel.'''
        left, top, width, height = self._box()
        inset = self.KNOB + 2
        return left + inset, top + inset, width - 2 * inset, height - 2 * inset

    def _knobPosition(self):
        left, top, width, height = self._area()
        # ratio 1 is the outer edge, so the knob walks towards OUT as it grows
        span = self.ratio_max - self.ratio_min
        fraction = (self.ratio - self.ratio_min) / span
        if not self.outward_right:
            fraction = 1.0 - fraction
        anterior = self.adjust * self.adjust_sign
        x = left + fraction * width
        y = top + (1.0 - (anterior + self.adjust_range) / (2 * self.adjust_range)) * height
        return x, y

    def _valuesAt(self, x, y):
        left, top, width, height = self._area()
        fraction = min(max((x - left) / float(width), 0.0), 1.0)
        if not self.outward_right:
            fraction = 1.0 - fraction
        ratio = self.ratio_min + fraction * (self.ratio_max - self.ratio_min)
        vertical = min(max((y - top) / float(height), 0.0), 1.0)
        anterior = (1.0 - vertical) * 2 * self.adjust_range - self.adjust_range
        return ratio, anterior * self.adjust_sign

    # ---- interaction ----------------------------------------------------

    def _isFine(self):
        return bool(slicer.app.keyboardModifiers() & Qt.ControlModifier)

    def mousePressEvent(self, event):
        self._dragging = True
        self._anchor = None
        x, y = _eventPosition(event)
        if self.spring_back or self._isFine():
            # Relative drag : hold the knob where it is and move from there,
            # rather than jumping it under the cursor.
            self._anchor = ((x, y), self._knobPosition())
            return
        self.setValues(*self._valuesAt(x, y), notify=True)

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        x, y = _eventPosition(event)
        if self.spring_back or self._isFine():
            if self._anchor is None:
                # Ctrl pressed mid-drag : rebase so the knob does not jump.
                self._anchor = ((x, y), self._knobPosition())
        else:
            self._anchor = None
        if self._anchor is not None:
            (anchor_x, anchor_y), (knob_x, knob_y) = self._anchor
            scale = self.FINE if self._isFine() else 1.0
            x = knob_x + (x - anchor_x) * scale
            y = knob_y + (y - anchor_y) * scale
        self.setValues(*self._valuesAt(x, y), notify=True)

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._anchor = None
        if self.spring_back:
            # The displacement has been dealt out : back to the centre,
            # silently, ready for the next push.
            self.setValues(self.default_ratio, self.default_adjust)
            if self.onReleased:
                self.onReleased(self)

    def setDefaults(self, ratio, adjust):
        '''Where a double-click sends the knob back to.'''
        self.default_ratio = float(ratio)
        self.default_adjust = float(adjust)

    def mouseDoubleClickEvent(self, event):
        self.setValues(self.default_ratio, self.default_adjust, notify=True)

    def wheelEvent(self, event):
        '''
        The wheel is a vertical gesture, so it drives the vertical axis :
        scrolling up walks the knob up, towards anterior. Shift swaps it onto
        the other axis, where up means outwards.
        '''
        steps = _wheelSteps(event)
        if self.onWheel is not None:
            # Both axes drive the movement in MGL, so the wheel is free for
            # whatever the caller wants to put on it.
            self.onWheel(self, steps)
            return
        if slicer.app.keyboardModifiers() & Qt.ShiftModifier:
            # scrolling up walks the point outwards, which is the ratio going up
            self.setValues(self.ratio + self.ratio_step * steps, self.adjust, notify=True)
        else:
            self.setValues(self.ratio, self.adjust + 0.1 * steps * self.adjust_sign, notify=True)

    def keyPressEvent(self, event):
        key = event.key
        if callable(key):
            key = key()
        # arrows are screen-directional : Right always walks the knob right
        horizontal = 1.0 if self.outward_right else -1.0
        if key == Qt.Key_Left:
            self.setValues(self.ratio - self.ratio_step * horizontal, self.adjust, notify=True)
        elif key == Qt.Key_Right:
            self.setValues(self.ratio + self.ratio_step * horizontal, self.adjust, notify=True)
        elif key == Qt.Key_Up:
            self.setValues(self.ratio, self.adjust + 0.1 * self.adjust_sign, notify=True)
        elif key == Qt.Key_Down:
            self.setValues(self.ratio, self.adjust - 0.1 * self.adjust_sign, notify=True)
        if self.spring_back:
            # each key press is one nudge, dealt out then sprung back
            self.setValues(self.default_ratio, self.default_adjust)
            if self.onReleased:
                self.onReleased(self)

    # ---- painting -------------------------------------------------------

    def _palette(self):
        # Same test applyDarkModeStyles uses, so the pad follows the rest of
        # the panel : the stylesheets it applies do not touch the palette.
        window = qt.QApplication.instance().palette().color(qt.QPalette.Window)
        if window.lightness() < 128:
            return {
                'background': qt.QColor('#2b3138'),
                'border': qt.QColor('#4a5560'),
                'grid': qt.QColor('#3d454e'),
                'text': qt.QColor('#8b97a3'),
                'label': qt.QColor('#b6c2ce'),
                'knob': qt.QColor('#4ba3ff'),
                'trail': qt.QColor('#3f5871'),
            }
        return {
            'background': qt.QColor('#f4f7fa'),
            'border': qt.QColor('#d3dce5'),
            'grid': qt.QColor('#e3eaf1'),
            'text': qt.QColor('#93a2b1'),
            'label': qt.QColor('#6b7c8d'),
            'knob': qt.QColor('#3498db'),
            'trail': qt.QColor('#bcd7ef'),
        }

    def paintEvent(self, event):
        colors = self._palette()
        box_left, box_top, box_width, box_height = self._box()
        centre_x = box_left + box_width / 2.0
        centre_y = box_top + box_height / 2.0

        painter = qt.QPainter(self)
        painter.setRenderHint(qt.QPainter.Antialiasing)

        painter.setPen(qt.QPen(colors['border'], 1))
        painter.setBrush(qt.QBrush(colors['background']))
        painter.drawRoundedRect(box_left, box_top, box_width - 1, box_height - 1, 6, 6)

        painter.setPen(qt.QPen(colors['grid'], 1))
        painter.drawLine(int(box_left + 4), int(centre_y), int(box_left + box_width - 4), int(centre_y))
        painter.drawLine(int(centre_x), int(box_top + 4), int(centre_x), int(box_top + box_height - 4))

        painter.setFont(qt.QFont('', 6))
        painter.setPen(qt.QPen(colors['text'], 1))
        painter.drawText(qt.QRect(box_left, 0, box_width, self.GUTTER), Qt.AlignCenter, 'ANT')
        painter.drawText(qt.QRect(box_left, self.SIDE - self.GUTTER, box_width, self.GUTTER),
                         Qt.AlignCenter, 'POST')

        # Which way the arch faces outwards. A ratio of 0 sits on the tooth
        # itself, so the knob travels towards OUT as the value goes down.
        painter.setPen(qt.QPen(colors['label'], 1))
        left_gutter = qt.QRect(0, 0, self.OUT_GUTTER, self.SIDE)
        right_gutter = qt.QRect(self.SIDE - self.OUT_GUTTER, 0, self.OUT_GUTTER, self.SIDE)
        if self.side_labels:
            painter.drawText(left_gutter, Qt.AlignCenter, '\n'.join(self.side_labels[0]))
            painter.drawText(right_gutter, Qt.AlignCenter, '\n'.join(self.side_labels[1]))
        else:
            painter.drawText(right_gutter if self.outward_right else left_gutter,
                             Qt.AlignCenter, 'O\nU\nT')

        if not self.enabled_preview:
            painter.setPen(qt.QPen(colors['text'], 1))
            painter.drawText(qt.QRect(box_left, box_top, box_width, box_height), Qt.AlignCenter, 'n/a')
            painter.end()
            return

        knob_x, knob_y = self._knobPosition()
        painter.setPen(qt.QPen(colors['trail'], 2))
        painter.drawLine(int(centre_x), int(centre_y), int(knob_x), int(knob_y))

        painter.setPen(qt.QPen(colors['knob'].darker(120), 1))
        painter.setBrush(qt.QBrush(colors['knob']))
        painter.drawEllipse(int(knob_x - self.KNOB), int(knob_y - self.KNOB), 2 * self.KNOB, 2 * self.KNOB)
        painter.end()


# Class with widget
class WidgetParameter:
    def __init__(self,layout,parent,title,scans=None) -> None:
        self.parent_layout = layout
        self.parent = parent
        self.surf = None
        self.curve = None
        self.glue = False
        self.middle_point = None
        self.matrix = None
        self.title=title
        self.camera = True
        # Which arch is registered; set from the module-level arch selector.
        self.lower_arch = False
        # The live list of scan widgets, shared by all of them, so the Copy
        # button can read the values of the one above. It is still being filled
        # while this one is built, hence the lookup at click time.
        self.scans = scans if scans is not None else []

        # Live preview of the butterfly patch. The contour and an approximate
        # fill are recomputed on every joystick move (a few ms); the real patch
        # still comes from the CLI when Update is pressed.
        self.pads = {}
        self.fields = {}
        self.preview = ButterflyPreview()
        self.preview_node = None
        # Touched by handleStackedWidgetChange, which fires as soon as the
        # first page is inserted -- before setupMGL has run.
        self.mgl_markups_node = None
        self.mgl_names = []
        self.preview_dirty = False
        self._syncing = False
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.refreshPreview)

        self.main_widget = QWidget()
        layout.addWidget(self.main_widget)
        self.maint_layout = QVBoxLayout(self.main_widget)
        self.setup(self.maint_layout,title)
        self.timer = QTimer()
        self.start_time = None
        self.documentsLocation = QStandardPaths.DocumentsLocation
        self.documents = QStandardPaths.writableLocation(self.documentsLocation)
        self.SlicerDownloadPath = os.path.join(
                self.documents,
                slicer.app.applicationName + "Downloads",
            )
        self.logic = FlexRegLogic()

    def setup(self,layout,title):
        '''
        Create the widget with all the qt design and the connection of the button
        '''

        self.layout_file = QHBoxLayout()
        layout.addLayout(self.layout_file)
        if title==2:
            self.label_1 = QLabel(f'Moving scan : ')
        else :
            self.label_1 = QLabel(f'Fix scan : ')
        self.lineedit = QLineEdit()
        self.button_select_scan = QPushButton('Select')
        self.button_select_scan.pressed.connect(self.selectFile)

        self.button_scene = QPushButton('Scene')
        self.button_scene.setToolTip(
            'Use a model already loaded in the scene instead of browsing for '
            'its file. Its own file is used when it has one; otherwise a .vtk '
            'copy is written. The original model is hidden while FlexReg '
            'shows its own centered copy of it.')
        self.button_scene.pressed.connect(self.selectSceneModel)

        self.button_test_file = QPushButton('TestFile')
        self.button_test_file.pressed.connect(self.testFile)


        self.layout_file.addWidget(self.label_1)
        self.layout_file.addWidget(self.lineedit)
        self.layout_file.addWidget(self.button_select_scan)
        self.layout_file.addWidget(self.button_scene)
        self.layout_file.addWidget(self.button_test_file)

        widgetView = QWidget()
        self.layoutView = QGridLayout(widgetView)
        self.button_view = QPushButton('View')
        self.button_view.pressed.connect(self.viewScan)
        self.layoutView.addWidget(self.button_view)
        layout.addWidget(widgetView)
        

        self.combobox_choice_method = QComboBox()
        self.combobox_choice_method.addItems(['Parameter','Landmark'])
        self.combobox_choice_method.activated.connect(self.changeMode)
        layout.addWidget(self.combobox_choice_method)



        self.stackedWidget = QStackedWidget()
        layout.addWidget(self.stackedWidget)
        self.stackedWidget.currentChanged.connect(self.handleStackedWidgetChange)


        #widget paramater
        widget_full_paramater = QWidget()
        self.stackedWidget.insertWidget(0,widget_full_paramater)
        self.layout_widget = QGridLayout(widget_full_paramater)

        self.layout_left_top = QGridLayout()
        self.layout_right_top = QGridLayout()
        self.layout_left_bot = QGridLayout()
        self. layout_right_bot = QGridLayout()

        self.layout_widget.addLayout(self.layout_left_top,0,0)
        self.layout_widget.addLayout(self.layout_right_top,0,1)
        self.layout_widget.addLayout(self.layout_left_bot,1,0)
        self.layout_widget.addLayout(self.layout_right_bot,1,1)


        # The 2x2 grid mirrors the arch : left column is the side of teeth 5/3,
        # right column the side of teeth 12/14, top row anterior.
        # Base values come from 'FIX: Fix base values for FlexReg' and go with
        # the (1 - ratio) / 2 mapping in make_butterfly.
        (self.lineedit_teeth_left_top ,
         self.lineedit_ratio_left_top ,
            self.lineedit_adjust_left_top) = self.displayParamater(self.layout_left_top,1,[5,0.345,-0.1],'anterior_left',False)

        (self.lineedit_teeth_right_top ,
         self.lineedit_ratio_right_top ,
            self.lineedit_adjust_right_top) = self.displayParamater(self.layout_right_top,2,[12,0.345,-0.1],'anterior_right',True)

        (self.lineedit_teeth_left_bot ,
         self.lineedit_ratio_left_bot ,
            self.lineedit_adjust_left_bot) = self.displayParamater(self.layout_left_bot,3,[3,0.32,-2],'posterior_left',False)

        (self.lineedit_teeth_right_bot ,
         self.lineedit_ratio_right_bot ,
            self.lineedit_adjust_right_bot) = self.displayParamater(self.layout_right_bot,4,[14,0.32,-2],'posterior_right',True)

        # Translation of the whole patch, on top of the four corners.
        self.layout_shift = QGridLayout()
        self.layout_widget.addLayout(self.layout_shift,2,0,1,2)
        self.lineedit_shift_lr, self.lineedit_shift_ap = self.displayShift(self.layout_shift)

        self.label_preview = QLabel('')
        self.label_preview.setVisible(False)
        self.layout_widget.addWidget(self.label_preview,3,0,1,2)

        self.button_update = QPushButton('Update')
        self.button_update.pressed.connect(self.processPatch)
        self.layout_widget.addWidget(self.button_update,4,0,1,2)

       


        

        #widget outline
        widget_outline = QWidget()
        self.stackedWidget.insertWidget(1,widget_outline)

        self.layout_outline = QGridLayout(widget_outline)
        self.button_loadmarkups = QPushButton('Load Landmarks')
        self.button_loadmarkups.pressed.connect(self.loadLandamrk)
        self.layout_outline.addWidget(self.button_loadmarkups,0,0,1,2)

        self.button_curvepoint = QPushButton('Point Curve')
        self.button_curvepoint.pressed.connect(self.curvePoint)
        self.layout_outline.addWidget(self.button_curvepoint,1,0,1,2)  

        self.add_points = QPushButton('Resample points')
        self.add_points.pressed.connect(self.addPoints)
        self.layout_outline.addWidget(self.add_points,2,0) 

        self.spin_add_points = QSpinBox()
        self.spin_add_points.setMinimum(4)
        self.spin_add_points.setValue(4)
        self.layout_outline.addWidget(self.spin_add_points,2,1) 

        self.button_placepoint = QPushButton('Middle point')
        self.button_placepoint.pressed.connect(self.placeMiddlePoint)
        self.layout_outline.addWidget(self.button_placepoint,3,0,1,2)

        self.button_draw = QPushButton('Draw')
        self.button_draw.pressed.connect(self.draw)
        self.layout_outline.addWidget(self.button_draw,4,0,1,2)

        

        

        # page 2 : the mucogingival patch of the lower arch
        widget_mgl = QWidget()
        self.stackedWidget.insertWidget(2, widget_mgl)
        self.setupMGL(QVBoxLayout(widget_mgl))

        self.layout_file2 = QHBoxLayout()
        layout.addLayout(self.layout_file2)

        self.combobox_patch = QComboBox()
        self.combobox_patch.addItems(['1'])
        self.label_patch = QLabel("Patch : ")
        self.label_patch.setVisible(False)
        self.combobox_patch.setVisible(False)

        self.layout_file2.addWidget(self.label_patch)
        self.layout_file2.addWidget(self.combobox_patch)

        self.layout_file3 = QHBoxLayout()
        layout.addLayout(self.layout_file3)

        self.add_patch = QCheckBox()
        self.add_patch.stateChanged.connect(self.onCheckboxStateChanged)
        self.add_patch.setVisible(False)

        self.label_addpatch = QLabel("Create new patch : ")
        self.label_addpatch.setVisible(False)
        
        self.delete_patch = QPushButton(f'Delete patch')
        self.delete_patch.pressed.connect(self.deletPatch)
        self.delete_patch.setVisible(False)

        
        
        self.layout_file3.addWidget(self.label_addpatch)
        self.layout_file3.addWidget(self.add_patch)
        spacer = QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.layout_file3.addSpacerItem(spacer)
        self.layout_file3.addWidget(self.delete_patch)
        

        
        self.layout_file2.setStretchFactor(self.combobox_patch, 1)

        self.layout_label_display = QGridLayout()
        layout.addLayout(self.layout_label_display)
        self.label_time = QLabel(f'time')
        self.layout_label_display.addWidget(self.label_time)
        self.label_time.setVisible(False)

        self.label_sep = QLabel('_'*100)
        self.layout_label_display.addWidget(self.label_sep)
        self.label_sep.setVisible(True)

        

    # ---------------- lower arch : the mucogingival patch ----------------

    def setupMGL(self, layout):
        '''
        Build the lower-arch page : where the landmarks come from, which ones
        the joystick acts on, and the joystick itself.
        '''
        self.mgl_builder = MGLPatchBuilder()
        self.mgl_landmarks = None
        self.mgl_names = []
        # Horizontal pad axis : slide along the mucogingival line (towards
        # LL6 or LR6). Vertical : towards the crown or the vestibule.
        self.mgl_tangent = np.zeros(len(MGL_ORDER))
        self.mgl_apical = np.zeros(len(MGL_ORDER))
        self.mgl_heights_up = np.full(len(MGL_ORDER), DEFAULT_HEIGHT)
        self.mgl_heights_down = np.full(len(MGL_ORDER), DEFAULT_HEIGHT)
        self.mgl_checkboxes = {}
        self.mgl_preview_node = None
        self.mgl_markups_node = None
        self._mgl_pad_origin = (0.0, 0.0)

        # --- landmarks ---
        row_landmarks = QHBoxLayout()
        row_landmarks.addWidget(QLabel('Landmarks :'))
        self.lineedit_mgl_landmarks = QLineEdit()
        self.lineedit_mgl_landmarks.setPlaceholderText('MG landmarks json, or press Compute')
        row_landmarks.addWidget(self.lineedit_mgl_landmarks)
        self.button_mgl_browse = QPushButton('Select')
        self.button_mgl_browse.pressed.connect(self.selectMGLLandmarks)
        row_landmarks.addWidget(self.button_mgl_browse)
        self.button_mgl_compute = QPushButton('Compute')
        self.button_mgl_compute.setToolTip('Predict the MG landmarks of this scan with ALI')
        self.button_mgl_compute.pressed.connect(self.computeMGLLandmarks)
        row_landmarks.addWidget(self.button_mgl_compute)
        layout.addLayout(row_landmarks)

        # --- which landmarks the pad moves ---
        row_select = QHBoxLayout()
        row_select.addWidget(QLabel('Selection :'))
        for text, action in (('All', self.selectAllMGL), ('None', self.selectNoMGL),
                             ('R side', self.selectRightMGL), ('L side', self.selectLeftMGL)):
            button = QPushButton(text)
            button.pressed.connect(action)
            row_select.addWidget(button)
        layout.addLayout(row_select)

        # Toggle buttons over three rows : the R side, the midline point on
        # its own, then the L side -- distal to mesial reading left to right.
        point_rows = (
            [name for name in reversed(MGL_ORDER) if name.startswith('LR')],
            ['L0MG'],
            list(reversed([name for name in MGL_ORDER if name.startswith('LL')])),
        )
        for row_names in point_rows:
            row_points = QHBoxLayout()
            row_points.setSpacing(8)
            row_points.addStretch()
            for name in row_names:
                button = QPushButton(self.shortMGLName(name))
                button.setCheckable(True)
                button.setChecked(True)
                button.setToolTip(f'{name} : include in what the joystick moves')
                button.setFixedWidth(34)
                # An explicit checked state : the module stylesheet leaves
                # checked and unchecked buttons looking the same.
                button.setStyleSheet(
                    'QPushButton { background-color: #555555; color: #dddddd; '
                    'border: 1px solid #777777; border-radius: 4px; padding: 3px 0px; }'
                    'QPushButton:checked { background-color: #2e7dd1; color: white; '
                    'border: 1px solid #5dade2; font-weight: bold; }'
                    'QPushButton:disabled { background-color: #3a3a3a; color: #666666; }'
                )
                button.toggled.connect(self.onMGLSelectionChanged)
                self.mgl_checkboxes[name] = button
                row_points.addWidget(button)
            row_points.addStretch()
            layout.addLayout(row_points)

        # --- the joystick and the height it carries ---
        row_pad = QHBoxLayout()
        # The clinician faces the patient : dragging left pushes towards the
        # patient's right (LR), and dragging up pushes towards the crowns.
        # adjust_sign flips the vertical axis accordingly; the horizontal one
        # is flipped where the delta is applied, in onMGLPadMoved.
        self.mgl_pad = JoystickPad(
            outward_right=True,
            adjust_range=MGL_OFFSET_RANGE,
            adjust_sign=-1.0,
            ratio_range=(-MGL_OFFSET_RANGE, MGL_OFFSET_RANGE),
            side_labels=('LR', 'LL'),
            spring_back=True,
            # Twice the default pad : the whole travel is only 12 mm, so
            # every extra pixel is precision under the finger.
            size=2 * PAD_SIZE,
        )
        self.mgl_pad.setValues(0.0, 0.0)
        self.mgl_pad.setDefaults(0.0, 0.0)
        self.mgl_pad.onChanged = self.onMGLPadMoved
        self.mgl_pad.onReleased = self.onMGLPadReleased
        self.mgl_pad.onWheel = self.onMGLPadWheel
        self.mgl_pad.setToolTip(
            'Drag to push the checked landmarks ALONG the mucogingival line : '
            'left towards the LR6 end, right towards the LL6 end, exactly as '
            'the arch faces you. Up pushes towards the crowns, down towards '
            'the vestibule. The knob springs back to the centre : each drag '
            'is one push, they add up. The wheel changes the heights.'
        )
        row_pad.addWidget(self.mgl_pad)

        column_values = QVBoxLayout()
        # Slider to rough out the height, field to type it exactly : both
        # drive the checked landmarks and mirror each other, 0.1 mm per step.
        self.slider_mgl_height_up, self.lineedit_mgl_height_up = self.mglHeightControls(
            column_values, 'up', 'Upper extension (mm)',
            'How far the band climbs from the line towards the teeth, '
            'for the checked landmarks. Both extensions at 0 leave the '
            'landmarks alone, with no surface around them : the registration '
            'then runs on those points only, which is the control case.')
        self.slider_mgl_height_down, self.lineedit_mgl_height_down = self.mglHeightControls(
            column_values, 'down', 'Lower extension (mm)',
            'How far the band descends from the line towards the vestibule, '
            'for the checked landmarks. Both extensions at 0 leave the '
            'landmarks alone, with no surface around them : the registration '
            'then runs on those points only, which is the control case.')
        self.label_mgl_state = QLabel('')
        column_values.addWidget(self.label_mgl_state)
        column_values.addStretch()
        row_pad.addLayout(column_values)
        layout.addLayout(row_pad)

        row_buttons = QHBoxLayout()
        self.button_mgl_copy = QPushButton('Copy the patch of the fix scan')
        self.button_mgl_copy.setToolTip('Apply the offsets and heights of the scan above to this '
                                        'one. The landmarks are not copied : each scan keeps its own.')
        self.button_mgl_copy.pressed.connect(self.copyMGLParameters)
        # The fix scan is the one being copied from, so it has nothing to copy.
        self.button_mgl_copy.setVisible(self.title != 1)
        row_buttons.addWidget(self.button_mgl_copy)
        self.button_mgl_reset = QPushButton('Reset')
        self.button_mgl_reset.setToolTip('Put the checked landmarks back where ALI placed them')
        self.button_mgl_reset.pressed.connect(self.resetMGL)
        row_buttons.addWidget(self.button_mgl_reset)
        self.button_mgl_update = QPushButton('Update')
        self.button_mgl_update.pressed.connect(self.applyMGLPatch)
        row_buttons.addWidget(self.button_mgl_update)
        layout.addLayout(row_buttons)

        self.selectAllMGL()

    # --- selection ---

    @staticmethod
    def shortMGLName(name):
        '''R6..R1, 0, L1..L6 : what fits on a one-row button.'''
        base = name.replace('MG', '')
        if base == 'L0':
            return '0'
        return base.replace('LL', 'L').replace('LR', 'R')

    def selectedMGL(self):
        '''Indices, in MGL_ORDER, of the landmarks the pad acts on.'''
        return [index for index, name in enumerate(MGL_ORDER)
                if self.mgl_checkboxes[name].isChecked()]

    def mglIndices(self):
        '''Row, in the MGL_ORDER-indexed arrays, of each loaded landmark.

        The builder compacts the loaded landmarks while the offset and height
        arrays keep one row per possible name : indexing by name is what keeps
        the L0 checkbox driving the L0 point when a tooth is missing in the
        middle of the arch.
        '''
        return [MGL_ORDER.index(name) for name in self.mgl_names]

    def setMGLSelection(self, names):
        self._syncing = True
        try:
            for name, checkbox in self.mgl_checkboxes.items():
                checkbox.setChecked(name in names)
        finally:
            self._syncing = False
        self.onMGLSelectionChanged()

    def selectAllMGL(self):
        self.setMGLSelection(set(MGL_ORDER))

    def selectNoMGL(self):
        self.setMGLSelection(set())

    def selectLeftMGL(self):
        self.setMGLSelection({name for name in MGL_ORDER if name.startswith('LL')})

    def selectRightMGL(self):
        self.setMGLSelection({name for name in MGL_ORDER if name.startswith('LR')})

    def onMGLSelectionChanged(self, _state=None):
        '''
        The pad and the height field show the average of the selection, so the
        knob always sits where the checked points sit.
        '''
        if self._syncing:
            return
        selected = self.selectedMGL()
        self._syncing = True
        try:
            if selected:
                for side in ('up', 'down'):
                    heights, slider, lineedit = self.mglHeightWidgets(side)
                    mean = float(np.mean(heights[selected]))
                    lineedit.setText(f'{mean:.1f}')
                    slider.setValue(int(round(mean * 10)))
            # The pad is relative and rests at the centre : only the origin
            # follows it, the knob is never parked at a mean that could
            # saturate the travel.
            self._mgl_pad_origin = (self.mgl_pad.ratio, self.mgl_pad.adjust)
            self.label_mgl_state.setText(f'{len(selected)} / {len(MGL_ORDER)} selected')
        finally:
            self._syncing = False

    # --- edition ---

    def onMGLPadMoved(self, pad):
        '''
        The joystick moved : every checked landmark takes the same step, so the
        differences the user already dialled in are kept.
        '''
        if self._syncing:
            return
        selected = self.selectedMGL()
        if not selected:
            # Say why nothing moves instead of silently doing nothing.
            self.label_mgl_state.setText('0 selected : check a point below to move it')
            return
        delta_tangent = pad.ratio - self._mgl_pad_origin[0]
        delta_apical = pad.adjust - self._mgl_pad_origin[1]
        self._mgl_pad_origin = (pad.ratio, pad.adjust)
        # Dragging left must push towards the patient's right, i.e. towards
        # LR6, which is the POSITIVE end of the tangent axis : flip the sign.
        self.mgl_tangent[selected] -= delta_tangent
        self.mgl_apical[selected] += delta_apical
        self.markPreviewDirty()

    def onMGLPadReleased(self, pad):
        '''The knob sprang back to the centre : follow it, without a delta.'''
        self._mgl_pad_origin = (pad.ratio, pad.adjust)

    def mglHeightControls(self, layout, side, title, tooltip):
        '''One height : a label, a slider and a field mirroring each other.'''
        layout.addWidget(QLabel(title))
        row = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(MIN_HEIGHT * 10), int(MAX_HEIGHT * 10))
        slider.setValue(int(DEFAULT_HEIGHT * 10))
        slider.setToolTip(tooltip)
        slider.valueChanged.connect(partial(self.onMGLHeightSlid, side))
        row.addWidget(slider)
        lineedit = QLineEdit(f'{DEFAULT_HEIGHT:.1f}')
        lineedit.setMaximumWidth(56)
        lineedit.setToolTip(tooltip)
        lineedit.textEdited.connect(partial(self.onMGLHeightEdited, side))
        row.addWidget(lineedit)
        layout.addLayout(row)
        return slider, lineedit

    def mglHeightWidgets(self, side):
        '''The array, slider and field of one side of the band.'''
        if side == 'up':
            return self.mgl_heights_up, self.slider_mgl_height_up, self.lineedit_mgl_height_up
        return self.mgl_heights_down, self.slider_mgl_height_down, self.lineedit_mgl_height_down

    def onMGLHeightSlid(self, side, value):
        '''A height slider moved : a tenth of a millimetre per step.'''
        if self._syncing:
            return
        selected = self.selectedMGL()
        if not selected:
            return
        heights, _slider, lineedit = self.mglHeightWidgets(side)
        height = value / 10.0
        heights[selected] = height
        self._syncing = True
        try:
            lineedit.setText(f'{height:.1f}')
        finally:
            self._syncing = False
        self.markPreviewDirty()

    def onMGLPadWheel(self, _pad, steps):
        '''The wheel raises or lowers both heights of the checked landmarks;
        the two fields are there for setting each side on its own.'''
        selected = self.selectedMGL()
        if not selected:
            return
        for heights in (self.mgl_heights_up, self.mgl_heights_down):
            heights[selected] = np.clip(
                heights[selected] + 0.5 * steps, MIN_HEIGHT, MAX_HEIGHT)
        self._syncing = True
        try:
            for side in ('up', 'down'):
                heights, slider, lineedit = self.mglHeightWidgets(side)
                mean = float(np.mean(heights[selected]))
                lineedit.setText(f'{mean:.1f}')
                slider.setValue(int(round(mean * 10)))
        finally:
            self._syncing = False
        self.markPreviewDirty()

    def onMGLHeightEdited(self, side, _text=None):
        '''A height typed in applies to every checked landmark, on one side of
        the line only : towards the crown or towards the vestibule.'''
        if self._syncing:
            return
        selected = self.selectedMGL()
        heights, slider, lineedit = self.mglHeightWidgets(side)
        height = self.readFloat(lineedit)
        if height is None or not selected:
            return  # half-typed number, wait for the rest
        clamped = min(max(height, MIN_HEIGHT), MAX_HEIGHT)
        heights[selected] = clamped
        self._syncing = True
        try:
            slider.setValue(int(round(clamped * 10)))
        finally:
            self._syncing = False
        self.markPreviewDirty()

    def resetMGL(self):
        '''Undo every offset and height on the checked landmarks.'''
        selected = self.selectedMGL()
        if not selected:
            return
        self.mgl_tangent[selected] = 0.0
        self.mgl_apical[selected] = 0.0
        self.mgl_heights_up[selected] = DEFAULT_HEIGHT
        self.mgl_heights_down[selected] = DEFAULT_HEIGHT
        self.onMGLSelectionChanged()
        self.markPreviewDirty()

    # --- landmarks ---

    def selectMGLLandmarks(self):
        path = QFileDialog.getOpenFileName(slicer.util.mainWindow(), 'Open the MG landmarks', '', 'Markups (*.json *.mrk.json)')
        if path:
            self.lineedit_mgl_landmarks.setText(path)
            self.loadMGLLandmarks()

    def loadMGLLandmarks(self):
        '''Read the landmark file and hand it to the patch builder.'''
        path = self.lineedit_mgl_landmarks.text.strip()
        if not path or not os.path.isfile(path):
            return False
        if self.surf is None:
            self.viewScan()
        if self.surf is None:
            return False

        try:
            self.mgl_landmarks = ReadLandmarks(path)
        except Exception as error:
            logger.error(f"Could not read the MG landmarks: {error}")
            self.warning(f'This landmark file cannot be read.\n{error}')
            return False

        # The displayed mesh is in RAS : Slicer converts model files on load
        # (a .vtk with no metadata is assumed LPS), while ReadLandmarks reads
        # the raw positions of the json. Flip LPS landmarks the way Slicer's
        # own markups loader would, or the curve is a mirror of the arch.
        import json
        with open(path) as handle:
            system = json.load(handle)['markups'][0].get('coordinateSystem', 'LPS')
        if system.upper() == 'LPS':
            for name, position in self.mgl_landmarks.items():
                self.mgl_landmarks[name] = position * np.array([-1.0, -1.0, 1.0])

        # The mesh was then centered by viewScan (the matrix is kept) : bring
        # the landmarks along, or the curve is built beside the scan.
        matrix = self.getMatrix()
        if matrix is not None and self.camera:
            for name, position in self.mgl_landmarks.items():
                moved = matrix.MultiplyPoint((*position, 1.0))
                self.mgl_landmarks[name] = np.array(moved[:3])

        if not self.mgl_builder.prepare(self.surf.GetPolyData(), self.mgl_landmarks):
            self.warning(self.mgl_builder.error or 'The MG landmarks do not fit this scan.')
            return False

        self.mgl_names = self.mgl_builder.names()
        for name, checkbox in self.mgl_checkboxes.items():
            checkbox.setEnabled(name in self.mgl_names)
            if name not in self.mgl_names:
                checkbox.setChecked(False)
        self.showMGLLandmarks()
        self.markPreviewDirty()

        # Say which points ALI doubted rather than dropping them silently: the
        # curve is then drawn on fewer landmarks than the file holds, and the
        # user is the one who knows whether that stretch matters.
        doubtful = DoubtfulLandmarks(path)
        if doubtful:
            self.label_mgl_state.setText(
                f'{len(doubtful)} landmark(s) left out : {", ".join(sorted(doubtful))}')
            logger.info(f"MG landmarks left out of the curve: {doubtful}")
        return True

    def showMGLLandmarks(self):
        '''Show the MG points on the scan, at their joystick-moved positions.

        They are displayed as soon as the landmarks are loaded, follow the
        offsets the user dials in, and are locked : the joystick is what moves
        them, a mouse drag would be invisible to the patch engine.

        The positions shown are the moved points snapped back onto the mesh —
        the raw offsets leave the surface, and a point floating in the air or
        buried in the scan says nothing about where the band will be.
        '''
        if not self.mgl_builder.ready:
            return
        rows = self.mglIndices()
        positions = self.mgl_builder.snappedLandmarks(
            np.zeros(len(rows)), self.mgl_apical[rows],
            tangent_offsets=self.mgl_tangent[rows])

        node = self.mgl_markups_node
        if node is None or node.GetScene() is None:
            node = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLMarkupsFiducialNode', f'MG landmarks {self.title}')
            node.SetSaveWithScene(False)
            node.SetLocked(True)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetTextScale(2.5)
            display.SetGlyphScale(1.5)
            display.SetSelectedColor(1.0, 0.85, 0.0)
            display.SetPointLabelsVisibility(True)
            # A point on the far side of the arch stays faintly visible
            # instead of vanishing behind the scan.
            display.SetOccludedVisibility(True)
            display.SetOccludedOpacity(0.4)
            view = self.previewViewNode()
            if view:
                display.SetViewNodeIDs([view.GetID()])
            self.mgl_markups_node = node

        if node.GetNumberOfControlPoints() == len(positions):
            for index, position in enumerate(positions):
                node.SetNthControlPointPosition(index, *position)
        else:
            node.RemoveAllControlPoints()
            for name, position in zip(self.mgl_names, positions):
                node.AddControlPoint(vtk.vtkVector3d(*position), name.replace('MG', ''))
        node.SetDisplayVisibility(1)

    def computeMGLLandmarks(self):
        '''Predict the MG landmarks of this scan with ALI, in the conda env.'''
        path = str(self.lineedit.text)
        if not os.path.isfile(path):
            self.warning('Select the scan first.')
            return

        # ALI runs in the conda environment, so this is one of the places the
        # environment check has to happen.
        if not ensureBooted(self):
            return

        models = self.mglModelsFolder()
        if models is None:
            return

        output = os.path.join(tempfile.mkdtemp(), 'MGL_landmarks')
        os.makedirs(output, exist_ok=True)
        arguments = [path, models, 'None', 'None', ' '.join(MGL_ORDER), output,
                     '224', '0', '1', os.path.join(output, 'process.log')]

        progress = QProgressDialog('Predicting the MG landmarks with ALI...', None, 0, 0,
                                   slicer.util.mainWindow())
        progress.setWindowTitle('ALI_IOS')
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        # The prediction runs in a thread while this loop keeps pumping events:
        # a window shown while the event loop is blocked is never painted and
        # sits on the screen as a transparent ghost.
        thread = threading.Thread(target=self.logic.runALI, args=(arguments,))
        thread.start()
        try:
            while thread.is_alive():
                slicer.app.processEvents()
                time.sleep(0.05)
        finally:
            thread.join()
            progress.close()

        produced = sorted(f for f in os.listdir(output) if f.endswith('.json'))
        if not produced:
            self.warning('ALI did not produce any landmark file, see the log for details.')
            return

        self.lineedit_mgl_landmarks.setText(os.path.join(output, produced[0]))
        if self.loadMGLLandmarks() and len(self.mgl_names) < len(MGL_ORDER):
            missing = [name for name in MGL_ORDER if name not in self.mgl_names]
            self.warning(
                f'ALI only placed {len(self.mgl_names)} of the {len(MGL_ORDER)} MG landmarks.\n'
                f'Missing : {", ".join(missing)}.\n\n'
                'A landmark is only placed when its tooth has a label in the '
                'segmentation of the scan (Universal_ID / PredictedID). Check '
                'that the scan is segmented tooth by tooth with universal '
                'numbering, and see the ALI log for details.'
            )

    def mglModelsFolder(self):
        '''Folder holding Lower_MG_*.pth, downloading it once if need be.'''
        folder = os.path.join(self.SlicerDownloadPath, 'ALI', 'ALI_IOS', 'Models', 'Prediction')
        if self.hasMGLModel(folder):
            return folder

        chosen = QFileDialog.getExistingDirectory(slicer.util.mainWindow(), 'Select the ALI models folder')
        if chosen and self.hasMGLModel(chosen):
            return chosen
        if chosen:
            self.warning('No MGL model in that folder, a file named "Lower_MG_*.pth" is expected.')
        return None

    @staticmethod
    def hasMGLModel(folder):
        if not folder or not os.path.isdir(folder):
            return False
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.endswith('.pth') and 'Lower' in name and name.split('_')[1:2] == ['MG']:
                    return True
        return False

    # --- preview and patch ---

    def refreshMGLPreview(self):
        '''Repaint the band for the current offsets and heights.'''
        if not self.mgl_builder.ready:
            return
        rows = self.mglIndices()
        labels, _samples = self.mgl_builder.compute(
            np.zeros(len(rows)), self.mgl_apical[rows],
            self.mgl_heights_up[rows], self.mgl_heights_down[rows],
            tangent_offsets=self.mgl_tangent[rows],
        )
        self.showMGLPreview(labels)
        self.showMGLLandmarks()

    def showMGLPreview(self, labels):
        polydata = self.surf.GetPolyData()
        array = self.mgl_builder.toArray(labels, MGL_PREVIEW_ARRAY_NAME)
        polydata.GetPointData().AddArray(array)
        polydata.GetPointData().SetActiveScalars(MGL_PREVIEW_ARRAY_NAME)
        display = self.surf.GetDisplayNode()
        if display is not None:
            display.SetActiveScalarName(MGL_PREVIEW_ARRAY_NAME)
            display.SetScalarVisibility(True)
        polydata.Modified()

    def clearMGLPreview(self):
        if (self.mgl_markups_node is not None
                and self.mgl_markups_node.GetScene() is not None):
            self.mgl_markups_node.SetDisplayVisibility(0)
        if self.surf is None:
            return
        polydata = self.surf.GetPolyData()
        if polydata.GetPointData().GetArray(MGL_PREVIEW_ARRAY_NAME):
            polydata.GetPointData().RemoveArray(MGL_PREVIEW_ARRAY_NAME)
            polydata.Modified()

    def applyMGLPatch(self):
        '''
        Write the patch into the scan. The band is computed here rather than by
        the CLI: it needs no GPU, so what the user sees is exactly what is
        written.
        '''
        if not self.checkArchMatchesScan():
            return
        if not self.mgl_builder.ready:
            self.warning('Load or compute the MG landmarks first.')
            return

        rows = self.mglIndices()
        labels, _samples = self.mgl_builder.compute(
            np.zeros(len(rows)), self.mgl_apical[rows],
            self.mgl_heights_up[rows], self.mgl_heights_down[rows],
            tangent_offsets=self.mgl_tangent[rows],
        )
        if labels.sum() == 0:
            self.warning('The patch is empty, raise the height or move the '
                         'landmarks off the crowns.')
            return

        path = str(self.lineedit.text)
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        polydata = reader.GetOutput()

        if polydata.GetNumberOfPoints() != len(labels):
            self.warning('The scan on disk no longer matches the one displayed.')
            return

        polydata.GetPointData().AddArray(self.mgl_builder.toArray(labels, MGL_ARRAY_NAME))

        # For a future training set : keep the refined line itself, not only
        # its band. Vertex ids are coordinate-free, so everything is stored
        # against the file's own geometry, inside the file.
        ids = self.mgl_builder.snappedLandmarkIds(
            np.zeros(len(rows)), self.mgl_apical[rows],
            tangent_offsets=self.mgl_tangent[rows])
        field = polydata.GetFieldData()
        for name, values, dtype in (
                ('MGL_landmark_ids', ids, np.int64),
                ('MGL_landmark_slots', rows, np.int64),
                ('MGL_offsets_tangent', self.mgl_tangent[rows], np.float64),
                ('MGL_offsets_apical', self.mgl_apical[rows], np.float64),
                ('MGL_heights_crown', self.mgl_heights_up[rows], np.float64),
                ('MGL_heights_vestibule', self.mgl_heights_down[rows], np.float64)):
            array = numpy_to_vtk(np.asarray(values, dtype=dtype), deep=True)
            array.SetName(name)
            field.AddArray(array)

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(polydata)
        writer.Write()

        # The same points as a markups file, in the naming and coordinates of
        # the training annotations : ready to join the corpus.
        markups_path = path.rsplit('.vtk', 1)[0] + '_MG_edited.mrk.json'
        WriteLandmarks(markups_path, self.mgl_names,
                       [polydata.GetPoint(point_id) for point_id in ids])
        logger.info(f"Refined MGL line saved for training : {markups_path}")

        # The scene node carries it too: that is what the registration reads to
        # know the patch exists, and what the user is left looking at.
        displayed = self.surf.GetPolyData()
        displayed.GetPointData().AddArray(self.mgl_builder.toArray(labels, MGL_ARRAY_NAME))
        displayed.GetPointData().SetActiveScalars(MGL_ARRAY_NAME)
        display = self.surf.GetDisplayNode()
        if display is not None:
            display.SetActiveScalarName(MGL_ARRAY_NAME)
            display.SetScalarVisibility(True)
        displayed.Modified()

        self.preview_dirty = False
        self.button_mgl_update.setText('Update')
        logger.info(f"Wrote the {MGL_ARRAY_NAME} patch of {int(labels.sum())} points to {path}")

    def warning(self, text):
        qt.QMessageBox.warning(slicer.util.mainWindow(), 'Warning', text)

    def handleStackedWidgetChange(self, index):
        # When stackedWidget change of page, this is called.
        # Check if the new page is page 0 (index 0) and called hideLandmark if its the case.
        if index == 0:
            self.clearMGLPreview()
            self.hideLandmark()
            self.schedulePreview()
        elif index == 2:
            self.clearPreview()
            self.hideLandmark()
            self.schedulePreview()
        else :
            # The joystick preview has no meaning in the drawn-curve mode.
            self.clearPreview()
            self.clearMGLPreview()
            self.viewLandmark()

    def onCheckboxStateChanged(self):
        ''''
        Change state when checkbox is True
        '''
        if self.add_patch.isChecked():
            self.combobox_patch.setDisabled(True)
            self.delete_patch.setDisabled(True)
        else:
            self.combobox_patch.setDisabled(False)
            self.delete_patch.setDisabled(False)

    def getMainWidget(self):
        return self.main_widget
    
    def getName(self):
        return "WidgetParameter"
    
    def getSurf(self):
        return self.surf
    
    def changeMode(self,index):
        self.stackedWidget.setCurrentIndex(index)

    def isLowerArch(self):
        return self.lower_arch

    def scanIsLower(self):
        '''Which arch the loaded scan actually is, read from its segmentation.

        The lower teeth carry Universal_ID 18 to 31 and the upper ones 2 to 15,
        so the labels say it without trusting the file name.
        '''
        if self.surf is None:
            return None
        point_data = self.surf.GetPolyData().GetPointData()
        for name in ("Universal_ID", "PredictedID", "UniversalID"):
            scalars = point_data.GetScalars(name) or point_data.GetArray(name)
            if scalars is not None:
                labels = vtk_to_numpy(scalars)
                lower = int(np.isin(labels, range(18, 32)).sum())
                upper = int(np.isin(labels, range(2, 16)).sum())
                if lower == upper:
                    return None
                return lower > upper
        return None

    def checkArchMatchesScan(self):
        '''Warn when the scan is not the arch the panel is set to.

        The two patches are not interchangeable: the palate does not exist on a
        mandible, and the mucogingival model was only trained on one.
        '''
        is_lower = self.scanIsLower()
        if is_lower is None:
            return True
        if is_lower == self.isLowerArch():
            return True

        scan_arch = "lower" if is_lower else "upper"
        chosen = "lower" if self.isLowerArch() else "upper"
        self.warning(f'This scan is a {scan_arch} arch but the module is set to the '
                     f'{chosen} arch.\nSwitch the arch selector at the top before going on.')
        return False

    def setArch(self, lower):
        '''
        Applied from the module-level arch selector, to every scan at once.
        Upper keeps the two ways of drawing the palatal patch; lower has a
        single one, built from the landmarks, so the method selector goes away.
        '''
        self.lower_arch = lower
        self.combobox_choice_method.setVisible(not lower)
        if self.surf is not None:
            self.checkArchMatchesScan()
        self.stackedWidget.setCurrentIndex(2 if lower else self.combobox_choice_method.currentIndex)
        for widget in (self.label_patch, self.combobox_patch,
                       self.add_patch, self.label_addpatch, self.delete_patch):
            widget.setVisible(False if lower else widget.isVisible())

    def getPath(self):
        return self.lineedit.text
    
    def getTitle(self):
        return self.title
    
    def getCurve(self):
        return self.curve
    
    def getMiddle(self):
        return self.middle_point
    
    def getMatrix(self):
        return self.matrix
    
    def setCamera(self,b:bool):
        self.camera=b

    def deletPatch(self):
        '''
        Call the cli to delete a patch. Launch onProcessUpdateDelete
        '''
        if not ensureBooted(self):
            return

        index = int(self.combobox_patch.currentText)
        self._processed3 = False
        self.logic = FlexRegLogic(str(self.lineedit.text),
                            int(self.lineedit_teeth_left_top.text),
                        int(self.lineedit_teeth_right_top.text),
                        int(self.lineedit_teeth_left_bot.text),
                        int(self.lineedit_teeth_right_bot.text),
                        float(self.lineedit_ratio_left_top.text),
                        float(self.lineedit_ratio_right_top.text),
                        float(self.lineedit_ratio_left_bot.text),
                        float(self.lineedit_ratio_right_bot.text),
                        float(self.lineedit_adjust_left_top.text),
                        float(self.lineedit_adjust_right_top.text),
                        float(self.lineedit_adjust_left_bot.text),
                        float(self.lineedit_adjust_right_bot.text),
                        "None",
                        "None",
                        "delete",
                        "None",
                        "None",
                        "None",
                        index)
        self.logic.process()
        self.start_time = time.time()
        self.timer.timeout.connect(self.onProcessUpdateDelete)
        self.timer.start(500)
        
    def DownloadUnzip(
        self, url, directory, folder_name=None, num_downl=1, total_downloads=1
    ):
        """
        Download and unzip a file from a given URL to a specified directory.

        Parameters:
        - url: The URL of the zip file to download.
        - directory: The directory where the file should be downloaded and unzipped.
        - folder_name: The name of the folder to create and unzip the contents into.
        - num_downl: The current download number (for progress display).
        - total_downloads: The total number of downloads (for progress display).

        Returns:
        - out_path: The path to the unzipped folder.
        """
        
        out_path = os.path.join(directory, folder_name)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

            temp_path = os.path.join(directory, "temp.zip")

            # Download the zip file from the url
            with urllib.request.urlopen(url) as response, open(
                temp_path, "wb"
            ) as out_file:
                # Pop up a progress bar with a QProgressDialog
                progress = QProgressDialog(
                    "Downloading {} (File {}/{})".format(
                        folder_name.split(os.sep)[0], num_downl, total_downloads
                    ),
                    "Cancel",
                    0,
                    100,
                    self.parent,
                )
                progress.setCancelButton(None)
                progress.setWindowModality(Qt.WindowModal)
                progress.setWindowTitle(
                    "Downloading {}...".format(folder_name.split(os.sep)[0])
                )
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
                        QApplication.processEvents()
                shutil.copyfileobj(response, out_file) 

            # Unzip the file
            with zipfile.ZipFile(temp_path, "r") as zip:
                zip.extractall(out_path)

            # Delete the zip file
            os.remove(temp_path)

        return out_path
        
    def testFile(self):
        url = "https://github.com/GaelleLeroux/SlicerAutomatedDentalTools/releases/download/testfileFlexReg/TestFiles.zip"
        

        _ = self.DownloadUnzip(
            url=url,
            directory=os.path.join(self.SlicerDownloadPath),
            folder_name=os.path.join("FlexReg"),
            num_downl=1,
            total_downloads=1,
        )
        model_folder = os.path.join(self.SlicerDownloadPath,"FlexReg", "TestFiles")
        path_file = os.path.join(model_folder,f"T{self.title}_test_file.vtk")
        self.lineedit.setText(path_file)
        self.viewScan()

    def onProcessUpdateDelete(self):
        '''
        Update time since the beginning of the cli. When it's the end of the cli, display the patch and update combo box
        '''
        if hasattr(self, "_processed3") and self._processed3:
            return
        
        elapsed_time = time.time() - self.start_time
        self.label_time.setVisible(True)
        self.label_time.setText(f"Patch deletion, time : {round(float(elapsed_time),2)}s")

        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
            self.label_time.setText(f"Patch deleted, time : {round(float(elapsed_time),2)}s")
            self._processed3 = True
            self.timer.stop()
            self.viewScan()
            indexC = self.combobox_patch.findText(str(int(self.addItemsCombobox())-1))
            if indexC!=0:
                self.combobox_patch.removeItem(indexC)
            self.displaySegmentation(self.surf)
            



    def displayParamater(self,layout,number,parameter,corner,outward_right):
        '''
        One corner of the patch : a joystick pad plus the values it drives.
        The pad and the line edits stay in sync in both directions, so the
        numbers remain readable and typeable.
        '''
        label_teeth= QLabel(f'Teeth {number}')
        lineedit_teeth= QLineEdit(str(parameter[0]))
        label_ratio= QLabel('Ratio (R-L)')
        lineedit_ratio= QLineEdit(str(parameter[1]))
        label_adjust = QLabel('Adjust (A-P)')
        lineedit_adjust = QLineEdit(str(parameter[2]))

        for lineedit in (lineedit_teeth, lineedit_ratio, lineedit_adjust):
            lineedit.setMaximumWidth(64)

        pad = JoystickPad(outward_right=outward_right, adjust_range=ADJUST_RANGE,
                          adjust_sign=ADJUST_SIGN[corner])
        pad.setValues(float(parameter[1]), float(parameter[2]))
        pad.setDefaults(float(parameter[1]), float(parameter[2]))
        pad.onChanged = partial(self.onPadMoved, corner)

        layout.addWidget(pad,0,0,3,1)
        layout.addWidget(label_teeth,0,1)
        layout.addWidget(lineedit_teeth,0,2)
        layout.addWidget(label_ratio,1,1)
        layout.addWidget(lineedit_ratio,1,2)
        layout.addWidget(label_adjust,2,1)
        layout.addWidget(lineedit_adjust,2,2)

        self.pads[corner] = pad
        self.fields[corner] = (lineedit_teeth, lineedit_ratio, lineedit_adjust)

        lineedit_ratio.textChanged.connect(partial(self.onFieldEdited, corner))
        lineedit_adjust.textChanged.connect(partial(self.onFieldEdited, corner))
        lineedit_teeth.textChanged.connect(partial(self.onTeethEdited, corner))

        return lineedit_teeth, lineedit_ratio, lineedit_adjust

    def displayShift(self,layout):
        '''
        The translation pad : it slides the whole patch without touching its
        shape. Each corner is an affine combination of two tooth centroids
        whose weights sum to 1, so the same vector added to all four centroids
        comes straight back out of the interpolation and every corner moves
        together. Both axes are millimetres in the oriented frame : horizontal
        towards the patient's right or left, vertical anterior or posterior.
        '''
        label_title = QLabel('Move the whole patch')
        label_lr = QLabel('Shift (R-L)')
        lineedit_lr = QLineEdit('0.0')
        label_ap = QLabel('Shift (A-P)')
        lineedit_ap = QLineEdit('0.0')

        for lineedit in (lineedit_lr, lineedit_ap):
            lineedit.setMaximumWidth(64)

        # The horizontal axis is millimetres here, not a ratio, and both of its
        # ends are named : the left column of the panel holds the teeth on the
        # patient's right.
        pad = JoystickPad(outward_right=True, adjust_range=SHIFT_RANGE, size=SHIFT_PAD_SIZE,
                          ratio_range=(-SHIFT_RANGE, SHIFT_RANGE), side_labels=('R', 'L'))
        pad.setToolTip(
            'Drag to slide the whole patch. Its shape and size do not change.\n'
            'Horizontal : towards the patient right or left, in mm\n'
            'Vertical : anterior or posterior, in mm\n'
            'Ctrl+drag : five times finer\n'
            'Wheel : antero-posterior step, Shift+wheel : medio-lateral step\n'
            'Arrow keys : one step, double-click : back to no shift'
        )
        pad.setDefaults(0.0, 0.0)
        pad.onChanged = self.onShiftPadMoved

        self.button_copy = QPushButton('Copy the parameters of the fix scan')
        self.button_copy.setToolTip('Read every teeth, ratio, adjust and shift value of the '
                                    'scan above and apply them here.')
        self.button_copy.pressed.connect(self.copyParameters)
        # The fix scan is the one being copied from, so it has nothing to copy.
        self.button_copy.setVisible(self.title != 1)

        layout.addWidget(pad,0,0,4,1)
        layout.addWidget(label_title,0,1,1,2)
        layout.addWidget(label_lr,1,1)
        layout.addWidget(lineedit_lr,1,2)
        layout.addWidget(label_ap,2,1)
        layout.addWidget(lineedit_ap,2,2)
        layout.addWidget(self.button_copy,3,1,1,2)

        self.shift_pad = pad

        lineedit_lr.textChanged.connect(self.onShiftFieldEdited)
        lineedit_ap.textChanged.connect(self.onShiftFieldEdited)

        return lineedit_lr, lineedit_ap


    # ---- live preview ---------------------------------------------------

    def onPadMoved(self, corner, pad):
        '''A joystick moved : mirror it into the fields and repaint the patch.'''
        if self._syncing:
            return
        self._syncing = True
        try:
            _, lineedit_ratio, lineedit_adjust = self.fields[corner]
            lineedit_ratio.setText(f'{pad.ratio:.3f}')
            lineedit_adjust.setText(f'{pad.adjust:.2f}')
        finally:
            self._syncing = False
        self.markPreviewDirty()

    def onFieldEdited(self, corner, _text=None):
        '''A value was typed : mirror it into the joystick.'''
        if self._syncing:
            return
        _, lineedit_ratio, lineedit_adjust = self.fields[corner]
        ratio = self.readFloat(lineedit_ratio)
        adjust = self.readFloat(lineedit_adjust)
        if ratio is None or adjust is None:
            return  # half-typed number, wait for the rest
        self._syncing = True
        try:
            self.pads[corner].setValues(ratio, adjust)
        finally:
            self._syncing = False
        self.markPreviewDirty()

    def onTeethEdited(self, corner, _text=None):
        '''A different tooth invalidates the cached centroids.'''
        self.preview.clear()
        self.markPreviewDirty()

    def onShiftPadMoved(self, pad):
        '''The translation pad moved : mirror it into the two shift fields.'''
        if self._syncing:
            return
        self._syncing = True
        try:
            self.lineedit_shift_lr.setText(f'{pad.ratio:.2f}')
            self.lineedit_shift_ap.setText(f'{pad.adjust:.2f}')
        finally:
            self._syncing = False
        self.markPreviewDirty()

    def onShiftFieldEdited(self, _text=None):
        '''A shift was typed : mirror it into the translation pad.'''
        if self._syncing:
            return
        shift_lr, shift_ap = self.readFloat(self.lineedit_shift_lr), self.readFloat(self.lineedit_shift_ap)
        if shift_lr is None or shift_ap is None:
            return  # half-typed number, wait for the rest
        self._syncing = True
        try:
            self.shift_pad.setValues(shift_lr, shift_ap)
        finally:
            self._syncing = False
        self.markPreviewDirty()

    def shiftValues(self):
        '''
        Translation of the whole patch, in mm : (medio-lateral,
        antero-posterior). The fields are what counts, so that a value typed
        beyond the travel of the pad is still honoured; the pad only stands in
        while a number is half-typed.
        '''
        shift_lr, shift_ap = self.readFloat(self.lineedit_shift_lr), self.readFloat(self.lineedit_shift_ap)
        return (self.shift_pad.ratio if shift_lr is None else shift_lr,
                self.shift_pad.adjust if shift_ap is None else shift_ap)

    # ---- copying one scan onto another ----------------------------------

    def sourceWidget(self):
        '''The scan the Copy button reads from : the first one that is not us.'''
        for widget in self.scans:
            if widget is not self:
                return widget
        return None

    def parameterValues(self):
        '''Every value of the patch panel, as the text of its field.'''
        values = {corner: tuple(field.text for field in fields)
                  for corner, fields in self.fields.items()}
        values['shift'] = (self.lineedit_shift_lr.text, self.lineedit_shift_ap.text)
        return values

    def setParameterValues(self, values):
        '''
        Write values in through the line edits : their textChanged already
        mirrors them into the pads and schedules the preview. Untouched fields
        are left alone, so copying the same teeth does not throw away the
        cached centroids.
        '''
        for corner, fields in self.fields.items():
            for field, text in zip(fields, values[corner]):
                if field.text != text:
                    field.setText(text)

        for field, text in zip((self.lineedit_shift_lr, self.lineedit_shift_ap), values['shift']):
            if field.text != text:
                field.setText(text)

    def copyParameters(self):
        '''Take every value of the fix scan and apply it to this one.'''
        source = self.sourceWidget()
        if source is None:
            return
        self.setParameterValues(source.parameterValues())

    def mglParameterValues(self):
        '''The hand-set part of the mucogingival patch : offsets and heights.'''
        return (self.mgl_tangent.copy(), self.mgl_apical.copy(),
                self.mgl_heights_up.copy(), self.mgl_heights_down.copy())

    def copyMGLParameters(self):
        '''
        Take the offsets and heights of the fix scan and apply them here. The
        landmarks themselves are not copied : each scan keeps its own, so the
        same adjustments land on this scan's mucogingival line.
        '''
        source = self.sourceWidget()
        if source is None:
            return
        (self.mgl_tangent, self.mgl_apical,
         self.mgl_heights_up, self.mgl_heights_down) = source.mglParameterValues()
        self.onMGLSelectionChanged()
        self.markPreviewDirty()

    def readFloat(self, lineedit):
        try:
            return float(lineedit.text)
        except ValueError:
            return None

    def teethMapping(self):
        '''Corner -> tooth number, in the order butterflyPatch expects.'''
        try:
            return {
                'anterior_left': int(self.lineedit_teeth_left_top.text),
                'anterior_right': int(self.lineedit_teeth_right_top.text),
                'posterior_left': int(self.lineedit_teeth_left_bot.text),
                'posterior_right': int(self.lineedit_teeth_right_bot.text),
            }
        except ValueError:
            return None

    def markPreviewDirty(self):
        self.preview_dirty = True
        button = (self.button_mgl_update if self.stackedWidget.currentIndex == 2
                  else self.button_update)
        button.setText('Update   (preview not applied)')
        self.schedulePreview()

    def schedulePreview(self):
        '''
        Coalesce the redraws : a drag fires far more events than the preview
        needs, and one pending pass is always enough.
        '''
        if not self.preview_timer.isActive():
            self.preview_timer.start(15)

    def refreshPreview(self):
        if self.surf is None:
            return
        try:
            page = self.stackedWidget.currentIndex
        except ValueError:
            # the panel was closed while a redraw was still pending
            return
        if page == 2:
            self.refreshMGLPreview()
            return
        if page != 0:
            return

        teeth = self.teethMapping()
        if teeth is None:
            return

        if not self.preview.matches(teeth):
            if not self.preview.prepare(self.surf.GetPolyData(), teeth):
                self.setPreviewAvailable(False, self.preview.error)
                return
            self.setPreviewAvailable(True)
            self.alignPads()

        ratios = {corner: pad.ratio for corner, pad in self.pads.items()}
        adjusts = {corner: pad.adjust for corner, pad in self.pads.items()}

        try:
            contour, labels, _ = self.preview.compute(ratios, adjusts, self.shiftValues(),
                                                      with_fill=self.preview_dirty)
        except Exception as error:
            logging.warning(f'FlexReg : preview failed ({error})')
            return

        self.showContour(contour)
        if self.preview_dirty:
            self.showPreviewFill(labels)

    def alignPads(self):
        '''
        Point each pad's OUT side at the real exterior of the arch, read from
        the centroids rather than assumed from the grid position -- the teeth
        numbers are editable.
        '''
        for corner, centroid in self.preview.centroids().items():
            if corner in self.pads:
                self.pads[corner].setOutwardRight(centroid[0] > 0)

    def setPreviewAvailable(self, available, message=None):
        for pad in list(self.pads.values()) + [self.shift_pad]:
            pad.setPreviewEnabled(available)
        self.label_preview.setVisible(not available)
        if not available:
            self.label_preview.setText(f'No live preview : {message}')

    def previewViewNode(self):
        viewNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLViewNode')
        viewNodes.UnRegister(None)
        if viewNodes.GetNumberOfItems() >= self.title:
            return viewNodes.GetItemAsObject(self.title - 1)
        return None

    def showContour(self, polydata):
        if self.preview_node is None:
            node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', f'FlexReg preview {self.title}')
            node.SetSaveWithScene(False)
            node.CreateDefaultDisplayNodes()

            display = node.GetDisplayNode()
            display.SetColor(1.0, 0.78, 0.05)
            display.SetLineWidth(3)
            display.SetScalarVisibility(False)
            display.SetLighting(False)
            display.SetSelectable(False)

            view = self.previewViewNode()
            if view:
                display.SetViewNodeIDs([view.GetID()])

            self.preview_node = node

        self.preview_node.SetAndObservePolyData(polydata)

    def showPreviewFill(self, labels):
        '''
        Paint the approximate patch on the scan itself, under a name of its
        own so the real Butterfly array is left untouched.
        '''
        polydata = self.surf.GetPolyData()
        array = numpy_to_vtk(labels, deep=1)
        array.SetName('ButterflyPreview')
        polydata.GetPointData().AddArray(array)

        display = self.surf.GetModelDisplayNode()
        if display:
            display.SetActiveScalarName('ButterflyPreview')
            display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
            display.SetScalarRange(0.0, 1.0)
            display.SetScalarVisibility(True)
        polydata.Modified()

    def clearPreviewFill(self):
        '''Hand the display back to the real patch.'''
        if self.surf is None:
            return
        polydata = self.surf.GetPolyData()
        if polydata is None:
            return
        if polydata.GetPointData().GetArray('ButterflyPreview'):
            polydata.GetPointData().RemoveArray('ButterflyPreview')
        display = self.surf.GetModelDisplayNode()
        if display and polydata.GetPointData().GetArray('Butterfly'):
            display.SetActiveScalarName('Butterfly')

    def clearPreview(self):
        self.preview_timer.stop()
        self.clearPreviewFill()
        if self.preview_node is not None:
            slicer.mrmlScene.RemoveNode(self.preview_node)
            self.preview_node = None
        self.preview.clear()
        self.preview_dirty = False
        self.button_update.setText('Update')


    def selectFile(self):
        path_file = QFileDialog.getOpenFileName(self.parent,'Open a file','', 'Surfaces (*.vtk *.stl)')

        self.lineedit.setText(path_file)

    def sceneModels(self):
        '''Models of the scene this scan could use.

        FlexReg's own working copies and previews are pinned to one of its
        views, so a model restricted to specific views is one of ours and is
        left out; a user's model shows everywhere and has no such restriction.
        '''
        models = []
        for node in slicer.util.getNodesByClass('vtkMRMLModelNode'):
            if node.GetHideFromEditors() or node.GetPolyData() is None:
                continue
            display = node.GetDisplayNode()
            if display is not None and display.GetViewNodeIDs():
                continue
            models.append(node)
        return models

    def selectSceneModel(self):
        '''Offer the models already loaded in the scene as this scan.'''
        models = self.sceneModels()
        if not models:
            self.warning('No model is loaded in the scene.')
            return

        menu = qt.QMenu(self.button_scene)
        for node in models:
            storage = node.GetStorageNode()
            path = storage.GetFullNameFromFileName() if storage else None
            if path and os.path.isfile(path) and path.endswith('.vtk'):
                hint = path
            else:
                hint = 'no .vtk file, a copy will be written'
            action = menu.addAction(f'{node.GetName()}   ({hint})')
            action.triggered.connect(partial(self.useSceneModel, node.GetID()))
        menu.exec_(qt.QCursor.pos())

    def useSceneModel(self, node_id, _checked=False):
        '''
        Use a model already loaded in the scene as this scan. The whole
        pipeline works on files, so the node's own file is used when it is a
        readable .vtk; otherwise a .vtk snapshot of the node is written first.
        '''
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is None or node.GetPolyData() is None:
            return

        storage = node.GetStorageNode()
        path = storage.GetFullNameFromFileName() if storage else None
        if not path or not os.path.isfile(path) or not path.endswith('.vtk'):
            path = self.exportNodeToVtk(node)
            logger.info(f"Scene model '{node.GetName()}' written to {path}")

        # FlexReg displays its own centered copy: the original stays in the
        # scene but is hidden so the scan is not shown twice.
        node.SetDisplayVisibility(0)
        self.lineedit.setText(path)
        self.viewScan()

    def exportNodeToVtk(self, node):
        '''A .vtk snapshot of a scene model, as displayed (world coordinates).'''
        polydata = node.GetPolyData()
        if node.GetParentTransformNode() is not None:
            to_world = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                node.GetParentTransformNode(), None, to_world)
            transformer = vtk.vtkTransformPolyDataFilter()
            transformer.SetTransform(to_world)
            transformer.SetInputData(polydata)
            transformer.Update()
            polydata = transformer.GetOutput()

        folder = os.path.join(slicer.app.temporaryPath, 'FlexReg_scene_models')
        os.makedirs(folder, exist_ok=True)
        name = re.sub(r'[^\w-]+', '_', node.GetName()).strip('_') or 'model'
        path = os.path.join(folder, f'{name}.vtk')
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(polydata)
        # Binary rather than the ASCII default: three to five times faster to
        # write, and the reader takes both.
        writer.SetFileTypeToBinary()
        writer.Write()
        return path

    def checkLineEdit(self)->bool:
        '''
        check if input path is a surface this module can work on
        '''
        fname, extension = os.path.splitext(os.path.basename(self.lineedit.text))
        return extension.lower() in ('.vtk', '.stl')

    def ensureVtkInput(self):
        '''Turn a selected .stl into the .vtk everything downstream needs.

        The patch is a point array stored inside the scan file, and the STL
        format holds no arrays at all: the CLIs read and write .vtk. The scan
        is therefore converted once, beside the file the user picked so the
        patch stays with their data, and the panel then works on that copy.
        '''
        path = str(self.lineedit.text)
        if not path.lower().endswith('.stl'):
            return

        converted = os.path.splitext(path)[0] + '.vtk'
        if not os.path.isfile(converted):
            reader = vtk.vtkSTLReader()
            reader.SetFileName(path)
            reader.Update()
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(converted)
            writer.SetInputData(reader.GetOutput())
            writer.SetFileTypeToBinary()
            if not writer.Write():
                # Read-only folder: keep going from a copy of our own rather
                # than refusing the scan.
                folder = os.path.join(slicer.app.temporaryPath, 'FlexReg_converted')
                os.makedirs(folder, exist_ok=True)
                converted = os.path.join(folder, os.path.basename(converted))
                writer.SetFileName(converted)
                writer.Write()
            logger.info(f"{os.path.basename(path)} converted to {converted}: "
                        "the patch is stored in the scan file, which .stl cannot do")

        self.lineedit.setText(converted)


    def viewScan(self):
        '''
        Display the scan in the correct window. If scan already loaded, delete it and display the new one
        '''
        # No environment check here: showing a scan is vtk and Slicer, nothing
        # else. The conda probes it used to run cost about 7 s and froze the
        # panel before the scan appeared -- ensureBooted() now runs them from
        # the actions that really need the environment, the first time one of
        # them is used.

        if self.surf == None :
            if self.checkLineEdit():
                self.ensureVtkInput()
                # Load model
                self.surf = slicer.util.loadModel(self.lineedit.text)

                # Get data model
                displayNode = self.surf.GetDisplayNode()
                
                # Retrieve all availables vtkMRMLViewNodes in the scene
                viewNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLViewNode')
                viewNodes.UnRegister(None) # Unregister to avoid memory leakage
                
                customLayoutId=501
                layoutManager = slicer.app.layoutManager()
                layoutManager.setLayout(customLayoutId)

                viewNode = viewNodes.GetItemAsObject(self.title - 1) if viewNodes.GetNumberOfItems() >= self.title else None
                
                if viewNode:
                    # Display model in windows
                    displayNode.SetViewNodeIDs([viewNode.GetID()])

                else:
                    slicer.util.errorDisplay(f"There is 3D windows available with the index : {self.title - 1}.")

                # Get center of model. One GetPoint() per vertex costs 40 ms on
                # a 100k scan and 70 ms on a 170k one, for a mean numpy reads
                # off the same buffer in under 2 ms.
                points = self.surf.GetPolyData().GetPoints()
                center = list(vtk_to_numpy(points.GetData()).mean(axis=0).astype(float))


                # Get the focal point of the camera
                render_view = slicer.app.layoutManager().threeDWidget(0).threeDView()
                camera = render_view.renderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
                focal_point = camera.GetFocalPoint() 
                center[0]-=focal_point[0]
                center[1]-=focal_point[1]
                center[2]-=focal_point[2]


                # Create matrix to center the vtk
                matrix = vtk.vtkMatrix4x4()
                matrix.Identity()  
                matrix.SetElement(0, 3, -center[0])  
                matrix.SetElement(1, 3, -center[1])  
                matrix.SetElement(2, 3, -center[2])  

                self.matrix = matrix

                transform_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
                transform_node.SetMatrixTransformToParent(matrix)
                model = self.surf

                if self.camera :
                    model.SetAndObserveTransformNodeID(transform_node.GetID())
                    model.HardenTransform()

                self.displaySegmentation(self.surf)
                if not self.combobox_patch.isVisible():
                    self.displayComboBox(self.surf)

                # Outline the patch the current values describe, without
                # touching the colours of the patch already stored in the scan.
                self.preview_dirty = False
                self.schedulePreview()

            else:
                slicer.util.infoDisplay("Enter a path to a .vtk or .stl surface")


        else :
            self.clearPreview()
            viewNode1 = slicer.mrmlScene.GetSingletonNode(str(self.title), "vtkMRMLViewNode")
            modelNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
            modelNodes.InitTraversal()
            modelsToDelete = []
            for i in range(modelNodes.GetNumberOfItems()):
                modelNode = modelNodes.GetNextItemAsObject()
                modelDisplayNode = modelNode.GetDisplayNode()
    
                if modelDisplayNode and modelDisplayNode.GetViewNodeIDs() and viewNode1.GetID() in modelDisplayNode.GetViewNodeIDs():
                    modelsToDelete.append(modelNode)
          
            for model in modelsToDelete:
                slicer.mrmlScene.RemoveNode(model)
            
            self.surf = None
            self.viewScan()

        
    def displayComboBox(self,model_node):
        '''
        Display combobox
        Add number of element to match number of patch in the model
        '''
        index = 1
        polydata = model_node.GetPolyData()
        self.combobox_patch.clear()
        self.combobox_patch.addItem("1")
        while True:
            array_name = f"Butterfly{index}"
            
            if self.isButterflyPatchAvailable(polydata,array_name):
                if index==1:
                    self.label_patch.setVisible(True)
                    self.combobox_patch.setVisible(True)
                    self.delete_patch.setVisible(True)
                    self.label_addpatch.setVisible(True)
                    self.add_patch.setVisible(True)
                
                else : 
                    self.combobox_patch.addItem(str(index))

                
                index += 1
            else:
                break

        


    def checkSurfExist(self)->bool:
        return not (self.surf==None)
    
    def update_message_box(self,msg_box, start_time):
        elapsed_time = time.time() - start_time
        msg_box.setText(f"Your file wasn't segmented.\nSegmentation in process. This task may take a few minutes.\ntime: {elapsed_time:.1f}s")

    def downloadModel(self):
        '''
        Download the latest model to do the segmentation of the teeth
        '''
        url = "https://github.com/DCBIA-OrthoLab/Fly-by-CNN/releases/download/3.0/07-21-22_val-loss0.169.pth"
        name = "Model_segmentation_teeh.pth"

        documentsLocation = QStandardPaths.DocumentsLocation
        documentsPath = QStandardPaths.writableLocation(documentsLocation)

        # Path for Slicer downloads
        slicerDownloadPath = os.path.join(documentsPath, slicer.app.applicationName + "Downloads")

        # Create the directory if it does not exist
        if not os.path.exists(slicerDownloadPath):
            os.makedirs(slicerDownloadPath)

        # Full path where the file will be saved
        modelFilePath = os.path.join(slicerDownloadPath, name)

        # Download the file
        if not os.path.isfile(modelFilePath):
            slicer.util.downloadFile(url, modelFilePath)

        # Now you can use the downloaded model file path as needed
        logger.info(f"Model file downloaded to: {modelFilePath}")
        return modelFilePath
    
    def checkSegmentation(self)->bool:
        '''
        This function is doing the first step of makebutterfly to be sure the segmentation and the tooth are existing.
        If the segmentation is not existing, calling the module crownsegmentation to do it
        '''
        # The scan is read here and segmented in place below, so it has to be
        # the .vtk copy and never the .stl the user selected: handed an .stl to
        # overwrite, the segmentation deletes it (shapeaxi, dental_model_seg.py
        # "if ext == '.stl': os.remove(args.stl)"). viewScan already converted,
        # this is for a path edited afterwards.
        self.ensureVtkInput()

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(self.lineedit.text))
        reader.Update()
        modelNode = reader.GetOutput()

        # Transform the data to read it in coordinate RAS (like slicer)
        transform = vtk.vtkTransform()
        transform.Scale(-1, -1, 1)

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(modelNode)
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        modelNode = transformFilter.GetOutput()
        surf_tmp = vtk.vtkPolyData()
        surf_tmp.DeepCopy(modelNode)

        try :
            surf_tmp = orientation_f(surf_tmp,[[-0.5,-0.5,0],[0,0,0],[0.5,-0.5,0]],
                                    ['3','5','12','14'])
            return True

        except ToothNoExist as error :
            slicer.util.infoDisplay(f' Error : {error}')
            return False
        
        except NoSegmentationSurf as error :
            sucess_segmentation = self.shapeaxi_conda()
            if sucess_segmentation:
                self.viewScan()
                # msg_box.hide()
                return True
            return False
        
    def check_lib_wsl(self) -> bool:
        # Ubuntu versions under 24.04
        required_libs_old = ["libxrender1", "libgl1-mesa-glx"]
        # Ubuntu versions after 24.04
        required_libs_new = ["libxrender1", "libgl1", "libglx-mesa0"]


        all_installed = lambda libs: all(
            subprocess.run(
                f"wsl -- bash -c \"dpkg -l | grep {lib}\"", capture_output=True, text=True
            ).stdout.encode("utf-16-le").decode("utf-8").replace("\x00", "").find(lib) >= 0
            for lib in libs
        )

        return all_installed(required_libs_old) or all_installed(required_libs_new)
            
    def shapeaxi_conda(self):
        slicer.app.processEvents()

        # Segmenting overwrites the file it is given, and an .stl would be
        # deleted rather than written to.
        self.ensureVtkInput()
        
        output_command = self.logic.conda.condaRunCommand(["which","dentalmodelseg"],self.logic.name_env).strip()
        clean_output = re.search(r"Result: (.+)", output_command)
        if clean_output is None:
            # 'which' found nothing, which means shapeaxi is missing from the
            # environment. Reading .group(1) off None here reported an
            # AttributeError instead of the actual problem.
            logger.error(f"dentalmodelseg not found in '{self.logic.name_env}' : {output_command}")
            slicer.util.errorDisplay(
                f"The conda environment '{self.logic.name_env}' does not provide "
                "'dentalmodelseg', so the scan cannot be segmented automatically.\n\n"
                "The shapeaxi package is missing from that environment. Delete it "
                "and let the module rebuild it, or segment the scan beforehand so "
                "it carries a Universal_ID array."
            )
            return False
        dentalmodelseg_path = clean_output.group(1).strip()
        dentalmodelseg_path_clean = dentalmodelseg_path.replace("\\n","")
        
        args = [self.lineedit.text,                 #surf
                "None",                             #input_csv
                os.path.dirname(self.lineedit.text),#out
                "1",                                #overwrite
                "latest",                           #model
                "0",                                #crownsegmentation
                "Universal_ID",                     #array_name
                "0",                                #fdi
                "None",                             #suffix 
                os.path.dirname(self.lineedit.text),#vtk_folder
                dentalmodelseg_path_clean]          #dentalmodelseg_path

        
        conda_exe = self.logic.conda.getCondaExecutable()
        command = [conda_exe, "run", "-n", self.logic.name_env, "python" ,"-m", f"CrownSegmentationcli"]
        for arg in args :
            command.append("\""+arg+"\"")

        # running in // to not block Slicer
        process = threading.Thread(target=self.logic.conda.condaRunCommand, args=(command,))
        process.start()
        self.label_time.setVisible(True)
        self.label_time.setText(f"Your file wasn't segmented.\nSegmentation in process. This task may take a few minutes.\ntime: 0.0s")
        start_time = time.time()
        previous_time = start_time
        while process.is_alive():
            slicer.app.processEvents()
            current_time = time.time()
            gap=current_time-previous_time
            if gap>0.3:
                previous_time = current_time
                elapsed_time = current_time - start_time
                self.label_time.setText(f"Your file wasn't segmented.\nSegmentation in process. This task may take a few minutes.\ntime: {elapsed_time:.1f}s")
        
        self.viewScan()

        return True

    def parall_process(self,function,arguments=[],message=""):
        '''
        to be able to run function in parralle with a message
        '''
        process = threading.Thread(target=function, args=tuple(arguments)) #run in paralle to not block slicer
        process.start()
        start_time = time.time()
        previous_time = time.time()
        self.label_time.setVisible(True)
        self.label_time.setText(f"{message}\ntime: 0s")
        while process.is_alive():
          slicer.app.processEvents()
          current_time = time.time()
          gap=current_time-previous_time
          if gap>0.3:
              previous_time = current_time
              elapsed_time = current_time - start_time
              self.label_time.setText(f"{message}\ntime: {elapsed_time:.1f}s")
              
    def onCheckRequirements(self):
        self.label_time.setHidden(False)
        
        if not self.logic.isCondaSetUp:
            messageBox = qt.QMessageBox()
            text = textwrap.dedent("""
            SlicerConda is not set up, please click 
            <a href=\"https://github.com/DCBIA-OrthoLab/SlicerConda/\">here</a> for installation.
            """).strip()
            messageBox.information(None, "Information", text)
            return False
        
        if platform.system() == "Windows":
            self.label_time.setText(f"Checking if wsl is installed, this task may take a moments")
            
            if self.logic.conda.testWslAvailable():
                self.label_time.setText(f"WSL installed")
                if not self.logic.check_lib_wsl():
                    self.label_time.setText(f"Checking if the required librairies are installed, this task may take a moments")
                    messageBox = qt.QMessageBox()
                    text = textwrap.dedent("""
                        WSL doesn't have all the necessary libraries, please download the installer 
                        and follow the instructions 
                        <a href=\"https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_WSL2.zip\">here</a> 
                        for installation. The link may be blocked by Chrome, just authorize it.""").strip()

                    messageBox.information(None, "Information", text)
                    return False
                
            else : # if wsl not install, ask user to install it ans stop process
                messageBox = qt.QMessageBox()
                text = textwrap.dedent("""
                    WSL is not installed, please download the installer and follow the instructions 
                    <a href=\"https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_WSL2.zip\">here</a> 
                    for installation. The link may be blocked by Chrome, just authorize it.""").strip()        

                messageBox.information(None, "Information", text)
                return False
            
        
        ## MiniConda
        
        
        self.label_time.setText(f"Checking if miniconda is installed")
        if "no setup" in self.logic.conda.condaRunCommand([self.logic.conda.getCondaExecutable(),"--version"]):
            messageBox = qt.QMessageBox()
            text = textwrap.dedent("""
            Code can't be launch. \nConda is not setup. 
            Please go the extension CondaSetUp in SlicerConda to do it.""").strip()
            messageBox.information(None, "Information", text)
            return False
        
        
        ## shapeAXI


        self.label_time.setText(f"Checking if environnement exists")
        if not self.logic.conda.condaTestEnv(self.logic.name_env) : # check is environnement exist, if not ask user the permission to do it
            userResponse = slicer.util.confirmYesNoDisplay("The environnement to run the classification doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
            if userResponse :
                start_time = time.time()
                previous_time = start_time
                formatted_time = self.format_time(0)
                self.label_time.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
                process = self.logic.install_shapeaxi()
                
                while self.logic.process.is_alive():
                    slicer.app.processEvents()
                    formatted_time = self.update_ui_time(start_time, previous_time)
                    self.label_time.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
            
                start_time = time.time()
                previous_time = start_time
                formatted_time = self.format_time(0)
                text = textwrap.dedent(f"""
                Installation of librairies into the new environnement. 
                This task may take a few minutes.\ntime: {formatted_time}""").strip()
                self.label_time.setText(text)
            else:
                return False
        else:
            self.label_time.setText(f"Ennvironnement already exists")
            
        
        ## pytorch3d


        self.label_time.setText(f"Checking if pytorch3d is installed")
        if not self.logic.install_pytorch3d():
            slicer.util.errorDisplay(
                "The pytorch3d installation could not be started : the conda "
                f"environment '{self.logic.name_env}' cannot import "
                "FlexReg_utils.install_pytorch.\n\n"
                "See the Python console for the underlying import error."
            )
            return False
        start_time = time.time()
        previous_time = start_time
        
        while self.logic.process.is_alive():
            slicer.app.processEvents()
            formatted_time = self.update_ui_time(start_time, previous_time)
            text = textwrap.dedent(f"""
            Installation of pytorch into the new environnement. 
            This task may take a few minutes.\ntime: {formatted_time}
            """).strip()
            self.label_time.setText(text)

        self.all_installed = True   
        return True
            
    def format_time(self,seconds):
        """ Convert seconds to H:M:S format. """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"
    
    def update_ui_time(self, start_time, previous_time):
        current_time = time.time()
        gap=current_time-previous_time
        if gap>0.3:
            previous_time = current_time
            self.elapsed_time = current_time - start_time
            formatted_time = self.format_time(self.elapsed_time)
            return formatted_time

    def shapeaxi(self):
        '''
        run shapeaxi (segmentation of the crown, dentalmodelseg) in slicer (for Linux system)
        '''
        slicer_path = slicer.app.applicationDirPath()
        dentalmodelseg_path = os.path.join(slicer_path,"..","lib","Python","bin","dentalmodelseg")

        moduleName = "CrownSegmentation"
        moduleAvailable = moduleName in slicer.app.moduleManager().modulesNames()
        self._processed2 = False
        if moduleAvailable : 
            parameters = {
                "surf" :self.lineedit.text,
                "input_csv":"None",
                "out" : "None",
                "overwrite":"1",
                "model": "latest",
                "crown_segmentation" : "0",
                "array_name":"Universal_ID",
                "fdi":"0",
                "suffix":"None",
                "vtk_folder":os.path.dirname(self.lineedit.text),
                "dentalmodelseg_path":dentalmodelseg_path
            }
            self.start_time = time.time()
            flybyProcess = slicer.modules.crownsegmentationcli
            self.start_time = time.time()
            try:
                self.timer.timeout.disconnect()
            except TypeError:
                pass
            self.timer.timeout.connect(self.onProcessUpdateSeg)
            self.timer.start(500)
            self.seg_clinode = slicer.cli.run(flybyProcess,None, parameters)    
            
            self._segmentationCompleted = False
            while not self._segmentationCompleted:
                slicer.app.processEvents()  # Process GUI events
            return True
            
        return True

            
    def onProcessUpdateSeg(self):
        '''
        Update time since the beginning of the segmentation. When it's the end of it, load the new scan segmented
        '''
        if hasattr(self, "_processed2") and self._processed2:
            return
        
        elapsed_time = time.time() - self.start_time
        self.label_time.setVisible(True)
        self.label_time.setText(f"Your file wasn't segmented.\nSegmentation in process. This task may take a few minutes.\ntime: {elapsed_time:.1f}s")


        if self.seg_clinode.GetStatus() & self.seg_clinode.Completed:
            self._processed2 = True
            self.timer.stop()
            self.viewScan() 
            self._segmentationCompleted = True
            

    def processPatch(self)->None:
        '''
        Call the cli for the butterfly patch. Launch onProcessUpdateButterfly
        '''
        if self.checkSurfExist() :
            if not ensureBooted(self):
                return
            seg = self.checkSegmentation()
            if seg:
                self._processed2 = False
                if self.add_patch.isChecked():
                    index=int(self.addItemsCombobox())
                else:
                    index=int(self.combobox_patch.currentText)

                self.logic = FlexRegLogic(str(self.lineedit.text),
                                          
                                int(self.lineedit_teeth_left_top.text),
                            int(self.lineedit_teeth_right_top.text),
                            int(self.lineedit_teeth_left_bot.text),
                            int(self.lineedit_teeth_right_bot.text),

                            float(self.lineedit_ratio_left_top.text),
                            float(self.lineedit_ratio_right_top.text),
                            float(self.lineedit_ratio_left_bot.text),
                            float(self.lineedit_ratio_right_bot.text),

                            float(self.lineedit_adjust_left_top.text),
                            float(self.lineedit_adjust_right_top.text),
                            float(self.lineedit_adjust_left_bot.text),
                            float(self.lineedit_adjust_right_bot.text),
                            "None",
                            "None",
                            "butterfly",
                            "None",
                            "None",
                            "None",
                            index,
                            "None",
                            *self.shiftValues())
                self.logic.process()
                self.start_time = time.time()
                try:
                    self.timer.timeout.disconnect()
                except TypeError:
                    pass
                self.timer.timeout.connect(self.onProcessUpdateButterfly)
                self.timer.start(500)
        else :
            slicer.util.infoDisplay(f"Load a vtk file in window number : {self.title} \nTo do this, enter the path to a vtk file and click on view.")


    def onProcessUpdateButterfly(self):
        '''
        Update time since the beginning of the cli. When it's the end of the cli, display the patch
        '''
        if hasattr(self, "_processed2") and self._processed2:
            return
        
        elapsed_time = time.time() - self.start_time
        self.label_time.setVisible(True)
        self.label_time.setText(f"Creation of the patch, time : {round(float(elapsed_time),2)}s")

        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
            self.label_time.setText(f"Patch created, time : {round(float(elapsed_time),2)}s")
            self._processed2 = True
            self.timer.stop()
            self.viewScan()
            self.displaySegmentation(self.surf)
            # The real patch is now on screen : stop overriding it with the
            # approximate fill, and keep the outline as a reference.
            self.preview_dirty = False
            self.button_update.setText('Update')
            self.schedulePreview()
            if self.add_patch.isChecked():
                number_to_add = self.addItemsCombobox()
                self.combobox_patch.addItem(number_to_add)
                self.add_patch.setChecked(False)
                index = self.combobox_patch.findText(number_to_add)  
                if index >= 0:  # -1 signify that the value hasn't been found
                    self.combobox_patch.setCurrentIndex(index)
            if not self.combobox_patch.isVisible():
                self.displayComboBox(self.surf)
            

    def loadLandamrk(self)->None:
        '''
        Load the landmars creating the curve. Center it in the middle of the load model
        '''
        
        bounding_box = [0, 0, 0, 0, 0, 0]
        self.surf.GetRASBounds(bounding_box)
        center = [(bounding_box[1] + bounding_box[0]) / 2, (bounding_box[3] + bounding_box[2]) / 2, (bounding_box[5] + bounding_box[4]) / 2]

        self.curve = slicer.app.mrmlScene().AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode", f'T{self.title} curve')

        self.curve.AddControlPoint([center[0]+10,center[1]-10,center[2]-5],f'F1')
        self.curve.AddControlPoint([center[0]+10,center[1]+10,center[2]-5],f'F2')
        self.curve.AddControlPoint([center[0]-10,center[1]+10,center[2]-5],f'F3')
        self.curve.AddControlPoint([center[0]-10,center[1]-10,center[2]-5],f'F4')
        
        self.viewLandmark()
        



    def viewLandmark(self)->None:
        '''
        Display the landmarks
        '''
        viewNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLViewNode')
        viewNodes.UnRegister(None)  # Unregister to avoid memory leakage

        if self.curve!=None:
            displayNode = self.curve.GetDisplayNode()
            if displayNode is not None:
                displayNode.SetVisibility2D(False)
                displayNode.SetVisibility3D(True)

                view_ids_to_display = [viewNodes.GetItemAsObject(self.title-1).GetID()]
                displayNode.SetViewNodeIDs(view_ids_to_display)

        if self.middle_point!=None:
            displayNode = self.middle_point.GetDisplayNode()
            if displayNode is not None:
                displayNode.SetVisibility2D(False)
                displayNode.SetVisibility3D(True)
                view_ids_to_display = [viewNodes.GetItemAsObject(self.title-1).GetID()]
                displayNode.SetViewNodeIDs(view_ids_to_display)

    def hideLandmark(self) -> None:
        '''
        Hide the landmarks
        '''
        viewNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLViewNode')
        viewNodes.UnRegister(None)  # Unregister to avoid memory leakage

        if self.curve!=None :
            displayNode = self.curve.GetDisplayNode()
            if displayNode is not None:
                displayNode.SetVisibility2D(True)  #Restore 2D view
                displayNode.SetVisibility3D(False)  # Hide 3D view

                view_ids_to_display = [viewNodes.GetItemAsObject(self.title-1).GetID()]
                displayNode.SetViewNodeIDs(view_ids_to_display)

        if self.middle_point!=None :
            displayNode = self.middle_point.GetDisplayNode()
            if displayNode is not None:
                displayNode.SetVisibility2D(True)  #Restore 2D view
                displayNode.SetVisibility3D(False)  # Hide 3D view

                view_ids_to_display = [viewNodes.GetItemAsObject(self.title-1).GetID()]
                displayNode.SetViewNodeIDs(view_ids_to_display)



    def curvePoint(self)->None:
        '''
        Match the points with the load model 
        '''

        self.curve.SetAndObserveSurfaceConstraintNode(self.surf)
        self.glue=True
        

        

    def addPoints(self)->None:
        '''
        Resample the curve with more control points.
        '''
        # Get your curve node
        curveNode = self.curve
        curvePolyData = curveNode.GetCurveWorld()
        points = curvePolyData.GetPoints()

        # Create splines to interpolate curve points
        splineX = vtk.vtkCardinalSpline()
        splineY = vtk.vtkCardinalSpline()
        splineZ = vtk.vtkCardinalSpline()

        # Add curve points to splines
        for i in range(points.GetNumberOfPoints()):
            p = points.GetPoint(i)
            splineX.AddPoint(i, p[0])
            splineY.AddPoint(i, p[1])
            splineZ.AddPoint(i, p[2])

        # Determine the desired number of points
        numberOfPoints = self.spin_add_points.value
        newCurveNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsClosedCurveNode',f'T{self.title} curve')

        # Evaluate the splines at regular intervals to obtain the new set of points
        for i in range(numberOfPoints):
            u = i / (numberOfPoints - 1.0) * (points.GetNumberOfPoints() - 1)
            if i == numberOfPoints-1:
                u = u -(points.GetNumberOfPoints() - 1)/(numberOfPoints*2)
            x = splineX.Evaluate(u)
            y = splineY.Evaluate(u)
            z = splineZ.Evaluate(u)
            newCurveNode.AddControlPoint(vtk.vtkVector3d(x, y, z))

        # If you wish, you can now delete the old curve node
        self.curve = newCurveNode
        slicer.mrmlScene.RemoveNode(curveNode)
        self.viewLandmark()
        if self.glue:
            self.curve.SetAndObserveSurfaceConstraintNode(self.surf)


    def placeMiddlePoint(self)->None:
        '''
        Place the middle point for the curve patch 
        '''

        bounding_box = [0, 0, 0, 0, 0, 0]
        self.surf.GetRASBounds(bounding_box)
        center = [(bounding_box[1] + bounding_box[0]) / 2, (bounding_box[3] + bounding_box[2]) / 2, (bounding_box[5] + bounding_box[4]) / 2]

        self.middle_point = slicer.app.mrmlScene().AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")

        self.middle_point.AddControlPoint(center,'F1')

        viewNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLViewNode')
        viewNodes.UnRegister(None)  # Unregister to avoid memory leakage

        displayNode = self.middle_point.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetVisibility2D(False)
            displayNode.SetVisibility3D(True)
            view_ids_to_display = [viewNodes.GetItemAsObject(self.title-1).GetID()]
            displayNode.SetViewNodeIDs(view_ids_to_display)


    def moveCurve(self,matrix)->None:
        '''
        apply the matrix to the landmarks
        '''
        transform_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
        transform_node.SetMatrixTransformToParent(matrix)

        self.curve.SetAndObserveTransformNodeID(transform_node.GetID())
        self.curve.HardenTransform()
        self.middle_point.SetAndObserveTransformNodeID(transform_node.GetID())
        self.middle_point.HardenTransform() 


    def draw(self)->None:
        '''
        launch the cli for the curve patch and lauch onProcessUpdateCurve
        '''
        if self.checkSurfExist():
            if not ensureBooted(self):
                return
            self._processed = False
            
            # Move the curve and the middle point where the original model is located
            inverse_matrix = vtk.vtkMatrix4x4()

            # Calculate invert matrix to reg curve and middle point with model not center in front of the camera
            inverse_matrix.DeepCopy(self.getMatrix()) 
            inverse_matrix.Invert()

            self.moveCurve(inverse_matrix)
            self.camera=False
            self.viewScan()
            self.curve.SetAndObserveSurfaceConstraintNode(self.surf)

            middle_point_vector3D = self.middle_point.GetNthControlPointPositionWorld(0)
            
            # put the data in str type
            vector_middle = ','.join([str(middle_point_vector3D.GetX()), str(middle_point_vector3D.GetY()), str(middle_point_vector3D.GetZ())])
            list_curve = list(vtk_to_numpy(self.curve.GetCurvePointsWorld().GetData()))
            list_curve_str = ','.join(map(str, list_curve))   
            vector_middle="["+vector_middle+"]"

            if self.add_patch.isChecked():
                index=int(self.addItemsCombobox())
            else:
                index=int(self.combobox_patch.currentText)

            # CLI 
            self.logic = FlexRegLogic(str(self.lineedit.text),
                            int(self.lineedit_teeth_left_top.text),
                        int(self.lineedit_teeth_right_top.text),
                        int(self.lineedit_teeth_left_bot.text),
                        int(self.lineedit_teeth_right_bot.text),
                        float(self.lineedit_ratio_left_top.text),
                        float(self.lineedit_ratio_right_top.text),
                        float(self.lineedit_ratio_left_bot.text),
                        float(self.lineedit_ratio_right_bot.text),
                        float(self.lineedit_adjust_left_top.text),
                        float(self.lineedit_adjust_right_top.text),
                        float(self.lineedit_adjust_left_bot.text),
                        float(self.lineedit_adjust_right_bot.text),
                        list_curve_str,
                        vector_middle,
                        "curve",
                        "None",
                        "None",
                        "None",
                        index,
                        "None")
            self.logic.process()

            self.start_time = time.time()
            try:
                self.timer.timeout.disconnect()
            except TypeError:
                pass
            self.timer.timeout.connect(self.onProcessUpdateCurve)
            self.timer.start(500)

        else :
            slicer.util.infoDisplay(f"Load a vtk file in window number : {self.title} \nTo do this, enter the path to a vtk file and click on view.")


        


    def onProcessUpdateCurve(self)->None:
        ''''
         Update time since the beginning of the cli. When it's the end of the cli, display the patch and move the curve at their original place
        '''
    # If already processed, do nothing.
        if hasattr(self, "_processed") and self._processed:
            return

        elapsed_time = time.time() - self.start_time
        self.label_time.setVisible(True)
        self.label_time.setText(f"Creation of the patch, time : {round(float(elapsed_time),2)}s")

        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
            #PLACE BACK THE CURVE AND THE MIDDLE POINT ON THE CENTER MODEL 
            self.label_time.setText(f"Patch created, time : {round(float(elapsed_time),2)}s")
            self.camera=True
            self.viewScan()
            self.moveCurve(self.matrix)
            # Load the new model and display the patch 
            self.curve.SetAndObserveSurfaceConstraintNode(self.surf)
            self.displaySegmentation(self.surf)
            self._processed = True  # set the flag to prevent reprocessing
            self.timer.stop()
            if self.add_patch.isChecked():
                number_to_add = self.addItemsCombobox()
                self.combobox_patch.addItem(number_to_add)
                self.add_patch.setChecked(False)
                index = self.combobox_patch.findText(number_to_add)
                if index >= 0:
                    self.combobox_patch.setCurrentIndex(index)
            if not self.combobox_patch.isVisible():
                self.displayComboBox(self.surf)
        
            

    def addItemsCombobox(self):
        '''
        Return the number of the last element of the combo box + 1
        '''
        max_num = -float('inf')

        for index in range(self.combobox_patch.count):
            try:
                num = int(self.combobox_patch.itemText(index))
                
                if num > max_num:
                    max_num = num
            except ValueError:
                pass

        return str(max_num + 1)



    def displaySurf(self,surf)->None:
        '''
        Display the model
        '''
        mesh = slicer.app.mrmlScene().AddNewNodeByClass("vtkMRMLModelNode", 'First data')
        mesh.SetAndObservePolyData(surf)
        mesh.CreateDefaultDisplayNodes()




    def displaySegmentation(self,model_node)->None:
        '''
        Display the patch
        '''

        self.createButterfly(model_node.GetPolyData())
        
        displayNode = model_node.GetModelDisplayNode()
        displayNode.SetScalarVisibility(False)
        disabledModify = displayNode.StartModify()
        displayNode.SetActiveScalarName("Butterfly")
        displayNode.SetScalarVisibility(True)
        displayNode.EndModify(disabledModify)


    def isButterflyPatchAvailable(self, model_node,name)->bool:
        """
        Check if the Butterfly patch is available for the provided model node.
        """
        polyData = model_node
        if polyData:
            scalars = polyData.GetPointData().GetScalars(name)
            return scalars is not None
        return False
    
    def  createButterfly(self,polydata):
        '''
        Check if a Butterfly1 exist, if no disable the display of the combobox
        '''
        # Only whether a patch exists is read below, so the merged array this
        # used to build with torch was thrown away every time -- and importing
        # torch for it cost a second the first time a scan was shown.
        has_patch = self.isButterflyPatchAvailable(polydata, "Butterfly1")

        if not has_patch and self.combobox_patch.isVisible():
            self.label_patch.setVisible(False)
            self.combobox_patch.setVisible(False)
            self.delete_patch.setVisible(False)
            self.label_addpatch.setVisible(False)
            self.add_patch.setVisible(False)
        
            self.combobox_patch.addItem(str(1))

class DummyFile(io.IOBase):
        def close(self):
            pass
        
class FlexRegBootManager:
    booted = False