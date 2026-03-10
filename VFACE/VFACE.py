import logging
import os
from typing import Annotated
import urllib.request
import shutil
import zipfile

import importlib
try:
    import Progress
    importlib.reload(Progress)
    from Progress import DisplayALICBCT,DisplayAMASSS,DisplayASOCBCT,Display
    
    import createlistprocess
    importlib.reload(createlistprocess)
    from createlistprocess import CreateListProcess

except Exception as e:
    from Progress import DisplayALICBCT,DisplayAMASSS,DisplayASOCBCT,Display
    from createlistprocess import CreateListProcess

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
# VFACE
#


class VFACE(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("V FACE")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.contributors = ["Alexandre Buisson (University of North Carolina at Chapel Hill)"] 
        self.parent.helpText = _("""This extension is created to help you to classified the facial asymettry of your patient.""")
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # VFACE1
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

    # VFACE2
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


#
# VFACEParameterNode
#


@parameterNodeWrapper
class VFACEParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    InputFolder: str
    OutputFolder: str
    MeasurementsFolder: str

#
# VFACEWidget
#
class PopUpWindow(qt.QDialog):
    """Class to generate a popup window with text and button (either radio or checkbox)"""

    def __init__(
        self,
        title="Title",
        text=None,
        listename=["1", "2", "3"],
        type=None,
        tocheck=None,
    ):
        qt.QWidget.__init__(self)
        self.setWindowTitle(title)
        layout = qt.QGridLayout()
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

    def checkbox(self, layout):
        j = 0
        for i in range(len(self.listename)):
            button = qt.QCheckBox(self.listename[i])
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
        TrueFalse = [button.isChecked() for button in self.ListButtons]
        self.checked = [
            self.listename[i] for i in range(len(self.listename)) if TrueFalse[i]
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



class VFACEWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
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

    def reloadCustomModules(self):
        import importlib
        import sys

        modules_to_reload = ['createlistprocess', 'Progress', 'functionaq3dc']
        
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
        print("All files have been reload properly")

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        self.reloadCustomModules()

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/VFACE.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

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
        self.ui.CheckDependecyButton.connect("clicked(bool)", self.CheckDepedency)
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

        if not self._parameterNode:
            self.ui.applyButton.enabled = False

        else:
            self._parameterNode.InputFolder = self.ui.PathLineEdit.currentPath
            self._parameterNode.OutputFolder = self.ui.PathLineEdit_2.currentPath
            self._parameterNode.MeasurementsFolder = self.ui.PathLineEdit_3.currentPath

            if self.ui.comboBox2.currentText == "Visualization (Heatmaps)" and self.ui.comboBox3.currentText == "File already Registered":
        
                if self._parameterNode.InputFolder != "" and self._parameterNode.OutputFolder!="" and self.ui.PathLineEdit_4.currentPath != "":
                    self.ui.applyButton.toolTip = _("Click to classify your patient asymettry")
                    self.ui.applyButton.enabled = True
                else:
                    self.ui.applyButton.toolTip = _("Fill in the different folder path")
                    self.ui.applyButton.enabled = False

            elif self.ui.comboBox3.currentText == "File already Registered":
        
                if self._parameterNode.InputFolder != "" and self._parameterNode.OutputFolder!="" and self._parameterNode.MeasurementsFolder != "" and self.ui.PathLineEdit_4.currentPath != "":
                    self.ui.applyButton.toolTip = _("Click to classify your patient asymettry")
                    self.ui.applyButton.enabled = True
                else:
                    self.ui.applyButton.toolTip = _("Fill in the different folder path")
                    self.ui.applyButton.enabled = False

            elif self.ui.comboBox2.currentText == "Visualization (Heatmaps)":
        
                if self._parameterNode.InputFolder != "" and self._parameterNode.OutputFolder!="":
                    self.ui.applyButton.toolTip = _("Click to classify your patient asymettry")
                    self.ui.applyButton.enabled = True
                else:
                    self.ui.applyButton.toolTip = _("Fill in the different folder path")
                    self.ui.applyButton.enabled = False

            else:
                if self._parameterNode.InputFolder != "" and self._parameterNode.OutputFolder!="" and self._parameterNode.MeasurementsFolder != "":
                    self.ui.applyButton.toolTip = _("Click to classify your patient asymettry")
                    self.ui.applyButton.enabled = True

                else:
                    self.ui.applyButton.toolTip = _("Fill in the different folder path")
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

    def CheckDepedency(self):
        print("=== Checking and installing Python dependencies ===")
        
        try:
            import joblib
            print(f"✓ joblib is already installed (version: {joblib.__version__})")
        except ImportError:
            print("✗ joblib not found, installing...")
            try:
                print("Installing joblib...")
                slicer.util.pip_install('joblib')
                import joblib
                print(f"✓ joblib successfully installed (version: {joblib.__version__})")
            except Exception as e:
                print(f"✗ Failed to install joblib: {str(e)}")
                raise e
        try:
            import lightgbm
            print(f"✓ lightgbm is already installed (version: {lightgbm.__version__})")
        except ImportError:
            print("✗ lightgbm not found, installing...")
            try:
                print("Installing lightgbm... (this may take a while)")
                slicer.util.pip_install('lightgbm')
                import lightgbm
                print(f"✓ lightgbm successfully installed (version: {lightgbm.__version__})")
            except Exception as e:
                print(f"✗ Failed to install lightgbm: {str(e)}")
                raise e
        
        print("\n=== Python dependencies check completed ===")
        print("--- Downloading model files ---")
        self.DownloadAllFiles()
        print(f"✓ Every dependency has been successfully installed")

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
        self.ui.CheckDependecyButton.enabled = False
        self.ui.cancelButton.setVisible(True)
        self.ui.label_3.setVisible(True)
        self.ui.progressBar.setVisible(True)

        self.NumberProcess = len(self.list_process)
        self.executeProcess(self.list_process[0])
        del self.list_process[0]

    def onContinueButton(self):
        print("Continuing process after visualization...")
        self.ui.continueButton.setVisible(False)
        self.ui.cancelButton.setVisible(True)  # Réafficher le bouton cancel
        self.paused_for_visualization = False
        self.ui.label_3.setVisible(True)
        self.ui.progressBar.setVisible(True)
        
        if self.list_process:
            self.ActualProcess += 1
            self.executeProcess(self.list_process[0])
            del self.list_process[0]
        else:
            self.OnEndProcess()

    def onCancelButton(self):
        """Called when the cancel button is clicked"""
        # Afficher une boîte de dialogue de confirmation
        msgBox = qt.QMessageBox()
        msgBox.setWindowTitle("Confirm Cancellation")
        msgBox.setText("Are you sure you want to cancel the current process?")
        msgBox.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        msgBox.setDefaultButton(qt.QMessageBox.No)
        
        result = msgBox.exec_()
        
        if result == qt.QMessageBox.Yes:
            self.cancelProcess()
    
    def cancelProcess(self):
        """Annule le processus en cours"""
        print("Canceling process...")
        
        # Arrêter le processus CLI s'il est en cours
        if hasattr(self, 'cliNode') and self.cliNode:
            try:
                self.cliNode.Cancel()
                print("CLI process canceled")
            except Exception as e:
                print(f"Error canceling CLI process: {e}")
        
        # Arrêter le processus Python s'il est en cours
        if hasattr(self, 'python_process') and self.python_process:
            try:
                # Si c'est run_bds, nous devons arrêter la segmentation
                if hasattr(self.python_process, '__name__') and 'bds' in self.python_process.__name__:
                    # Importer et arrêter la logique de segmentation
                    try:
                        import sys
                        import os
                        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        if parent_dir not in sys.path:
                            sys.path.insert(0, parent_dir)
                        
                        from segmentation_logic import SegmentationLogic
                        # Créer une instance temporaire pour arrêter tous les processus
                        temp_logic = SegmentationLogic()
                        temp_logic.stop()
                        print("Segmentation process canceled")
                    except Exception as e:
                        print(f"Error stopping segmentation: {e}")
                
                self.python_process_completed = True
                print("Python process canceled")
            except Exception as e:
                print(f"Error canceling Python process: {e}")
        
        # Réinitialiser l'interface
        self.list_process = []
        self.ui.applyButton.enabled = True
        self.ui.CheckDependecyButton.enabled = True
        self.ui.cancelButton.setVisible(False)
        self.ui.label_3.setVisible(False)
        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setValue(0)
        
        if hasattr(self, 'continueButton'):
            self.ui.continueButton.setVisible(False)
        
        print("Process cancellation completed")
        self.cancelCurrentProcess()

    def cancelCurrentProcess(self):
        """Cancel the currently running process"""
        print("Cancelling current process...")
        
        if hasattr(self, 'cliNode') and self.cliNode is not None:
            try:
                self.cliNode.Cancel()
                self.removeObserver(self.cliNode, vtk.vtkCommand.ModifiedEvent, self.onCliUpdated)
                print("Process cancelled")
            except Exception as e:
                print(f"Error cancelling CLI process: {e}")
        
        self.list_process.clear()

        self.resetUIAfterCancel()

        s = PopUpWindow(title="Process Cancelled", text="The process has been successfully cancelled.")
        s.exec_()

    def resetUIAfterCancel(self):
        """Reset UI elements after cancellation"""
        self.ui.label_3.setVisible(False)
        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setValue(0)
        self.ui.continueButton.setVisible(False)
        self.ui.cancelButton.setVisible(False)
        
        self.ui.applyButton.enabled = True
        self.ui.CheckDependecyButton.enabled = True

        self.paused_for_visualization = False
        self.current_output_to_load = None
        self.current_process_info = None
        self.cliNode = None
        self.ActualProcess = 1
        self.NumberProcess = 0
        
        print("Interface reset after cancellation")

    def loadOutputInSlicer(self, output_path):
        """Charge l'output dans le visualiseur Slicer"""
        try:
            if output_path and os.path.exists(output_path):

                if output_path.endswith(('.nrrd', '.nii', '.nii.gz')):
                    # Volume
                    volume_node = slicer.util.loadVolume(output_path)
                    if volume_node:

                        slicer.util.setSliceViewerLayers(background=volume_node)
                        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
                        slicer.util.resetSliceViews()
                        print(f"Volume loaded : {output_path}")
                        return True
                elif output_path.endswith('.vtk'):
                    # Model
                    model_node = slicer.util.loadModel(output_path)
                    if model_node:
                        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
                        threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
                        threeDView.resetFocalPoint()
                        print(f"Model loaded : {output_path}")
                        return True
        except Exception as e:
            print(f"Error during the loading of {output_path}: {e}")
        
        return False

    def shouldPauseAfterProcess(self, process_info):
        return process_info.get("pause_for_visualization", False)

    def getOutputPathForModule(self, module_name):
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

        print(f"PROCESS DONE in {timer}")

        self.ui.label_3.setVisible(False)
        self.ui.progressBar.setVisible(False)
        self.ui.continueButton.setVisible(False)
        self.ui.cancelButton.setVisible(False)
        self.ui.CheckDependecyButton.enabled = True
        self.paused_for_visualization = False
        self.ActualProcess = 1
        self.NumberProcess = 0
        self.cliNode = None
        self.list_process = []
        self.current_output_to_load = None
        self.current_process_info = None

        s = PopUpWindow(title="Process Done",text="Successfully done")
        s.exec_()
        self._checkCanApply()

        if not self.ui.checkBox.isChecked():
            list_not_to_clean = []
            if "Visualization" in self.ui.comboBox2.currentText:
                list_not_to_clean.append("Heatmaps")
                list_not_to_clean.append("VTK Files")
            if "Quantification" in self.ui.comboBox2.currentText:
                list_not_to_clean.append("Measurements")
                list_not_to_clean.append("Classification")

            output_path = Path(self._parameterNode.OutputFolder)
            for item in output_path.iterdir():
                if item.is_dir():
                    if item.name not in list_not_to_clean:
                        shutil.rmtree(item)
            
            
            

class VFACELogic(ScriptedLoadableModuleLogic):
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

    def getParameterNode(self):
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
