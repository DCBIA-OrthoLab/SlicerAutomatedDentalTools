
import logging
import os,sys,time,zipfile,urllib.request,shutil

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install

import qt
import glob
import numpy as np
from qt import QFileDialog,QMessageBox,QGridLayout,QWidget,QPixmap
from functools import partial
import SimpleITK as sitk


from MedX_Method.summarize import MedX_Summarize_Method
from MedX_Method.dashboard import MedX_Dashboard_Method
from MedX_Method.Method import Method
from MedX_Method.Progress import Display


import signal
import time
import textwrap
import platform
import threading
import subprocess
import pkg_resources
import io


def check_lib_installed(lib_name, required_version=None):
    '''
    Check if the library is installed and meets the required version constraint (if any).
    - lib_name: "torch"
    - required_version: ">=1.10.0", "==0.7.0", "<2.0.0", etc.
    '''
    try:
        if required_version:
            # Use full requirement spec (e.g., "torch>=1.10.0")
            pkg_resources.require(f"{lib_name}{required_version}")
        else:
            # Just check if it's installed
            pkg_resources.get_distribution(lib_name)
        return True
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
        print(f"Version check failed: {e}")
        return False

# import csv

def install_function(self,list_libs:list):
    '''
    Test the necessary libraries and install them with the specific version if needed
    User is asked if he wants to install/update-by changing his environment- the libraries with a pop-up window
    '''
    libs = list_libs
    libs_to_install = []
    libs_to_update = []
    installation_errors = []
    for lib, version_constraint,url in libs:
        if not check_lib_installed(lib, version_constraint):
            try:
            # check if the library is already installed
                if pkg_resources.get_distribution(lib).version:
                    libs_to_update.append((lib, version_constraint))
            except:
                libs_to_install.append((lib, version_constraint))

    if libs_to_install or libs_to_update:
          message = "The following changes are required for the libraries:\n"

          #Specify which libraries will be updated with a new version
          #and which libraries will be installed for the first time
          if libs_to_update:
              message += "\n --- Libraries to update (version mismatch): \n"
              message += "\n".join([f"{lib} (current: {pkg_resources.get_distribution(lib).version}) -> {version_constraint.replace('==','').replace('<=','').replace('>=','').replace('<','').replace('>','')}" for lib, version_constraint in libs_to_update])
              message += "\n"
          if libs_to_install:

              message += "\n --- Libraries to install:  \n"
          message += "\n".join([f"{lib}{version_constraint}" if version_constraint else lib for lib, version_constraint in libs_to_install])

          message += "\n\nDo you agree to modify these libraries? Doing so could cause conflicts with other installed Extensions."
          message += "\n\n (If you are using other extensions, consider downloading another Slicer to use AutomatedDentalTools exclusively.)"

          user_choice = slicer.util.confirmYesNoDisplay(message)

          if user_choice:
            self.ui.label_LibsInstallation.setVisible(True)
            try:
                for lib, version_constraint in libs_to_install + libs_to_update:
                    if not version_constraint:
                        pip_install(lib)

                    elif "https:/" in version_constraint:
                        print("version_constraint", version_constraint)
                        # download the library from the url
                        pip_install(version_constraint)
                    else:
                        print("version_constraint else", version_constraint)
                        lib_version = f'{lib}{version_constraint}' if version_constraint else lib
                        pip_install(lib_version)

                return True
            except Exception as e:
                    installation_errors.append((lib, str(e)))

            if installation_errors:
                error_message = "The following errors occured during installation:\n"
                error_message += "\n".join([f"{lib}: {error}" for lib, error in installation_errors])
                slicer.util.errorDisplay(error_message)
                return False
          else :
            return False

    else:
        return True


#
# MedX
#test

class MedX(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MedX"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Automated Dental Tools"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Gaydamour Alban"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#MedX">module documentation</a>.
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

    # MedX1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='MedX',
        sampleName='MedX1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'MedX1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='MedX1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='MedX1'
    )

    # MedX2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='MedX',
        sampleName='MedX2',
        thumbnailFileName=os.path.join(iconsPath, 'MedX2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='MedX2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='MedX2'
    )


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

#
# MedXWidget
#

class MedXWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
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

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/MedX.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MedXLogic()
        
        self.summarize = MedX_Summarize_Method(self)
        self.dashboard = MedX_Dashboard_Method(self)
        self.dashboardPixmap = None  # store original pixmap

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        
        self.nb_scans = 0
        self.time_log = 0
        
        self.log_path = os.path.join(slicer.util.tempDirectory(), "process.log")
        
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documentsLocation)
        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
            "MedX",
        )

        if not os.path.exists(self.SlicerDownloadPath):
            os.makedirs(self.SlicerDownloadPath)

        # Summarization
        self.ui.SearchButtonModel.connect("clicked(bool)",partial(self.openFinder,"Model"))
        self.ui.SearchButtonPatient.connect("clicked(bool)",partial(self.openFinder,"Patient"))
        self.ui.SearchButtonOutput.connect("clicked(bool)",partial(self.openFinder,"Output"))
        self.ui.DownloadModel.pressed.connect(
            lambda: self.downloadModel(
                self.ui.LineEditModel, "MedX"
            )
        )
        self.ui.SummarizeButton.connect("clicked(bool)", self.onSummarizeButton)
        self.ui.ButtonCancel.connect("clicked(bool)", self.onCancel)
        
        
        # Dashboard
        self.ui.SearchButtonSummary.connect("clicked(bool)",partial(self.openFinder,"Summary"))
        self.ui.SearchButtonOutDashboard.connect("clicked(bool)",partial(self.openFinder,"OutDashboard"))
        self.ui.DashboardButton.connect("clicked(bool)", self.onDashboardButton)
        

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # VARIABLES
        self.progress=0

        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setRange(0,100)
        self.ui.progressBar.setTextVisible(True)
        self.ui.label_info.setVisible(False)
        self.ui.label_time.setVisible(False)
        self.ui.ButtonCancel.setVisible(False)
        self.ui.LabelNameExtension.setVisible(False)
        self.ui.label_LibsInstallation.setVisible(False)

        self.timer_should_continue = True

    def openFinder(self,nom : str,_) -> None :
        """
         Open finder to let the user choose is files or folder
        """

        if nom=="Model":
            surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.LineEditModel.setText(surface_folder)

        elif nom=="Patient":
            surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.LineEditClinicalNotes.setText(surface_folder)

        elif nom=="Output":
            surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.LineEditOutput.setText(surface_folder)
            
        elif nom=="Summary":
            surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditSummaries.setText(surface_folder)
            
        elif nom=="OutDashboard":
            surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditOutDashboard.setText(surface_folder)


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


        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)
        
        
    def DownloadUnzip(
        self, url, directory, folder_name=None, num_downl=1, total_downloads=1
    ):
        out_path = os.path.join(directory, folder_name)

        if not os.path.exists(out_path):
            # print("Downloading {}...".format(folder_name.split(os.sep)[0]))
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

            # Unzip the file
            with zipfile.ZipFile(temp_path, "r") as zip:
                zip.extractall(out_path)

            # Delete the zip file
            os.remove(temp_path)

        return out_path
        
        
    def downloadModel(self, lineEdit, name, test=False):
        """Function to download the model files from the link in the getModelUrl function"""

        listmodel = self.summarize.getModelUrl()

        urls = listmodel[name]
        for i, (name_bis, url) in enumerate(urls.items()):
            _ = self.DownloadUnzip(
                url=url,
                directory=os.path.join(self.SlicerDownloadPath),
                folder_name=os.path.join(self.SlicerDownloadPath),
                num_downl=i + 1,
                total_downloads=len(urls),
            )
        model_folder = os.path.join(self.SlicerDownloadPath)

        if not model_folder == "":
            error = self.summarize.TestModel(model_folder)

            if isinstance(error, str):
                qt.QMessageBox.warning(self.parent, "Warning", error)

            else:
                lineEdit.setText(model_folder)


    def onSummarizeButton(self)->None:
        """
        Run processing when user clicks "Summarize" button.
        """
        list_libs = ['transformers',
                     'torch',
                     'pymupdf',
                     'python-docx',
                     'evaluate',
                     'scikit-learn',
                     'peft',
                     'bitsandbytes',
                     'matplotlib',]

        try:
            check_env = self.onCheckRequirements(list_libs)
            print("seg_env: ", check_env)
        except Exception as e:
            qt.QMessageBox.warning(self.parent, "Warning", f"An error occurred while checking requirements: {str(e)}")
            return
        
        self.logic.check_cli_script()
        
        self.ui.label_LibsInstallation.setVisible(False)

        error = self.summarize.TestProcess(
            input_notes=self.ui.LineEditClinicalNotes.text,
            input_model=self.ui.LineEditModel.text,
            output_folder=self.ui.LineEditOutput.text,
        )

        # print('error',error)
        if isinstance(error, str):
            qt.QMessageBox.warning(self.parent, "Warning", error.replace(",", "\n"))

        self.list_Processes_Parameters = self.summarize.Process(
            input_notes=self.ui.LineEditClinicalNotes.text,
            input_model=self.ui.LineEditModel.text,
            output_folder=self.ui.LineEditOutput.text,
            log_path=self.log_path,
        )
        
        self.nb_scans = self.summarize.NbScan(
            file_folder=self.ui.LineEditClinicalNotes.text,
        )
        
        self.nb_extension_launch = len(self.list_Processes_Parameters)
        self.onProcessStarted()
        
        module = self.list_Processes_Parameters[0]["Module"]
        # /!\ Launch of the first process /!\
        print("module name : ", module)
        
        self.ui.SummarizeButton.setEnabled(False)
        self.run_conda_tool()
        self.OnEndProcess()
        # self.process = slicer.cli.run(
        #     self.list_Processes_Parameters[0]["Process"],
        #     None,
        #     self.list_Processes_Parameters[0]["Parameter"],
        # )
        
        # self.module_name = self.list_Processes_Parameters[0]["Module"]
        # self.displayModule = self.list_Processes_Parameters[0]["Display"]
        # self.processObserver = self.process.AddObserver(
        #     "ModifiedEvent", self.onProcessUpdate
        # )

        # del self.list_Processes_Parameters[0]
        
    def onDashboardButton(self)->None:
        """
        Run processing when user clicks "Dashboard" button.
        """
        list_libs = [('numpy', None, None),
                     ('pandas', None, None),
                     ('matplotlib', None, None)]
        
        is_installed = install_function(self, list_libs)
        
        if not is_installed:
            qt.QMessageBox.warning(self.parent, 'Warning', 'The module will not work properly without the required libraries.\nPlease install them and try again.')
            return

        error = self.dashboard.TestProcess(
            summary_folder=self.ui.lineEditSummaries.text,
            output_folder=self.ui.lineEditOutDashboard.text,
        )

        # print('error',error)
        if isinstance(error, str):
            qt.QMessageBox.warning(self.parent, "Warning", error.replace(",", "\n"))
            return

        self.list_Processes_Parameters = self.dashboard.Process(
            summary_folder=self.ui.lineEditSummaries.text,
            output_folder=self.ui.lineEditOutDashboard.text,
            log_path=self.log_path,
        )
        
        self.nb_scans = self.dashboard.NbScan(
            file_folder=self.ui.lineEditSummaries.text,
        )
        
        self.nb_extension_launch = len(self.list_Processes_Parameters)
        self.onProcessStarted()
        
        module = self.list_Processes_Parameters[0]["Module"]
        # /!\ Launch of the first process /!\
        print("module name : ", module)
        
        self.ui.DashboardButton.setEnabled(False)
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
        
    def showDashboardImageInSliceView(self):
        dashboardPath = os.path.join(self.ui.lineEditOutDashboard.text, "dashboard.png")
        if not os.path.exists(dashboardPath):
            qt.QMessageBox.warning(self.parent, "Warning", f"No dashboard image found at:\n{dashboardPath}")
            return

        volumeNode = slicer.util.loadVolume(dashboardPath)
        if not volumeNode:
            qt.QMessageBox.warning(self.parent, "Warning", "Failed to load dashboard image.")
            return

        # Switch to single slice layout (Red)
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

        # Show image in Red slice viewer
        redWidget = layoutManager.sliceWidget("Red")
        redLogic = redWidget.sliceLogic()
        redLogic.GetSliceCompositeNode().SetBackgroundVolumeID(volumeNode.GetID())

        # Fit the image to view
        redLogic.FitSliceToAll()
            
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
            
    def onCheckRequirements(self, list_libs:list) -> bool:
        if not self.logic.isCondaSetUp:
            messageBox = qt.QMessageBox()
            text = textwrap.dedent("""
            SlicerConda is not set up, please click 
            <a href=\"https://github.com/DCBIA-OrthoLab/SlicerConda/\">here</a> for installation.
            """).strip()
            messageBox.information(None, "Information", text)
            return False
        
        self.ui.label_LibsInstallation.setVisible(True)
        
        if platform.system() == "Windows":
            self.ui.label_LibsInstallation.setText(f"Checking if wsl is installed, this task may take a moments")
            
            if self.logic.testWslAvailable():
                self.ui.label_LibsInstallation.setText(f"WSL installed")
                if not self.logic.check_lib_wsl():
                    self.ui.label_LibsInstallation.setText(f"Checking if the required librairies are installed, this task may take a moments")
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
        
        
        self.ui.label_LibsInstallation.setText(f"Checking if miniconda is installed")
        if "no setup" in self.logic.conda.condaRunCommand([self.logic.conda.getCondaExecutable(),"--version"]):
            messageBox = qt.QMessageBox()
            text = textwrap.dedent("""
            Code can't be launch. \nConda is not setup. 
            Please go the extension CondaSetUp in SlicerConda to do it.""").strip()
            messageBox.information(None, "Information", text)
            return False
        
        self.ui.label_LibsInstallation.setText(f"Checking if environnement exists")
        
        
        ## summaries
        
        
        if not self.logic.conda.condaTestEnv(self.logic.name_env) : # check is environnement exist, if not ask user the permission to do it
            userResponse = slicer.util.confirmYesNoDisplay("The environnement to run the summarization doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
            if userResponse :
                start_time = time.time()
                previous_time = start_time
                formatted_time = self.format_time(0)
                self.ui.label_LibsInstallation.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
                process = self.logic.install_summaries(list_libs=list_libs)
                
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
            self.ui.label_LibsInstallation.setText(f"Environnement already exists")


        self.all_installed = True
        return True
        
    def run_conda_tool(self):
        args = self.list_Processes_Parameters[0]["Parameter"]
        print("args : ", args)
        conda_exe = self.logic.conda.getCondaExecutable()
        command = [conda_exe, "run", "-n", self.logic.name_env, "python" ,"-m", f"MedX_Summarize"]
        for key, value in args.items():
            print("key : ", key)
            if isinstance(value, str) and ("\\" in value or (len(value) > 1 and value[1] == ":")):
                value = self.logic.windows_to_linux_path(value)
            command.append(f"\"{value}\"")
        print("command : ",command)

        # running in // to not block Slicer
        self.process = threading.Thread(target=self.logic.condaRunCommand, args=(command,))
        self.process.start()
        self.module_name = self.list_Processes_Parameters[0]["Module"]
        self.ui.LabelNameExtension.setText(f"Running {self.module_name}")
        self.ui.label_time.setVisible(True)
        self.ui.label_time.setText(f"time : 0.00s")
        previous_time = self.startTime
        while self.process.is_alive():
            self.ui.ButtonCancel.setVisible(True)
            self.onCondaProcessUpdate()
            slicer.app.processEvents()
            current_time = time.time()
            gap=current_time-previous_time
            if gap>0.3:
                currentTime = time.time() - self.startTime
                previous_time = currentTime
                if currentTime < 60:
                    timer = f"Time : {int(currentTime)}s"
                elif currentTime < 3600:
                    timer = f"Time : {int(currentTime/60)}min and {int(currentTime%60)}s"
                else:
                    timer = f"Time : {int(currentTime/3600)}h, {int(currentTime%3600/60)}min and {int(currentTime%60)}s"
                    
                self.ui.label_time.setText(timer)

        del self.list_Processes_Parameters[0]
    
    def read_log_path(self):
        with open(self.log_path, 'r') as f:
            line = f.readline()
            if line != '':
                return line
    
    def onCondaProcessUpdate(self):
        if os.path.isfile(self.log_path):
            time_progress = os.path.getmtime(self.log_path)
            line = self.read_log_path()
            if (time_progress != self.time_log) and line:
                progress = line.strip()
            
                self.progress = int(progress)
                self.ui.label_info.setText(f"Number of processed files : {self.progress}/{self.nb_scans}")
                progress_bar_value = round((self.progress) / self.nb_scans * 100,2)
                self.time_log = time_progress
                
                self.ui.progressBar.setValue(progress_bar_value)
                self.ui.progressBar.setFormat(f"{progress_bar_value:.2f}%")

    def onProcessUpdate(self, caller, event):
        # timer = f"Time : {time.time()-self.startTime:.2f}s"
        currentTime = time.time() - self.startTime
        if currentTime < 60:
            timer = f"Time : {int(currentTime)}s"
        elif currentTime < 3600:
            timer = f"Time : {int(currentTime/60)}min and {int(currentTime%60)}s"
        else:
            timer = f"Time : {int(currentTime/3600)}h, {int(currentTime%3600/60)}min and {int(currentTime%60)}s"

        self.ui.label_time.setText(timer)
        progress = caller.GetProgress()
        self.ui.LabelNameExtension.setText(f"Running {self.module_name}")
        
        if progress == 0:
            self.updateProgessBar = False
            
        if self.displayModule.isProgress(
            progress=progress, updateProgessBar=self.updateProgessBar
        ):
            progress_bar, message = self.displayModule()
            self.ui.progressBar.setValue(progress_bar)
            self.ui.progressBar.setFormat(f"{progress_bar:.2f}%")
            self.ui.label_info.setText(message)
            
        if caller.GetStatus() & caller.Completed:
            if caller.GetStatus() & caller.ErrorsMask:
                # error
                print("\n\n ========= PROCESSED ========= \n")

                print(self.process.GetOutputText())
                print("\n\n ========= ERROR ========= \n")
                errorText = self.process.GetErrorText()
                print("CLI execution failed: \n \n" + errorText)
                # error
                # errorText = caller.GetErrorText()
                # print("\n"+ 70*"=" + "\n\n" + errorText)
                # print(70*"=")
                self.onCancel()

            else:
                print("\n\n ========= PROCESSED ========= \n")
                # print("PROGRESS :",self.displayModule.progress)

                print(self.process.GetOutputText())
                try:
                    print("name process : ",self.list_Processes_Parameters[0]["Process"])
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
                    # self.displayModule.progress = 0
                except IndexError:
                    self.OnEndProcess()

    def saveOutput(self, outputVolumeNode, outputFilePath)->None:
        """
        Saves the output volume in the specified file with the .nii.gz extension.

        :param outputVolumeNode: The output volume node in Slicer MRML scene.
        :param outputFilePath: The full path where the file is to be saved.
        """
        if not os.path.exists(os.path.dirname(outputFilePath)):
            os.makedirs(os.path.dirname(outputFilePath))

        slicer.util.exportNode(outputVolumeNode, outputFilePath,world=True)




    def UpdateTime(self)->None:
        '''
        Update the time since the beginning
        '''
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        gap = current_time - self.previous_time
        if gap>0.5:
            self.ui.label_time.setText(f"time : {round(elapsed_time,2)}s")
            self.previous_time = current_time


    def UpdateProgressBar(self,end:bool)->None:
        '''
        Update the progress bar every time it's call
        '''
        if not end:
            self.progress+=1
            progressbar_value = (self.progress-1) /self.nbFiles * 100
            if progressbar_value < 100 :
                self.ui.progressBar.setValue(progressbar_value)
                self.ui.progressBar.setFormat(str(round(progressbar_value,2))+"%")
            else:
                self.ui.progressBar.setValue(99)
                self.ui.progressBar.setFormat("99%")
            self.ui.label_info.setText("Number of processed files : "+str(self.progress-1)+"/"+str(self.nbFiles))

        else :
            # success
            print('PROCESS DONE.')
            self.ui.progressBar.setValue(100)
            self.ui.progressBar.setFormat("100%")
            self.ui.label_info.setText("Number of processed files : "+str(self.nbFiles)+"/"+str(self.nbFiles))
            time.sleep(0.5)

            # qt.QMessageBox.information(self.parent,"Matrix applied with sucess")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)

            # setting message for Message Box
            msg.setText("Matrix applied with success")

            # setting Message box window title
            msg.setWindowTitle("Information")

            # declaring buttons on Message Box
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            self.ui.progressBar.setVisible(False)
            self.ui.label_info.setVisible(False)
            self.ui.label_time.setVisible(False)




    def onProcessStarted(self)->None:
        """
        Initialize the variables and progress bar.
        """
        self.startTime = time.time()
        self.ui.progressBar.setValue(0)
        self.progress = 0
        
        self.ui.label_LibsInstallation.setVisible(False)
        
        self.RunningUI(True)
        self.ui.progressBar.setEnabled(True)
        self.ui.progressBar.setHidden(False)
        self.ui.progressBar.setTextVisible(True)
        self.ui.progressBar.setFormat("0%")
        
        
    def OnEndProcess(self):
        if self.module_name == "MedX Dashboard" and self.ui.DisplayDashboard.isChecked():
            self.showDashboardImageInSliceView()
            
        total_time = time.time() - self.startTime
        average_time = total_time / self.nb_scans
        print("PROCESS DONE.")
        print(
            "Done in {} min and {} sec".format(
                int(total_time / 60), int(total_time % 60)
            )
        )
        print(
            "Average time per patient : {} min and {} sec".format(
                int(average_time / 60), int(average_time % 60)
            )
        )
        self.RunningUI(False)

        stopTime = time.time()

        logging.info(f"Processing completed in {stopTime-self.startTime:.2f} seconds")

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

    def RunningUI(self,run:bool)->None:
        self.ui.LabelNameExtension.setVisible(run)
        self.ui.label_info.setVisible(run)
        self.ui.ButtonCancel.setVisible(run)
        self.ui.progressBar.setVisible(run)
        self.ui.label_time.setVisible(run)
        self.ui.SummarizeButton.setEnabled(not run)
        self.ui.DashboardButton.setEnabled(not run)
        
    def onCancel(self):
        self.logic.cancel_process()
        print("\n\n ========= PROCESS CANCELED ========= \n")
        
        self.RunningUI(False)

#
# MedXLogic
#
class DummyFile(io.IOBase):
        def close(self):
            pass

class MedXLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self,path_patient_intput=None,path_matrix_intput=None,path_patient_output=None,suffix=None,logPath=None):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.path_patient_intput = path_patient_intput
        self.path_matrix_intput = path_matrix_intput
        self.path_patient_output = path_patient_output
        self.suffix = suffix
        self.logPath = logPath
        self.cliNode = None
        self.installCliNode = None
        self.isCondaSetUp = False
        self.conda = self.init_conda()
        self.name_env = "summaries"
        self.cliNode = None
        
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

    def install_summaries(self, list_libs: list):
        self.run_conda_command(target=self.conda.condaCreateEnv, command=(self.name_env,"3.12", list_libs))
        
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
    
    def check_pythonpath_windows(self,file):
        '''
        Check if the environment env_name in wsl know the path to a specific file (ex : Crownsegmentationcli.py)
        return : bool
        '''
        conda_exe = self.conda.getCondaExecutable()
        command = [conda_exe, "run", "-n", self.name_env, "python" ,"-c", f"\"import {file} as check;import os; print(os.path.isfile(check.__file__))\""]
        result = self.conda.condaRunCommand(command)
        print("output CHECK python path: ", result)
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
        print("output GIVE python path: ", results)
        
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
    
    def cancel_process(self):
        if platform.system() == 'Windows':
            self.subpro.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(self.subpro.pid), signal.SIGTERM)
        print("Cancellation requested. Terminating process...")

        self.subpro.wait() ## important
        self.cancel = True
    
    def check_cli_script(self):
        if not self.check_pythonpath_windows("MedX_Summarize"):
            self.give_pythonpath_windows()
            results = self.check_pythonpath_windows("MedX_Summarize")

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
            print("command_to_execute in condaRunCommand : ",command_to_execute)

            self.subpro = subprocess.Popen(command_to_execute, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, encoding='utf-8', errors='replace', env=slicer.util.startupEnvironment(),
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # For Windows
                              )
        else:
            path_conda_exe = self.conda.getCondaExecutable()
            command_execute = f"{path_conda_exe} run -n {self.name_env}"
            for com in command :
                command_execute = command_execute+ " "+com

            print("command_to_execute in conda run : ",command_execute)
            self.subpro = subprocess.Popen(command_execute, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', env=slicer.util.startupEnvironment(), executable="/bin/bash", preexec_fn=os.setsid)
    
        self.stdout, self.stderr = self.subpro.communicate()


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
         Call the process with the parameters
        """

        pass



#
# MedXTest
#

class MedXTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_MedX1()

    def test_MedX1(self):
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

        # import SampleData
        # registerSampleData()
        # inputVolume = SampleData.downloadSample('MedX1')
        # self.delayDisplay('Loaded test data set')

        # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)

        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # threshold = 100

        # # Test the module logic

        # logic = MedXLogic()

        # # Test algorithm with non-inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, True)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], threshold)

        # # Test algorithm with inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, False)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        #This module is not using MedXLogic

        self.delayDisplay('Test passed')