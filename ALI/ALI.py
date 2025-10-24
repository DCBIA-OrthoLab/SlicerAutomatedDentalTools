
import os, sys, time, logging, zipfile, urllib.request, shutil, glob, re
import vtk, qt, slicer
from qt import (
    QWidget,
    QGridLayout,
)
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install, pip_uninstall
import webbrowser
import textwrap
import importlib.metadata
import signal

from pathlib import Path
import platform
import threading
import subprocess
from multiprocessing import Process, Value

from ALI_Method.IOS import Auto_IOS
from ALI_Method.CBCT import Auto_CBCT
from ALI_Method.Method import Method
from ALI_Method.Progress import Display


def check_lib_installed(lib_name, required_version=None):
  try:
    installed_version =importlib.metadata.version(lib_name)
    if required_version and installed_version != required_version:
      return False
    return True
  except importlib.metadata.PackageNotFoundError:
    return False

# import csv
def install_function(self, libs=None):
  libs_to_install = []
  for lib, version in libs:
    if not check_lib_installed(lib, version):
      libs_to_install.append((lib, version))

  if libs_to_install:
    message = "The following libraries are not installed or need updating:\n"
    message += "\n".join([f"{lib}=={version}" if version else lib for lib, version in libs_to_install])
    message += "\n\nDo you want to install/update these libraries?\n Doing it could break other modules"
    user_choice = slicer.util.confirmYesNoDisplay(message)

    if user_choice:
      self.ui.label_LibsInstallation.setVisible(True)
      for lib, version in libs_to_install:
        lib_version = f'{lib}=={version}' if version else lib
        pip_install(lib_version)
    else:
      return False
  return True

#region ========== FUNCTIONS ==========

def PathFromNode(node):
  storageNode=node.GetStorageNode()
  if storageNode is not None:
    filepath=storageNode.GetFullNameFromFileName()
  else:
    filepath=None
  return filepath


TEST_SCAN = {
  "CBCT": 'https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/MG_test_scan.nii.gz',
  "IOS" : 'https://github.com/baptistebaquero/ALIDDM/releases/tag/v1.0.4',
}

MODELS_LINK = {
  "CBCT": [
    'https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/tag/v0.1-v2.0_models',
  ],
  "IOS" : [
    'https://github.com/baptistebaquero/ALIDDM/releases/tag/v1.0.3',
  ],
}


GROUPS_LANDMARKS = {
  'Impacted canine' : ['UR3OI','UL3OI','UR3RI','UL3RI'],

  'Cranial base' : ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4'],

  'Lower' : ['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R'],

  'Upper' : ['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R'],
}


TEETH = {
  'Upper teeth' : ['UL7','UL6','UL5','UL4','UL3','UL2','UL1','UR1','UR2','UR3','UR4','UR5','UR6','UR7'],
  'Lower teeth' : ['LL7','LL6','LL5','LL4','LL3','LL2','LL1','LR1','LR2','LR3','LR4','LR5','LR6','LR7'],
}

SURFACE_LANDMARKS = {
  'Cervical' : ['CL','CB','R','RIP','OIP'],
  'Occlusal' : ['O','DB','MB'],
}

SURFACE_NETWORK = {
  '_O_' : 'Occlusal',
  '_C_' : 'Cervical'
}



  # "Dental" :  ['LL7','LL6','LL5','LL4','LL3','LL2','LL1','LR1','LR2','LR3','LR4','LR5','LR6','LR7','UL7','UL6','UL5','UL4','UL3','UL2','UL1','UR1','UR2','UR3','UR4','UR5','UR6','UR7'] ,

  # "Landmarks type" : ['CL','CB','O','DB','MB','R','RIP','OIP']

import json

class ALI(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ALI"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Automated Dental Tools"]  # set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = ["CondaSetUp"]  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Maxime Gillot (UoM), Baptiste Baquero (UoM)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#ALI">module documentation</a>.
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

  # ALI1
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='ALI',
    sampleName='ALI1',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'ALI1.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='ALI1.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='ALI1'
  )

  # ALI2
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='ALI',
    sampleName='ALI2',
    thumbnailFileName=os.path.join(iconsPath, 'ALI2.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='ALI2.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='ALI2'
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
# ALIWidget
#

class ALIWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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

    self.folder_as_input = False # If use a folder as input

    self.MRMLNode_scan = None # MRML node of the selected scan
    self.input_path = None # path to the folder containing the scans
    self.model_folder = None

    self.available_landmarks = [] # list of available landmarks to predict

    self.output_folder = None # If save the output in a folder
    self.goup_output_files = False

    self.scan_count = 0 # number of scans in the input folder
    self.landmark_cout = 0 # number of landmark to identify
    self.nb_patient = 0 # number of patients to process

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/ALI.ui'))
    self.layout.addWidget(uiWidget)

    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = ALILogic()
    
    
    self.MethodDic = {
      "IOS": Auto_IOS(self),
      "CBCT": Auto_CBCT(self),
    }
    self.ActualMeth = Method
    self.ActualMeth = self.MethodDic["CBCT"]
    self.type = "CBCT"
    self.display = Display
    self.selected_tooth = None
    
    self.nb_patient = 0
    self.time_log = 0
    
    self.log_path = os.path.join(slicer.util.tempDirectory(), "process.log")
    
    documentsLocation = qt.QStandardPaths.DocumentsLocation
    self.documents = qt.QStandardPaths.writableLocation(documentsLocation)
    self.SlicerDownloadPath = os.path.join(
      self.documents,
      slicer.app.applicationName + "Downloads",
      "ALI",
      "ALI_" + self.type,
    )

    if not os.path.exists(self.SlicerDownloadPath):
      os.makedirs(self.SlicerDownloadPath)

    
    self.HideComputeItems()
    
    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).


    self.lm_selection_area = qt.QWidget()
    self.lm_selection_layout = qt.QHBoxLayout(self.lm_selection_area)
    self.ui.OptionVLayout.addWidget(self.lm_selection_area)
    self.tooth_lm = LMTab()
    self.tooth_lm.Clear()
    self.tooth_lm.FillTab(TEETH,True)
    self.lm_selection_layout.addWidget(self.tooth_lm.widget)

    self.lm_tab = LMTab()
    # LM_tab_widget,LM_buttons_dic = GenLandmarkTab(Landmarks_group)
    self.lm_selection_layout.addWidget(self.lm_tab.widget)


    #region ===== INPUTS =====

    self.ui.InputTypeComboBox.currentIndexChanged.connect(self.SwitchInputType)
    self.SwitchInputType(0)
    self.ui.ExtensioncomboBox.currentIndexChanged.connect(self.SwitchInputExtension)
    self.SwitchInputExtension(0)
    self.ui.MRMLNodeComboBox.setMRMLScene(slicer.mrmlScene)
    self.ui.MRMLNodeComboBox.currentNodeChanged.connect(self.onNodeChanged)
    self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
    self.ui.InputComboBox.currentIndexChanged.connect(self.SwitchInput)
    self.SwitchInput(0)
    self.ui.DownloadTestPushButton.connect('clicked(bool)',self.TestFiles)
    #endregion

    self.ui.SavePredictCheckBox.connect("toggled(bool)", self.UpdateSaveType)

    self.ui.SearchSaveFolder.setHidden(False)
    self.ui.SaveFolderLineEdit.setHidden(False)
    self.ui.PredictFolderLabel.setHidden(False)
    # Buttons
    self.ui.SearchScanFolder.pressed.connect(
      lambda: self.onSearchScanButton(self.ui.lineEditScanPath)
    )
    self.ui.SearchModelsFolder.connect('clicked(bool)',self.onSearchModelButton)
    self.ui.SearchModelFolder.pressed.connect(
      lambda: self.downloadModel(
        self.ui.lineEditModelPath
      )
    )

    self.ui.SearchSaveFolder.connect('clicked(bool)',self.onSearchSaveButton)


    self.ui.PredictionButton.connect('clicked(bool)', self.onPredictButton)
    self.ui.CancelButton.connect('clicked(bool)', self.onCancel)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  #region ===== FUNCTIONS =====

  #region ===== INPUTS =====

  def SwitchInputType(self,index):

    self.lm_tab.Clear()

    if index == 1:
      self.SwitchInputExtension(0)
      self.type = "IOS"
      self.ui.MRMLNodeComboBox.nodeTypes = ['vtkMRMLModelNode']
      self.lm_tab.FillTab(SURFACE_LANDMARKS)
      self.ui.ExtensionLabel.setVisible(False)
      self.ui.ExtensioncomboBox.setVisible(False)
      self.ui.label_LibsInstallation.setVisible(False)

    else:
      self.type = "CBCT"
      self.ui.MRMLNodeComboBox.nodeTypes = ['vtkMRMLVolumeNode']
      self.lm_tab.FillTab(GROUPS_LANDMARKS)
      self.ui.ExtensionLabel.setVisible(True)
      self.ui.ExtensioncomboBox.setVisible(True)
      self.ui.label_LibsInstallation.setVisible(False)

    self.ui.lineEditModelPath.setText("")
    self.model_folder = None

    self.tooth_lm.widget.setHidden(True if self.type == "CBCT" else False)
    
    self.ActualMeth = self.MethodDic[self.type]
    self.SlicerDownloadPath = os.path.join(
      self.documents,
      slicer.app.applicationName + "Downloads",
      "ALI",
      "ALI_" + self.type,
    )

  def SwitchInputExtension(self,index):

    if index == 0: # NIFTI, NRRD, GIPL Files
      self.SwitchInput(0)
      self.isDCMInput = False

      self.ui.label_11.setVisible(True)
      self.ui.InputComboBox.setVisible(True)

    if index == 1: # DICOM Files
      self.SwitchInput(1)
      self.ui.label_11.setVisible(False)
      self.ui.InputComboBox.setVisible(False)
      self.ui.ScanPathLabel.setText('DICOM\'s Folder')
      self.isDCMInput = True

  def SwitchInput(self,index):

    if index == 0:
      self.folder_as_input = True
      self.input_path = None

    else:
      self.folder_as_input = False
      self.ui.SavePredictCheckBox.setChecked(False)
      self.onNodeChanged()

    # print("Input type : ", index)

    self.ui.ScanPathLabel.setVisible(self.folder_as_input)
    self.ui.lineEditScanPath.setVisible(self.folder_as_input)
    self.ui.SearchScanFolder.setVisible(self.folder_as_input)
    self.ui.SavePredictCheckBox.setEnabled(self.folder_as_input)
  
    self.ui.SelectNodeLabel.setVisible(not self.folder_as_input)
    self.ui.MRMLNodeComboBox.setVisible(not self.folder_as_input)
    self.ui.FillNodeLlabel.setVisible(not self.folder_as_input)


  def onNodeChanged(self):
    selected = False
    self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
    if self.MRMLNode_scan is not None:
      print(PathFromNode(self.MRMLNode_scan))
      self.input_path = PathFromNode(self.MRMLNode_scan)
      self.nb_patient = 1

      self.ui.LabelInfoPreProc.setText("Number of scans to process : 1")
      selected = True

    return selected

  def onTestDownloadButton(self):
    webbrowser.open(TEST_SCAN[self.type])


  def onModelDownloadButton(self):
    for link in MODELS_LINK[self.type]:
      webbrowser.open(link)

  def UpdateSaveType(self,caller=None, event=None):
    # print(caller,event)
    state = self.ui.SavePredictCheckBox.isChecked()
    self.ui.SearchSaveFolder.setEnabled(not state)
    self.ui.SaveFolderLineEdit.setEnabled(not state)

    if state:
      self.output_folder = self.ui.lineEditScanPath.text
      self.ui.SaveFolderLineEdit.text = self.output_folder

  def CountFileWithExtention(self,path,extentions = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"], exception = ["Seg", "seg", "Pred"]):

    count = 0
    normpath = os.path.normpath("/".join([path, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(img_fn)

        if True in [ext in basename for ext in extentions]:
            if not True in [ex in basename for ex in exception]:
                count += 1

    return count

  def onSearchScanButton(self, lineEdit):
    scan_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
    if scan_folder != '':
      lineEdit.setText(scan_folder)
      if self.type == "CBCT":
        if self.isDCMInput:
          print("DICOM")
          nbr_scans = len(os.listdir(scan_folder))
        else:
          nbr_scans = self.CountFileWithExtention(scan_folder, [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"],[])
      else:
        nbr_scans = self.CountFileWithExtention(scan_folder, [".vtk", ".stl"],[])

      if nbr_scans == 0:
        qt.QMessageBox.warning(self.parent, 'Warning', 'No scans found in the selected folder')

      else:
        self.input_path = scan_folder
        self.ui.lineEditScanPath.setText(self.input_path)
        # self.ui.PrePredInfo.setText("Number of scans to process : " + str(nbr_scans))
        self.scan_count = nbr_scans
        self.CheckScan()
      
        
  def TestFiles(self):
    """Function to download and select all the test files"""
    if self.isDCMInput:
      name, url = self.ActualMeth.getTestFileListDCM()
    else:
      name, url = self.ActualMeth.getTestFileList()

    print("name : ",name)
    print("url : ",url)

    scan_folder = self.DownloadUnzip(
      url=url,
      directory=os.path.join(self.SlicerDownloadPath),
      folder_name=os.path.join("Test_Files", name)
      if not self.isDCMInput
      else os.path.join("Test_Files", "DCM", name),
    )

    print("scan folder : ",scan_folder)

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
      self.ui.lineEditScanPath.setText(scan_folder)
      self.ui.LabelInfoPreProc.setText(
          "Number of Patients to process : " + str(nb_scans)
      )

    if self.ui.SaveFolderLineEdit.text == "":
      dir, spl = os.path.split(scan_folder)
      self.ui.SaveFolderLineEdit.setText(os.path.join(dir, spl, "Predicted"))
        
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
                shutil.copyfileobj(response, out_file)

            # Unzip the file
            with zipfile.ZipFile(temp_path, "r") as zip:
                zip.extractall(out_path)

            # Delete the zip file
            os.remove(temp_path)

        return out_path
  
  def CheckScan(self):
    """Function to test scan folder"""
    if self.isDCMInput:
      nb_scans = self.ActualMeth.NumberScanDCM(
        self.ui.lineEditScanPath.text
      )
      error = self.ActualMeth.TestScanDCM(
        self.ui.lineEditScanPath.text
      )

    else:
      nb_scans = self.ActualMeth.NumberScan(
        self.ui.lineEditScanPath.text
      )
      error = self.ActualMeth.TestScan(
        self.ui.lineEditScanPath.text
      )

    if isinstance(error, str):
      qt.QMessageBox.warning(self.parent, "Warning", error)

    else:
      self.nb_patient = nb_scans
      self.ui.LabelInfoPreProc.setText(
        "Number of Patients to process : " + str(nb_scans)
      )
      self.ui.LabelProgressPatient.setText(
        "Patient process : 0 /" + str(nb_scans)
      )

  def SearchScan(self, lineEdit):
    scan_folder = qt.QFileDialog.getExistingDirectory(
      self.parent, "Select a scan folder for Input"
    )

    if scan_folder != "":
      lineEdit.setText(scan_folder)

      if (self.ui.lineEditScanPath.text != ""):
        self.CheckScan()
        
  def loadModelFolder(self, model_folder):
    if self.type == "CBCT":
      lm_group = GetLandmarkGroup(GROUPS_LANDMARKS)
      available_lm, brain_dic = GetAvailableLm(model_folder, lm_group)

      if len(available_lm.keys()) == 0:
        qt.QMessageBox.warning(
          self.parent,
          'Warning',
          'No models found in the selected folder\nPlease select a folder containing .pth files\nYou can download the latest models with\n  "Download latest models" button'
        )
        return False
      else:
        self.model_folder = model_folder
        self.ui.lineEditModelPath.setText(self.model_folder)
        self.available_landmarks = available_lm.keys()
        self.lm_tab.Clear()
        self.lm_tab.FillTab(available_lm, enable=True)
        return True
    else:
      available_lm = self.GetAvailableSurfLm(model_folder)

      if len(available_lm.keys()) == 0:
        qt.QMessageBox.warning(
          self.parent,
          'Warning',
          'No models found in the selected folder\nPlease select a folder containing .pth files\nYou can download the latest models with\n  "Download latest models" button'
        )
        return False
      else:
        self.model_folder = model_folder
        self.ui.lineEditModelPath.setText(self.model_folder)
        self.available_landmarks = available_lm.keys()
        self.lm_tab.Clear()
        self.lm_tab.FillTab(available_lm, enable=True)
        return True

  def downloadModel(self, lineEdit):
    """Function to download the model files from the link in the getModelUrl function"""
      
    name = "Prediction" if self.type == "IOS" else "Landmark"
    
    listmodel = self.ActualMeth.getModelUrl()

    urls = listmodel[name]
    if isinstance(urls, str):
      url = urls
      _ = self.DownloadUnzip(
          url=url,
          directory=os.path.join(self.SlicerDownloadPath),
          folder_name=os.path.join("Models", name),
          num_downl=1,
          total_downloads=1,
      )
      model_folder = os.path.join(self.SlicerDownloadPath, "Models", name)

    elif isinstance(urls, dict):
      for i, (name_bis, url) in enumerate(urls.items()):
        _ = self.DownloadUnzip(
          url=url,
          directory=os.path.join(self.SlicerDownloadPath),
          folder_name=os.path.join("Models", name, name_bis),
          num_downl=i + 1,
          total_downloads=len(urls),
        )
      model_folder = os.path.join(self.SlicerDownloadPath, "Models", name)

    if not model_folder == "":
      error = self.ActualMeth.TestModel(model_folder, lineEdit.name)

      if isinstance(error, str):
        qt.QMessageBox.warning(self.parent, "Warning", error)

      else:
        lineEdit.setText(model_folder)
        self.loadModelFolder(model_folder)

  def onSearchModelButton(self):
    model_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a model folder")
    if model_folder != '':
      self.loadModelFolder(model_folder)

  def GetAvailableSurfLm(self,model_folder):
    available_lm = {}
    networks = self.GetNetworks(model_folder)
    for net in networks:
      available_lm[net] = SURFACE_LANDMARKS[net]

    return available_lm

  def GetNetworks(self,dir_path):
    networks = []
    normpath = os.path.normpath("/".join([dir_path, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and ".pth" in img_fn:
          for id, group in SURFACE_NETWORK.items():
            if id in os.path.basename(img_fn):
              networks.append(group)
    return networks

  def onSearchSaveButton(self):
    save_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
    if save_folder != '':
      self.output_folder = save_folder
      self.ui.SaveFolderLineEdit.setText(save_folder)

  def onPredictButton(self):
    if self.type == "CBCT":
      list_libs_CBCT = [('itk', None), ('dicom2nifti', '2.3.0'), ('pydicom', '2.2.2')]
      monai_version = '1.5.0' if sys.version_info >= (3, 10) else '0.7.0'
      list_libs_CBCT.append(('monai', monai_version))
      
      is_installed = install_function(self,list_libs_CBCT)
    
    else:  
      is_installed = False
      check_env = self.onCheckRequirements()
      print("seg_env : ",check_env)
      
      if check_env:
        list_libs_IOS = [('itk', None), ('dicom2nifti', '2.3.0'), ('pydicom', '2.2.2')]
        monai_version = '1.5.0' if sys.version_info >= (3, 10) else '0.7.0'
        list_libs_IOS.append(('monai', monai_version))

        is_installed = install_function(self,list_libs_IOS)
      
    if not is_installed:
      qt.QMessageBox.warning(self.parent, 'Warning', 'The module will not work properly without the required libraries.\nPlease install them and try again.')
      return
    
    self.logic.check_cli_script()
    
    self.ui.label_LibsInstallation.setVisible(False)
      
    if self.type == "IOS":
      selected_tooth_lst = self.tooth_lm.GetSelected()
      if len(selected_tooth_lst) == 0:
        qt.QMessageBox.warning(self.parent, 'Warning', 'Please select at least one tooth')
        return
      self.selected_tooth = " ".join(selected_tooth_lst)

      
    selected_lm_lst = self.lm_tab.GetSelected()
    self.landmark_cout = len(selected_lm_lst)
    if len(selected_lm_lst) == 0:
      qt.QMessageBox.warning(self.parent, 'Warning', 'Please select at least one landmark')
      return
    self.selected_lm = " ".join(selected_lm_lst)
      
    error = self.ActualMeth.TestProcess(
      input_folder=self.input_path,
      dir_models=self.model_folder,
      output_dir=self.ui.SaveFolderLineEdit.text,
    )
    
    if isinstance(error, str):
      qt.QMessageBox.warning(self.parent, "Warning", error.replace(",", "\n"))
      
    self.list_Processes_Parameters = self.ActualMeth.Process(
      input_folder=self.input_path,
      dir_models=self.model_folder,
      lm_type=self.selected_lm,
      teeth=self.selected_tooth,
      output_dir=self.ui.SaveFolderLineEdit.text,
      logPath=self.log_path,
      DCMInput=self.isDCMInput,
    )
    
    self.nb_extension_launch = len(self.list_Processes_Parameters)
    if self.type == "IOS":
      self.nb_lm = self.ActualMeth.NumberLandmark(self.selected_tooth)
    else:
      self.nb_lm = self.ActualMeth.NumberLandmark(self.selected_lm)
    self.onProcessStarted()
    
    self.module_name = self.list_Processes_Parameters[0]["Module"]
    self.displayModule = self.list_Processes_Parameters[0]["Display"]
    

    if "CrownSegmentationcli" in self.module_name:
      print("module name : ", self.module_name)
      self.nb_extension_did += 1
      self.run_conda_tool("seg")
      self.module_name = self.list_Processes_Parameters[0]["Module"]
    if "ALI_IOS" in self.module_name:
      self.nb_extension_did += 1
      print("module name : ", self.module_name)
      self.run_conda_tool("ali")
      self.OnEndProcess()
        
     
    else: 
      self.process = slicer.cli.run(
        self.list_Processes_Parameters[0]["Process"],
        None,
        self.list_Processes_Parameters[0]["Parameter"],
      )
      self.processObserver = self.process.AddObserver(
        "ModifiedEvent", self.onProcessUpdate
      )
    
      del self.list_Processes_Parameters[0]
    
  def onProcessStarted(self):
    self.ui.label_LibsInstallation.setHidden(True)
    self.ui.LabelInfoPreProc.setHidden(True)
    self.startTime = time.time()

    self.ui.progressBar.setValue(0)
    
    self.ui.LabelProgressPatient.setText(f"Patient : 0 / {self.nb_patient}")
    self.ui.LabelProgressExtension.setText(f"Extension : 1 / {self.nb_extension_launch}")
    
    self.nb_extension_did = 0
    self.module_name_before = 0
    self.nb_change_bystep = 0

    self.RunningUI(True)

  def read_txt(self):
    '''
    Read a file and return the last line
    '''
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_path,"tempo.txt")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return lines[-1] if lines else None
      
  def read_log_path(self):
      with open(self.log_path, 'r') as f:
          line = f.readline()
          if line != '':
              return line
  
  def onCondaProcessUpdate(self):
      if os.path.isfile(self.log_path):
          self.ui.LabelProgressExtension.setText(
              f"Extension : {self.nb_extension_did} / {self.nb_extension_launch}"
          )
          time_progress = os.path.getmtime(self.log_path)
          line = self.read_log_path()
          if (time_progress != self.time_log) and line:
              progress = line.strip()
          
              self.progress = int(progress)
              self.ui.LabelProgressPatient.setText(f"Patient : {self.progress}/{self.nb_patient}")
              
              progress_bar_value = round((self.progress) / self.nb_patient * 100,2)
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

        self.ui.TimerLabel.setText(timer)
        progress = caller.GetProgress()
        # self.module_name = caller.GetModuleTitle() if self.module_name_bis is None else self.module_name_bis
        self.ui.LabelNameExtension.setText(f"Running {self.module_name}")
        # self.displayModule = self.displayModule_bis if self.displayModule_bis is not None else self.display[self.module_name.split(' ')[0]]

        if self.module_name_before != self.module_name:
            self.ui.LabelProgressPatient.setText(f"Landmarks : 0 / {self.nb_lm*self.nb_patient} | Patient : 0 / {self.nb_patient}")
            self.nb_extension_did += 1
            self.ui.LabelProgressExtension.setText(
                f"Extension : {self.nb_extension_did} / {self.nb_extension_launch}"
            )
            self.ui.progressBar.setValue(0)

            # if self.nb_change_bystep == 0 and self.module_name_before:
            #     print(f'Error this module doesn\'t work {self.module_name_before}')

            self.module_name_before = self.module_name
            self.nb_change_bystep = 0

        if progress == 0:
            self.updateProgessBar = False

        if self.displayModule.isProgress(
            progress=progress, updateProgessBar=self.updateProgessBar
        ):
            progress_bar, message = self.displayModule()
            self.ui.progressBar.setValue(progress_bar)
            self.ui.LabelProgressPatient.setText(message)
            self.nb_change_bystep += 1

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
                    if self.list_Processes_Parameters[0]["Module"]=="ALI_IOS":
                        print("name process : ",self.list_Processes_Parameters[0]["Process"])
                        self.run_conda_tool("ali")
                        
                except IndexError:
                    self.OnEndProcess()
                    
  def OnEndProcess(self):
    self.ui.LabelProgressPatient.setText(f"Patient : 0 / {self.nb_lm}")
    self.ui.LabelProgressExtension.setText(
      f"Extension : {self.nb_extension_did} / {self.nb_extension_launch}"
    )
    self.ui.progressBar.setValue(0)
    
    self.module_name_before = self.module_name
    self.nb_change_bystep = 0
    total_time = time.time() - self.startTime
    average_time = total_time / self.nb_patient
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
    
    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    csv_file = os.path.join(folder_path,"ALI_Method","liste_csv_file.csv")
    print("csv_file : ",csv_file)
    if os.path.exists(csv_file):
      os.remove(csv_file)

  def onCancel(self):
    # print(self.logic.cliNode.GetOutputText())
    try:
      self.process.Cancel()
    except Exception as e:
      self.logic.cancel_process()

    print("\n\n ========= PROCESS CANCELED ========= \n")

    self.RunningUI(False)

  def RunningUI(self, run = False):

    self.HideComputeItems(run)
    
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
    
  def run_conda_tool(self, type):
    if type == "seg":
      output_command = self.logic.conda.condaRunCommand(["which","dentalmodelseg"],self.logic.name_env).strip()
      clean_output = re.search(r"Result: (.+)", output_command)
      if clean_output:
        dentalmodelseg_path = clean_output.group(1).strip()
        dentalmodelseg_path_clean = dentalmodelseg_path.replace("\\n","")
      else:
        print("Error: Unable to find dentalmodelseg path.")
        return
      
      args = self.list_Processes_Parameters[0]["Parameter"]
      print("args : ",args)
      conda_exe = self.logic.conda.getCondaExecutable()
      command = [conda_exe, "run", "-n", self.logic.name_env, "python" ,"-m", f"CrownSegmentationcli"]
      for key, value in args.items():
        if key in ["out","input_csv","vtk_folder","dentalmodelseg_path"]:
          value = self.logic.windows_to_linux_path(value)
        if key == "dentalmodelseg_path":
          value = dentalmodelseg_path_clean
        command.append(f"\"{value}\"")
      print("*"*50)
      print("command : ",command)
      
      # running in // to not block Slicer
      self.process = threading.Thread(target=self.logic.condaRunCommand, args=(command,))
      self.process.start()
      self.ui.LabelNameExtension.setText(f"Running {self.module_name}")
      self.ui.TimerLabel.setHidden(False)
      self.ui.TimerLabel.setText(f"Time : 0.00s")
      previous_time = self.startTime
      while self.process.is_alive():
        self.ui.CancelButton.setVisible(True)
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
          
          self.ui.TimerLabel.setText(timer)
      
      del self.list_Processes_Parameters[0]
    
    elif type == "ali":
      args = self.list_Processes_Parameters[0]["Parameter"]
      print("args : ", args)
      conda_exe = self.logic.conda.getCondaExecutable()
      command = [conda_exe, "run", "-n", self.logic.name_env, "python" ,"-m", f"ALI_IOS"]
      for key, value in args.items():
        print("key : ", key)
        if isinstance(value, str) and ("\\" in value or (len(value) > 1 and value[1] == ":")):
            value = self.logic.windows_to_linux_path(value)
        command.append(f"\"{value}\"")
      print("command : ",command)

      # running in // to not block Slicer
      self.process = threading.Thread(target=self.logic.condaRunCommand, args=(command,))
      self.process.start()
      self.ui.LabelNameExtension.setText(f"Running {self.module_name}")
      self.ui.TimerLabel.setHidden(False)
      self.ui.TimerLabel.setText(f"time : 0.00s")
      previous_time = self.startTime

      while self.process.is_alive():
        self.ui.CancelButton.setVisible(True)
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
            
          self.ui.TimerLabel.setText(timer)

      del self.list_Processes_Parameters[0]
      
    
  def onCheckRequirements(self):
    if not self.logic.isCondaSetUp:
      messageBox = qt.QMessageBox()
      text = textwrap.dedent("""
      SlicerConda is not set up, please click 
      <a href=\"https://github.com/DCBIA-OrthoLab/SlicerConda/\">here</a> for installation.
      """).strip()
      messageBox.information(None, "Information", text)
      return False
    
    if platform.system() == "Windows":
      self.ui.label_LibsInstallation.setHidden(False)
      self.ui.label_LibsInstallation.setText(f"Checking if wsl is installed, this task may take a moments")
      
      if self.logic.conda.testWslAvailable():
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
    
    
    ## shapeAXI


    self.ui.label_LibsInstallation.setText(f"Checking if environnement exists")
    if not self.logic.conda.condaTestEnv(self.logic.name_env) : # check is environnement exist, if not ask user the permission to do it
      userResponse = slicer.util.confirmYesNoDisplay("The environnement to run the classification doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
      if userResponse :
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
      print("pytorch3d already installed")

    self.all_installed = True   
    return True


  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    if self.logic.cliNode is not None:
      # if self.logic.cliNode.GetStatus() & self.logic.cliNode.Running:
      self.logic.cliNode.Cancel()

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
      self.updateGUIFromParameterNode
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


  def setParameterNode(self, inputParameterNode):
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
        self.updateGUIFromParameterNode
      )
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(
        self._parameterNode,
        vtk.vtkCommand.ModifiedEvent,
        self.updateGUIFromParameterNode
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

    wasModified = (
      self._parameterNode.StartModify()  # Modify all properties in a single batch
    )

    self._parameterNode.SetNodeReferenceID(
      "InputVolume", self.ui.inputSelector.currentNodeID
    )
    self._parameterNode.SetNodeReferenceID(
      "OutputVolume", self.ui.outputSelector.currentNodeID
    )
    # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
    self._parameterNode.SetParameter(
      "Invert", "true" if self.ui.invertOutputCheckBox.checked else "false"
    )
    self._parameterNode.SetNodeReferenceID(
      "OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID
    )

    self._parameterNode.EndModify(wasModified)


  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
      # Compute output
      self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
        self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

      # Compute inverted output (if needed)
      if self.ui.invertedOutputSelector.currentNode():
        # If additional output volume is selected then result with inverted threshold is written there
        self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
          self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()
      
  def HideComputeItems(self, run=False):
    self.ui.PredictionButton.setVisible(not run)

    self.ui.CancelButton.setVisible(run)

    self.ui.LabelProgressPatient.setVisible(run)
    self.ui.LabelProgressExtension.setVisible(run)
    self.ui.LabelNameExtension.setVisible(run)
    self.ui.progressBar.setVisible(run)

    self.ui.TimerLabel.setVisible(run)

class LMTab:
    def __init__(self) -> None:

      self.widget = qt.QWidget()
      layout = qt.QVBoxLayout(self.widget)

      self.LM_tab_widget = qt.QTabWidget()
      self.LM_tab_widget.minimumSize = qt.QSize(100,200)
      self.LM_tab_widget.maximumSize = qt.QSize(800,400)
      self.LM_tab_widget.setMovable(True)


      # print(self.lm_status_dic)
      # print(lcbd)
      buttons_wid = qt.QWidget()
      buttons_layout = qt.QHBoxLayout(buttons_wid)
      self.select_all_btn = qt.QPushButton("Select All")
      self.select_all_btn.setEnabled(False)
      self.select_all_btn.connect('clicked(bool)', self.SelectAll)
      self.clear_all_btn = qt.QPushButton("Clear All")
      self.clear_all_btn.setEnabled(False)
      self.clear_all_btn.connect('clicked(bool)', self.ClearAll)

      buttons_layout.addWidget(self.select_all_btn)
      buttons_layout.addWidget(self.clear_all_btn)

      layout.addWidget(self.LM_tab_widget)
      layout.addWidget(buttons_wid)
      self.lm_status_dic = {}

    def Clear(self):
      self.LM_tab_widget.clear()

    def FillTab(self,lm_dic, enable = False):
      self.select_all_btn.setEnabled(enable)
      self.clear_all_btn.setEnabled(enable)

      self.lm_group_dic = lm_dic.copy()
      self.lm_group_dic["All"] = []

      cbd = {}
      lmsd = {}
      for group,lm_lst in lm_dic.items():
          # lm_lst = lm_lst.sort()
          for lm in sorted(lm_lst):
              if lm not in lmsd.keys():
                  lmsd[lm] = False
                  self.lm_group_dic["All"].append(lm)

      self.check_box_dic = cbd
      self.lm_status_dic = lmsd


      for group,lm_lst in self.lm_group_dic.items():
        lst_wid = []
        # lm_lst = lm_lst.sort()
        for lm in sorted(lm_lst):
          new_cb = qt.QCheckBox(lm)
          self.check_box_dic[new_cb] = lm
          lst_wid.append(new_cb)

        new_lm_tab = self.GenNewTab(lst_wid,enable)
        # new_lm_tab.setEnabled(False)
        self.LM_tab_widget.insertTab(0,new_lm_tab,group)

      self.LM_tab_widget.currentIndex = 0

      # print(self.check_box_dic)
      lcbd = {}
      for cb,lm in self.check_box_dic.items():
        if lm not in lcbd.keys():
          lcbd[lm] = [cb]
        else:
          lcbd[lm].append(cb)

      self.lm_cb_dic = lcbd

      # for lm in lm_dic["U"]:
      #   self.UpdateLmSelect(lm,True)

      for cb in self.check_box_dic.keys():
        cb.connect("toggled(bool)", self.CheckBox)
        cb.setEnabled(enable)

    def CheckBox(self, caller=None, event=None):
      for cb,lm in self.check_box_dic.items():
        if cb.checkState():
          state = True
        else:
          state = False

        if self.lm_status_dic[lm] != state:
          self.UpdateLmSelect(lm,state)

    def ToggleSelection(self):
      idx = self.LM_tab_widget.currentIndex
      # print(idx)
      group = self.LM_tab_widget.tabText(idx)
      # print(group)
      new_state = not self.lm_status_dic[self.lm_group_dic[group][0]]
      # print(new_state)
      for lm in self.lm_group_dic[group]:
        self.UpdateLmSelect(lm,new_state)


    def GenNewTab(self,widget_lst,enable = False):
        new_widget = qt.QWidget()
        vb = qt.QVBoxLayout(new_widget)
        scr_box = qt.QScrollArea()
        vb.addWidget(scr_box)
        pb = qt.QPushButton('Switch group selection')
        pb.connect('clicked(bool)', self.ToggleSelection)
        pb.setEnabled(enable)
        vb.addWidget(pb)

        wid = qt.QWidget()
        vb2 = qt.QVBoxLayout()
        for widget in widget_lst:
          vb2.addWidget(widget)
        vb2.addStretch()
        wid.setLayout(vb2)

        scr_box.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOn)
        scr_box.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        scr_box.setWidgetResizable(True)
        scr_box.setWidget(wid)

        return new_widget

    def UpdateLmSelect(self,lm_id,state):
      for cb in self.lm_cb_dic[lm_id]:
        cb.setChecked(state)
      self.lm_status_dic[lm_id] = state

    def UpdateAll(self,state):
      for lm_id,cb_lst in self.lm_cb_dic.items():
        for cb in cb_lst:
          cb.setChecked(state)
        self.lm_status_dic[lm_id] = state

    def GetSelected(self):
      selectedLM = []
      for lm,state in self.lm_status_dic.items():
        if state:
          selectedLM.append(lm)
      return selectedLM

    def SelectAll(self):
      self.UpdateAll(True)

    def ClearAll(self):
      self.UpdateAll(False)

def GetAvailableLm(mfold,lm_group):
  brain_dic = GetBrain(mfold)
  # print(brain_dic)
  available_lm = {}
  for lm in brain_dic.keys():
    if lm in lm_group.keys():
      group = lm_group[lm]
    else:
      group = "Other"
    if group not in available_lm.keys():
      available_lm[group] = [lm]
    else:
      available_lm[group].append(lm)

  # available_lm = brain_dic.keys()

  return available_lm,brain_dic


def GetLandmarkGroup(group_landmark):
  lm_group = {}
  for group,labels in group_landmark.items():
      for label in labels:
          lm_group[label] = group
  return lm_group

def GetBrain(dir_path):
    brainDic = {}
    normpath = os.path.normpath("/".join([dir_path, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        if os.path.isfile(img_fn) and ".pth" in img_fn:
            lab = os.path.basename(os.path.dirname(os.path.dirname(img_fn)))
            num = os.path.basename(os.path.dirname(img_fn))
            if lab in brainDic.keys():
                brainDic[lab][num] = img_fn
            else:
                network = {num : img_fn}
                brainDic[lab] = network

    return brainDic

#
# ALILogic
#

class ALILogic(ScriptedLoadableModuleLogic):
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

    self.cliNode = None
    self.isCondaSetUp = False
    self.conda = self.init_conda()
    self.name_env = "shapeaxi"
    self.cliNode = None
    self.pythonVersion = "3.9"  # Default Python version for the conda environment

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
    self.run_conda_command(target=self.conda.condaCreateEnv, command=(self.name_env,self.pythonVersion,["shapeaxi==1.0.10"],)) #run in parallel to not block slicer
    
  def check_if_pytorch3d(self):
    conda_exe = self.conda.getCondaExecutable()
    command = [conda_exe, "run", "-n", self.name_env, "python" ,"-c", f"\"import pytorch3d;import pytorch3d.renderer\""]
    return self.conda.condaRunCommand(command)

  def install_pytorch3d(self):
    result_pythonpath = self.check_pythonpath_windows("ALI_Method.install_pytorch")
    if not result_pythonpath :
      self.give_pythonpath_windows()
      result_pythonpath = self.check_pythonpath_windows("ALI_Method.install_pytorch")
    
    if result_pythonpath : 
      conda_exe = self.conda.getCondaExecutable()
      path_pip = self.conda.getCondaPath()+f"/envs/{self.name_env}/bin/pip"
      command = [conda_exe, "run", "-n", self.name_env, "python" ,"-m", f"ALI_Method.install_pytorch",path_pip]

    self.run_conda_command(target=self.conda.condaRunCommand, command=(command,))
    
  def setup_cli_command(self):
    args = self.find_cli_parameters()
    conda_exe = self.conda.getCondaExecutable()
    command = [conda_exe, "run", "-n", self.name_env, "python" ,"-m", f"ALI_IOS"]
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
  
  def cancel_process(self):
    if platform.system() == 'Windows':
      self.subpro.send_signal(signal.CTRL_BREAK_EVENT)
    else:
      os.killpg(os.getpgid(self.subpro.pid), signal.SIGTERM)
    print("Cancellation requested. Terminating process...")

    self.subpro.wait() ## important
    self.cancel = True

  def check_cli_script(self):
    if not self.check_pythonpath_windows("ALI_IOS"): 
      self.give_pythonpath_windows()
      results = self.check_pythonpath_windows("ALI_IOS")
        
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