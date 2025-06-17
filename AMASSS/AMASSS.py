"""
Automatic multi-anatomical skull structure segmentation (AMASSS) of cone-beam computed tomography scans (CBCT)

Authors :
- Maxime Gillot (UoM)
- Baptiste Baquero (UoM)

"""

import os
import logging
import glob
import time
import shutil
import vtk, qt, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install

import webbrowser
import pkg_resources
import importlib
import platform
import threading
def check_lib_installed(lib_name, required_version=None,system="Windows"):
    '''
    Check if the library with the good version (if needed) is already installed in the slicer environment
    input: lib_name (str) : name of the library
            required_version (str) : required version of the library (if None, any version is accepted)
    output: bool : True if the library is installed with the good version, False otherwise
    '''
    if system == "Windows":
      lib_torch = ["torch","torchvision","torchaudio"]
      list_cuda_version =[]
      if lib_name in lib_torch:
        for lib_str in lib_torch:
          try:
            lib = importlib.import_module(lib_str)
            cuda_version = lib.__version__.split('cu')[1]
            list_cuda_version.append(cuda_version)
          except:
            return False
        for i in range(len(list_cuda_version)-1):
          if list_cuda_version[i] != list_cuda_version[i+1]:
            return False
          else:
            return True

    try:
        installed_version = pkg_resources.get_distribution(lib_name).version
        # check if the version is the good one - if required_version != None it's considered as a True
        if required_version and installed_version != required_version:
          return False
        else:
          return True
    except pkg_resources.DistributionNotFound:
        return False

# import csv

def install_function(self,list_libs:list,system:str):
    '''
    Test the necessary libraries and install them with the specific version if needed
    User is asked if he wants to install/update-by changing his environment- the libraries with a pop-up window
    '''
    libs = list_libs
    libs_to_install = []
    libs_to_update = []
    for lib, version in libs:
        if not check_lib_installed(lib, version,system):
            try:
              # check if the library is already installed
              if pkg_resources.get_distribution(lib).version:
                libs_to_update.append((lib, version))
            except:
              libs_to_install.append((lib, version))

    if libs_to_install or libs_to_update:
          message = "The following changes are required for the libraries:\n"

          #Specify which libraries will be updated with a new version
          #and which libraries will be installed for the first time
          if libs_to_update:

              message += "\nLibraries to update (version mismatch):\n"
              message += "\n".join([f"{lib} (current: {pkg_resources.get_distribution(lib).version}) -> {version}" for lib, version in libs_to_update])

          if libs_to_install:
              message += "\nLibraries to install:\n"
              message += "\n".join([f"{lib}=={version}" if version else lib for lib, version in libs_to_install])

          message += "\n\nDo you agree to modify these libraries? Doing so could cause conflicts with other installed Extensions."
          message += "\n\n (If you are using other extensions, consider downloading another Slicer to use AutomatedDentalTools exclusively.)"

          user_choice = slicer.util.confirmYesNoDisplay(message)

          if user_choice:
              self.ui.label_installation.setVisible(True)
              self.ui.nb_package.setVisible(True)
              len_libs = len(libs_to_install+libs_to_update)
              self.ui.nb_package.setText(f"Package: 0/{len_libs}")
              nb_installed = 0

              if system == "Windows":
                # Installation specified for Windows system
                already_installed =False
                for lib, version in libs_to_install:
                  if lib == "torch" or lib=="torchvision" or lib== "torchaudio":
                    try:
                      import torch
                      if torch.cuda.is_available():
                        cuda_version = torch.version.cuda
                        if cuda_version =="11.8" or cuda_version=="12.1":
                          cuda_version= f"cu{cuda_version.replace('.','')}"
                        elif float(cuda_version) > 12.1:
                          cuda_version = "cu121"
                        elif 11.8 < float(cuda_version) < 12.1:
                          cuda_version = "cu118"
                        else:
                          cuda_version = "cu118"
                        print('required cuda version:',cuda_version)
                      else:
                        raise RuntimeError("CUDA is not available")
                    except ImportError:
                      cuda_version = "cu121"
                    except RuntimeError:
                      cuda_version = "cu121"

                    if not already_installed:
                      already_installed = True
                      pip_install(f'torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/{cuda_version}')
                      nb_installed += 3

                  else:
                    lib_version = f'{lib}=={version}' if version else lib
                    pip_install(lib_version)
                    nb_installed += 1
                  self.ui.nb_package.setText(f"Package: {nb_installed}/{len_libs}")

                return True

              else:


                pip_install(f'torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118')
                for lib, version in libs_to_install:
                  print('lib:',lib)
                  lib_version = f'{lib}=={version}' if version else lib
                  pip_install(lib_version)
                  nb_installed += 1
                  self.ui.nb_package.setText(f"Package: {nb_installed}/{len_libs}")


                for lib, version in libs_to_update:
                  lib_version = f'{lib}=={version}' if version else lib
                  pip_install(lib_version)
                  nb_installed += 1
                  self.ui.nb_package.setText(f"Package: {nb_installed}/{len_libs}")

                return True
          else :
            return False
    else:
        return True

#region ========== FUNCTIONS ==========
def GetSegGroup(group_landmark):
  seg_group = {}
  for group,labels in group_landmark.items():
      for label in labels:
          seg_group[label] = group
  return seg_group



def PathFromNode(node):
  storageNode=node.GetStorageNode()
  if storageNode is not None:
    filepath=storageNode.GetFullNameFromFileName()
  else:
    filepath=None
  return filepath

#endregion


def createProgressDialog(parent=None, value=0, maximum=100, windowTitle="Starting..."):
    # import qt # qt.qVersion()
    progressIndicator = qt.QProgressDialog()  #(parent if parent else self.mainWindow())
    progressIndicator.minimumDuration = 0
    progressIndicator.maximum = maximum
    progressIndicator.value = value
    progressIndicator.windowTitle = windowTitle
    return progressIndicator


#region ========= GLOBAL VARIABLES =========

# MODEL_LINK = 'https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.0-alpha/ALL_MODELS.zip'
MODEL_LINK = 'https://github.com/lucanchling/AMASSS_CBCT/releases/tag/v1.0.2'
SCAN_LINK = 'https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/MG_test_scan.nii.gz'


GROUPS_FF_SEG = {
  "Bones" : ["Mandible","Maxilla","Cranial base","Cervical vertebra"],
  "Soft tissue" :['Upper airway','Skin',],
  "Masks" : ["Cranial Base (Mask)","Mandible (Mask)","Maxilla (Mask)"],
}


GROUPS_HD_SEG = {
  "Bones" : ["Mandible","Maxilla"],
  "Teeth" : ['Teeth','Root canal'],
  "Nerves" : ['Mandibular canal'],


}


DEFAULT_SELECT = ["Mandible","Maxilla","Cranial base","Cervical vertebra","Upper airway","Root canal"]

UNAVAILABLE_MODELS = ["Teeth","Mandibular canal"]

TRANSLATE ={
  "Mandible" : "MAND",
  "Maxilla" : "MAX",
  "Cranial base" : "CB",
  "Cervical vertebra" : "CV",
  "Root canal" : "RC",
  "Mandibular canal" : "MCAN",
  "Upper airway" : "UAW",
  "Skin" : "SKIN",
  "Teeth" : "TEETH",
  "Cranial Base (Mask)" : "CBMASK",
  "Mandible (Mask)" : "MANDMASK",
  "Maxilla (Mask)" : "MAXMASK",
}


LOADED_VTK_FILES = {}

#endregion


#
# AMASSS
#

class AMASSS(ScriptedLoadableModule):
  """
  AMASSS (https://github.com/Maxlo24/Slicer_Automatic_Tools/blob/main/AMASSS/AMASSS.py)
  Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "AMASSS"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Automated Dental Tools"]  # set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Maxime Gillot (CPE Lyon & UoM), Baptiste Baquero (CPE Lyon & UoM), Lucia Cevidanes (UoM), Juan Carlos Prieto (UoNC)"]  # TODO: replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
      This is a module that will allow you to automatically perform segmentation of skull structures in your CBCT scans.
      """
    self.parent.acknowledgementText = """
    This file was developed by Maxime Gillot (CPE Lyon & UoM), Baptiste Baquero (CPE Lyon & UoM)
    and was supported by NIDCR R01 024450, AA0F Dewel Memorial Biomedical Research award and by
    Research Enhancement Award Activity 141 from the University of the Pacific, Arthur A. Dugoni School of Dentistry.
    """

#
# AMASSSWidget
#

class AMASSSWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """
  AMASSS (https://github.com/Maxlo24/Slicer_Automatic_Tools/blob/main/AMASSS/AMASSS.py)
  Uses ScriptedLoadableModuleWidget base class, available at:
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

    self.MRMLNode_scan = None # MRML node of the selected scan


    self.input_path = None # path to the folder containing the scans
    self.folder_as_input = False # 0 for file, 1 for folder

    self.isSegmentInput = False # Is the input (folder or file) is a Segmentation
    self.isDCMInput = False

    self.output_folder = None # path to the folder where the segmentations will be saved
    self.vtk_output_folder = None
    self.all_installed = False


    self.use_small_FOV = False # use high resolution model


    # self.model_ready = False # model selected
    # self.scan_ready = False # scan is selected

    self.save_surface = True # True: save surface .vtk of the segmentation
    self.output_selection = "MERGE" # m: merged, s: separated, ms: both
    self.prediction_ID = "Seg_Pred" # ID to put in the prediction name
    self.save_in_input_folder = True # path to the folder where the results will be saved


    self.center_all = False # True: center all the scan seg and surfaces in the same position
    self.save_adjusted = False # True: save the contrast adjusted scan
    self.precision = 50 # Default precision for the segmentation
    self.smoothing = 5 # Default smoothing value for the generated surface


    self.scan_count = 0 # number of scans in the input folder
    self.seg_cout = 0 # number of segmentations to perform

    self.total_seg_progress = 0 # total number of step to perform

    self.prediction_step = 0 # step of the prediction
    self.progress = 0 # progress of the prediction

    self.startTime = 0 # start time of the prediction


  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    #region ===== WIDGET =====


    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/AMASSS.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = AMASSSLogic()

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
    #endregion

    #region ===== CONNECTIONS =====


      #region == INPUT ==

    # Input type
    self.ui.input_type_select.currentIndexChanged.connect(self.SwitchInputType)
    self.SwitchInputType(0)

    # For NIFTI, NRRD, GIPL, DICOM or Segmentation as input
    self.ui.InputTypecomboBox.currentIndexChanged.connect(self.SwitchInputExtension)

    # Input scan
    self.ui.MRMLNodeComboBox_file.setMRMLScene(slicer.mrmlScene)
    self.ui.MRMLNodeComboBox_file.currentNodeChanged.connect(self.onNodeChanged)
    self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox_file.currentNodeID)

    self.ui.SearchScanFolder.connect('clicked(bool)',self.onSearchScanButton)

    # model folder
    self.ui.SearchModelFolder.connect('clicked(bool)',self.onSearchModelButton)

    # Download model
    self.ui.DownloadButton.connect('clicked(bool)',self.onDownloadButton)
    self.ui.DownloadScanButton.connect('clicked(bool)',self.onDownloadScanButton)

    #endregion

      #region == SEGMENTATION SELECTION ==

    self.ui.smallFOVCheckBox.connect("toggled(bool)",self.onSmallFOVCheckBox)
    self.seg_tab = LMTab()
    # seg_tab_widget,seg_buttons_dic = GenLandmarkTab(Landmarks_group)
    self.ui.OptionVLayout.addWidget(self.seg_tab.widget)
    self.seg_tab.Clear()
    self.seg_tab.FillTab(GROUPS_FF_SEG)

    #endregion

      #region == OUTPUT ==

    # Output type
    self.ui.OutputTypecomboBox.currentIndexChanged.connect(self.SwitchOutputType)

    # Generate vtk file
    self.ui.checkBoxSurfaceSelect.connect("toggled(bool)", self.UpdateSaveSurface)

    # Save in a folder
    self.ui.SavePredictCheckBox.connect("toggled(bool)", self.UpdateSaveFolder)

    # folder selection
    self.ui.SearchSaveFolder.connect('clicked(bool)',self.onSearchSaveButton)
    self.ui.SearchSaveFolder.setHidden(True)
    self.ui.SaveFolderLineEdit.setHidden(True)
    self.ui.PredictFolderLabel.setHidden(True)

    # Checkbox to enable usage of CPU memory in case GPU memory is not enough
    self.ui.host_memory.setChecked(False)

    self.ui.label_4.setVisible(False)
    self.ui.horizontalSliderCPU.setVisible(False)
    self.ui.spinBoxCPU.setVisible(False)

    #endregion


      #region == MORE OPTIONS ==

    # Center all
    # self.ui.CenterAllCheckBox.connect("toggled(bool)", self.UpdateCenterAll)

    # Save adjusted
    # self.ui.SaveAdjustedCheckBox.connect("toggled(bool)", self.UpdateSaveAdjusted)

    # precision
    self.ui.horizontalSliderPrecision.valueChanged.connect(self.onPrecisionSlider)
    self.ui.horizontalSliderSmoothing.setVisible(True)
    self.ui.spinBoxPrecision.valueChanged.connect(self.onPrecisionSpinbox)
    self.ui.spinBoxPrecision.setVisible(True)

    # smoothing
    self.ui.horizontalSliderSmoothing.valueChanged.connect(self.onSmoothingSlider)
    self.ui.spinBoxSmoothing.valueChanged.connect(self.onSmoothingSpinbox)

    self.ui.horizontalSliderGPU.valueChanged.connect(self.onGPUSlider)
    self.ui.spinBoxGPU.valueChanged.connect(self.onGPUSpinbox)

    self.ui.horizontalSliderCPU.valueChanged.connect(self.onCPUSlider)
    self.ui.spinBoxCPU.valueChanged.connect(self.onCPUSpinbox)

    self.UpdateSaveSurface(False)

    self.ui.nb_package.setVisible(False)
    self.ui.label_installation.setVisible(False)

    #endregion

      #region == RUN ==


    self.ui.PredictionButton.connect('clicked(bool)', self.onPredictButton)

    self.ui.CancelButton.connect('clicked(bool)', self.onCancel)
    self.RunningUI(False)



      #endregion

    #endregion

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()
    qt.QTimer.singleShot(100, self.onCheckRequirements)



  #region ====== FUNCTIONS ======

    #region == INPUT ==

  def SwitchInputType(self,index):

    if index == 1:
      self.folder_as_input = True
      self.input_path = None

    else:
      self.folder_as_input = False
      self.onNodeChanged()

    self.ui.label_folder_select.setVisible(self.folder_as_input)
    self.ui.lineEditScanPath.setVisible(self.folder_as_input)
    self.ui.SearchScanFolder.setVisible(self.folder_as_input)

    self.ui.label_node_select.setVisible(not self.folder_as_input)
    self.ui.MRMLNodeComboBox_file.setVisible(not self.folder_as_input)
    self.ui.emptyLabelNodeSelect.setVisible(not self.folder_as_input)

  def SwitchInputExtension(self,index):
    if index == 0: # NIFTI, NRRD, GIPL Files
      self.isSegmentInputFunction(False)
      self.SwitchInputType(0)
      self.isDCMInput = False

      self.ui.label.setVisible(True)
      self.ui.input_type_select.setVisible(True)
    if index == 1: # DICOM Files
      self.isSegmentInputFunction(False)
      self.SwitchInputType(1)
      self.ui.label.setVisible(False)
      self.ui.input_type_select.setVisible(False)
      self.ui.label_folder_select.setText('DICOM\'s Folder')
      self.isDCMInput = True

    if index == 2: # Segmentation Files
      self.isSegmentInputFunction(True)

  def isSegmentInputFunction(self,SegInput):

    # Set the value to True when checked and vice-versa
    # self.isSegmentInput = not self.isSegmentInput

    if SegInput:
      self.isSegmentInput = True
      self.isDCMInput = False
      self.ui.label_folder_select.setText("Segmentation's Folder")
      self.ui.OptionVLayout.removeWidget(self.seg_tab.widget)
      self.ui.PrePredInfo.setText("Number of segmentation to process : 0")

    else:
      self.isSegmentInput = False
      self.ui.label_folder_select.setText("Scan's Folder")
      self.ui.OptionVLayout.addWidget(self.seg_tab.widget)
      self.seg_tab.Clear()
      self.seg_tab.FillTab(GROUPS_FF_SEG)
      self.ui.PrePredInfo.setText("Number of scans to process : 0")
    # Set to invisble all the unnecessary input

    self.ui.DownloadScanButton.setVisible(not SegInput)
    self.ui.DownloadButton.setVisible(not SegInput)
    self.ui.label_model_select.setVisible(not SegInput)
    self.ui.lineEditModelPath.setVisible(not SegInput)
    self.ui.SearchModelFolder.setVisible(not SegInput)

    self.ui.smallFOVCheckBox.setVisible(not SegInput)
    self.ui.label_6.setVisible(not SegInput)

    # OUTPUT
    self.ui.CenterAllCheckBox.setVisible(not SegInput)
    self.ui.SaveAdjustedCheckBox.setVisible(not SegInput)

    self.ui.label_2.setVisible(not SegInput)
    self.ui.OutputTypecomboBox.setVisible(not SegInput)

    self.ui.label_9.setVisible(not SegInput)
    self.ui.SaveId.setVisible(not SegInput)

    self.ui.checkBoxSurfaceSelect.setVisible(not SegInput)




    # ADVANCED
    self.ui.labelSmoothing.setVisible(SegInput)
    self.ui.horizontalSliderSmoothing.setVisible(SegInput)
    self.ui.spinBoxSmoothing.setVisible(SegInput)

    self.ui.saveInFolder.setVisible(not SegInput)

    self.ui.labelPrecision.setVisible(not SegInput)
    self.ui.horizontalSliderPrecision.setVisible(not SegInput)
    self.ui.spinBoxPrecision.setVisible(not SegInput)

    self.ui.label_3.setVisible(not SegInput)
    self.ui.horizontalSliderGPU.setVisible(not SegInput)
    self.ui.spinBoxGPU.setVisible(not SegInput)

    self.ui.label_CPU_use.setVisible(not SegInput)
    self.ui.host_memory.setVisible(not SegInput)



    # self.ui..setVisible(not self.isSegmentInput)


  def onNodeChanged(self):
    selected = False
    self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox_file.currentNodeID)
    if self.MRMLNode_scan is not None:
      print(PathFromNode(self.MRMLNode_scan))
      self.input_path = PathFromNode(self.MRMLNode_scan)
      self.scan_count = 1


      self.ui.PrePredInfo.setText("Number of scans to process : 1")
      selected = True

    return selected

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

  def onSearchScanButton(self):
    file_explorer = qt.QFileDialog()
    # file_explorer.setFileMode(qt.QFileDialog.AnyFile)
    scan_folder = file_explorer.getExistingDirectory(self.parent, "Select a scan folder")

    if scan_folder != '':
      if self.isSegmentInput:
        nbr_scans = self.CountFileWithExtention(scan_folder, [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"],exception=["scan"])
      elif self.isDCMInput:
        nbr_scans = len(os.listdir(scan_folder))
      else:
        nbr_scans = self.CountFileWithExtention(scan_folder, [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"])
      if nbr_scans == 0:
        qt.QMessageBox.warning(self.parent, 'Warning', 'No scans found in the selected folder')

      else:
        self.input_path = scan_folder
        self.ui.lineEditScanPath.setText(self.input_path)
        self.ui.PrePredInfo.setText("Number of scans to process : " + str(nbr_scans))
        self.scan_count = nbr_scans


  def onSearchModelButton(self):
    model_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a model folder")
    if model_folder != '':
      nbr_model = self.CountFileWithExtention(model_folder, [".pth"], [])
      if nbr_model == 0:
        qt.QMessageBox.warning(self.parent, 'Warning', 'No models found in the selected folder\nPlease select a folder containing .pth files\nYou can download the latest models with\n  "Download latest models" button')

      else:
        self.ui.lineEditModelPath.setText(model_folder)
        self.model_ready = True

  def onDownloadButton(self):
    webbrowser.open(MODEL_LINK)

  def onDownloadScanButton(self):
    webbrowser.open(SCAN_LINK)

    #endregion


    #region == SEGMENTATION SELECTION ==
  def onSmallFOVCheckBox(self,checked):
    self.use_small_FOV = checked
    if self.use_small_FOV:
      self.seg_tab.Clear()
      self.seg_tab.FillTab(GROUPS_HD_SEG)
    else:
      self.seg_tab.Clear()
      self.seg_tab.FillTab(GROUPS_FF_SEG)
    #region == OUTPUT ==

  def UpdateSaveFolder(self,checked):

    # print(caller,event)
    hide = checked

    self.ui.SearchSaveFolder.setHidden(hide)
    self.ui.SaveFolderLineEdit.setHidden(hide)
    self.ui.PredictFolderLabel.setHidden(hide)

    # self.ui.SearchSaveFolder.setEnabled(not caller)
    # self.ui.SaveFolderLineEdit.setEnabled(not caller)

    self.save_in_input_folder = checked

  def UpdateSaveSurface(self,checked):

    self.ui.labelSmoothing.setVisible(checked)
    self.ui.horizontalSliderSmoothing.setVisible(checked)
    self.ui.spinBoxSmoothing.setVisible(checked)

    self.save_surface = checked
    if checked:
      self.ui.saveInFolder.checked = True

  def onSearchSaveButton(self):
    save_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
    if save_folder != '':
      self.save_folder = save_folder
      self.ui.SaveFolderLineEdit.setText(self.save_folder)

  def SwitchOutputType(self,index):

    print("Selected output type:",index)
    if index == 0:
      self.output_selection = "MERGE"
      self.ui.saveInFolder.checked = False
    elif index == 1:
      self.output_selection = "SEPARATE"
      self.ui.saveInFolder.checked = True

    elif index == 2:
      self.output_selection = "MERGE SEPARATE"
      self.ui.saveInFolder.checked = True
    #endregion

    #region == MORE OPTIONS ==

  def onPrecisionSlider(self):
    self.precision = self.ui.horizontalSliderPrecision.value
    self.ui.spinBoxPrecision.value = self.precision

  def onPrecisionSpinbox(self):
    self.precision = self.ui.spinBoxPrecision.value
    self.ui.horizontalSliderPrecision.value = self.precision

  def onSmoothingSlider(self):
    self.smoothing = self.ui.horizontalSliderSmoothing.value
    self.ui.spinBoxSmoothing.value = self.smoothing

  def onSmoothingSpinbox(self):
    self.smoothing = self.ui.spinBoxSmoothing.value
    self.ui.horizontalSliderSmoothing.value = self.smoothing

  def onGPUSlider(self):
    self.ui.spinBoxGPU.value = self.ui.horizontalSliderGPU.value

  def onGPUSpinbox(self):
    self.ui.horizontalSliderGPU.value = self.ui.spinBoxGPU.value

  def onCPUSlider(self):
    self.ui.spinBoxCPU.value = self.ui.horizontalSliderCPU.value

  def onCPUSpinbox(self):
    self.ui.horizontalSliderCPU.value = self.ui.spinBoxCPU.value


    #region == Requirements ==

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
        self.ui.label_installation.setHidden(False)
        self.ui.label_installation.setText(f"Checking if wsl is installed, this task may take a moments")
        
        if self.logic.conda.testWslAvailable():
            self.ui.label_installation.setText(f"WSL installed")
            if not self.logic.check_lib_wsl():
                self.ui.label_installation.setText(f"Checking if the required librairies are installed, this task may take a moments")
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
    
    
    self.ui.label_installation.setText(f"Checking if miniconda is installed")
    if "Error" in self.logic.conda.condaRunCommand([self.logic.conda.getCondaExecutable(),"--version"]):
        messageBox = qt.QMessageBox()
        text = textwrap.dedent("""
        Code can't be launch. \nConda is not setup. 
        Please go the extension CondaSetUp in SlicerConda to do it.""").strip()
        messageBox.information(None, "Information", text)
        return False
    
    
    ## shapeAXI


    self.ui.label_installation.setText(f"Checking if environnement exists")
    if not self.logic.conda.condaTestEnv(self.logic.name_env) : # check is environnement exist, if not ask user the permission to do it
        userResponse = slicer.util.confirmYesNoDisplay("The environnement to run the classification doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
        if userResponse :
            start_time = time.time()
            previous_time = start_time
            formatted_time = self.format_time(0)
            self.ui.label_installation.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
            process = self.logic.install_shapeaxi()
            
            while self.logic.process.is_alive():
                slicer.app.processEvents()
                formatted_time = self.update_ui_time(start_time, previous_time)
                self.ui.label_installation.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
        
            start_time = time.time()
            previous_time = start_time
            formatted_time = self.format_time(0)
            text = textwrap.dedent(f"""
            Installation of librairies into the new environnement. 
            This task may take a few minutes.\ntime: {formatted_time}""").strip()
            self.ui.label_installation.setText(text)
        else:
            return False
    else:
        self.ui.label_installation.setText(f"Ennvironnement already exists")
        
    self.all_installed = True   
    return True

    #endregion


    #region == RUN ==
  def onPredictButton(self):

    if not self.all_installed:
      qt.QMessageBox.warning(self.parent, 'Warning', 'The module will not work properly without the required libraries.\nPlease install them and try again.')
      return  # stop the function

    self.ui.nb_package.setVisible(False)
    self.ui.label_installation.setVisible(False)

    ready = True

    if self.folder_as_input:
      if self.ui.lineEditScanPath.text == "":
        qt.QMessageBox.warning(self.parent, 'Warning', 'Please select a scan folder')
        ready = False
    else:
      if not self.onNodeChanged():
        qt.QMessageBox.warning(self.parent, 'Warning', 'Please select an input file')
        ready = False

    if self.ui.lineEditModelPath.text == "" and not self.isSegmentInput:
      qt.QMessageBox.warning(self.parent, 'Warning', 'Please select a model folder')
      ready = False

    if not ready:
      return

    selected_seg = []
    for struct in self.seg_tab.GetSelected():
      selected_seg.append(TRANSLATE[struct])


    if len(selected_seg) == 0:
      qt.QMessageBox.warning(self.parent, 'Warning', 'No segmentation selected')
      return

    self.seg_cout = len(selected_seg)

    param = {}

    param["inputVolume"] = self.ui.lineEditScanPath.text if self.folder_as_input else self.input_path

    param["modelDirectory"] = '/' if self.isSegmentInput else self.ui.lineEditModelPath.text
    param["highDefinition"] = self.use_small_FOV

    param["skullStructure"] = " ".join(selected_seg)
    param["merge"] = self.output_selection
    param["genVtk"] = self.save_surface
    param["save_in_folder"] = self.ui.saveInFolder.isChecked() or self.ui.checkBoxSurfaceSelect.isChecked()


    if self.save_in_input_folder:
      if os.path.isfile(self.input_path):
        self.output_folder = os.path.dirname(self.input_path)

        baseName = os.path.basename(self.input_path)
        scan_name= baseName.split(".")

        outputdir = self.output_folder
        if param["save_in_folder"]:
            outputdir += "/" + scan_name[0] + "_" + "SegOut"

        self.vtk_output_folder = outputdir + "/VTK files"


      else:
        self.output_folder = self.input_path
        self.vtk_output_folder = None

    else:
      self.output_folder = self.ui.SaveFolderLineEdit.text
      self.vtk_output_folder = None


    if not self.ui.checkBoxSurfaceSelect.isChecked():
      self.vtk_output_folder = None


    param["output_folder"] = self.output_folder


    param["precision"] = self.precision
    param["vtk_smooth"] = self.smoothing

    param["prediction_ID"] = self.ui.SaveId.text

    param["gpu_usage"] = self.ui.spinBoxGPU.value
    param["host_memory"] = self.ui.host_memory.isChecked()


    documentsLocation = qt.QStandardPaths.DocumentsLocation
    documents = qt.QStandardPaths.writableLocation(documentsLocation)
    temp_dir = os.path.join(documents, slicer.app.applicationName+"_temp_AMASSS")

    print(temp_dir)

    param["temp_fold"] = temp_dir

    param["SegmentInput"] = self.isSegmentInput

    param["DCMInput"] = self.isDCMInput

    # self.logic.process(param)
    # self.processObserver = self.logic.cliNode.AddObserver('ModifiedEvent',self.onProcessUpdate)
    # self.onProcessStarted()

    self.logic.check_cli_script(f"AMASSS_CLI")

    conda_exe = self.logic.conda.getCondaExecutable()
    command = [conda_exe, "run", "-n", self.logic.name_env, "python" ,"-m", "AMASSS_CLI"]

    for key, value in param.items():
        print("key : ",key)
        if isinstance(value, str) and ("\\" in value or (len(value) > 1 and value[1] == ":")):
            value = self.logic.windows_to_linux_path(value)
        command.append(f"\"{value}\"")
    print("command : ",command)

    self.process = threading.Thread(target=self.logic.conda.condaRunCommand, args=(command,))
    self.process.start()
    # self.ui.TimerLabel.setHidden(False)
    self.ui.TimerLabel.setText(f"time : 0.00s")
    previous_time = self.startTime
    self.onProcessStarted()
    while self.process.is_alive():
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

    # self.OnEndProcess()

    # def onCliModified(self, caller, event):
    #     self.progressBar.setValue(caller.GetProgress())
    #     if caller.GetStatus() == 32:
    #         self.progressBar.close()



  def onProcessStarted(self):

    # self.progressBar = createProgressDialog(None, 0, 100)
    # self.progressBar.setWindowTitle("Starting...")
    # self.progressBar.connect('canceled()', self.onCancel)

    self.startTime = time.time()

    self.ui.PredScanProgressBar.setMaximum(self.scan_count)
    self.ui.PredScanProgressBar.setValue(0)
    if not self.isSegmentInput:
      self.ui.PredScanLabel.setText(f"Scan ready for segmentation : 0 / {self.scan_count}")
    else:
      self.ui.PredScanLabel.setText(f"Ouput generated for segmentation : 0 / {self.scan_count}")

    self.total_seg_progress = self.scan_count * self.seg_cout

    self.ui.PredSegProgressBar.setMaximum(self.total_seg_progress)
    self.ui.PredSegProgressBar.setValue(0)
    self.ui.PredSegLabel.setText(f"Segmented structures : 0 / {self.total_seg_progress}")

    self.prediction_step = 0
    self.progress = 0


    self.RunningUI(True)

    return


  def UpdateRunBtn(self):
    self.ui.PredictionButton.setEnabled(self.scan_ready and self.model_ready)
    # self.ui.PredictionButton.setEnabled(True)



  def UpdateProgressBar(self,progress):

    # print("UpdateProgressBar")
    if progress == 200:
      self.prediction_step += 1

      if self.prediction_step == 1:
        self.progress = 0
        # self.progressBar.maximum = self.scan_count
        # self.progressBar.windowTitle = "Correcting contrast..."
        # self.progressBar.setValue(0)

      if self.prediction_step == 2:
        self.progress = 0
        self.ui.PredScanProgressBar.setValue(self.scan_count)

        # self.progressBar.maximum = self.total_seg_progress
        # self.progressBar.windowTitle = "Segmenting scans..."
        # self.progressBar.setValue(0)


    if progress == 100:
      # self.prediction_step += 1

      if self.prediction_step == 1:
        # self.progressBar.setValue(self.progress)
        self.ui.PredScanProgressBar.setValue(self.progress)
        if not self.isSegmentInput:
          self.ui.PredScanLabel.setText(f"Scan ready for segmentation : {self.progress} / {self.scan_count}")
        else:
          self.ui.PredScanLabel.setText(f"Ouput generated for segmentation : {self.progress} / {self.scan_count}")

      if self.prediction_step == 2:

        # self.progressBar.setValue(self.progress)
        self.ui.PredSegProgressBar.setValue(self.progress)
        self.ui.PredSegLabel.setText(f"Segmented structures : {self.progress} / {self.total_seg_progress}")

      self.progress += 1



  def onProcessUpdate(self,caller,event):

    # print(caller.GetProgress(),caller.GetStatus())

    # self.ui.TimerLabel.setText(f"Time : {self.startTime:.2f}s")
    self.ui.TimerLabel.setText(f"Time : {time.time()-self.startTime:.2f}s")

    progress = caller.GetProgress()

    # print("Progress : ",progress)

    if progress == 0:
      self.updateProgessBar = False

    if progress != 0 and self.updateProgessBar == False:
      self.updateProgessBar = True
      self.UpdateProgressBar(progress)

    # print(progress)



    if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
      # process complete


      if self.logic.cliNode.GetStatus() & self.logic.cliNode.ErrorsMask:
        # error
        print(self.logic.cliNode.GetOutputText())
        print("\n\n ========= ERROR ========= \n")
        errorText = self.logic.cliNode.GetErrorText()
        print("CLI execution failed: \n \n" + errorText)

        # self.progressBar.windowTitle = "FAILED 1"
        # self.progressBar.setValue(100)

        # msg = qt.QMessageBox()
        # msg.setText(f'There was an error during the process:\n \n {errorText} ')
        # msg.setWindowTitle("Error")
        # msg.exec_()

      else:
        # success

        self.OnEndProcess()

  def onCancel(self):
    # print(self.logic.cliNode.GetOutputText())
    self.logic.cliNode.Cancel()

    # self.ui.CLIProgressBar.setValue(0)
    # self.progressBar.close()

    self.RunningUI(False)



    print("Cancelled")

  def RunningUI(self, run = False):

    self.ui.PredictionButton.setVisible(not run)

    self.ui.CancelButton.setVisible(run)
    self.ui.PredScanLabel.setVisible(run)
    self.ui.PredScanProgressBar.setVisible(run)
    if not self.isSegmentInput:
      self.ui.PredSegLabel.setVisible(run)
      self.ui.PredSegProgressBar.setVisible(run)
    self.ui.TimerLabel.setVisible(run)



    #endregion


    #region == SLICER BASICS ==

  def OnEndProcess(self):


    print('PROCESS DONE.')
    # self.progressBar.setValue(100)
    # self.progressBar.close()
    print(self.logic.cliNode.GetOutputText())

    stopTime = time.time()
    # print(self.startTime)
    logging.info(f'Processing completed in {stopTime-self.startTime:.2f} seconds')

    self.RunningUI(False)


    if self.vtk_output_folder is not None:
      # print(self.vtk_output_folder)
      files = []
      # get all .vtk files in self.vtk_output_folder
      for file in os.listdir(self.vtk_output_folder):
        if file.endswith(".vtk"):
          files.append(self.vtk_output_folder + '/' +file)


      # print(files)
      # files = [files[0]]
      print("Loading surface files...")
      for models in files:
        if models not in LOADED_VTK_FILES.keys():
          #open model vtk file
          modelNode = slicer.util.loadModel(models)
          LOADED_VTK_FILES[models] = modelNode

          # Edit display properties
          displayNode = modelNode.GetDisplayNode()
          displayNode.SetSliceIntersectionVisibility(True)
          displayNode.SetSliceIntersectionThickness(2)
          if "Skin" in models or "SKIN" in models:
            displayNode.SetOpacity(0.1)



      if True in ["Root-canal" in name for name in LOADED_VTK_FILES.keys()]:
        for model,node in LOADED_VTK_FILES.items():
          if True in [x in model for x in ["Mandible","Maxilla"]]:
            displayNode = node.GetDisplayNode()
            displayNode.SetOpacity(0.2)


    self.RunningUI(False)


  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """

    temp_fold = os.path.join("..", "temp")
    try:
        shutil.rmtree(temp_fold)
    except OSError as e:
        print("Error: %s : %s" % (temp_fold, e.strerror))


    # if self.logic.cliNode is not None:
    #   if self.logic.cliNode.GetStatus() & self.logic.cliNode.Running:
    #     self.logic.cliNode.Cancel()
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

    # self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    # if not self._parameterNode.GetNodeReference("InputVolume"):
    #   firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    #   if firstVolumeNode:
    #     self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    # if inputParameterNode:
    #   self.logic.setDefaultParameters(inputParameterNode)

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


    self._parameterNode.EndModify(wasModified)

    #endregion

class LMTab:
    def __init__(self) -> None:

      self.default_checkbox_state = True


      self.widget = qt.QWidget()
      layout = qt.QVBoxLayout(self.widget)


      self.seg_tab_widget = qt.QTabWidget()
      # self.seg_tab_widget.connect('currentChanged(int)',self.Test)

      self.seg_tab_widget.minimumSize = qt.QSize(100,200)
      self.seg_tab_widget.maximumSize = qt.QSize(800,400)
      self.seg_tab_widget.setMovable(True)


      # print(self.seg_status_dic)
      # print(lcbd)
      buttons_wid = qt.QWidget()
      buttons_layout = qt.QHBoxLayout(buttons_wid)
      self.select_all_btn = qt.QPushButton("Select All")
      self.select_all_btn.setEnabled(False)
      self.select_all_btn.connect('clicked(bool)', self.SelectAll)
      self.clear_all_btn = qt.QPushButton("Clear Selection")
      self.clear_all_btn.setEnabled(False)
      self.clear_all_btn.connect('clicked(bool)', self.ClearAll)

      buttons_layout.addWidget(self.select_all_btn)
      buttons_layout.addWidget(self.clear_all_btn)

      layout.addWidget(self.seg_tab_widget)
      layout.addWidget(buttons_wid)
      self.seg_status_dic = {}


    # def Test(self, index):
    #   print(index)

    def Clear(self):
      self.seg_tab_widget.clear()

    def FillTab(self,seg_dic):


      self.select_all_btn.setEnabled(True)
      self.clear_all_btn.setEnabled(True)

      self.seg_status_dic = {}
      self.seg_group_dic = {}
      self.seg_group_dic["All"] = []
      for group,seg_lst in seg_dic.items():
        self.seg_group_dic[group] = seg_lst
        for seg in seg_lst:
          self.seg_group_dic["All"].append(seg)
          if seg not in self.seg_status_dic.keys():
            self.seg_status_dic[seg] = seg in DEFAULT_SELECT

      # print(self.seg_group_dic)


      self.check_box_dic = {}



      for group,seg_lst in self.seg_group_dic.items():
        lst_wid = []
        for seg in seg_lst:
          new_cb = qt.QCheckBox(seg)
          new_cb.setChecked(seg in DEFAULT_SELECT)
          if seg in UNAVAILABLE_MODELS:
            new_cb.setEnabled(False)

          self.check_box_dic[new_cb] = seg
          lst_wid.append(new_cb)

        new_seg_tab = self.GenNewTab(lst_wid)
        self.seg_tab_widget.insertTab(-1,new_seg_tab,group)

      self.seg_tab_widget.currentIndex = 0

      # print(self.check_box_dic)
      self.seg_cb_dic = {}
      for cb,seg in self.check_box_dic.items():
        if seg not in self.seg_cb_dic.keys():
          self.seg_cb_dic[seg] = [cb]
        else:
          self.seg_cb_dic[seg].append(cb)


      # for seg in seg_dic["U"]:
      #   self.UpdateSegSelect(seg,True)

      for cb in self.check_box_dic.keys():
        cb.connect("toggled(bool)", self.CheckBox)

    def CheckBox(self, caller=None, event=None):
      for cb,seg in self.check_box_dic.items():
        if cb.checkState():
          state = True
        else:
          state = False

        if self.seg_status_dic[seg] != state:
          self.UpdateSegSelect(seg,state)

    def ToggleSelection(self):
      idx = self.seg_tab_widget.currentIndex
      # print(idx)
      group = self.seg_tab_widget.tabText(idx)
      # print(group)
      new_state = not self.seg_status_dic[self.seg_group_dic[group][0]]
      # print(new_state)
      for seg in self.seg_group_dic[group]:
        self.UpdateSegSelect(seg,new_state)


    def GenNewTab(self,widget_lst):
        new_widget = qt.QWidget()
        vb = qt.QVBoxLayout(new_widget)
        scr_box = qt.QScrollArea()
        vb.addWidget(scr_box)
        pb = qt.QPushButton('Switch tab selection')
        pb.connect('clicked(bool)', self.ToggleSelection)
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

    def UpdateSegSelect(self,seg_id,state):
      for cb in self.seg_cb_dic[seg_id]:
        cb.setChecked(state)
      self.seg_status_dic[seg_id] = state

    def UpdateAll(self,state):
      for seg_id,cb_lst in self.seg_cb_dic.items():
        for cb in cb_lst:
          cb.setChecked(state)
        self.seg_status_dic[seg_id] = state

    def GetSelected(self):
      selected = []
      for seg,state in self.seg_status_dic.items():
        if state:
          selected.append(seg)
      return selected

    def SelectAll(self):
      self.UpdateAll(True)

    def ClearAll(self):
      self.UpdateAll(False)




#
# AMASSSLogic
#

class AMASSSLogic(ScriptedLoadableModuleLogic):
  """
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
      self.run_conda_command(target=self.conda.condaCreateEnv, command=(self.name_env,"3.9",["shapeaxi==1.0.10"],)) #run in parallel to not block slicer

  def setup_cli_command(self,file):
      args = self.find_cli_parameters()
      conda_exe = self.conda.getCondaExecutable()
      command = [conda_exe, "run", "-n", self.name_env, "python" ,"-m", f"{file}"]
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
  
  def check_cli_script(self, name):
      if not self.check_pythonpath_windows(f"{name}"): 
          self.give_pythonpath_windows()
          results = self.check_pythonpath_windows(f"{name}")
          
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

  # def process(self, parameters, showResult=True):
  #   """
  #   Run the processing algorithm.
  #   Can be used without GUI widget.
  #   """



  #   logging.info('Processing started')


  #   print ('parameters : ', parameters)


  #   AMASSSProcess = slicer.modules.amasss_cli

  #   self.cliNode = slicer.cli.run(AMASSSProcess, None, parameters)

  #   # We don't need the CLI module node anymore, remove it to not clutter the scene with it
  #   # slicer.mrmlScene.RemoveNode(cliNode)



  #   return AMASSSProcess


#region OLD CODE



# def GetAvailableSeg(mfold,seg_group):
#   brain_dic = GetBrain(mfold)
#   # print(brain_dic)
#   available_lm = {"Other":[]}
#   for lm in brain_dic.keys():
#     if lm in seg_group.keys():
#       group = seg_group[lm]
#     else:
#       group = "Other"
#     if group not in available_lm.keys():
#       available_lm[group] = [lm]
#     else:
#       available_lm[group].append(lm)

#   return available_lm,brain_dic



# def GetBrain(dir_path):
#     brainDic = {}
#     normpath = os.path.normpath("/".join([dir_path, '**', '']))
#     for img_fn in sorted(glob.iglob(normpath, recursive=True)):
#         #  print(img_fn)
#         if os.path.isfile(img_fn) and ".pth" in img_fn:
#             lab = os.path.basename(os.path.dirname(os.path.dirname(img_fn)))
#             num = os.path.basename(os.path.dirname(img_fn))
#             if lab in brainDic.keys():
#                 brainDic[lab][num] = img_fn
#             else:
#                 network = {num : img_fn}
#                 brainDic[lab] = network

#     # print(brainDic)
#     out_dic = {}
#     for l_key in brainDic.keys():
#         networks = []
#         for n_key in range(len(brainDic[l_key].keys())):
#             networks.append(brainDic[l_key][str(n_key)])

#         out_dic[l_key] = networks

#     return out_dic




  # def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
  #   """
  #   Run the processing algorithm.
  #   Can be used without GUI widget.
  #   """

  #   if not inputVolume or not outputVolume:
  #     raise ValueError("Input or output volume is invalid")

  #   import time
  #   startTime = time.time()
  #   logging.info('Processing started')

  #   # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module

  #   cliParams = {
  #     'InputVolume': inputVolume.GetID(),
  #     'OutputVolume': outputVolume.GetID(),
  #     'ThresholdValue' : imageThreshold,
  #     'ThresholdType' : 'Above' if invert else 'Below'
  #     }
  #   cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
  #   # We don't need the CLI module node anymore, remove it to not clutter the scene with it
  #   slicer.mrmlScene.RemoveNode(cliNode)

  #   stopTime = time.time()
  #   logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# AMASSSTest
#

# class AMASSSTest(ScriptedLoadableModuleTest):
#   """
#   This is the test case for your scripted module.
#   Uses ScriptedLoadableModuleTest base class, available at:
#   https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
#   """

#   def setUp(self):
#     """ Do whatever is needed to reset the state - typically a scene clear will be enough.
#     """
#     slicer.mrmlScene.Clear()

#   def runTest(self):
#     """Run as few or as many tests as needed here.
#     """
#     self.setUp()
#     self.test_AMASSS1()

#   def test_AMASSS1(self):
#     """ Ideally you should have several levels of tests.  At the lowest level
#     tests should exercise the functionality of the logic with different inputs
#     (both valid and invalid).  At higher levels your tests should emulate the
#     way the user would interact with your code and confirm that it still works
#     the way you intended.
#     One of the most important features of the tests is that it should alert other
#     developers when their changes will have an impact on the behavior of your
#     module.  For example, if a developer removes a feature that you depend on,
#     your test should break so they know that the feature is needed.
#     """

#     self.delayDisplay("Starting the test")

#     # Get/create input data

#     import SampleData
#     registerSampleData()
#     inputVolume = SampleData.downloadSample('AMASSS1')
#     self.delayDisplay('Loaded test data set')

#     inputScalarRange = inputVolume.GetImageData().GetScalarRange()
#     self.assertEqual(inputScalarRange[0], 0)
#     self.assertEqual(inputScalarRange[1], 695)

#     outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
#     threshold = 100

#     # Test the module logic

#     logic = AMASSSLogic()

#     # Test algorithm with non-inverted threshold
#     logic.process(inputVolume, outputVolume, threshold, True)
#     outputScalarRange = outputVolume.GetImageData().GetScalarRange()
#     self.assertEqual(outputScalarRange[0], inputScalarRange[0])
#     self.assertEqual(outputScalarRange[1], threshold)

#     # Test algorithm with inverted threshold
#     logic.process(inputVolume, outputVolume, threshold, False)
#     outputScalarRange = outputVolume.GetImageData().GetScalarRange()
#     self.assertEqual(outputScalarRange[0], inputScalarRange[0])
#     self.assertEqual(outputScalarRange[1], inputScalarRange[1])

#     self.delayDisplay('Test passed')

#endregion
