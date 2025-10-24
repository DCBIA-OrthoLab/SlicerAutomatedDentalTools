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
import subprocess

import vtk, qt, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install
from slicer import vtkMRMLCommandLineModuleNode
import webbrowser
import pkg_resources
import importlib

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
                      pip_install(f'torch>=2.6.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/{cuda_version}')
                      nb_installed += 3

                  else:
                    lib_version = f'{lib}=={version}' if version else lib
                    pip_install(lib_version)
                    nb_installed += 1
                  self.ui.nb_package.setText(f"Package: {nb_installed}/{len_libs}")

                return True

              else:
                pip_install(f'torch>=2.6.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118')
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

def createProgressDialog(parent=None, value=0, maximum=100, windowTitle="Starting..."):
    # import qt # qt.qVersion()
    progressIndicator = qt.QProgressDialog()  #(parent if parent else self.mainWindow())
    progressIndicator.minimumDuration = 0
    progressIndicator.maximum = maximum
    progressIndicator.value = value
    progressIndicator.windowTitle = windowTitle
    return progressIndicator

#========= GLOBAL VARIABLES =========

# MODEL_LINK = 'https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.0-alpha/ALL_MODELS.zip'
MODEL_LINK = 'https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/tag/AMASSS_CBCT'
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
    self.use_small_FOV = False # use high resolution model
    self.save_surface = True # True: save surface .vtk of the segmentation
    self.output_selection = "MERGE" # m: merged, s: separated, ms: both
    self.prediction_ID = "Seg_Pred" # ID to put in the prediction name
    self.save_in_input_folder = True # path to the folder where the results will be saved
    self.center_all = False # True: center all the scan seg and surfaces in the same position
    self.save_adjusted = False # True: save the contrast adjusted scan
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
    #===== WIDGET =====
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

    # smoothing
    self.ui.horizontalSliderSmoothing.valueChanged.connect(self.onSmoothingSlider)
    self.ui.spinBoxSmoothing.valueChanged.connect(self.onSmoothingSpinbox)
    self.UpdateSaveSurface(False)

    self.ui.nb_package.setVisible(False)
    self.ui.label_installation.setVisible(False)

    self.ui.PredictionButton.connect('clicked(bool)', self.onPredictButton)

    self.ui.CancelButton.connect('clicked(bool)', self.onCancel)
    self.RunningUI(False)
    self.initializeParameterNode()


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

  def openUrlInBrowser(self,url):
    # on choisit firefox si dispo
    browser = 'firefox' if shutil.which('firefox') else 'xdg-open'
    env = os.environ.copy()
    # empêche le chargement d’atk-bridge
    env['NO_AT_BRIDGE'] = '1'
    subprocess.Popen(
        [browser, url],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
  def onDownloadButton(self):
        self.openUrlInBrowser(MODEL_LINK)

  def onDownloadScanButton(self):
        self.openUrlInBrowser(SCAN_LINK)

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

  def onSmoothingSlider(self):
    self.smoothing = self.ui.horizontalSliderSmoothing.value
    self.ui.spinBoxSmoothing.value = self.smoothing

  def onSmoothingSpinbox(self):
    self.smoothing = self.ui.spinBoxSmoothing.value
    self.ui.horizontalSliderSmoothing.value = self.smoothing
    #region == RUN ==
    #endregion


    #region == RUN ==
  def onPredictButton(self):
    print('onPredictButton called: starting prediction process...', flush=True)
    import platform
    # first, install the required libraries and their version
    list_libs = [('torch','2.6.0'),('torchvision', "0.21.0"),('blosc2', None), ('torchaudio',"2.6.0"),('itk', None), ('dicom2nifti', '2.3.0'), ('pydicom', '2.2.2'),('einops',None),('nibabel',None),('nnunetv2',None)]

    print('Installing/checking required libraries...', flush=True)
    libs_installation = install_function(self,list_libs,platform.system())
    if not libs_installation:
      qt.QMessageBox.warning(self.parent, 'Warning', 'The module will not work properly without the required libraries.\nPlease install them and try again.')
      return  # stop the function

    self.ui.nb_package.setVisible(False)
    self.ui.label_installation.setVisible(False)

    print('Libraries installed successfully. Checking inputs...', flush=True)
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

    param["skullStructure"] = ",".join(selected_seg)
    param["merge"] = self.output_selection
    param["genVtk"] = self.save_surface
    param["save_in_folder"] = self.ui.saveInFolder.isChecked() or self.ui.checkBoxSurfaceSelect.isChecked()


    if self.save_in_input_folder:
      if os.path.isfile(self.input_path):
        self.output_folder = os.path.dirname(self.input_path)

        baseName = os.path.basename(self.input_path)
        scan_name= baseName.split(".")

        outputdir = self.output_folder
        # if param["save_in_folder"]:
        #     outputdir +

        self.vtk_output_folder = outputdir 


      else:
        self.output_folder = self.input_path
        self.vtk_output_folder = None

    else:
      self.output_folder = self.ui.SaveFolderLineEdit.text
      self.vtk_output_folder = None


    if not self.ui.checkBoxSurfaceSelect.isChecked():
      self.vtk_output_folder = None


    param["output_folder"] = self.output_folder


 
    param["vtk_smooth"] = self.smoothing

    param["prediction_ID"] = self.ui.SaveId.text


    documentsLocation = qt.QStandardPaths.DocumentsLocation
    documents = qt.QStandardPaths.writableLocation(documentsLocation)
    temp_dir = os.path.join(documents, slicer.app.applicationName+"_temp_AMASSS")

    print(temp_dir)

    param["temp_fold"] = temp_dir

    param["SegmentInput"] = self.isSegmentInput

    param["DCMInput"] = self.isDCMInput

    print('All parameters set, starting logic process...', flush=True)
    self.logic.process(param)
    print('Logic process initiated.', flush=True)
    self.processObserver = self.logic.cliNode.AddObserver("ModifiedEvent", self.onProcessUpdate)
    self.onProcessStarted()

  def onProcessStarted(self):

    self.startTime = time.time()
    print('Process started at:', time.strftime('%H:%M:%S', time.localtime(self.startTime)), flush=True)
    if not self.isSegmentInput:
      self.ui.PredScanLabel.setText(f"Scan ready for segmentation : 0 / {self.scan_count}")
    else:
      self.ui.PredScanLabel.setText(f"Ouput generated for segmentation : 0 / {self.scan_count}")

    self.total_seg_progress = self.scan_count * self.seg_cout
    self.ui.PredSegLabel.setText(f"Segmented structures : 0 / {self.total_seg_progress}")

    self.prediction_step = 0
    self.progress = 0
    self.RunningUI(True)

    return


  def UpdateRunBtn(self):
    self.ui.PredictionButton.setEnabled(self.scan_ready and self.model_ready)
    # self.ui.PredictionButton.setEnabled(True)

  def onProcessUpdate(self, caller, event):
      # Met à jour le timer
      elapsed = time.time() - self.startTime
      self.ui.TimerLabel.setText(f"Time : {elapsed:.2f}s")

      status = caller.GetStatus()

      # 1) Si le CLI est terminé :
      if status & caller.Completed:
          print('Process completed.', flush=True)
          if status & caller.ErrorsMask:
              # Gestion de l’erreur
              errorText = caller.GetErrorText()
              qt.QMessageBox.critical(
                  slicer.util.mainWindow(),
                  "AMASSS Error",
                  f"Une erreur est survenue :\n\n{errorText}"
              )
          else:
              # Fin normale du process
              self.OnEndProcess()
          return

      # 2) Sinon, on est toujours en cours – calcul de la progression
      progress = caller.GetProgress()
      if progress > 1:
          progress /= 100.0

      # Nombre de scans prêts
      done_scans = int(round(progress * self.scan_count))
      self.ui.PredScanLabel.setText(
          f"Scan ready for segmentation : {done_scans} / {self.scan_count}"
      )

      # Nombre de structures segmentées
      done_segs = int(round(progress * self.total_seg_progress))

      self.ui.PredSegLabel.setText(
          f"Segmented structures : {done_segs} / {self.total_seg_progress}"
      )


  def onCancel(self):
    print('Cancelling process...', flush=True)
    # print(self.logic.cliNode.GetOutputText())
    self.logic.cliNode.Cancel()

    # self.ui.CLIProgressBar.setValue(0)
    # self.progressBar.close()

    self.RunningUI(False)


    print("Cancelled", flush=True)

  def RunningUI(self, run = False):

    self.ui.PredictionButton.setVisible(not run)

    self.ui.CancelButton.setVisible(run)
    self.ui.PredScanLabel.setVisible(False)
    self.ui.PredScanProgressBar.setVisible(False)
    if not self.isSegmentInput:
      self.ui.PredSegLabel.setVisible(run)
      self.ui.PredSegProgressBar.setVisible(False)
    self.ui.TimerLabel.setVisible(run)

    #region == SLICER BASICS ==

  def OnEndProcess(self):


    print('PROCESS DONE.')
    print('Retrieving output text...', flush=True)
    # self.progressBar.setValue(100)
    # self.progressBar.close()
    print(self.logic.cliNode.GetOutputText())

    stopTime = time.time()
    # print(self.startTime)
    logging.info(f'Processing completed in {stopTime-self.startTime:.2f} seconds')
    print(f'Processing completed in {stopTime-self.startTime:.2f} seconds', flush=True)

    self.RunningUI(False)


    if self.vtk_output_folder is not None:
      print(self.vtk_output_folder)
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


    if self.logic.cliNode is not None:
      if self.logic.cliNode.GetStatus() & self.logic.cliNode.Running:
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


  def process(self, parameters, showResult=True):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    """



    logging.info('Processing started')

    print('AMASSS processing started with parameters:', parameters, flush=True)

    print ('parameters : ', parameters)


    AMASSSProcess = slicer.modules.amasss_cli

    print('Running AMASSS CLI module...', flush=True)
    self.cliNode = slicer.cli.run(AMASSSProcess, None, parameters)

    print('CLI node created and running.', flush=True)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    # slicer.mrmlScene.RemoveNode(cliNode)

    return AMASSSProcess