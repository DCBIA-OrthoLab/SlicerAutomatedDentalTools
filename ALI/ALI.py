import os
import unittest
import logging
import glob
import time
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import webbrowser

# import csv


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
  "IOS" : 'https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/MG_test_scan.nii.gz',
}

MODELS_LINK = {
  "CBCT": [
    'https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/MG_test_scan.nii.gz',
  ],
  "IOS" : [
    'https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/MG_test_scan.nii.gz',
  ],
}


GROUPS_LANDMARKS = {
  'Impacted canine' : ['UR3OIP','UL3OIP','UR3RIP','UL3RIP'],

  'Cranial base' : ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4'],

  'Lower' : ['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R'],

  'Upper' : ['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R'],
}


SURFACE_LANDMARKS = {
  'Landmarks' : ['CL','CB','O','DB','MB','R','RIP','OIP'],
  'Upper teeth' : ['UL7','UL6','UL5','UL4','UL3','UL2','UL1','UR1','UR2','UR3','UR4','UR5','UR6','UR7'],
  'Lower teeth' : ['LL7','LL6','LL5','LL4','LL3','LL2','LL1','LR1','LR2','LR3','LR4','LR5','LR6','LR7'],
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
    self.parent.categories = ["Automated dental tools"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Maxime Gillot (UoM)"]  # TODO: replace with "Firstname Lastname (Organization)"
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


    self.CBCT_as_input = True # True : CBCT image, False : surface IOS
    self.folder_as_input = False # If use a folder as input

    self.MRMLNode_scan = None # MRML node of the selected scan
    self.input_path = None # path to the folder containing the scans

    self.available_landmarks = [] # list of available landmarks to predict
    


    self.scan_count = 0 # number of scans in the input folder
    self.landmark_cout = 0 # number of landmark to identify 




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

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).


    self.lm_tab = LMTab()
    # LM_tab_widget,LM_buttons_dic = GenLandmarkTab(Landmarks_group)
    self.ui.OptionVLayout.addWidget(self.lm_tab.widget)

    #region ===== INPUTS =====

    self.ui.InputTypeComboBox.currentIndexChanged.connect(self.SwitchInputType)
    self.SwitchInputType(0)

    self.ui.MRMLNodeComboBox.setMRMLScene(slicer.mrmlScene)
    self.ui.MRMLNodeComboBox.currentNodeChanged.connect(self.onNodeChanged)
    self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)


    self.ui.InputComboBox.currentIndexChanged.connect(self.SwitchInput)
    self.SwitchInput(0)

    self.ui.DownloadTestPushButton.connect('clicked(bool)',self.onTestDownloadButton)
    self.ui.DownloadModelPushButton.connect('clicked(bool)',self.onModelDownloadButton)


    #endregion

    self.ui.SavePredictCheckBox.connect("toggled(bool)", self.UpdateSaveType)

    self.ui.SearchSaveFolder.setHidden(True)
    self.ui.SaveFolderLineEdit.setHidden(True)
    self.ui.PredictFolderLabel.setHidden(True)




    # Buttons
    self.ui.SearchScanFolder.connect('clicked(bool)',self.onSearchScanButton)
    self.ui.SearchModelFolder.connect('clicked(bool)',self.onSearchModelButton)
    self.ui.SearchSaveFolder.connect('clicked(bool)',self.onSearchSaveButton)

    self.ui.PredictionButton.connect('clicked(bool)', self.onPredictButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  #region ===== FUNCTIONS =====

  #region ===== INPUTS =====

  def SwitchInputType(self,index):

    if index == 1:
      self.CBCT_as_input = False
      self.ui.MRMLNodeComboBox.nodeTypes = ['vtkMRMLModelNode']
      self.lm_tab.Clear()
      self.lm_tab.FillTab(SURFACE_LANDMARKS)

    else:
      self.CBCT_as_input = True
      self.ui.MRMLNodeComboBox.nodeTypes = ['vtkMRMLVolumeNode']
      self.lm_tab.Clear()
      self.lm_tab.FillTab(GROUPS_LANDMARKS)

    # print()


  def SwitchInput(self,index):

    if index == 1:
      self.folder_as_input = True

    else:
      self.folder_as_input = False
      self.onNodeChanged()

    print("Input type : ", index)

    self.ui.ScanPathLabel.setVisible(self.folder_as_input)
    self.ui.lineEditScanPath.setVisible(self.folder_as_input)
    self.ui.SearchScanFolder.setVisible(self.folder_as_input)

    self.ui.SelectNodeLabel.setVisible(not self.folder_as_input)
    self.ui.MRMLNodeComboBox.setVisible(not self.folder_as_input)
    self.ui.FillNodeLlabel.setVisible(not self.folder_as_input)


  def onNodeChanged(self):
    selected = False
    self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentNodeID)
    if self.MRMLNode_scan is not None:
      print(PathFromNode(self.MRMLNode_scan))
      self.input_path = PathFromNode(self.MRMLNode_scan)
      self.scan_count = 1


      self.ui.PrePredInfo.setText("Number of scans to process : 1")
      selected = True

    return selected

  def onTestDownloadButton(self):
    if self.CBCT_as_input:
      webbrowser.open(TEST_SCAN["CBCT"])
    else:
      webbrowser.open(TEST_SCAN["IOS"])


  def onModelDownloadButton(self):
    if self.CBCT_as_input:
      for link in MODELS_LINK["CBCT"]:
        webbrowser.open(link)
    else:
      for link in MODELS_LINK["IOS"]:
        webbrowser.open(link)



  def updateProgressBare(self,caller=None, event=None):
    self.ui.progressBar.value = 50
    # print(self.ui.horizontalSlider.value)
    # print(self.ui.inputSelector.currentNode())

  def UpdateSaveType(self,caller=None, event=None):
    # print(caller,event)
    self.ui.SearchSaveFolder.setHidden(caller)
    self.ui.SaveFolderLineEdit.setHidden(caller)
    self.ui.PredictFolderLabel.setHidden(caller)

    # self.ui.SearchSaveFolder.setEnabled(not caller)
    # self.ui.SaveFolderLineEdit.setEnabled(not caller)
    
    self.save_scan_folder = caller

  def onSearchScanButton(self):
    surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
    if surface_folder != '':
      self.surface_folder = surface_folder
      self.ui.lineEditScanPath.setText(self.surface_folder)

      

  def onSearchModelButton(self):
    model_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a model folder")
    if model_folder != '':


      if self.CBCT_as_input:
        lm_group = GetLandmarkGroup(GROUPS_LANDMARKS)
        available_lm,brain_dic = GetAvailableLm(model_folder,lm_group)

        if len(available_lm.keys()) == 0:
          qt.QMessageBox.warning(self.parent, 'Warning', 'No models found in the selected folder\nPlease select a folder containing .pth files\nYou can download the latest models with\n  "Download latest models" button')
          return
        else:
          self.model_folder = model_folder
          self.ui.lineEditModelPath.setText(self.model_folder)
          self.available_landmarks = available_lm.keys()
          self.lm_tab.Clear()
          self.lm_tab.FillTab(available_lm, enable = True)
          # print(available_lm)
          # print(brain_dic)


  def onSearchSaveButton(self):
    save_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
    if save_folder != '':
      self.save_folder = save_folder
      self.ui.SaveFolderLineEdit.setText(self.save_folder)

  def onPredictButton(self):

    # print(self.addLog)

    scan_folder = self.ui.lineEditScanPath.text

    scans = []
    if scan_folder != '':
      normpath = os.path.normpath("/".join([scan_folder, '**', '']))
      for img_fn in sorted(glob.iglob(normpath, recursive=True)):
          #  print(img_fn)
          basename = os.path.basename(img_fn)

          if True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            scans.append({"name" : basename, "path":img_fn})

    # print(scans)

    selectedLM = self.lm_tab.GetSelectedLM()
    LM_nbr = len(selectedLM)
    self.ui.LandmarkProgressBar.maximum = LM_nbr
    scan_nbr = len(scans)
    self.ui.TotalrogressBar.maximum = scan_nbr
    if LM_nbr == 0 or scan_nbr ==0:
      print("Error")
      msg = qt.QMessageBox()
      msg_txt = "Missing parameters : \n"
      if scan_nbr == 0:
        msg_txt += "- No scan found in folder '" + scan_folder + "'\n"
      if LM_nbr == 0:
        msg_txt += "- 0 Landmark selected"


      msg.setText(msg_txt)
      msg.setWindowTitle("Error")
      msg.exec_()
      
      return

    self.scans = scans
    self.selectedLM = selectedLM
    self.scan_nbr = scan_nbr
    self.LM_nbr = LM_nbr


    # self.logic = ALILogic()
    # self.logic.process(self.selectedLM)

    # self.logic.cliNode.AddObserver('ModifiedEvent',self.onProcessUpdate)
    # self.onProcessStarted()

    # th = threading.Thread(target=self.threadFunc)
    # th.start()

  def onProcessUpdate(self,caller,event):
    print("DONE")
    # print(self.logic.cliNode.GetStatus())

 
    


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

    # Update node selectors and sliders
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
    # self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
    self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

    # Update buttons states and tooltips
    # if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
    #   self.ui.applyButton.toolTip = "Compute output volume"
    #   self.ui.applyButton.enabled = True
    # else:
    #   self.ui.applyButton.toolTip = "Select input and output volume nodes"
    #   self.ui.applyButton.enabled = False

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
    # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
    self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
    self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

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
          for lm in lm_lst:
              if lm not in lmsd.keys():
                  lmsd[lm] = False
                  self.lm_group_dic["All"].append(lm)

      self.check_box_dic = cbd
      self.lm_status_dic = lmsd


      for group,lm_lst in lm_dic.items():
        lst_wid = []
        for lm in lm_lst:
          new_cb = qt.QCheckBox(lm)
          self.check_box_dic[new_cb] = lm
          lst_wid.append(new_cb)

        new_lm_tab = self.GenNewTab(lst_wid,enable)
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

    def GetSelectedLM(self):
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
  print(brain_dic)
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

    # print(brainDic)
    # out_dic = {}
    # for l_key in brainDic.keys():
    #     networks = []
    #     for n_key in range(len(brainDic[l_key].keys())):
    #         networks.append(brainDic[l_key][str(n_key)])

    #     out_dic[l_key] = networks

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

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("Threshold"):
      parameterNode.SetParameter("Threshold", "100.0")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("Invert", "false")

  def process(self, landmark_lst):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded

    """
    self.landmark_lst = landmark_lst

    # import time
    # startTime = time.time()
    logging.info('Processing started')

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    cliParams = {
      'landmark_lst': landmark_lst,
      }
    # env = slicer.util.startupEnvironment()
    # print('\n\n\n\n')
    # #print ('parameters : ', parameters)

    # with open('env.json', 'w') as convert_file:
    #   convert_file.truncate(0)
    #   convert_file.write(json.dumps(env))
    
    PredictProcess = slicer.modules.ali_cli
    self.cliNode = slicer.cli.run(PredictProcess,None, cliParams)    

    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    slicer.mrmlScene.RemoveNode(self.cliNode)

    # stopTime = time.time()
    # logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))
    

#
# ALITest
#

class ALITest(ScriptedLoadableModuleTest):
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
    self.test_ALI1()

  def test_ALI1(self):
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
    inputVolume = SampleData.downloadSample('ALI1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = ALILogic()

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

