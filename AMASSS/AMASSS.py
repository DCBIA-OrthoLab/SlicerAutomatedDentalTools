import os
import unittest
import logging
import glob
import time
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import webbrowser


def GetSegGroup(group_landmark):
  seg_group = {}
  for group,labels in group_landmark.items():
      for label in labels:
          seg_group[label] = group
  return seg_group


MODEL_LINK = 'https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.0-alpha/ALL_MODELS.zip'

GROUPS_SEG = {
  "Bones" : ["Mandible","Maxilla","Cranial base","Cervical vertebra"],
  "Teeth" : ['Root canal'],
  "Nerfs" : ['Mandibular canal'],
  "Soft tissue" :['Skin','Upper airway'],
}


SEG_GROUP = GetSegGroup(GROUPS_SEG)




#
# AMASSS
#

class AMASSS(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "AMASSS"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Automatic Tools"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#AMASSS">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

#
# Register sample data sets in Sample Data module
#


#
# AMASSSWidget
#

class AMASSSWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
    
    self.save_folder = None


  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
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

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)



    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    # self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    # self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    # self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    # self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    # self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    self.ui.SavePredictCheckBox.connect("toggled(bool)", self.UpdateSaveType)

    self.ui.SearchSaveFolder.setHidden(True)
    self.ui.SaveFolderLineEdit.setHidden(True)
    self.ui.PredictFolderLabel.setHidden(True)

    self.SwitchInputType(0)



    # Buttons
    # self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    self.ui.input_type_select.currentIndexChanged.connect(self.SwitchInputType)

    self.ui.DownloadButton.connect('clicked(bool)',self.onDownloadButton)

    self.ui.SearchScanFolder.connect('clicked(bool)',self.onSearchScanButton)
    self.ui.SearchModelFolder.connect('clicked(bool)',self.onSearchModelButton)
    self.ui.SearchSaveFolder.connect('clicked(bool)',self.onSearchSaveButton)

    self.ui.PredictionButton.connect('clicked(bool)', self.onPredictButton)


    self.seg_tab = LMTab()
    # seg_tab_widget,seg_buttons_dic = GenLandmarkTab(Landmarks_group)
    self.ui.OptionVLayout.addWidget(self.seg_tab.widget)
    self.seg_tab.Clear()
    self.seg_tab.FillTab(GROUPS_SEG)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()



  def UpdateSaveType(self,caller=None, event=None):
    # print(caller,event)
    self.ui.SearchSaveFolder.setHidden(caller)
    self.ui.SaveFolderLineEdit.setHidden(caller)
    self.ui.PredictFolderLabel.setHidden(caller)

    # self.ui.SearchSaveFolder.setEnabled(not caller)
    # self.ui.SaveFolderLineEdit.setEnabled(not caller)
    
    self.save_scan_folder = caller

  def SwitchInputType(self,index):
    if index == 0:
      index =False
    else:
      index = True

    self.ui.label_folder_select.setHidden(index)
    self.ui.lineEditScanPath.setHidden(index)
    self.ui.SearchScanFolder.setHidden(index)

    self.ui.label_node_select.setHidden(not index)
    self.ui.MRMLNodeComboBox_file.setHidden(not index)
    self.ui.emptyLabelNodeSelect.setHidden(not index)
  

  def onDownloadButton(self):
    webbrowser.open(MODEL_LINK)


  def onSearchScanButton(self):
    surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
    if surface_folder != '':
      self.surface_folder = surface_folder
      self.ui.lineEditScanPath.setText(self.surface_folder)

      
  def onSearchModelButton(self):
    model_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a model folder")
    if model_folder != '':
      self.model_folder = model_folder
      self.ui.lineEditModelPath.setText(self.model_folder)

      # available_lm,brain_dic = GetAvailableSeg(self.model_folder,SEG_GROUP)
      # print(brain_dic)
      
      # self.seg_tab.Clear()
      # self.seg_tab.FillTab(GROUPS_SEG)

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

    selectedLM = self.seg_tab.GetSelectedLM()
    seg_nbr = len(selectedLM)
    self.ui.LandmarkProgressBar.maximum = seg_nbr
    scan_nbr = len(scans)
    self.ui.TotalrogressBar.maximum = scan_nbr
    if seg_nbr == 0 or scan_nbr ==0:
      print("Error")
      msg = qt.QMessageBox()
      msg_txt = "Missing parameters : \n"
      if scan_nbr == 0:
        msg_txt += "- No scan found in folder '" + scan_folder + "'\n"
      if seg_nbr == 0:
        msg_txt += "- 0 Landmark selected"


      msg.setText(msg_txt)
      msg.setWindowTitle("Error")
      msg.exec_()
      
      return

    self.scans = scans
    self.selectedLM = selectedLM
    self.scan_nbr = scan_nbr
    self.seg_nbr = seg_nbr

    self.threadFunc()






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



class LMTab:
    def __init__(self) -> None:

      self.default_checkbox_state = True


      self.widget = qt.QWidget()
      layout = qt.QVBoxLayout(self.widget)

      self.seg_tab_widget = qt.QTabWidget()
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
            self.seg_status_dic[seg] = self.default_checkbox_state
      
      # print(self.seg_group_dic)


      self.check_box_dic = {}



      for group,seg_lst in self.seg_group_dic.items():
        lst_wid = []
        for seg in seg_lst:
          new_cb = qt.QCheckBox(seg)
          new_cb.setChecked(self.default_checkbox_state)
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

    def GetSelectedLM(self):
      selectedLM = []
      for seg,state in self.seg_status_dic.items():
        if state:
          selectedLM.append(seg)
      return selectedLM

    def SelectAll(self):
      self.UpdateAll(True)
    
    def ClearAll(self):
      self.UpdateAll(False)

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




#
# AMASSSLogic
#

class AMASSSLogic(ScriptedLoadableModuleLogic):
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

  def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
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
    logging.info('Processing started')

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    cliParams = {
      'InputVolume': inputVolume.GetID(),
      'OutputVolume': outputVolume.GetID(),
      'ThresholdValue' : imageThreshold,
      'ThresholdType' : 'Above' if invert else 'Below'
      }
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    slicer.mrmlScene.RemoveNode(cliNode)

    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# AMASSSTest
#

class AMASSSTest(ScriptedLoadableModuleTest):
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
    self.test_AMASSS1()

  def test_AMASSS1(self):
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
    inputVolume = SampleData.downloadSample('AMASSS1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = AMASSSLogic()

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
