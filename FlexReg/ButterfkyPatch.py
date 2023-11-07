import os
from functools import partial
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
import vtk
import time


import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import torch

from qt import (QGridLayout,
                QHBoxLayout,
                QVBoxLayout,
                QCheckBox,
                QLabel,
                QLineEdit,
                QStackedWidget,
                QComboBox,
                QPushButton,
                QFileDialog,
                QSpinBox,
                QWidget,
                QTimer,
                QDialog,
                QSizePolicy,
                QSpacerItem)




#
# ButterfkyPatch
#

class ButterfkyPatch(ScriptedLoadableModule):
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
See more information in <a href="https://github.com/organization/projectname#ButterfkyPatch">module documentation</a>.
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

    # ButterfkyPatch1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='ButterfkyPatch',
        sampleName='ButterfkyPatch1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'ButterfkyPatch1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='ButterfkyPatch1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='ButterfkyPatch1'
    )

    # ButterfkyPatch2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='ButterfkyPatch',
        sampleName='ButterfkyPatch2',
        thumbnailFileName=os.path.join(iconsPath, 'ButterfkyPatch2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='ButterfkyPatch2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='ButterfkyPatch2'
    )


#
# ButterfkyPatchWidget
#

class ButterfkyPatchWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        self.reg = Reg() #Creation os an object reg for the registration

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/ButterfkyPatch.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ButterfkyPatchLogic()

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


        self.number_widget_scan = 0
        self.list_widget_scan = []
        self.manageNumberWidgetScan(2)
        self.ui.applyButton.enabled = True
        self.ui.buttonSelectOutput.connect("clicked(bool)",partial(self.openFinder,"Output"))
        self.ui.applyButton.connect("clicked(bool)",self.on_apply_button_clicked)

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
        self.reg.run(output_text, suffix_text)

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
        self.list_widget_scan.append(WidgetParameter(self.ui.verticalLayout_2,self.parent,title))

    def openFinder(self,nom : str,_) -> None : 
        """
         Open finder to let the user choose is folder
        """ 
        # Path ordi lucia lab
        # self.ui.lineEditOutput.setText("/home/luciacev/Documents/Gaelle/Data/FlexReg/output")

        self.ui.lineEditOutput.setText("/home/luciacev/Documents/Gaelle/Data/Flex_Reg/output")
        self.ui.lineEditSuffix.setText("_REG")

        # if nom=="Output":
        #     surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
        #     self.ui.lineEditOutput.setText(surface_folder)




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
# ButterfkyPatchLogic
#

class ButterfkyPatchLogic(ScriptedLoadableModuleLogic):
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
                 index_patch=0):
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

        self.curve=curve
        self.middle_point=middle_point

        self.type=type

        self.path_reg=path_reg
        self.path_output=path_output
        self.suffix=suffix
        
        self.index_patch=index_patch

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

        parameters ["curve"] = self.curve
        parameters ["middle_point"] = self.middle_point

        parameters ["type"] = self.type

        parameters ["path_reg"] = self.path_reg
        parameters["path_output"] = self.path_output
        parameters["suffix"] = self.suffix

        parameters["index_patch"] = self.index_patch



        flybyProcess = slicer.modules.flex_reg_cli
        self.cliNode = slicer.cli.run(flybyProcess,None, parameters)  
        return flybyProcess


#
# ButterfkyPatchTest
#

class ButterfkyPatchTest(ScriptedLoadableModuleTest):
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
        self.test_ButterfkyPatch1()

    def test_ButterfkyPatch1(self):
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
        inputVolume = SampleData.downloadSample('ButterfkyPatch1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = ButterfkyPatchLogic()

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
        self.timer = QTimer()

    def run(self,output_folder:str,suffix:str)->None:
        '''
        call the cli for the registration with icp method and launch onProcessUpdateICP
        '''
        if self.T1.getSurf()!=None and  self.T2.getSurf()!=None :
            if self.isButterflyPatchAvailable(self.T1.getSurf()) and self.isButterflyPatchAvailable(self.T2.getSurf()) :
                self.output_folder=output_folder
                self.suffix=suffix
                self._processed = False # To allow onProcessUpdateICP to display the time and launch endProcess
                # CLI 
                self.logic = ButterfkyPatchLogic(self.T2.getPath(),
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
                            "icp",
                            self.T1.getPath(),
                            output_folder,
                            suffix)
                self.logic.process()

                self.start_time = time.time()
                self.timer.timeout.connect(self.onProcessUpdateICP)
                self.timer.start(500)

            else:
                slicer.util.infoDisplay("Create patch on T1 and T2 before registration")
        else :
            slicer.util.infoDisplay(f"Load a vtk file in window number : 1 and 2 \nTo do this, enter the path to a vtk file and click on view.")

    def isButterflyPatchAvailable(self, model_node)->bool:
        """
        Check if the Butterfly patch is available for the provided model node.
        """
        polyData = model_node.GetPolyData()
        if polyData:
            scalars = polyData.GetPointData().GetScalars("Butterfly")
            return scalars is not None
        return False


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
            self.timerDialog = TimerDialog()
            self.timerDialog.show()
            self.timerDialog.startTimer()

        # If end cli launch display and end timer
        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
            self._processed = True
            self.timer.stop()
            self.timerDialog.endTimer()
            del self.timerDialog
            self.endProcess()



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


# Class with widget
class WidgetParameter:
    def __init__(self,layout,parent,title) -> None:
        self.parent_layout = layout
        self.parent = parent
        self.surf = None
        self.curve = None
        self.glue = False
        self.middle_point = None
        self.matrix = None
        self.title=title
        self.camera = True
        self.main_widget = QWidget()
        layout.addWidget(self.main_widget)
        self.maint_layout = QVBoxLayout(self.main_widget)
        self.setup(self.maint_layout,title)
        self.timer = QTimer()
        self.start_time = None

    def setup(self,layout,title):
        '''
        Create the widget with all the qt design and the connection of the button
        '''

        self.layout_file = QHBoxLayout()
        layout.addLayout(self.layout_file)
        self.label_1 = QLabel(f'Scan T{title}')
        self.lineedit = QLineEdit()
        self.button_select_scan = QPushButton('Select')
        self.button_select_scan.pressed.connect(self.selectFile)
        

        self.layout_file.addWidget(self.label_1)
        self.layout_file.addWidget(self.lineedit)
        self.layout_file.addWidget(self.button_select_scan)

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


        (self.lineedit_teeth_left_top , 
         self.lineedit_ratio_left_top ,
            self.lineedit_adjust_left_top) = self.displayParamater(self.layout_left_top,1,[5,0.3,0])
        
        (self.lineedit_teeth_right_top , 
         self.lineedit_ratio_right_top ,
            self.lineedit_adjust_right_top) = self.displayParamater(self.layout_right_top,2,[12,0.3,0])
        
        (self.lineedit_teeth_left_bot , 
         self.lineedit_ratio_left_bot ,
            self.lineedit_adjust_left_bot) = self.displayParamater(self.layout_left_bot,3,[3,0.33,0])

        (self.lineedit_teeth_right_bot , 
         self.lineedit_ratio_right_bot ,
            self.lineedit_adjust_right_bot) = self.displayParamater(self.layout_right_bot,4,[14,0.33,0])
        
       
        self.button_update = QPushButton('Update')
        self.button_update.pressed.connect(self.processPatch)
        self.layout_widget.addWidget(self.button_update,2,0,1,2)

       


        

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

        

    def handleStackedWidgetChange(self, index):
        # Lorsque le stackedWidget change de page, cette méthode est appelée.
        # Vérifiez si la nouvelle page est la page 0 (index 0) et appelez hideLandmark si c'est le cas.
        if index == 0:
            print("r","ooo"*100)
            self.hideLandmark()
        else :
            print("h","aaa"*100)
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

        index = int(self.combobox_patch.currentText)
        self._processed3 = False
        self.logic = ButterfkyPatchLogic(str(self.lineedit.text),
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
        


    def onProcessUpdateDelete(self):
        '''
        Update time since the beginning of the cli. When it's the end of the cli, display the patch and update combo box
        '''
        if hasattr(self, "_processed3") and self._processed3:
            return
        
        elapsed_time = time.time() - self.start_time
        self.label_time.setVisible(True)
        self.label_time.setText(f"time : {round(float(elapsed_time),2)}s")

        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
            self._processed3 = True
            self.timer.stop()
            self.viewScan()
            indexC = self.combobox_patch.findText(str(int(self.addItemsCombobox())-1))
            self.combobox_patch.removeItem(indexC)
            self.displaySegmentation(self.surf)
            



    def displayParamater(self,layout,number,parameter):
        label_teeth= QLabel(f'Teeth {number}')
        lineedit_teeth= QLineEdit(str(parameter[0]))
        label_ratio= QLabel('Ratio')
        lineedit_ratio= QLineEdit(str(parameter[1]))
        label_adjust = QLabel('Adjust')
        lineedit_adjust = QLineEdit(str(parameter[2]))

        layout.addWidget(label_teeth,0,0)
        layout.addWidget(lineedit_teeth,0,1)
        layout.addWidget(label_ratio,1,0)
        layout.addWidget(lineedit_ratio,1,1)
        layout.addWidget(label_adjust,2,0)
        layout.addWidget(lineedit_adjust,2,1)

        return lineedit_teeth, lineedit_ratio, lineedit_adjust


    def selectFile(self):
        path_file = QFileDialog.getOpenFileName(self.parent,
                                                'Open a file',
                                                '/home',
                                                'VTK File (*.vtk) ;; STL File (*.stl)')
        self.lineedit.insert(path_file)


 
        
        # if self.title==1:
        #     self.lineedit.insert('/home/luciacev/Documents/Gaelle/Data/Flex_Reg/P16_T1.vtk')
        # else : 
        #     self.lineedit.insert('/home/luciacev/Documents/Gaelle/Data/Flex_Reg/P16_T2.vtk')

        # Path for ordi lucia lab
        # if self.title==1:
        #     self.lineedit.insert('/home/luciacev/Documents/Gaelle/Data/FlexReg/P16_T1.vtk')
        # else : 
        #     self.lineedit.insert('/home/luciacev/Documents/Gaelle/Data/FlexReg/P16_T2.vtk')

    def checkLineEdit(self)->bool:
        '''
        check if input path is a vtk file
        '''
        fname, extension = os.path.splitext(os.path.basename(self.lineedit.text))
        return extension=='.vtk'


    def viewScan(self):
        '''
        Display the scan in the correct window. If scan already loaded, delete it and display the new one
        '''
        if self.surf == None :
            if self.checkLineEdit():
                # Load model
                self.surf = slicer.util.loadModel(self.lineedit.text)

                # Get data model
                displayNode = self.surf.GetDisplayNode()
                
                # Récupérer tous les vtkMRMLViewNodes disponibles dans la scène
                viewNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLViewNode')
                viewNodes.UnRegister(None) # Désenregistrer pour éviter les fuites de mémoire
                
                customLayoutId=501
                layoutManager = slicer.app.layoutManager()
                layoutManager.setLayout(customLayoutId)

                viewNode = viewNodes.GetItemAsObject(self.title - 1) if viewNodes.GetNumberOfItems() >= self.title else None
                
                if viewNode:
                    # Display model in windows
                    displayNode.SetViewNodeIDs([viewNode.GetID()])

                else:
                    slicer.util.errorDisplay(f"There is 3D windows available with the index : {self.title - 1}.")

                # Get data of model
                points = self.surf.GetPolyData().GetPoints()

                # Get center of model
                center = [0.0, 0.0, 0.0]
                for i in range(points.GetNumberOfPoints()):
                    x, y, z = points.GetPoint(i)
                    center[0] += x
                    center[1] += y
                    center[2] += z

                center[0] /= points.GetNumberOfPoints()
                center[1] /= points.GetNumberOfPoints()
                center[2] /= points.GetNumberOfPoints()


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

            else:
                slicer.util.infoDisplay("Enter a path to a vtk file")


        else :
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

    def processPatch(self)->None:
        '''
        Call the cli for the butterfly patch. Launch onProcessUpdateButterfly
        '''
        if self.checkSurfExist() :
            self._processed2 = False
            if self.add_patch.isChecked():
                index=int(self.addItemsCombobox())
            else:
                index=int(self.combobox_patch.currentText)

            self.logic = ButterfkyPatchLogic(str(self.lineedit.text),
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
                        index)
            self.logic.process()
            self.start_time = time.time()
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
        self.label_time.setText(f"time : {round(float(elapsed_time),2)}s")

        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
            self._processed2 = True
            self.timer.stop()
            self.viewScan()
            self.displaySegmentation(self.surf)
            if self.add_patch.isChecked():
                number_to_add = self.addItemsCombobox()
                self.combobox_patch.addItem(number_to_add)
                self.add_patch.setChecked(False)
                index = self.combobox_patch.findText(number_to_add)  # Remplacez "VotreValeur" par la valeur que vous souhaitez sélectionner
                if index >= 0:  # -1 signifie que la valeur n'a pas été trouvée
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
        viewNodes.UnRegister(None)  # Désenregistrer pour éviter les fuites de mémoire

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
        viewNodes.UnRegister(None)  # Désenregistrer pour éviter les fuites de mémoire

        if self.curve!=None :
            displayNode = self.curve.GetDisplayNode()
            if displayNode is not None:
                displayNode.SetVisibility2D(True)  # Rétablir la visibilité 2D
                displayNode.SetVisibility3D(False)  # Masquer la visibilité 3D

                view_ids_to_display = [viewNodes.GetItemAsObject(self.title-1).GetID()]
                displayNode.SetViewNodeIDs(view_ids_to_display)

        if self.middle_point!=None :
            displayNode = self.middle_point.GetDisplayNode()
            if displayNode is not None:
                displayNode.SetVisibility2D(True)  # Rétablir la visibilité 2D
                displayNode.SetVisibility3D(False)  # Masquer la visibilité 3D

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
        viewNodes.UnRegister(None)  # Désenregistrer pour éviter les fuites de mémoire

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
            self.logic = ButterfkyPatchLogic(str(self.lineedit.text),
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
                        index)
            self.logic.process()

            self.start_time = time.time()
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
        self.label_time.setText(f"time : {round(float(elapsed_time),2)}s")

        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
            #PLACE BACK THE CURVE AND THE MIDDLE POINT ON THE CENTER MODEL 
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
                index = self.combobox_patch.findText(number_to_add)  # Remplacez "VotreValeur" par la valeur que vous souhaitez sélectionner
                if index >= 0:  # -1 signifie que la valeur n'a pas été trouvée
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
        polyData = model_node#.GetPolyData()
        if polyData:
            scalars = polyData.GetPointData().GetScalars(name)
            return scalars is not None
        return False
    
    def  createButterfly(self,polydata):
        '''
        Check if a Butterfly1 exist, if no disable the display of the combobox
        '''
        index = 1
        final_array = None

        while True:
            array_name = f"Butterfly{index}"
            
            if self.isButterflyPatchAvailable(polydata,array_name):
                current_array = polydata.GetPointData().GetArray(array_name)
                current_tensor = torch.tensor(vtk_to_numpy(current_array)).to(torch.float32).cuda()
                
                if final_array is None:
                    final_array = current_tensor
                else:
                    # Utiliser une opération OR pour combiner les patches
                    final_array = torch.logical_or(final_array, current_tensor).to(torch.float32)
                
                index += 1
            else:
                break


        if final_array is None and self.combobox_patch.isVisible():
            self.label_patch.setVisible(False)
            self.combobox_patch.setVisible(False)
            self.delete_patch.setVisible(False)
            self.label_addpatch.setVisible(False)
            self.add_patch.setVisible(False)
        
            self.combobox_patch.addItem(str(1))

