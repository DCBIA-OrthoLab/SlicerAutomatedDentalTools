import logging
import os,json

import vtk
import SimpleITK as sitk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import qt
from qt import QFileDialog, QMessageBox

import glob
import numpy as np
from functools import partial

from pathlib import Path
import time
#import Crop_Volumes_CLI.Crop_Volumes_utils as cpu

#
# t_crop_volumes
#

class t_crop_volumes(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Crop Volumes Auto"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Tutorials"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Jeanne Claret (DCBIA lab)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#t_crop_volumes">module documentation</a>.
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

    # t_crop_volumes1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='t_crop_volumes',
        sampleName='t_crop_volumes1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 't_crop_volumes1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='t_crop_volumes1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='t_crop_volumes1'
    )

    # t_crop_volumes2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='t_crop_volumes',
        sampleName='t_crop_volumes2',
        thumbnailFileName=os.path.join(iconsPath, 't_crop_volumes2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='t_crop_volumes2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='t_crop_volumes2'
    )


#
# t_crop_volumesWidget
#

class t_crop_volumesWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        
        self.startTime = time.time()

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/t_crop_volumes.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = t_crop_volumesLogic()

        # Connections
        self.ui.SearchPathButtonF.connect("clicked(bool)", partial(self.SearchPath,"Folder_file"))
        self.ui.SearchPathButtonV.connect("clicked(bool)", partial(self.SearchPath,"Volume"))
        self.ui.SearchPathButtonOut.connect("clicked(bool)", partial(self.SearchPath,"Output"))
        self.ui.TestFiles.connect("clicked(bool)",self.Autofill)
        #self.ui.chooseType.connect("clicked(bool)", self.SearchPath)
       

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Progress Bar
        self.log_path = os.path.join(slicer.util.tempDirectory(), 'process.log')
        self.time_log = 0
        self.cliNode = None
        self.installCliNode = None  
        self.progress=0

        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setRange(0,100)
        self.ui.progressBar.setTextVisible(True)
        self.ui.label_4.setVisible(False)
        self.ui.label_time.setVisible(False)
        

       

    def Autofill(self):
        self.ui.editPathF.setText("/home/luciacev/Desktop/Jeanne/Data/Input")
        self.ui.editPathVolume.setText("/home/luciacev/Desktop/Jeanne/Data/Volume/Crop_Volume_ROI_1.mrk.json")
        self.ui.editPathOutput.setText("/home/luciacev/Desktop/Jeanne/Data/Output")
        self.ui.chooseType.setCurrentIndex(1)

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
        

        # Update buttons states and tooltips
        

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

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        isValid = self.CheckInput()
        if isValid :
        #     cpu.Crop(dict_patient_files,self.ui.editPathF.text,self.ui.editPathVolume.text,self.ui.editPathOutput.text,self.ui.editSuffix.text)

        # else :
        #     print("Error input")


            self.logic = t_crop_volumesLogic(self.ui.editPathF.text,
                                            self.ui.editPathVolume.text,
                                            self.ui.editPathOutput.text, 
                                            self.ui.editSuffix.text,
                                            self.log_path)


            self.logic.process()
            self.addObserver(self.logic.cliNode,vtk.vtkCommand.ModifiedEvent,self.onProcessUpdate)
            self.onProcessStarted()
    

        ## A VOIR l'utilite ##
        #self.addObserver(self.logic.cliNode,vtk.vtkCommand.ModifiedEvent,self.onProcessUpdate)
        #self.onProcessStarted()
            


        # with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

        #     # Compute output
        #     self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
        #                        self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

        #     # Compute inverted output (if needed)
        #     if self.ui.invertedOutputSelector.currentNode():
        #         # If additional output volume is selected then result with inverted threshold is written there
        #         self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
        #                            self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)
    

    def onProcessStarted(self):    
        self.nbFiles = 0
        for key,data in self.list_patient.items() :
            self.nbFiles += len(self.list_patient[key]) 

        self.ui.progressBar.setValue(0)
        self.progress = 0
        self.ui.label_4.setVisible(True)
        self.ui.label_4.setText("Number of processed files : "+str(self.progress)+"/"+str(self.nbFiles))
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setEnabled(True)
        self.ui.progressBar.setHidden(False)
        self.ui.progressBar.setTextVisible(True)
    

    def onProcessUpdate(self,caller,event):
    # check log file
        if os.path.isfile(self.log_path):
            
            time_progress = os.path.getmtime(self.log_path)
            if time_progress != self.time_log and self.progress < self.nbFiles:
                # if progress was made
                self.time_log = time_progress
                self.progress += 1
                progressbar_value = round((self.progress-1) /self.nbFiles * 100,2)
                
                if progressbar_value < 100 :
                    self.ui.progressBar.setValue(progressbar_value)
                    self.ui.progressBar.setFormat(str(progressbar_value)+"%")
                else:
                    self.ui.progressBar.setValue(99)
                    self.ui.progressBar.setFormat("99%")
                self.ui.label_4.setText("Number of processed files : "+str(self.progress-1)+"/"+str(self.nbFiles))
                
                

        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed:
            # process complete
            self.ui.applyButton.setEnabled(True)
            self.ui.label_4.setText("Number of processed files : "+str(self.progress)+"/"+str(self.nbFiles))
            print(f"self.progress : {self.progress}")

            if self.logic.cliNode.GetStatus() & self.logic.cliNode.ErrorsMask:
                # error
                errorText = self.logic.cliNode.GetErrorText()
                print("CLI execution failed: \n \n" + errorText)
                msg = qt.QMessageBox()
                msg.setText(f'There was an error during the process:\n \n {errorText} ')
                msg.setWindowTitle("Error")
                msg.exec_()

            else:
                # success
                print('PROCESS DONE.')
                print(self.logic.cliNode.GetOutputText())
                self.ui.progressBar.setValue(100)
                self.ui.progressBar.setFormat("100%")

                #qt.QMessageBox.information(self.parent,"Matrix applied with sucess")
                msg = qt.QMessageBox()
                msg.setIcon(qt.QMessageBox.Information)
            
                # setting message for Message Box
                msg.setText("Scan(s) cropped with success")
                
                # setting Message box window title
                msg.setWindowTitle("End of Process")
                
                # declaring buttons on Message Box
                msg.setStandardButtons(qt.QMessageBox.Ok)
                msg.exec_()

                self.ui.progressBar.setVisible(False)
                self.ui.label_4.setVisible(False)
                self.ui.editPathF.setText("")
                self.ui.editPathVolume.setText("")
                self.ui.editPathOutput.setText("")
                self.ui.chooseType.setCurrentIndex(0)


                processTime = round(time.time() - self.startTime,3)
                self.ui.label_time.setVisible(True)
                self.ui.label_time.setText("done in "+ str(processTime)+ "s")

    def SearchPath(self,object : str,_):
        """
        Function to choose if the path needed is a file or a folder
        input : Str with the name of the "editLine" we want to fill
        output : None
        """

        self.ui.applyButton.setEnabled(True)
        if object == "Folder_file":

            if self.ui.chooseType.currentIndex == 0:
                path_folder = qt.QFileDialog.getOpenFileName(self.parent,'Open a file')

            else:   
                path_folder = qt.QFileDialog.getExistingDirectory(
                    self.parent, "Select a scan folder for Input"
                )

            if path_folder != "":
                self.ui.editPathF.setText(path_folder)

        if object == "Volume":
            path_folder = qt.QFileDialog.getOpenFileName(self.parent,'Open a file')
            self.ui.editPathVolume.setText(path_folder)

        if object == "Output":
            path_folder = qt.QFileDialog.getExistingDirectory(
                self.parent, "Select a scan folder for Output"
            )
            self.ui.editPathOutput.setText(path_folder)

    
        #self.ValidApplyButton()



    def Search(self,path : str,*args ) :
        """
        Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key
        Example:
        args = ('json',['.nii.gz','.nrrd'])
        return:
            {
                'json' : ['path/a.json', 'path/b.json','path/c.json'],
                '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
                '.nrrd.gz' : ['path/c.nrrd']
            }

        Input : Path of the folder/file, list of the type (str) of file we need 
        Output : dictionnary with the key and the associated path
        """
        
        arguments=[]
        
        for arg in args:
            if type(arg) == list:
                arguments.extend(arg)
                
            else:
                arguments.append(arg)
                
        #result = {key: [i for i in glob.iglob(os.path.join(path,'**','*'),recursive=True),if i.endswith(key)] for key in arguments}
       
        result = {}  # Initialize an empty dictionary
        
        for key in arguments:

            files_matching_key = [] # empty list 'files_matching_key' to store the file paths that end with the current 'key'

            true_path = str(path)
            if os.path.isdir(true_path):
                # Use 'glob.iglob' to find all file paths ending with the current 'key' in the 'path' directory
                # and store the generator object returned by 'glob.iglob' in a variable 'files_generator'
                
                files_list = glob.iglob(os.path.join(true_path,'**', '*'),recursive=True)
                for i in files_list:
            
                    if i.endswith(key):
                        # If the file path ends with the current 'key', append it to the 'files_matching_key' list
                        files_matching_key.append(i)
                    
                    

            else :  # if a file is choosen
                if true_path.endswith(key) :
                    files_matching_key.append(path)
            
            # Assign the resulting list to the 'key' in the 'result' dictionary
            result[key] = files_matching_key
       
        return result
        

    def CheckInput(self):
        """
        function to check all input and put a pop "error" window 
        Input: /
        Output: Boolean , Dictionnary with the key and path of the files 
        """
        warning_text = ""
        if self.ui.editPathF.text=="":
            if self.ui.chooseType.currentIndex == 1 : #Folder option
                warning_text = warning_text + "Enter a Folder in input" + "\n"
            else:
                warning_text = warning_text + "Enter a File in input (.nii.gz,.nrrd.gz,.gipl.gz)" + "\n"
    
        if self.ui.editPathVolume.text=="":
            
            warning_text = warning_text + "Choose a ROI file (.json)" + "\n"
            
        else :
            self.list_patient=self.Search(self.ui.editPathF.text,".nii.gz",".nrrd.gz",".gipl.gz") #dictionnary with all path of file (working on folder or file)
           
            isfile = 0
            for key,data in self.list_patient.items() :
                
                if len(self.list_patient[key])!=0 :
                    isfile = 1 # There are good types of files in the folder

            if self.ui.chooseType.currentIndex==1 and isfile ==0 :
                warning_text = warning_text + "Folder empty or wrong type of patient files " + "\n"
                warning_text = warning_text + "File authorized : .nii.gz, .nrrd.gz, .gipl.gz" + "\n"

            elif self.ui.chooseType.currentIndex==0 and isfile ==0 :
                warning_text = warning_text + "Wrong type of patient file  detected" + "\n"
                warning_text = warning_text + "File authorized : .nii.gz, .nrrd.gz, .gipl.gz" + "\n"


        if self.ui.editPathOutput.text=="":
            
            warning_text = warning_text + "Enter the output Folder" + "\n"
           
    
        if warning_text=="":
            result = True
            return result

        else :
            qt.QMessageBox.warning(self.parent, "Warning", warning_text)
            result = False
            return result
#
# t_crop_volumesLogic
#

class t_crop_volumesLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self,scan_files_path=None,path_ROI_file=None,output_path=None,suffix=None,logPath=None):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.scan_files_path = scan_files_path
        self.path_ROI_file = path_ROI_file
        self.output_path = output_path
        self.suffix = suffix
        self.logPath = logPath

        self.cliNode = None
        self.installCliNode = None

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")


    def process(self):
        """
        Run the processing algorithm.
        """
        parameters = {}
        
        parameters ["scan_files_path"] = self.scan_files_path
        parameters ["path_ROI_file"] = self.path_ROI_file
        parameters ["output_path"] = self.output_path
        parameters ["suffix"] = self.suffix
        parameters ["logPath"] = self.logPath
        
        flybyProcess = slicer.modules.crop_volumes_cli
        self.cliNode = slicer.cli.run(flybyProcess,None, parameters)  
        
        return flybyProcess

    # def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
    #     """
    #     Run the processing algorithm.
    #     Can be used without GUI widget.
    #     :param inputVolume: volume to be thresholded
    #     :param outputVolume: thresholding result
    #     :param imageThreshold: values above/below this threshold will be set to 0
    #     :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    #     :param showResult: show output volume in slice viewers
    #     """

    #     if not inputVolume or not outputVolume:
    #         raise ValueError("Input or output volume is invalid")

    #     import time
    #     startTime = time.time()
    #     logging.info('Processing started')

    #     # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    #     cliParams = {
    #         'InputVolume': inputVolume.GetID(),
    #         'OutputVolume': outputVolume.GetID(),
    #         'ThresholdValue': imageThreshold,
    #         'ThresholdType': 'Above' if invert else 'Below'
    #     }
    #     cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    #     # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    #     slicer.mrmlScene.RemoveNode(cliNode)

    #     stopTime = time.time()
    #     logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# t_crop_volumesTest
#

class t_crop_volumesTest(ScriptedLoadableModuleTest):       
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
        self.test_t_crop_volumes1()

    def test_t_crop_volumes1(self):
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
        inputVolume = SampleData.downloadSample('t_crop_volumes1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = t_crop_volumesLogic()

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
