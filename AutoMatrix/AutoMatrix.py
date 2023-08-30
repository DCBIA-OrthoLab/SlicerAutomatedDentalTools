
import logging
import os,sys,time,zipfile,urllib.request,shutil

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import qt
import glob
import numpy as np
from qt import QFileDialog,QMessageBox
from functools import partial
import SimpleITK as sitk

from Matrix_CLI.Apply_matrix_utils.GZ_tools import GetPatients

#
# AutoMatrix
#test

class AutoMatrix(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "AutoMatrix"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Automated Dental Tools"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Leroux Gaelle"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#AutoMatrix">module documentation</a>.
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

    # AutoMatrix1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='AutoMatrix',
        sampleName='AutoMatrix1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'AutoMatrix1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='AutoMatrix1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='AutoMatrix1'
    )

    # AutoMatrix2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='AutoMatrix',
        sampleName='AutoMatrix2',
        thumbnailFileName=os.path.join(iconsPath, 'AutoMatrix2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='AutoMatrix2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='AutoMatrix2'
    )


#
# AutoMatrixWidget
#

class AutoMatrixWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/AutoMatrix.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AutoMatrixLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # BUTTONS
        self.ui.SearchButtonMatrix.connect("clicked(bool)",partial(self.openFinder,"Matrix"))
        self.ui.SearchButtonPatient.connect("clicked(bool)",partial(self.openFinder,"Patient"))
        self.ui.SearchButtonOutput.connect("clicked(bool)",partial(self.openFinder,"Output"))
        self.ui.ButtonAutoFill.connect("clicked(bool)",self.Autofill)
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.CheckBoxMirror.connect('clicked(bool)', self.Mirror)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # VARIABLES
        self.log_path = os.path.join(slicer.util.tempDirectory(), 'process.log')
        self.time_log = 0 # for progress bar
        self.cliNode = None
        self.installCliNode = None  
        self.progress=0

        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setRange(0,100)
        self.ui.progressBar.setTextVisible(True)
        self.ui.label_info.setVisible(False)
        self.ui.ComboBoxPatient.setCurrentIndex(1)
        self.ui.ComboBoxMatrix.setCurrentIndex(1)

        self.ui.ButtonAutoFill.setVisible(True)

    def Mirror(self):
        if self.ui.CheckBoxMirror.isChecked():
            self.ui.SearchButtonMatrix.setEnabled(False)
            self.ui.LineEditMatrix.setEnabled(False)
            self.ui.ComboBoxMatrix.setCurrentIndex(0)
            self.ui.ComboBoxMatrix.setEnabled(False)
            self.DownloadMirror()

        else : 
            self.ui.SearchButtonMatrix.setEnabled(True)
            self.ui.LineEditMatrix.setEnabled(True)
            self.ui.ComboBoxMatrix.setCurrentIndex(1)
            self.ui.ComboBoxMatrix.setEnabled(True)
            self.ui.LineEditMatrix.setText("")


    def DownloadUnzip(self, url, directory, folder_name=None, num_downl=1, total_downloads=1):
        """Function to download and unzip a file from a url with a progress bar"""
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


    def DownloadMirror(self):
        url = "https://github.com/GaelleLeroux/DCBIA_Apply_matrix/releases/download/AutoMatrixMirror/Mirror.zip"
        name = "Mirror_matrix"

        documentsLocation = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documentsLocation)
        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
        )
        self.isDCMInput = False
        if not os.path.exists(self.SlicerDownloadPath):
            os.makedirs(self.SlicerDownloadPath)

        scan_folder = self.DownloadUnzip(
                url=url,
                directory=os.path.join(self.SlicerDownloadPath),
                folder_name=os.path.join(name)
                if not self.isDCMInput
                else os.path.join(name),
            )
        self.ui.LineEditMatrix.setText(os.path.join(scan_folder,"Mirror/Matrix_mirror.tfm"))

    def Autofill(self):

        #SEG 47
        self.ui.LineEditPatient.setText("/home/luciacev/Desktop/AutoMatrix/AutoMatrixRelease4/r_2_patient_files")


        #MATRIX 47
        self.ui.LineEditMatrix.setText("/home/luciacev/Desktop/AutoMatrix/AutoMatrixRelease4/r_2_matrix")



        self.ui.LineEditOutput.setText("/home/luciacev/Desktop/AutoMatrix/output")

        self.ui.ComboBoxPatient.setCurrentIndex(1)
        self.ui.ComboBoxMatrix.setCurrentIndex(1)

    
    def openFinder(self,nom : str,_) -> None : 
        """
         Open finder to let the user choose is files or folder
        """ 
        
        
        if nom=="Matrix":
            if self.ui.ComboBoxMatrix.currentIndex==1:
                  surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            else :
                  surface_folder = QFileDialog.getOpenFileName(self.parent,'Open a file',)

            self.ui.LineEditMatrix.setText(surface_folder)

        elif nom=="Patient":
            if self.ui.ComboBoxPatient.currentIndex==1:
                surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            else :
                surface_folder = QFileDialog.getOpenFileName(self.parent,'Open a file',)
            self.ui.LineEditPatient.setText(surface_folder)

        elif nom=="Output":
            surface_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.LineEditOutput.setText(surface_folder)
        

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

        # Update buttons states and tooltips
        self.ui.applyButton.toolTip = "Apply Matrix"
        self.ui.applyButton.enabled = True


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


    
    def search(self,path, *args):
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
        """
        arguments = []
        for arg in args:
            if type(arg) == list:
                arguments.extend(arg)
            else:
                arguments.append(arg)
        return {
            key: [
                i
                for i in glob.iglob(
                    os.path.normpath("/".join([path, "**", "*"])), recursive=True
                )
                if i.endswith(key)
            ]
            for key in arguments
        }
    

    def onApplyButton(self,_)->None:
        """
        Run processing when user clicks "Apply" button.
        """
        if self.CheckGoodEntre():
            self.ui.progressBar.setVisible(True)
            self.ui.progressBar.setEnabled(True)
            self.ui.progressBar.setTextVisible(True)
            
            self.ui.label_info.setVisible(True)
            output = self.ui.LineEditOutput.text
            input = self.ui.LineEditPatient.text
            suffix = self.ui.LineEditSuffix.text
            self.logic = AutoMatrixLogic(self.ui.LineEditPatient.text,
                                            self.ui.LineEditMatrix.text,
                                            self.ui.LineEditOutput.text, 
                                            self.ui.LineEditSuffix.text,
                                            self.log_path)


            self.logic.process()
            self.addObserver(self.logic.cliNode,vtk.vtkCommand.ModifiedEvent,self.onProcessUpdate)
            self.onProcessStarted()

            
           

                            

        
    def ProcessVolume(self)->None:
        patients,nb_files = GetPatients(self.ui.LineEditPatient.text,self.ui.LineEditMatrix.text)

        if nb_files!=0:
            for key,values in patients.items():
                for scan in values['scan']:
                    image = slicer.util.loadVolume(scan)
                    for matrix in values['matrix']:
                        try:
                            tform = slicer.util.loadTransform(matrix)
                            image.SetAndObserveTransformNodeID(tform.GetID())
                            image.HardenTransform()
                            outpath = scan.replace(self.ui.LineEditPatient.text,self.ui.LineEditOutput.text)
                            try : 
                                matrix_name = os.path.basename(matrix).split('.tfm')[0].split(key)[1]
                            except : 
                                matrix_name = os.path.basename(matrix).split('.tfm')[0]
                            
                            if not os.path.exists(os.path.dirname(outpath)):
                                os.makedirs(os.path.dirname(outpath))

                            slicer.util.saveNode(image,outpath.split('.nii.gz')[0]+self.ui.LineEditSuffix.text+matrix_name+'.nii.gz')

                        except:
                            print("An issue occured")
                            pass
                    self.UpdateProgressBar(False)
        


              

    def UpdateProgressBar(self,end:bool)->None:
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
            print(self.logic.cliNode.GetOutputText())
            self.ui.progressBar.setValue(100)
            self.ui.progressBar.setFormat("100%")

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
            self.ui.LineEditOutput.setText("")
            self.ui.LineEditPatient.setText("")
            self.ui.LineEditMatrix.setText("")
            self.ui.ComboBoxMatrix.setCurrentIndex(1)
            self.ui.ComboBoxPatient.setCurrentIndex(1)


            

    def onProcessStarted(self)->None:   
        """
        Initialize the variables and progress bar.
        """ 
        if os.path.isdir(self.ui.LineEditPatient.text):
            self.nbFiles = len(self.dico_patient[".vtk"]) + len(self.dico_patient['.vtp']) + len(self.dico_patient['.stl']) + len(self.dico_patient['.off']) + len(self.dico_patient['.obj']) + len(self.dico_patient['.nii.gz'])
        else:
            self.nbFiles = 1
        self.ui.progressBar.setValue(0)
        self.progress = 0
        self.ui.label_info.setText("Number of processed files : "+str(self.progress)+"/"+str(self.nbFiles))
        self.ui.progressBar.setEnabled(True)
        self.ui.progressBar.setHidden(False)
        self.ui.progressBar.setTextVisible(True)
        self.ui.progressBar.setFormat("0%")


    def onProcessUpdate(self,caller,event)->None:
        """
        Call at each event to update the progress bar.
        """ 
    # check log file
        if os.path.isfile(self.log_path):
            time = os.path.getmtime(self.log_path)
            if time != self.time_log:
                # if progress was made
                self.time_log = time
                self.progress += 1
                progressbar_value = (self.progress-1) /self.nbFiles * 100
                if progressbar_value < 100 :
                    self.ui.progressBar.setValue(progressbar_value)
                    self.ui.progressBar.setFormat(str(round(progressbar_value,2))+"%")
                else:
                    self.ui.progressBar.setValue(99)
                    self.ui.progressBar.setFormat("99%")
                self.ui.label_info.setText("Number of processed files : "+str(self.progress-1)+"/"+str(self.nbFiles))
                
                

        if self.logic.cliNode.GetStatus() & self.logic.cliNode.Completed :
            self.ui.applyButton.setEnabled(True)

            if self.logic.cliNode.GetStatus() & self.logic.cliNode.ErrorsMask:
                # error
                errorText = self.logic.cliNode.GetErrorText()
                print("CLI execution failed: \n \n" + errorText)
                msg = qt.QMessageBox()
                msg.setText(f'There was an error during the process :\n \n {errorText} ')
                msg.setWindowTitle("Error")
                msg.exec_()

            else:
                # success
                print('PROCESS DONE.')
                print(self.logic.cliNode.GetOutputText())

                self.ProcessVolume()
                self.UpdateProgressBar(True)
                
    

    
    def CheckGoodEntre(self)->bool:
        """
        Check if the folder and/or files have the right type of files in entries, return true or false
        """ 

        warning_text = ""
        if self.ui.LineEditOutput.text=="":
            warning_text = warning_text + "Enter folder output" + "\n"
    
        if self.ui.LineEditPatient.text=="":
            if self.ui.ComboBoxPatient.currentIndex==1 : # folder option
                warning_text = warning_text + "Enter folder patients" + "\n"
            elif self.ui.ComboBoxPatient.currentIndex==0 : # file option
                warning_text = warning_text + "Enter file patient" + "\n"
        else :
            if self.ui.ComboBoxPatient.currentIndex==1 : #folder option
                self.dico_patient=self.search(self.ui.LineEditPatient.text,'.vtk','.vtp','.stl','.off','.obj','.nii.gz')
                if len(self.dico_patient['.vtk'])==0 and len(self.dico_patient['.vtp']) and len(self.dico_patient['.stl']) and len(self.dico_patient['.off']) and len(self.dico_patient['.obj']) and len(self.dico_patient['.nii.gz']) :
                    warning_text = warning_text + "Folder empty or wrong type of file patient" + "\n"
                    warning_text = warning_text + "File authorized : .vtk / .vtp / .stl / .off / .obj / .nii.gz" + "\n"
            elif self.ui.ComboBoxPatient.currentIndex==0 : # file option
                fname, extension = os.path.splitext(os.path.basename(self.ui.LineEditPatient.text))
                try : 
                    fname, extension2 = os.path.splitext(os.path.basename(fname))
                    extension = extension2+extension
                except : 
                    print("not a .nii.gz")
                if extension != ".vtk" and extension != ".vtp" and extension != ".stl" and extension != ".off" and extension != ".obj" and extension != ".nii.gz" :
                        warning_text = warning_text + "Wrong type of file patient detected" + "\n"
                        warning_text = warning_text + "File authorized : .vtk / .vtp / .stl / .off / .obj / .nii.gz" + "\n"
        

        if self.ui.LineEditMatrix.text=="":
            if self.ui.ComboBoxMatrix.currentIndex==1 : # folder option
                warning_text = warning_text + "Enter folder matrix" + "\n"
            elif self.ui.ComboBoxMatrix.currentIndex==0 and self.ui.CheckBoxMirror.isChecked()==False : # file option
                warning_text = warning_text + "Enter file matrix" + "\n"
        else :
            if self.ui.ComboBoxMatrix.currentIndex==1 : # folder option
                dico_matrix=self.search(self.ui.LineEditMatrix.text,'.npy','.h5','.tfm','.mat','.txt')
                if len(dico_matrix['.npy'])==0 and len(dico_matrix['.h5'])==0 and len(dico_matrix['.tfm'])==0 and len(dico_matrix['.mat'])==0 and len(dico_matrix['.txt'])==0 :
                    warning_text = warning_text + "Folder empty or wrong type of files matrix " + "\n"
                    warning_text = warning_text + "File authorized : .npy / .h5 / .tfm / . mat / .txt" + "\n"
            elif self.ui.ComboBoxMatrix.currentIndex==0 and self.ui.CheckBoxMirror.isChecked()==False: # file option
                fname, extension = os.path.splitext(os.path.basename(self.ui.LineEditMatrix.text))
                if extension != ".npy"  and extension != ".h5" and extension != ".tfm" and extension != ".mat" and extension != ".txt":
                        warning_text = warning_text + "Wrong type of file matrix detect" + "\n"
                        warning_text = warning_text + "File authorized : .npy / .h5 / .tfm / . mat / .txt" + "\n"

        if warning_text=='':
            return True

        else :
            qt.QMessageBox.warning(self.parent, "Warning", warning_text)
            return False
        


#
# AutoMatrixLogic
#

class AutoMatrixLogic(ScriptedLoadableModuleLogic):
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
        
        parameters = {}
        
        parameters ["path_patient_intput"] = self.path_patient_intput
        parameters ["path_matrix_intput"] = self.path_matrix_intput
        parameters ["path_patient_output"] = self.path_patient_output
        parameters ["suffix"] = self.suffix
        parameters ["logPath"] = self.logPath
        
        flybyProcess = slicer.modules.matrix_cli
        self.cliNode = slicer.cli.run(flybyProcess,None, parameters)  
        return flybyProcess

   

#
# AutoMatrixTest
#

class AutoMatrixTest(ScriptedLoadableModuleTest):
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
        self.test_AutoMatrix1()

    def test_AutoMatrix1(self):
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
        inputVolume = SampleData.downloadSample('AutoMatrix1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = AutoMatrixLogic()

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
