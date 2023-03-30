import os, sys, time, logging, zipfile, urllib.request, shutil, glob
import vtk, qt, slicer
from qt import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QTabWidget,
    QCheckBox,
    QPushButton,
    QPixmap,
    QIcon,
    QSize,
    QLabel,
    QHBoxLayout,
    QGridLayout,
    QMediaPlayer,
)
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from functools import partial

from Methode.IOS import Auto_IOS, Semi_IOS
from Methode.CBCT import Semi_CBCT, Auto_CBCT
from Methode.Methode import Methode
from Methode.Progress import Display


class ASO(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = (
            "ASO"  # TODO: make this more human readable by adding spaces
        )
        self.parent.categories = [
            "Automated Dental Tools"
        ]  # set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = (
            []
        )  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "Nathan Hutin (UoM), Luc Anchling (UoM)"
        ]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
        This is an example of scripted loadable module bundled in an extension.
        See more information in <a href="https://github.com/organization/projectname#ASO">module documentation</a>.
        """
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.registerSampleData)

        #
        # Register sample data sets in Sample Data module
        #

    def registerSampleData(self):
        """
        Add data sets to Sample Data module.
        """
        # It is always recommended to provide sample data for users to make it easy to try the module,
        # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

        import SampleData

        iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

        # To ensure that the source code repository remains small (can be downloaded and installed quickly)
        # it is recommended to store data sets that are larger than a few MB in a Github release.

        # ALI1
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            # Category and sample name displayed in Sample Data module
            category="ASO",
            sampleName="ASO1",
            # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
            # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
            thumbnailFileName=os.path.join(iconsPath, "ASO1.png"),
            # Download URL and target file name
            uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
            fileNames="ASO1.nrrd",
            # Checksum to ensure file integrity. Can be computed by this command:
            #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
            checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
            # This node name will be used when the data set is loaded
            nodeNames="ASO1",
        )

        # ASO2
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            # Category and sample name displayed in Sample Data module
            category="ASO",
            sampleName="ASO2",
            thumbnailFileName=os.path.join(iconsPath, "ASO2.png"),
            # Download URL and target file name
            uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
            fileNames="ASO2.nrrd",
            checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
            # This node name will be used when the data set is loaded
            nodeNames="ASO2",
        )


        # self.ui.ButtonSearchModelAli.clicked.connect(
        #     lambda: self.SearchModel(self.ui.lineEditModelAli)
        # )
class PopUpWindow(qt.QDialog):
    """PopUpWindow class
    This class is used to create a pop-up window with a list of buttons
    """
    def __init__(self,title="Title",listename=["1","2","3"],type="radio", tocheck=None):
        QWidget.__init__(self)
        self.setWindowTitle(title)
        layout = QGridLayout()
        self.setLayout(layout)
        self.ListButtons = []
        self.listename = listename
        self.type = type

        if self.type == 'radio':
            self.radiobutton(layout)

        elif self.type == 'checkbox':
            self.checkbox(layout)
            if tocheck is not None:
                self.toCheck(tocheck)
    
    def checkbox(self,layout):
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
        layout.addWidget(button, len(self.listename)+1, j-2)
        button = qt.QPushButton("Deselect All")
        button.connect("clicked()", self.onClickedDeselectAll)
        layout.addWidget(button, len(self.listename)+1, j-1)

        # Add a button to close the dialog
        button = qt.QPushButton("OK")
        button.connect("clicked()", self.onClickedCheckbox)
        layout.addWidget(button, len(self.listename)+1, j)

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
        self.checked = [self.listename[i] for i in range(len(self.listename)) if TrueFalse[i]]
        self.accept()
    
    def radiobutton(self,layout):
        for i in range(len(self.listename)):
            radiobutton = qt.QRadioButton(self.listename[i])
            self.ListButtons.append(radiobutton)
            radiobutton.connect("clicked(bool)",self.onClickedRadio)
            layout.addWidget(radiobutton, i, 0)
    
    def onClickedRadio(self):
        self.checked = self.listename[[button.isChecked() for button in self.ListButtons].index(True)]
        self.accept()

#
# ASOWidget
#

class ASOWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initiASOzed.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.nb_patient = 0  # number of scans in the input folder

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initiASOzed.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ASO.ui"))
        self.layout.addWidget(uiWidget)

        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ASOLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).

        """
            888     888        d8888 8888888b.  8888888        d8888 888888b.   888      8888888888  .d8888b.  
            888     888       d88888 888   Y88b   888         d88888 888  "88b  888      888        d88P  Y88b 
            888     888      d88P888 888    888   888        d88P888 888  .88P  888      888        Y88b.      
            Y88b   d88P     d88P 888 888   d88P   888       d88P 888 8888888K.  888      8888888     "Y888b.   
             Y88b d88P     d88P  888 8888888P"    888      d88P  888 888  "Y88b 888      888            "Y88b. 
              Y88o88P     d88P   888 888 T88b     888     d88P   888 888    888 888      888              "888 
               Y888P     d8888888888 888  T88b    888    d8888888888 888   d88P 888      888        Y88b  d88P 
                Y8P     d88P     888 888   T88b 8888888 d88P     888 8888888P"  88888888 8888888888  "Y8888P"
        """

        self.MethodeDic = {
            "Semi_IOS": Semi_IOS(self),
            "Auto_IOS": Auto_IOS(self),
            "Semi_CBCT": Semi_CBCT(self),
            "Auto_CBCT": Auto_CBCT(self),
        }
        self.reference_lm = []
        self.ActualMeth = Methode
        self.ActualMeth = self.MethodeDic["Auto_CBCT"]
        self.type = "CBCT"
        self.nb_scan = 0
        self.startprocess = 0
        self.patient_process = 0
        self.dicchckbox = {}
        self.dicchckbox2 = {}
        self.display = Display
        """
        exemple dic = {'teeth'=['A,....],'Type'=['O',...]}
        """

        self.log_path = os.path.join(slicer.util.tempDirectory(), "process.log")
        self.time = 0

        # use messletter to add big comment with univers as police

        documentsLocation = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documentsLocation)
        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
            "ASO",
            "ASO_" + self.type,
        )

        if not os.path.exists(self.SlicerDownloadPath):
            os.makedirs(self.SlicerDownloadPath)

        """
                              
                                        8888888 888b    888 8888888 88888888888 
                                          888   8888b   888   888       888     
                                          888   88888b  888   888       888     
                                          888   888Y88b 888   888       888     
                                          888   888 Y88b888   888       888     
                                          888   888  Y88888   888       888     
                                          888   888   Y8888   888       888     
                                        8888888 888    Y888 8888888     888 
                              
        """

        # self.initCheckbox(self.MethodeDic['Semi_IOS'],self.ui.LayoutLandmarkSemiIOS,self.ui.tohideIOS)
        # self.initCheckbox(self.MethodeDic['Auto_IOS'],self.ui.LayoutLandmarkAutoIOS,self.ui.tohideIOS)
        self.initCheckboxIOS(
            self.MethodeDic["Auto_IOS"],
            self.ui.LayoutAutoIOS_tooth,
            self.ui.tohideAutoIOS_tooth,
            self.ui.LayoutLandmarkAutoIOS,
            self.ui.checkBoxOcclusionAutoIOS
        )
        self.initCheckboxIOS(
            self.MethodeDic["Semi_IOS"],
            self.ui.LayoutSemiIOS_tooth,
            self.ui.tohideSemiIOS_tooth,
            self.ui.LayoutLandmarkSemiIOS,
            self.ui.checkBoxOcclusionSemiIOS
        )

        self.initCheckbox(
            self.MethodeDic["Semi_CBCT"],
            self.ui.LayoutLandmarkSemiCBCT,
            self.ui.tohideCBCT,
        )  # a decommmente
        self.initCheckbox(
            self.MethodeDic["Auto_CBCT"],
            self.ui.LayoutLandmarkAutoCBCT,
            self.ui.tohideCBCT,
        )
        self.HideComputeItems()
        # self.initTest(self.MethodeDic['Semi_IOS'])

        # self.dicchckbox=self.ActualMeth.getcheckbox()
        # self.dicchckbox2=self.ActualMeth.getcheckbox2()

        # self.enableCheckbox()

        # self.SwitchMode(0)
        self.SwitchType()

        """
                                                                                       
                     .d8888b.   .d88888b.  888b    888 888b    888 8888888888  .d8888b.  88888888888 
                    d88P  Y88b d88P" "Y88b 8888b   888 8888b   888 888        d88P  Y88b     888     
                    888    888 888     888 88888b  888 88888b  888 888        888    888     888     
                    888        888     888 888Y88b 888 888Y88b 888 8888888    888            888     
                    888        888     888 888 Y88b888 888 Y88b888 888        888            888     
                    888    888 888     888 888  Y88888 888  Y88888 888        888    888     888     
                    Y88b  d88P Y88b. .d88P 888   Y8888 888   Y8888 888        Y88b  d88P     888     
                     "Y8888P"   "Y88888P"  888    Y888 888    Y888 8888888888  "Y8888P"      888 
                                                                                                            
        """

        self.ui.ButtonSearchScanLmFolder.connect("clicked(bool)", self.SearchScanLm)
        self.ui.ButtonSearchReference.connect("clicked(bool)", self.SearchReference)
        self.ui.ButtonSearchModelSegOr.connect("clicked(bool)", self.SearchModelSegOr)
        self.ui.ButtonSearchModelAli.connect("clicked(bool)", self.SearchModelALI)
        self.ui.ButtonOriented.connect("clicked(bool)", self.onPredictButton)
        self.ui.ButtonOutput.connect("clicked(bool)", self.ChosePathOutput)
        self.ui.ButtonCancel.connect("clicked(bool)", self.onCancel)
        self.ui.ButtonSuggestLmIOS.clicked.connect(self.SelectSuggestLandmark)
        self.ui.ButtonSuggestLmIOSSemi.clicked.connect(self.SelectSuggestLandmark)
        self.ui.CbInputType.currentIndexChanged.connect(self.SwitchType)
        self.ui.CbModeType.currentIndexChanged.connect(self.SwitchType)
        self.ui.ButtonTestFiles.clicked.connect(lambda: self.SearchScanLm(True))
        self.ui.checkBoxOcclusionAutoIOS.toggled.connect(partial(self.OcclusionCheckbox,self.MethodeDic['Auto_IOS'].getcheckbox()['Jaw']['Upper'],self.MethodeDic['Auto_IOS'].getcheckbox()['Jaw']['Lower'],self.MethodeDic['Semi_IOS'].getcheckbox()['Teeth']))

    """

                                                                                                                                                                    
                888888b.   888     888 88888888888 88888888888  .d88888b.  888b    888  .d8888b.  
                888  "88b  888     888     888         888     d88P" "Y88b 8888b   888 d88P  Y88b 
                888  .88P  888     888     888         888     888     888 88888b  888 Y88b.      
                8888888K.  888     888     888         888     888     888 888Y88b 888  "Y888b.   
                888  "Y88b 888     888     888         888     888     888 888 Y88b888     "Y88b. 
                888    888 888     888     888         888     888     888 888  Y88888       "888 
                888   d88P Y88b. .d88P     888         888     Y88b. .d88P 888   Y8888 Y88b  d88P 
                8888888P"   "Y88888P"      888         888      "Y88888P"  888    Y888  "Y8888P"
                                                                                                                                                                    
                                                                                                                                                                    

    """

    def SwitchMode(self, index):
        """Function to change the UI depending on the mode selected (Semi or Fully Automated)"""
        if index == 1:  # Semi-Automated
            self.ui.label_3.setText("Scan / Landmark Folder")
            self.ui.label_6.setVisible(False)
            self.ui.label_7.setVisible(False)
            self.ui.lineEditModelAli.setVisible(False)
            self.ui.lineEditModelAli.setText(" ")
            self.ui.lineEditModelSegOr.setVisible(False)
            self.ui.lineEditModelSegOr.setText(" ")
            self.ui.ButtonSearchModelAli.setVisible(False)
            self.ui.ButtonSearchModelSegOr.setVisible(False)
            self.ui.checkBoxSmallFOV.setVisible(False)

        if index == 0:  # Fully Automated
            self.ui.label_3.setText("Scan Folder")
            # self.ui.label_6.setVisible(True)
            self.ui.label_7.setVisible(True)
            self.ui.lineEditModelSegOr.setVisible(True)
            self.ui.ButtonSearchModelSegOr.setVisible(True)
            self.ui.checkBoxSmallFOV.setVisible(True)
            if isinstance(self.ActualMeth,(Auto_IOS,Semi_IOS)):
                self.ui.lineEditModelAli.setVisible(False)
                self.ui.ButtonSearchModelAli.setVisible(False)
                self.ui.label_6.setVisible(False)
            else :
                self.ui.lineEditModelAli.setVisible(True)
                self.ui.ButtonSearchModelAli.setVisible(True)
                self.ui.label_6.setVisible(True)              

    def SwitchType(self):
        """Function to change the UI and the Method in ASO depending on the selected type (Semi CBCT, Fully CBCT...)"""
        if (
            self.ui.CbInputType.currentIndex == 0
            and self.ui.CbModeType.currentIndex == 1
        ):
            self.ActualMeth = self.MethodeDic["Semi_CBCT"]
            self.ui.stackedWidget.setCurrentIndex(0)
            self.type = "CBCT"

        elif (
            self.ui.CbInputType.currentIndex == 0
            and self.ui.CbModeType.currentIndex == 0
        ):
            self.ActualMeth = self.MethodeDic["Auto_CBCT"]
            self.ui.stackedWidget.setCurrentIndex(1)
            self.type = "CBCT"
            self.ui.label_7.setText("Orientation Model Folder")

        elif (
            self.ui.CbInputType.currentIndex == 1
            and self.ui.CbModeType.currentIndex == 1
        ):
            self.ActualMeth = self.MethodeDic["Semi_IOS"]
            self.ui.stackedWidget.setCurrentIndex(2)
            self.type = "IOS"

        elif (
            self.ui.CbInputType.currentIndex == 1
            and self.ui.CbModeType.currentIndex == 0
        ):
            self.ActualMeth = self.MethodeDic["Auto_IOS"]
            self.ui.stackedWidget.setCurrentIndex(3)
            self.type = "IOS"
            self.ui.label_7.setText("Segmentation Model Folder")
        # UI Changes
        self.SwitchMode(self.ui.CbModeType.currentIndex)

        self.dicchckbox = self.ActualMeth.getcheckbox()
        self.dicchckbox2 = self.ActualMeth.getcheckbox2()

        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
            "ASO",
            "ASO_" + self.type,
        )

        self.ClearAllLineEdits()

        self.enableCheckbox()

        self.HideComputeItems()

        # best = ['Ba','N','RPo']
        # for checkbox in self.logic.iterillimeted(self.dicchckbox):
        #     if checkbox.text in best and checkbox.isEnabled():
        #         checkbox.setCheckState(True)

    def ClearAllLineEdits(self):
        """Function to clear all the line edits"""
        self.ui.lineEditScanLmPath.setText("")
        self.ui.lineEditRefFolder.setText("")
        self.ui.lineEditModelAli.setText("")
        self.ui.lineEditModelSegOr.setText("")
        self.ui.lineEditOutputPath.setText("")
    
    def DownloadUnzip(self, url, directory, folder_name=None, num_downl=1, total_downloads=1):
        """Function to download and unzip a file from a url with a progress bar"""
        out_path = os.path.join(directory, folder_name)

        if not os.path.exists(out_path):
            # print("Downloading {}...".format(folder_name.split("/")[0]))
            os.makedirs(out_path)

            temp_path = os.path.join(directory, "temp.zip")

            # Download the zip file from the url
            with urllib.request.urlopen(url) as response, open(
                temp_path, "wb"
            ) as out_file:
                # Pop up a progress bar with a QProgressDialog
                progress = qt.QProgressDialog(
                    "Downloading {} (File {}/{})".format(folder_name.split("/")[0],num_downl, total_downloads), "Cancel", 0, 100, self.parent
                )
                progress.setWindowModality(qt.Qt.WindowModal)
                progress.setWindowTitle("Downloading {}...".format(folder_name.split("/")[0]))
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

    def SearchScanLm(self,test=False):
        """Function to search the scan folder and to check if the scans are valid"""
        if not test:
            scan_folder = qt.QFileDialog.getExistingDirectory(
                    self.parent, "Select a scan folder for Input"
                )
        else:
            name,url = self.ActualMeth.getTestFileList()
            
            scan_folder = self.DownloadUnzip(
                url=url,
                directory=os.path.join(self.SlicerDownloadPath),
                folder_name=os.path.join("Test_Files", name),
            )
            self.SearchReference(test=True)
            self.SearchModelSegOr()
            if self.type == "CBCT":
                self.SearchModelALI(test=True)

        if not scan_folder == "":
            nb_scans = self.ActualMeth.NumberScan(scan_folder)
            error = self.ActualMeth.TestScan(scan_folder)

            if isinstance(error, str):
                qt.QMessageBox.warning(self.parent, "Warning", error)
            else:
                self.nb_patient = nb_scans
                self.ui.lineEditScanLmPath.setText(scan_folder)
                self.ui.LabelInfoPreProc.setText(
                    "Number of scans to process : " + str(nb_scans)
                )
                self.ui.LabelProgressPatient.setText(
                    "Patient process : 0 /" + str(nb_scans)
                )
                self.enableCheckbox()

                if self.ui.lineEditOutputPath.text == "":
                    dir, spl = os.path.split(scan_folder)
                    self.ui.lineEditOutputPath.setText(os.path.join(dir, spl + "Or"))


    def SearchReference(self,test=False):
        """Function to search the reference folder and to check if the reference is valid"""
        referenceList = self.ActualMeth.getReferenceList()
        refList = list(referenceList.keys())
        refList.append("Select your own folder")
        
        if test:
            ret = refList[0]
        
        else:    
            s = PopUpWindow(title="Choice of Reference Files",listename=refList,type="radio")
            s.exec_()
            ret = s.checked
            
        if ret == "Select your own folder":
            ref_folder = qt.QFileDialog.getExistingDirectory(
                self.parent, "Select a scan folder for Reference"
            )

        else:  # Automatically Download the reference, unzip it and set the path
            ref_folder = self.DownloadUnzip(
                url=referenceList[ret],
                directory=os.path.join(self.SlicerDownloadPath),
                folder_name=os.path.join("Reference", ret),
            )

        # print(ref_folder)

        if not ref_folder == "":
            error = self.ActualMeth.TestReference(ref_folder)

            if isinstance(error, str):
                qt.QMessageBox.warning(self.parent, "Warning", error)

            else:
                self.ui.lineEditRefFolder.setText(ref_folder)
                self.enableCheckbox()
                self.reference_lm = self.ActualMeth.ListLandmarksJson(self.ActualMeth.search(ref_folder,'json')['json'][0])

    def SearchModelSegOr(self):
        """Function to search the model folder of either the segmentation or the orientation model and to check if the model is valid"""

        name, url = self.ActualMeth.getSegOrModelList()

        model_folder = self.DownloadUnzip(
            url=url,
            directory=os.path.join(self.SlicerDownloadPath),
            folder_name=os.path.join("Models", name)
        )
        

        if not model_folder == "":
            error = self.ActualMeth.TestModel(model_folder, self.ui.lineEditModelSegOr.name)

            if isinstance(error, str):
                qt.QMessageBox.warning(self.parent, "Warning", error)

            else:
                self.ui.lineEditModelSegOr.setText(model_folder)
                self.enableCheckbox()

    def SearchModelALI(self,test=False):
        """Function to search the model folder of the ALI model and to check if the model is valid"""
        listeLandmark = []
        for key,data in self.ActualMeth.DicLandmark()["Landmark"].items():
            listeLandmark += data
        
        if test:
            ret = self.reference_lm

        else:
            
            s = PopUpWindow(title="Chose ALI Models to Download",listename=sorted(listeLandmark),type="checkbox",tocheck=self.reference_lm)
            s.exec_()
            ret = s.checked

        name, url = self.ActualMeth.getALIModelList()

        for i,model in enumerate(ret):
            _ = self.DownloadUnzip(
                url=os.path.join(url,"{}.zip".format(model)),
                directory=os.path.join(self.SlicerDownloadPath),
                folder_name=os.path.join("Models", name,model),
                num_downl=i+1,
                total_downloads=len(ret)
            )
        
        model_folder = os.path.join(self.SlicerDownloadPath,"Models", name)
        
        if not model_folder == "":
            error = self.ActualMeth.TestModel(model_folder, self.ui.lineEditModelAli.name)

            if isinstance(error, str):
                qt.QMessageBox.warning(self.parent, "Warning", error)

            else:
                self.ui.lineEditModelAli.setText(model_folder)
                self.enableCheckbox()

    def ChosePathOutput(self):
        out_folder = qt.QFileDialog.getExistingDirectory(
            self.parent, "Select a scan folder"
        )
        if not out_folder == "":
            self.ui.lineEditOutputPath.setText(out_folder)

    def SelectSuggestLandmark(self):
        best = self.ActualMeth.Suggest()
        for checkbox in self.logic.iterillimeted(self.dicchckbox):
            if checkbox.text in best and checkbox.isEnabled():
                checkbox.setCheckState(True)
            # else :
            #     checkbox.setCheckState(False)


    def enableCheckbox(self):
        """Function to enable the checkbox depending on the presence of landmarks"""
        status = self.ActualMeth.existsLandmark(
            self.ui.lineEditScanLmPath.text,
            self.ui.lineEditRefFolder.text,
            self.ui.lineEditModelAli.text,
        )

        if status is None:
            return

        if self.type == "IOS":
            for checkbox, checkbox2 in zip(
                self.logic.iterillimeted(self.dicchckbox),
                self.logic.iterillimeted(self.dicchckbox),
            ):
                try:
                    checkbox.setCheckable(status[checkbox.text])
                    checkbox2.setCheckable(status[checkbox2.text])

                except:
                    pass

        if self.type == "CBCT":
            for checkboxs, checkboxs2 in zip(self.dicchckbox.values(), self.dicchckbox2.values()):
                for checkbox, checkbox2 in zip(checkboxs, checkboxs2):
                    checkbox.setVisible(status[checkbox.text])
                    checkbox2.setVisible(status[checkbox2.text])
                    if status[checkbox.text]:
                        checkbox.setChecked(True)
                        checkbox2.setChecked(True)

    """
                                                                                    
                    8888888b.  8888888b.   .d88888b.   .d8888b.  8888888888  .d8888b.   .d8888b.  
                    888   Y88b 888   Y88b d88P" "Y88b d88P  Y88b 888        d88P  Y88b d88P  Y88b 
                    888    888 888    888 888     888 888    888 888        Y88b.      Y88b.      
                    888   d88P 888   d88P 888     888 888        8888888     "Y888b.    "Y888b.   
                    8888888P"  8888888P"  888     888 888        888            "Y88b.     "Y88b. 
                    888        888 T88b   888     888 888    888 888              "888       "888 
                    888        888  T88b  Y88b. .d88P Y88b  d88P 888        Y88b  d88P Y88b  d88P 
                    888        888   T88b  "Y88888P"   "Y8888P"  8888888888  "Y8888P"   "Y8888P"  
                                                                                
                                                                                    
    """

    def onPredictButton(self):
        """Function to launch the prediction"""
        error = self.ActualMeth.TestProcess(
            input_folder=self.ui.lineEditScanLmPath.text,
            gold_folder=self.ui.lineEditRefFolder.text,
            folder_output=self.ui.lineEditOutputPath.text,
            model_folder_ali=self.ui.lineEditModelAli.text,
            model_folder_segor=self.ui.lineEditModelSegOr.text,
            add_in_namefile=self.ui.lineEditAddName.text,
            dic_checkbox=self.dicchckbox,
            smallFOV=str(self.ui.checkBoxSmallFOV.isChecked()),
        )

        # print('error',error)
        if isinstance(error, str):
            qt.QMessageBox.warning(self.parent, "Warning", error.replace(",", "\n"))

        else:
            self.list_Processes_Parameters, self.display = self.ActualMeth.Process(
                input_folder=self.ui.lineEditScanLmPath.text,
                gold_folder=self.ui.lineEditRefFolder.text,
                folder_output=self.ui.lineEditOutputPath.text,
                model_folder_ali=self.ui.lineEditModelAli.text,
                model_folder_segor=self.ui.lineEditModelSegOr.text,
                add_in_namefile=self.ui.lineEditAddName.text,
                dic_checkbox=self.dicchckbox,
                logPath=self.log_path,
                smallFOV=str(self.ui.checkBoxSmallFOV.isChecked()),
            )

            self.nb_extension_launch = len(self.list_Processes_Parameters)
            self.onProcessStarted()

            # /!\ Launch of the first process /!\
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
        self.startTime = time.time()

        # self.ui.progressBar.setMaximum(self.nb_patient)
        self.ui.progressBar.setValue(0)

        self.ui.LabelProgressPatient.setText(f"Patient : 0 / {self.nb_patient}")
        self.ui.LabelProgressExtension.setText(
            f"Extension : 0 / {self.nb_extension_launch}"
        )
        self.nb_extnesion_did = 0

        self.module_name_before = 0
        self.nb_change_bystep = 0

        self.RunningUI(True)

    def onProcessUpdate(self, caller, event):
        timer = f"Time : {time.time()-self.startTime:.2f}s"
        self.ui.LabelTimer.setText(timer)
        progress = caller.GetProgress()
        self.module_name = caller.GetModuleTitle()
        self.ui.LabelNameExtension.setText(self.module_name)

        if self.module_name_before != self.module_name:
            self.ui.LabelProgressPatient.setText(f"Patient : 0 / {self.nb_patient}")
            self.nb_extnesion_did += 1
            self.ui.LabelProgressExtension.setText(
                f"Extension : {self.nb_extnesion_did} / {self.nb_extension_launch}"
            )
            self.ui.progressBar.setValue(0)

            # if self.nb_change_bystep == 0 and self.module_name_before:
            #     print(f'Error this module doesn\'t work {self.module_name_before}')

            self.module_name_before = self.module_name
            self.nb_change_bystep = 0

        if progress == 0:
            self.updateProgessBar = False

        if self.display[self.module_name].isProgress(
            progress=progress, updateProgessBar=self.updateProgessBar
        ):
            progress_bar, message = self.display[self.module_name]()
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

                print(self.process.GetOutputText())
                try:
                    self.process = slicer.cli.run(
                        self.list_Processes_Parameters[0]["Process"],
                        None,
                        self.list_Processes_Parameters[0]["Parameter"],
                    )
                    self.processObserver = self.process.AddObserver(
                        "ModifiedEvent", self.onProcessUpdate
                    )
                    del self.list_Processes_Parameters[0]
                except IndexError:
                    self.OnEndProcess()

    def OnEndProcess(self):
        """Function called when the process is finished."""
        self.ui.LabelProgressPatient.setText(f"Patient : 0 / {self.nb_patient}")
        self.nb_extnesion_did += 1
        self.ui.LabelProgressExtension.setText(
            f"Extension : {self.nb_extnesion_did} / {self.nb_extension_launch}"
        )
        self.ui.progressBar.setValue(0)

        # if self.nb_change_bystep == 0:
        #     print(f'Erreur this module didnt work {self.module_name_before}')

        self.module_name_before = self.module_name
        self.nb_change_bystep = 0

        print("PROCESS DONE.")
        self.RunningUI(False)

        stopTime = time.time()

        logging.info(f"Processing completed in {stopTime-self.startTime:.2f} seconds")

    def onCancel(self):
        self.process.Cancel()

        self.RunningUI(False)

    def RunningUI(self, run=False):
        self.ui.ButtonOriented.setVisible(not run)

        self.ui.progressBar.setVisible(run)
        self.ui.LabelTimer.setVisible(run)

        self.HideComputeItems(run)

    """
                                                                                                                        
            8888888888 888     888 888b    888  .d8888b.      8888888 888b    888 8888888 88888888888 
            888        888     888 8888b   888 d88P  Y88b       888   8888b   888   888       888     
            888        888     888 88888b  888 888    888       888   88888b  888   888       888     
            8888888    888     888 888Y88b 888 888              888   888Y88b 888   888       888     
            888        888     888 888 Y88b888 888              888   888 Y88b888   888       888     
            888        888     888 888  Y88888 888    888       888   888  Y88888   888       888     
            888        Y88b. .d88P 888   Y8888 Y88b  d88P       888   888   Y8888   888       888     
            888         "Y88888P"  888    Y888  "Y8888P"      8888888 888    Y888 8888888     888     
                                                                                                                                      
                                                                                                                                    
                                                                                                                        
                                                                                                                        
    """

    def initCheckbox(self, methode, layout, tohide: qt.QLabel):
        """Function to create the checkbox at the beginning of the program"""
        if not tohide is None:
            tohide.setHidden(True)
        dic = methode.DicLandmark()
        # status = methode.existsLandmark('','')
        dicchebox = {}
        dicchebox2 = {}
        for type, tab in dic.items():
            Tab = QTabWidget()
            layout.addWidget(Tab)
            listcheckboxlandmark = []
            listcheckboxlandmark2 = []

            all_checkboxtab = self.CreateMiniTab(Tab, "All", 0)
            for i, (name, listlandmark) in enumerate(tab.items()):
                widget = self.CreateMiniTab(Tab, name, i + 1)
                for landmark in listlandmark:
                    checkbox = QCheckBox()
                    checkbox2 = QCheckBox()
                    checkbox.setText(landmark)
                    checkbox2.setText(landmark)
                    # checkbox.setEnabled(status[landmark])
                    # checkbox2.setEnabled(status[landmark])
                    checkbox2.toggled.connect(checkbox.setChecked)
                    checkbox.toggled.connect(checkbox2.setChecked)
                    widget.addWidget(checkbox)
                    all_checkboxtab.addWidget(checkbox2)

                    listcheckboxlandmark.append(checkbox)
                    listcheckboxlandmark2.append(checkbox2)

            dicchebox[type] = listcheckboxlandmark
            dicchebox2[type] = listcheckboxlandmark2

        methode.setcheckbox(dicchebox)
        methode.setcheckbox2(dicchebox2)

        return dicchebox, dicchebox2

    def CreateMiniTab(self, tabWidget: QTabWidget, name: str, index: int):
        """Function to create a new tab in the tabWidget"""
        new_widget = QWidget()
        # new_widget.setMinimumHeight(3)
        new_widget.resize(tabWidget.size)

        layout = QGridLayout(new_widget)

        scr_box = QScrollArea(new_widget)
        # scr_box.setMinimumHeight(50)
        scr_box.resize(tabWidget.size)

        layout.addWidget(scr_box, 0, 0)

        new_widget2 = QWidget(scr_box)
        layout2 = QVBoxLayout(new_widget2)

        scr_box.setWidgetResizable(True)
        scr_box.setWidget(new_widget2)

        tabWidget.insertTab(index, new_widget, name)

        return layout2

    def HideComputeItems(self, run=False):
        self.ui.ButtonOriented.setVisible(not run)

        self.ui.ButtonCancel.setVisible(run)

        self.ui.LabelProgressPatient.setVisible(run)
        self.ui.LabelProgressExtension.setVisible(run)
        self.ui.LabelNameExtension.setVisible(run)
        self.ui.progressBar.setVisible(run)

        self.ui.LabelTimer.setVisible(run)

    def initCheckboxIOS(
        self,
        methode: Auto_IOS,
        layout: QGridLayout,
        tohide: QLabel,
        layout2: QVBoxLayout,
        occlusion : QCheckBox
    ):
        """Function to create the checkbox at the beginning of the program for IOS"""
        diccheckbox = {"Adult": {}, "Child": {}}
        tohide.setHidden(True)
        dic_teeth = {
            1: "A",
            2: "B",
            3: "C",
            4: "D",
            5: "E",
            6: "F",
            7: "G",
            8: "H",
            9: "I",
            10: "J",
            11: "T",
            12: "S",
            13: "R",
            14: "Q",
            15: "P",
            16: "O",
            17: "N",
            18: "M",
            19: "L",
            20: "K",
        }
        upper = []
        lower = []

        list = []
        for i in range(1, 11):
            label = QLabel()
            pixmap = QPixmap(self.resourcePath(f"Image/{i}_resize_child.png"))
            label.setPixmap(pixmap)
            widget = QWidget()
            check = QCheckBox()
            check.setText(dic_teeth[i])
            check.setEnabled(False)
            layout_check = QHBoxLayout(widget)
            layout_check.addWidget(check)

            layout.addWidget(widget, 1, i + 3)
            layout.addWidget(label, 0, i + 3)
            list.append(check)
        diccheckbox["Child"]["Upper"] = list
        upper += list

        dic = {
            1: "UR8",
            2: "UR7",
            3: "UR6",
            4: "UR5",
            5: "UR4",
            6: "UR3",
            7: "UR2",
            8: "UR1",
            9: "UL1",
            10: "UL2",
            11: "UL3",
            12: "UL4",
            13: "UL5",
            14: "UL6",
            15: "UL7",
            16: "UL8",
            17: "LL8",
            18: "LL7",
            19: "LL6",
            20: "LL5",
            21: "LL4",
            22: "LL3",
            23: "LL2",
            24: "LL1",
            25: "LR1",
            26: "LR2",
            27: "LR3",
            28: "LR4",
            29: "LR5",
            30: "LR6",
            31: "LR7",
            32: "LR8",
        }

        list = []
        for i in range(1, 17):
            label = QLabel()
            pixmap = QPixmap(self.resourcePath(f"Image/{i}_resize.png"))
            label.setPixmap(pixmap)
            widget = QWidget()
            check = QCheckBox()
            check.setText(dic[i])
            check.setEnabled(False)
            layout_check = QHBoxLayout(widget)
            layout_check.addWidget(check)

            layout.addWidget(widget, 3, i)
            layout.addWidget(label, 2, i)

            list.append(check)

        diccheckbox["Adult"]["Upper"] = list
        upper += list

        list = []
        for i in range(1, 17):
            label = QLabel()
            pixmap = QPixmap(self.resourcePath(f"Image/{i+16}_resize.png"))
            label.setPixmap(pixmap)
            widget = QWidget()
            check = QCheckBox()
            check.setText(dic[i + 16])
            check.setEnabled(False)
            layout_check = QHBoxLayout(widget)
            layout_check.addWidget(check)

            layout.addWidget(widget, 4, 17 - i)
            layout.addWidget(label, 5, 17 - i)

            list.append(check)

        diccheckbox["Adult"]["Lower"] = list
        lower += list

        list = []
        for i in range(1, 11):
            label = QLabel()
            pixmap = QPixmap(self.resourcePath(f"Image/{i+10}_resize_child.png"))
            label.setPixmap(pixmap)
            widget = QWidget()
            check = QCheckBox()
            check.setText(dic_teeth[i + 10])
            check.setEnabled(False)
            layout_check = QHBoxLayout(widget)
            layout_check.addWidget(check)

            layout.addWidget(widget, 6, i + 3)
            layout.addWidget(label, 7, i + 3)

            list.append(check)

        diccheckbox["Child"]["Lower"] = list
        lower += list

        upper_checbox = QCheckBox()
        upper_checbox.setText("Upper")
        upper_checbox.toggled.connect(
            partial(self.UpperLowerCheckbox, {"Upper": upper, "Lower": lower}, "Upper")
        )
        layout.addWidget(upper_checbox, 3, 0)
        lower_checkbox = QCheckBox()
        lower_checkbox.setText("Lower")
        lower_checkbox.toggled.connect(
            partial(self.UpperLowerCheckbox, {"Upper": upper, "Lower": lower}, "Lower")
        )
        layout.addWidget(lower_checkbox, 4, 0)

        upper_checbox.toggled.connect(partial(self.UpperLowerChooseOcclusion,lower_checkbox,occlusion))
        lower_checkbox.toggled.connect(partial(self.UpperLowerChooseOcclusion,upper_checbox,occlusion))




        if isinstance(methode,Semi_IOS):
            dic1, dic2 = self.initCheckbox(methode, layout2, None)

            methode.setcheckbox(
                {
                    "Teeth": diccheckbox,
                    "Landmark": dic1,
                    "Jaw": {"Upper": upper_checbox, "Lower": lower_checkbox},
                    "Occlusion" : occlusion
                }
            )
            methode.setcheckbox2(
                {
                    "Teeth": diccheckbox,
                    "Landmark": dic2,
                    "Jaw": {"Upper": upper_checbox, "Lower": lower_checkbox},
                    "Occlusion" : occlusion
                }
            )
        else : 

            methode.setcheckbox(
                {
                    "Teeth": diccheckbox,
                    "Jaw": {"Upper": upper_checbox, "Lower": lower_checkbox},
                    "Occlusion" : occlusion
                }
            )
            methode.setcheckbox2(
                {
                    "Teeth": diccheckbox,
                    "Jaw": {"Upper": upper_checbox, "Lower": lower_checkbox},
                    "Occlusion" : occlusion
                }
            )
    def UpperLowerCheckbox(self, all_checkbox: dict, jaw, boolean):
       
        for checkbox in all_checkbox[jaw]:
            checkbox.setEnabled(boolean)
            if (not boolean) and checkbox.isChecked():
                checkbox.setChecked(False)
        self.enableCheckbox()

    def OcclusionCheckbox(self, Upper : QCheckBox, Lower : QCheckBox, all_checkbox : dict , boolean : bool):
       if boolean :
           if Upper.isChecked() and Lower.isChecked():
               Lower.setChecked(False)
               Lower.setEnabled(True)


    def UpperLowerChooseOcclusion(self,opposit_jaw : QCheckBox, Occlusion_checkbox : QCheckBox ,booleean : bool):
        if booleean and Occlusion_checkbox.isChecked() and opposit_jaw.isChecked() :
            opposit_jaw.setChecked(False)


    """
                          .d88888b.  88888888888 888    888 8888888888 8888888b.   .d8888b.  
                         d88P" "Y88b     888     888    888 888        888   Y88b d88P  Y88b 
                         888     888     888     888    888 888        888    888 Y88b.      
                         888     888     888     8888888888 8888888    888   d88P  "Y888b.   
                         888     888     888     888    888 888        8888888P"      "Y88b. 
                         888     888     888     888    888 888        888 T88b         "888 
                         Y88b. .d88P     888     888    888 888        888  T88b  Y88b  d88P 
                          "Y88888P"      888     888    888 8888888888 888   T88b  "Y8888P"    
    """

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
            self.updateGUIFromParameterNode,
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
                self.updateGUIFromParameterNode,
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
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

        wasModified = (
            self._parameterNode.StartModify()
        )  # Modify all properties in a single batch

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


"""
                d8888  .d8888b.   .d88888b.      888       .d88888b.   .d8888b.  8888888  .d8888b.  
               d88888 d88P  Y88b d88P" "Y88b     888      d88P" "Y88b d88P  Y88b   888   d88P  Y88b 
              d88P888 Y88b.      888     888     888      888     888 888    888   888   888    888 
             d88P 888  "Y888b.   888     888     888      888     888 888          888   888        
            d88P  888     "Y88b. 888     888     888      888     888 888  88888   888   888        
           d88P   888       "888 888     888     888      888     888 888    888   888   888    888 
          d8888888888 Y88b  d88P Y88b. .d88P     888      Y88b. .d88P Y88b  d88P   888   Y88b  d88P 
         d88P     888  "Y8888P"   "Y88888P"      88888888  "Y88888P"   "Y8888P88 8888888  "Y8888P"                                                                                  
"""


class ASOLogic(ScriptedLoadableModuleLogic):
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

    def process(self, parameters):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded

        """

        # import time
        # startTime = time.time()

        logging.info("Processing started")

        PredictProcess = slicer.modules.aso_ios

        self.cliNode = slicer.cli.run(PredictProcess, None, parameters)

        return PredictProcess

    def iterillimeted(self, iter):
        out = []
        if isinstance(iter, dict):
            iter = list(iter.values())

        for thing in iter:
            if isinstance(thing, (dict, list, set)):
                out += self.iterillimeted(thing)
            else:
                out.append(thing)

        return out
