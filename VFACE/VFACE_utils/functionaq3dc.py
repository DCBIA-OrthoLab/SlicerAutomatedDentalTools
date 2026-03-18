import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import logging
import vtk, qt, ctk, slicer
import glob
import numpy as np

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import json
from functools import partial
from qt import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QCheckBox,
    QTableWidgetItem,
    QTabWidget,
    QGridLayout,
)


from typing import Union

#
# AQ3DC
#
try:
    import pandas as pd

except:
    slicer.util.pip_install("pandas")
    import pandas as pd  # news users will not need to refresh the AQ3DC for the first 


try:
    # we need this package for pandas package
    import openpyxl
except:
    slicer.util.pip_install("openpyxl")
    import openpyxl

import importlib.util
import sys

modules_to_remove = [name for name in sys.modules.keys() if name.startswith('Classes.')]
for module_name in modules_to_remove:
    del sys.modules[module_name]

local_classes_path = os.path.join(current_dir, "Classes")
sys.path.insert(0, local_classes_path)

try:
    from Classes import (
        Angle,
        Distance,
        Diff2Measure,
        Measure,
        Point,
        Line,
        Group_landmark,
        MyList,
        MyDict,
    )
    
except ImportError as e:
    spec = importlib.util.spec_from_file_location("Classes.Measure", os.path.join(local_classes_path, "Measure.py"))
    measure_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measure_module)
    sys.modules["Classes.Measure"] = measure_module
    Measure = measure_module.Measure
"""
TODO:
    -change __setitem__ of lines, point and measure (very bad names, very confusing)
    -remove str in getitem component of measure, because in excel file is impossible to use the number to make calcul
    - make protection to have the same patient in T1 and T2

"""

class AQ3DC(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = (
            "AQ3DC"  # TODO: make this more human readable by adding spaces
        )
        self.parent.categories = [
            "Quantification"
        ]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = (
            []
        )  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "Baptiste Baquero (University of Michigan)"
        ]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
  This is an example of scripted loadable module bundled in an extension.
  See more information in <a href="https://github.com/organization/projectname#AQ3DC">module documentation</a>.
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

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # AQ3DC1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="AQ3DC",
        sampleName="AQ3DC1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "AQ3DC1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="AQ3DC1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="AQ3DC1",
    )

    # AQ3DC2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="AQ3DC",
        sampleName="AQ3DC2",
        thumbnailFileName=os.path.join(iconsPath, "AQ3DC2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="AQ3DC2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="AQ3DC2",
    )


#
# AQ3DCWidget
#


class AQ3DCWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent=None):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
    Called when the user opens the module the first time and the widget is initialized.
    """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AQ3DC.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AQ3DCLogic()


        # Make sure parameter node is initialized (needed for module reload)

        # need initialise the tab measurement
        # if we dont do this, if user add a measure without click on tabmeasurement or chekcbox. we get an issue
        self.updateComboboxListMeasurement()

        # ====================================================================================================================================
        # Variable
        # ====================================================================================================================================

        # list all ComboBox have to contain the landmark selected by user
        self.list_CbLandmark = [
            self.ui.CbMidpointP1,
            self.ui.CbMidpointP2,
            self.ui.CbAB2LT1P1,
            self.ui.CbAB2LT1P2,
            self.ui.CbAB2LT1P3,
            self.ui.CbAB2LT1P4,
            self.ui.CbAB2LT1T2P1T1,
            self.ui.CbAB2LT1T2P1T2,
            self.ui.CbAB2LT1T2P2T1,
            self.ui.CbAB2LT1T2P2T2,
            self.ui.CbAB2LT1T2P3T1,
            self.ui.CbAB2LT1T2P4T1,
            self.ui.CbAB2LT1T2P3T2,
            self.ui.CbAB2LT1T2P4T2,
            self.ui.CbALT1T2L1P1,
            self.ui.CbALT1T2L1P2,
            self.ui.CbALT1T2L2P3,
            self.ui.CbALT1T2L2P4,
            self.ui.CbD2PP1,
            self.ui.CbD2PP2,
            self.ui.CbDPLT1T2L1T1,
            self.ui.CbDPLT1T2L2T1,
            self.ui.CbDPLT1T2L2T2,
            self.ui.CbDPLT1T2P1T1,
            self.ui.CbDPLT1T2P1T2,
            self.ui.CbDPLT1T2L1T2,
            self.ui.CbDPLT1L1,
            self.ui.CbDPLT1L2,
            self.ui.CbDPLT1P1,
        ]

        self.list_LandMarkCheck = []

        self.dict_patient_T1 = {}
        self.dict_patient_T2 = {}
        self.dict_patient_extraction = {}
        """
    exemple of dict_patient T1 and T2


    dict = {"001":{"A":[0,0,2],"B":[0,2,3],..},
        ...,
        '29':{"A":[0,3,5],"B":[3,6,2],...}
          }
    The number is patient and the letter are landmark

    The patient and landmark have to be string
    The postion of landmark have to be a list
    """

        self.list_measure = []  # list of measurement
        self.list_landmark = []  # need comment
        self.exeption_display_group_landmark = (
            {}
        )  # link between the name of group and index of stackedwidget, add comment how is organize
        self.dict_checkbox = {}  # dict = {'A':checkbox , 'B':checkbox,...}

        # landmark available on tab landmark
        # example of the group are given in Group_landmark class
        self.GROUPS_LANDMARKS : Group_landmark

        #midpoint create by the user
        self.mid_point = []

        # init self.group landmark
        self.selectFileImportListLandmark(path_listlandmarks=self.resourcePath("name_landmark.xlsx"))


        # ==============================================================================================================================================================
        # Widget conect
        # ==============================================================================================================================================================

        # self.ui.TabLandmarks.clear()
        self.ui.TabMeasure.currentChanged.connect(self.updateComboboxListMeasurement)
        self.ui.CbListMeasurement.activated.connect(self.manageDisplayComboboxMeasurement)
        self.ui.CheckBoxT1T2.toggled.connect(self.updateComboboxListMeasurement)
        self.ui.ButtonPathT1.clicked.connect(self.selectFolderT1Patient)
        self.ui.ButtonPathT2.clicked.connect(self.selectFolderT2Patients)
        self.ui.ButtonImportLandmarks.clicked.connect(partial(self.selectFileImportListLandmark,None))
        self.ui.ButtonSelectTabLandmarks.clicked.connect(self.checkAllLandmarks)
        self.ui.ButtonClearTabLandmarks.clicked.connect(self.decheckAllLandmark)
        self.ui.ButtonAddMidpoint.clicked.connect(self.addMidpoint)
        self.ui.ButtonFolderMidpoint.clicked.connect(self.selectFolderSaveMidpoint)
        self.ui.ButtonSaveMidpoint.clicked.connect(self.saveMidpoint)
        self.ui.combineWithOriginal.clicked.connect(self.updateSaveMidpoint)
        self.ui.ButtonAddMeasure.clicked.connect(self.createMeasurement)
        self.ui.ButtonDeleteMeasurement.clicked.connect(self.deleteMeasurement)
        self.ui.CbImportExportMeasure.activated.connect(
            self.manageDisplayImportExportMeasurementPage
        )
        self.ui.ButtonFolderExportMeasure.clicked.connect(self.selectFolderExportMeasurement)
        self.ui.ButtonFileImportMeasure.clicked.connect(self.selectFolderImportMeasurement)
        self.ui.ButtonExportMeasure.clicked.connect(self.exportMeasurement)
        self.ui.ButtonImportMeasure.clicked.connect(self.importMeasurement)
        self.ui.ButtonCompute.clicked.connect(self.saveComputationMeasuement)
        self.ui.ButtonFolderCompute.clicked.connect(self.selectFolderComputeMeasure)
        self.ui.TabLandmarks.currentChanged.connect(self.manageStackedLandmark)

        self.ui.TableAngle.horizontalHeader().sectionDoubleClicked.connect(
            partial(self.selectAllMeasurement, "Angle")
        )
        self.ui.TableDistance.horizontalHeader().sectionDoubleClicked.connect(
            partial(self.selectAllMeasurement, "Distance")
        )

        self.ui.LabelExcelFormat.setVisible(True)
        self.ui.ComboBoxExcelFormat.setVisible(True)
        self.ui.ButtonFolderMidpoint.setEnabled(False)
        self.ui.LineEditPathMidpoint.setEnabled(False)

#==========================================================================================================================================================
# Computation
#==========================================================================================================================================================


    def selectFolderComputeMeasure(self):
        """Ask user, which folder he want to compute his result
    Display window to selecte the folder
    Display folder's path in LineEditFolderComputation

    Call by ButtonFolderCompute
    """
        computation_folder = qt.QFileDialog.getExistingDirectory(
            self.parent, "Select a scan folder"
        )
        if computation_folder != "":
            self.ui.LineEditFolderComputation.setText(computation_folder)

    def saveComputationMeasuement(self):
        """
    Compute measurement 

    Call by ButtonCompute
    """
        path = self.ui.LineEditFolderComputation.text
        file_name = self.ui.LineEditComputationFile.text
        if path != "" and (file_name != ".xlsx" or ""):
            # concatenate patient T1 and T2
            dict_patient = self.logic.concatenateT1T2Patient(
                self.dict_patient_T1, self.dict_patient_T2
            )

            # compute all measure
            patient_compute = self.logic.computeMeasurement(self.list_measure, dict_patient)
            print("*"*150)
            print("self.list_measure : ",self.list_measure)
            # patient_compute = self.allowSign(patient_compute,self.list_measure)

            print("self.ui.ComboBoxExcelFormat.currentText : ",self.ui.ComboBoxExcelFormat.currentText)
            if self.ui.ComboBoxExcelFormat.currentText == "Statistics":
                patient_compute = self.reorganizeStat(patient_compute)
            else :
                patient_compute = self.renameTimepoint(patient_compute)
            # write measure
            #test windows
            self.logic.writeMeasurementExcel(patient_compute, path, file_name)
            # self.logic.WriteMeasurementExcel2(test, path, file_name)

    def allowSign(self,patient_compute,list_measure):
        for i in range(len(patient_compute["Type of measurement"])) :
            T1 = False
            T2 = False
            if "T1" in patient_compute["Type of measurement"][i]:
                T1=True
            if "T2" in patient_compute["Type of measurement"][i]:
                T2=True

            if T1 and T2 :
                continue
            else :

                patient_compute["R-L Meaning"][i] = "x"
                patient_compute["A-P Meaning"][i] = "x"
                patient_compute["S-I Meaning"][i] = "x"

                patient_compute["Yaw Meaning"][i] = "x"
                patient_compute["Pitch Meaning"][i] = "x"
                patient_compute["Roll Meaning"][i] = "x"
        return patient_compute

    def checkKeepSign(self,list_measure,landmarks,measurement):
        for measure in list_measure :
            if measurement in measure["Type of measurement"]:
                if measure["group"]=="Angle":
                    land = measure["line1"] + " " + measure["line2"]



    def renameTimepoint(self,patient_compute):
        for i in range(len(patient_compute["Type of measurement"])) :
            measure = patient_compute["Type of measurement"][i]
            if measure == "Distance point line T1":
                patient_compute["Type of measurement"][i] = f"Distance point line {self.ui.lineEditNameT1.text}"

            elif measure == "Distance point line T2":
                patient_compute["Type of measurement"][i] = f"Distance point line {self.ui.lineEditNameT2.text}"

            elif measure == "Distance point line T1 T2" :
                patient_compute["Type of measurement"][i] = f"Distance point line {self.ui.lineEditNameT1.text} {self.ui.lineEditNameT2.text}"

            elif measure == "Distance between 2 points T1" :
                patient_compute["Type of measurement"][i] = f"Distance between 2 points {self.ui.lineEditNameT1.text}"

            elif measure == "Distance between 2 points T2" :
                patient_compute["Type of measurement"][i] = f"Distance between 2 points {self.ui.lineEditNameT2.text}"

            elif measure == "Distance between 2 points T1 T2":
                patient_compute["Type of measurement"][i] = f"Distance between 2 points {self.ui.lineEditNameT1.text} {self.ui.lineEditNameT2.text}"

            elif measure == "Angle between 2 lines T1":
                patient_compute["Type of measurement"][i] = f"Angle between 2 lines {self.ui.lineEditNameT1.text}"

            elif measure == "Angle between 2 lines T2":
                patient_compute["Type of measurement"][i] = f"Angle between 2 lines {self.ui.lineEditNameT2.text}"

            elif measure == "Angle between 2 lines T1 T2":
                patient_compute["Type of measurement"][i] = f"Angle between 2 lines {self.ui.lineEditNameT1.text} {self.ui.lineEditNameT2.text}"

            elif measure == "Angle line T1 and line T2":
                patient_compute["Type of measurement"][i] = f"Angle line {self.ui.lineEditNameT1.text} and line {self.ui.lineEditNameT2.text}"

        return patient_compute










    def reorganizeStat(self,patient_compute):
        dic_stats = {
                "ID":[],
                "Landmarks":[],
                "Time":[],
                "Arch":[],
                "Segment":[],
                "Transverse":[],
                "AP":[],
                "Vertical":[],
                "3D":[],
                "Yaw":[],
                "Pitch":[],
                "Roll":[],
                "BL":[],
                "MD":[],
                "Rotation":[]
            }

        TOOTHS = ["UR8", "UR7", "UR6", "UR5", "UR4", "UR3","UR1", "UR2","UL8", "UL7", "UL6", "UL5", "UL4", "UL3","UL1", "UL2",
                  "LR8", "LR7", "LR6", "LR5", "LR4", "LR3","LR1", "LR2","LL8", "LL7", "LL6", "LL5", "LL4", "LL3","LL1", "LL2"]

        for i in range(len(patient_compute["Patient"])) :


            if patient_compute["Patient"][i][1].lower()=="p" :
                dic_stats["ID"].append(patient_compute["Patient"][i][1:])
            elif patient_compute["Patient"][i][:3].lower() == "pat" :
                dic_stats["ID"].append(patient_compute["Patient"][i][3:])
            else :
                dic_stats["ID"].append(patient_compute["Patient"][i])

            dic_stats["Landmarks"].append(patient_compute["Landmarks"][i])

            T1=False
            T2=False
            if "T1" in patient_compute["Type of measurement"][i]:
                T1=True
            if "T2" in patient_compute["Type of measurement"][i]:
                T2=True

            if T1 and T2 :
                dic_stats["Time"].append(f"{self.ui.lineEditNameT1.text}-{self.ui.lineEditNameT2.text}")
            elif T1 :
                dic_stats["Time"].append(str(self.ui.lineEditNameT1.text))
            else :
                dic_stats["Time"].append(str(self.ui.lineEditNameT2.text))


            type = "skeletal"
            tooth = None
            for t in TOOTHS :
                if t in patient_compute["Landmarks"][i] :
                    tooth = t

            if tooth != None:
                #Arch : upper=0, lower=1
                if "U" in tooth :
                    dic_stats["Arch"].append(0)
                else :
                    dic_stats["Arch"].append(1)

                #Segment : posterior=0,anterior=1
                if "1" in tooth or "2" in tooth:
                    dic_stats["Segment"].append(1)
                else :
                    dic_stats["Segment"].append(0)


                if dic_stats["Segment"][len(dic_stats["Segment"])-1] == 1 : # is anterior teeth

                    #AP
                    ap = patient_compute["A-P Component"][i]
                    if ap!="x" and ap!="":
                        ap=float(ap)
                        if patient_compute["A-P Meaning"][i]=="L":
                            ap=-ap
                    dic_stats["AP"].append(str(ap))

                    #Transverse-RL
                    rl = patient_compute["R-L Component"][i]
                    if rl!="x" and rl!="":
                        rl=float(rl)
                        if patient_compute["R-L Meaning"][i]=="D":
                            rl=-rl
                    dic_stats["Transverse"].append(str(rl))


                    #BL
                    pitch = patient_compute["Pitch Component"][i]
                    if pitch!="x" and pitch!="":
                        pitch=float(pitch)
                        if patient_compute["Pitch Meaning"][i]=="L":
                            pitch=-pitch
                    dic_stats["BL"].append(str(pitch))

                    #MD
                    roll = patient_compute["Roll Component"][i]
                    if roll!="x" and roll!="":
                        roll=float(roll)
                        if patient_compute["Roll Meaning"][i]=="D":
                            roll=-roll
                    dic_stats["MD"].append(str(roll))


                else :
                    #AP
                    ap = patient_compute["A-P Component"][i]
                    if ap!="x" and ap!="":
                        ap=float(ap)
                        if patient_compute["A-P Meaning"][i]=="D":
                            ap=-ap
                    dic_stats["AP"].append(str(ap))

                    #Transverse-RL
                    rl = patient_compute["R-L Component"][i]
                    if rl!="x" and rl!="":
                        rl=float(rl)
                        if patient_compute["R-L Meaning"][i]=="B":
                            rl=-rl
                    dic_stats["Transverse"].append(str(rl))

                    #MD
                    pitch = patient_compute["Pitch Component"][i]
                    if pitch!="x" and pitch!="":
                        pitch=float(pitch)
                        if patient_compute["Pitch Meaning"][i]=="D":
                            pitch=-pitch
                    dic_stats["MD"].append(str(pitch))

                    #BL
                    roll = patient_compute["Roll Component"][i]
                    if roll!="x" and roll!="":
                        roll=float(roll)
                        if patient_compute["Roll Meaning"][i]=="L":
                            roll=-roll
                    dic_stats["BL"].append(str(roll))

                #Vertical
                si = patient_compute["S-I Component"][i]
                if si!="x" and si!="":
                    si=float(si)
                    if patient_compute["S-I Meaning"][i]=="I":
                        si=-si
                dic_stats["Vertical"].append(str(si))

                #Rotation
                yaw = patient_compute["Yaw Component"][i]
                if yaw!="x" and yaw!="":
                    yaw=float(yaw)
                    if patient_compute["Yaw Meaning"][i]=="DR":
                        yaw=-yaw
                dic_stats["Rotation"].append(str(yaw))

                #3D
                ThreeD = patient_compute["3D Distance"][i]
                dic_stats["3D"].append(str(ThreeD))

                dic_stats["Yaw"].append(str("x"))
                dic_stats["Pitch"].append(str("x"))
                dic_stats["Roll"].append(str("x"))

            else :
                dic_stats["Arch"].append("x")
                dic_stats["Segment"].append("x")

                dic_stats["BL"].append(str("x"))
                dic_stats["MD"].append(str("x"))
                dic_stats["Rotation"].append(str("x"))

                #Transverse-RL
                rl = patient_compute["R-L Component"][i]
                if rl!="x" and rl!="":
                    rl=float(rl)
                    if patient_compute["R-L Meaning"][i]=="Medial" or patient_compute["R-L Meaning"][i]=="L":
                        rl=-rl
                dic_stats["Transverse"].append(str(rl))


                #AP
                ap = patient_compute["A-P Component"][i]
                if ap!="x" and ap!="":
                    ap=float(ap)
                    if patient_compute["A-P Meaning"][i]=="P":
                        ap=-ap
                dic_stats["AP"].append(str(ap))

                #Vertical
                si = patient_compute["S-I Component"][i]
                if si!="x" and si!="":
                    si=float(si)
                    if patient_compute["S-I Meaning"][i]=="S":
                        si=-si
                dic_stats["Vertical"].append(str(si))

                #Yaw
                yaw = patient_compute["Yaw Component"][i]
                if yaw!="x" and yaw!="":
                    yaw=float(yaw)
                    if patient_compute["Yaw Meaning"][i]=="CounterC":
                        yaw=-yaw
                dic_stats["Yaw"].append(str(yaw))

                #Pitch
                pitch = patient_compute["Pitch Component"][i]
                if pitch!="x" and pitch!="":
                    pitch=float(pitch)
                    if patient_compute["Pitch Meaning"][i]=="CounterC":
                        pitch=-pitch
                dic_stats["Pitch"].append(str(pitch))

                #BL
                roll = patient_compute["Roll Component"][i]
                if roll!="x" and roll!="":
                    roll=float(roll)
                    if patient_compute["Roll Meaning"][i]=="CounterC":
                        roll=-roll
                dic_stats["Roll"].append(str(roll))

                #3D
                ThreeD = patient_compute["3D Distance"][i]
                dic_stats["3D"].append(str(ThreeD))


        keys_to_delete = []
        for key, value in dic_stats.items():
            if not value:  # Check if the list is empty
                keys_to_delete.append(key)
            elif all(item == "x" for item in value):  # Check if all items in the list are "x"
                keys_to_delete.append(key)

        # Deleting the keys where the condition is not met
        for key in keys_to_delete:
            del dic_stats[key]


        return dic_stats







#====================================================================================================================================================================
# Export/Import List Measurement
#====================================================================================================================================================================

    def manageDisplayImportExportMeasurementPage(self):
        """
    Manage Interface if user want export list of measurement or export

    Call by CbImportExportMeasure
    """
        indexes = {
            "None": 0,
            "Import list of measurements": 1,
            "Export list of measurements": 2,
        }
        choice = self.ui.CbImportExportMeasure.currentText
        self.ui.StackedImportExport.setCurrentIndex(indexes[choice])

    def selectFolderExportMeasurement(self):
        """Ask to user, which foler he want to put his measurement file,
        Display window to select folder
        Display folder's path in LineEditFolderExportMeasure

    Call by ButtonFolderExportMeasure
    """
        measure_folder = qt.QFileDialog.getExistingDirectory(
            self.parent, "Select a scan folder"
        )
        if measure_folder != "":
            self.ui.LineEditFolderExportMeasure.setText(measure_folder)

    def exportMeasurement(self):
        """Export Measure

      call by ButtonExportMeasure
      """
        name_file = self.ui.LineEditFileExportMeasure.text
        measure_folder = self.ui.LineEditFolderExportMeasure.text
        if name_file != "" and measure_folder != "":

            self.logic.exportMeasurement(
                measure_folder + "/" + name_file, self.list_measure
            )

    def selectFolderImportMeasurement(self):
        """Ask to user, Which folder he want to put his measurement file,
        Display window to select folder
        Display path folder in LineEditImportMeasure

      call by ButtonFileImportMeasure
    """
        file_measure = qt.QFileDialog.getOpenFileName(self.parent, "Select file")
        if file_measure != "":
            self.ui.LineEditImportMeasure.setText(file_measure)

    def importMeasurement(self):
        """Import Measure

    ButtonImportMeasure
    """
        list_measure = self.logic.importMeasurement(self.ui.LineEditImportMeasure.text)
        for measure in list_measure:
            self.addMeasurementToTabMeasurement(measure)



#=================================================================================================================================================
# CheckBox Landmark
#=================================================================================================================================================

    def manageTabLandmarks(self, list_landmarks: list, dict_landmarks: Group_landmark):
        """_summary_
    Manage Creation of Tab landmark

    Args:
        list_landmarks (list): for checkbox status
        dict_landmarks (dict): group landmark
    """
        self.ui.TabLandmarks.clear()
        self.dict_checkbox = {}
        self.dict_Checkbox2Landmark = {}
        self.dict_Landmark2Checkbox = {}
        self.dict_Group2Layout = {}

        # create dic to know each landmark is available
        status = dict_landmarks.existsInDict(list_landmarks)

        index = 0
        for group, landmarks in dict_landmarks.items():

            # create tab
            self.addTabLandmarks(self.ui.TabLandmarks, group, index)

            if not isinstance(landmarks, MyDict):
                # if landmarks need only one tab to be display
                for landmark in landmarks:
                    self.addLandmarkToTabLandmarks(group, landmark, status[landmark])

            else:
                # to display 2 tab landmark
                prefix, suffix = landmarks.getSeparatePreSuf()
                for suf in suffix:
                    self.addLandmarkToTabLandmarks(group, suf, status[suf])

                self.addQTabWidgetToTabLandmark(index, prefix, group)
                for key, values in prefix.items():
                    for value in values:
                        self.addLandmarkToTabLandmarks(key + group, value, status[value])

            index += 1

        self.list_LandMarkCheck = []
        self.updateAllLandmarks()

    def addQTabWidgetToTabLandmark(self, i, dict, parent):
        """add QTabWidget if the group need 2 TabWidget

    Args:
        i (int): index
        dict (dict): _description_
        parent (str): group link
    """

        new_tabwidget = QTabWidget()
        new_tabwidget.setHidden(True)
        self.ui.LayoutLandmarks.addWidget(new_tabwidget)
        self.exeption_display_group_landmark[i] = new_tabwidget

        for group in dict.keys():
            self.addTabLandmarks(new_tabwidget, group, i, parent=parent)

    def addTabLandmarks(
        self, tabWidget: QTabWidget, group: str, index: int, parent: str = ""
    ):
        """Add a new Tab in tabWidget

    Args:
        tabWidget (QTabWidget): _description_
        group (str): tab's name
        index (int): index tab
        parent (str, optional):to make a link with another tabwidget Defaults to ''.
    """

        new_widget = QWidget()
        new_widget.setMinimumHeight(250)

        layout = QGridLayout(new_widget)

        scr_box = QScrollArea(new_widget)
        scr_box.setMinimumHeight(200)

        layout.addWidget(scr_box, 0, 0)

        new_widget2 = QWidget(scr_box)
        layout2 = QVBoxLayout(new_widget2)

        scr_box.setWidgetResizable(True)
        scr_box.setWidget(new_widget2)

        tabWidget.insertTab(index, new_widget, group)

        self.dict_Group2Layout[group + parent] = [layout2, scr_box]

    def addLandmarkToTabLandmarks(self, group: str, landmark: str, status: bool):
        """Add landmark in tab

      Args:
          group (str): to know in which tab landmark to add
          landmark (str): landmark to add
          status (bool): to know if the checkbox of landmark is enable
      """
        check = QCheckBox(landmark)
        check.setEnabled(status)
        self.dict_Checkbox2Landmark[check] = [landmark, status, group]
        self.dict_Landmark2Checkbox[landmark] = [check, status]
        self.dict_checkbox[landmark] = check
        check.connect("toggled(bool)", partial(self.toggleCheckboxLandmark,landmark))
        self.dict_Group2Layout[group][0].addWidget(check)

    def manageStackedLandmark(self):
        """ manage the 2 tab widget
    """
        for tablandmark in self.exeption_display_group_landmark.values():
            tablandmark.setHidden(True)
        if self.ui.TabLandmarks.currentIndex in self.exeption_display_group_landmark:
            self.exeption_display_group_landmark[
                self.ui.TabLandmarks.currentIndex
            ].setHidden(False)

    def toggleCheckboxLandmark(self,landmark,enabled):
        """function connect to checkbox in tablelandmark

      update list landmark checked
    """
        if enabled :
            self.list_LandMarkCheck.append(landmark)
        else :
            self.list_LandMarkCheck.remove(landmark)
        self.updateAllLandmarks()

    def updateAllLandmarks(self):
        """Update Combobox containing landmarks
    """
        enable_landmark = self.logic.getEnableLandmarks(self.list_LandMarkCheck, self.GROUPS_LANDMARKS)
        for Cb in self.list_CbLandmark:
            Cb.clear()
            Cb.addItems(enable_landmark)



#=========================================================================================================================================
# Landmark
#=========================================================================================================================================

    def selectFileImportListLandmark(self, path_listlandmarks=None,nothing=None):
        """
    Ask user, which excel file is use for group landmark
    add landmark in the tablandmarks and update combo box with all landmark

    Call by ButtonImportLandmarks

    """
        if path_listlandmarks is None:
            path_listlandmarks = qt.QFileDialog.getOpenFileName(
                self.parent, "Select file"
            )

        if path_listlandmarks != "":
            self.ui.LineEditImportLandmarks.setText(path_listlandmarks)

            self.GROUPS_LANDMARKS = Group_landmark(path_listlandmarks)
            list_landmarks, self.GROUPS_LANDMARKS = self.logic.updateGroupLandmark(
                self.dict_patient_T1, self.GROUPS_LANDMARKS
            )
            self.manageTabLandmarks(list_landmarks, self.GROUPS_LANDMARKS)

            self.updateComboboxListMeasurement()

    def checkAllLandmarks(self):
        self.setCheckStateCurrentTabLandmakrs(True)

    def decheckAllLandmark(self):
        self.setCheckStateCurrentTabLandmakrs(False)

    def setCheckStateCurrentTabLandmakrs(self, status: bool):
        """Check or decheck all checkbox enable in tablandmark open in utilisator interface

    Args:
        status (bool): True -> check all checkbox in tablandmark open
                        False -> decheck all checkbox in tablandmark open
    """
        index = self.ui.TabLandmarks.currentIndex
        group = self.ui.TabLandmarks.tabText(index)

        for landmark in self.GROUPS_LANDMARKS[group]:
            if self.dict_Landmark2Checkbox[landmark][1]:
                self.dict_Landmark2Checkbox[landmark][0].setChecked(status)

        self.updateAllLandmarks()


#==========================================================================================================================================================================
# MidPoint
#==========================================================================================================================================================================

    def addMidpoint(self):
        """
    Add midpoint in tablandmark
    """
        P1 = self.ui.CbMidpointP1.currentText
        P2 = self.ui.CbMidpointP2.currentText
        mid_point = "Mid_" + P1 + "_" + P2
        self.GROUPS_LANDMARKS["Midpoint"] = mid_point

        self.addLandmarkToTabLandmarks("Midpoint", mid_point, True)
        self.GROUPS_LANDMARKS["Midpoint"] = mid_point
        self.mid_point.append((P1,P2))
        self.dict_patient_T1 = self.logic.addMidpointToPatient(
            self.dict_patient_T1, P1, P2
        )
        if len(self.dict_patient_T2) > 0:
            self.dict_patient_T2 = self.logic.addMidpointToPatient(
                self.dict_patient_T2, P1, P2
            )

    def selectFolderSaveMidpoint(self):
        """Ask user, which folder he want to save midpoint
      Display the folder chose

      Call by ButtonFolderMidpoint
    """
        folder_midpoint = qt.QFileDialog.getExistingDirectory(
            self.parent, "Select a scan folder"
        )
        if folder_midpoint != "":
            self.ui.LineEditPathMidpoint.setText(folder_midpoint)

    def saveMidpoint(self):
        """Save Midpoint in folder T1 and T2
    """
        if self.ui.combineWithOriginal.isChecked():
            # Save directly in original folders
            for patient, landmarks in self.dict_patient_T1.items():
                originalFile = self.logic.findOriginalJson(self.ui.LineEditPathT1.text, patient)
                if originalFile:
                    self.logic.appendMidpointsToJson(originalFile, landmarks, self.mid_point)

            if self.ui.LineEditPathT2.text != "":
                for patient, landmarks in self.dict_patient_T2.items():
                    originalFile = self.logic.findOriginalJson(self.ui.LineEditPathT2.text, patient)
                    if originalFile:
                        self.logic.appendMidpointsToJson(originalFile, landmarks, self.mid_point)
        else:
            out_path_T1 = os.path.join(self.ui.LineEditPathMidpoint.text, "T1")
            out_path_T2 = os.path.join(self.ui.LineEditPathMidpoint.text, "T2")
            if not os.path.exists(out_path_T1):
                os.makedirs(out_path_T1)
            self.logic.saveMidpoint(
                self.dict_patient_T1,
                out_path_T1,
                self.mid_point,
            )
            if self.ui.LineEditPathT2.text != "":
                if not os.path.exists(out_path_T2):
                    os.makedirs(out_path_T2)
                self.logic.saveMidpoint(
                    self.dict_patient_T2,
                    out_path_T2,
                    self.mid_point,
                )
    
    def updateSaveMidpoint(self):
        """Update the save midpoint button
    """
        show = False if self.ui.combineWithOriginal.isChecked() else True
            
        self.ui.ButtonFolderMidpoint.setEnabled(show)
        self.ui.LineEditPathMidpoint.setEnabled(show)

#===============================================================================================================
# Measurement
#===============================================================================================================

    def manageDisplayComboboxMeasurement(self):
        """_summary_
      Manage StackedMeasure to display the good page in function TabMeasure, CheckboxT1T2 and ComboBox

      This function is called by self.ui.CbListMeasurement and self.UpdateComboboxListMeasurement
      """
        text = self.ui.CbListMeasurement.currentText
        currentTab = self.ui.TabMeasure.currentWidget().name
        indexes = {
            "TabDistance": {
                False: {"Distance point line": 1, "Distance between 2 points": 0},
                True: {"Distance point line": 2, "Distance between 2 points": 0},
            },
            "TabAngle": {
                False: {"Angle between 2 lines": 3},
                True: {"Angle between 2 lines": 4, f"Angle line T1 and line T2": 5},
            },
        }

        self.ui.StackedMeasure.setCurrentIndex(
            indexes[currentTab][self.ui.CheckBoxT1T2.isChecked()][text]
        )

    def updateComboboxListMeasurement(self):
        """_summary_
    Manage items in CbListMeasurement in function TabMeasure and CheckboxT1T2
    And update StackedMeasure with self.ManageDisplayComboboxMeasurement

    This function is calld by TabMeasure and CheckBoxT1T2
    """
        currentTab = self.ui.TabMeasure.currentWidget().name
        for i in range(self.ui.CbListMeasurement.count):
            self.ui.CbListMeasurement.removeItem(0)

        if currentTab == "TabDistance":
            self.ui.CbListMeasurement.addItems(
                ["Distance point line", "Distance between 2 points"]
            )
        else:
            if self.ui.CheckBoxT1T2.isChecked() == True:
                self.ui.CbListMeasurement.addItem(f"Angle line T1 and line T2")
            self.ui.CbListMeasurement.addItem("Angle between 2 lines")
        self.manageDisplayComboboxMeasurement()

    def addMeasurementToTabMeasurement(self, measure : Union[Angle,Distance,Diff2Measure]):
        """Add new measure in tabmeasure

    Args:
        measure (Measure): new measure to add
    """
        for allmeasure in self.list_measure:

            if allmeasure == measure:

                return

        num = 0
        group = measure["group"]
        for lmeasure in self.list_measure:
            if group == lmeasure["group"]:
                num += 1
        dict = {"Distance": self.ui.TableDistance, "Angle": self.ui.TableAngle}
        dict[group].setRowCount(num + 1)
        a = QCheckBox()
        dict[group].setCellWidget(num, 0, a)


        #use function to iter all informations needed
        for count, value in enumerate(measure.iterBasicInformation()):
            b = QTableWidgetItem(value)
            dict[group].setItem(num, count + 1, b)

        # if group == "Angle":

        checkbox_keep_sign = QCheckBox()
        dict[group].setCellWidget(num, 4, checkbox_keep_sign)
        if "T1" in measure['Type of measurement + time'] and "T2" in measure['Type of measurement + time'] :
            checkbox_keep_sign.setChecked(True)
            checkbox_keep_sign.setEnabled(False)
        measure["keep_sign"] = checkbox_keep_sign

        measure["checkbox"] = a

        self.list_measure.append(measure)



    def selectAllMeasurement(self, group, column):
        if column == 0:
            list_checkbox = []
            for mea in self.list_measure:
                if mea["group"] == group:
                    list_checkbox.append(mea["checkbox"])

            for checkbox in list_checkbox:
                if checkbox.isChecked() :
                    checkbox.setChecked(False)
                else :
                    checkbox.setChecked(True)

        if column == 4:
            list_checkbox = []
            for mea in self.list_measure:
                if mea["group"] == group:
                    if "T1" not in mea['Type of measurement + time'] or "T2" not in mea['Type of measurement + time'] :
                        list_checkbox.append(mea["keep_sign"])

            for checkbox in list_checkbox:
                if checkbox.isChecked() :
                    checkbox.setChecked(False)
                else :
                    checkbox.setChecked(True)


    def deleteMeasurement(self):
        """
    Remove all measurement with checkbox checked

    call by ButtonDeleteMeasurement
    """

        text = self.ui.TabMeasure.currentWidget().name
        text = text[3:]

        dict_table = {"Distance": self.ui.TableDistance, "Angle": self.ui.TableAngle}

        row_remove = []
        i = 0
        for count, measure in enumerate(self.list_measure):

            if measure["group"] == text:
                if measure["checkbox"].checkState():
                    dict_table[text].removeRow(i - len(row_remove))
                    row_remove.append(count - len(row_remove))

                i += 1

        for idremove in row_remove:
            self.list_measure.pop(idremove)


    def createMeasurement(self):
        """
    call by ButtonAddMeasure

    """
        out = []

        dict_page2combobox = {
            "PageAngleBetween2LinesT1": [
                self.ui.CbAB2LT1P1,
                self.ui.CbAB2LT1P2,
                self.ui.CbAB2LT1P3,
                self.ui.CbAB2LT1P4,
            ],
            "PageAngleBetween2LinesT1T2": [
                self.ui.CbAB2LT1T2P1T1,
                self.ui.CbAB2LT1T2P2T1,
                self.ui.CbAB2LT1T2P1T2,
                self.ui.CbAB2LT1T2P2T2,
                self.ui.CbAB2LT1T2P3T1,
                self.ui.CbAB2LT1T2P4T1,
                self.ui.CbAB2LT1T2P3T2,
                self.ui.CbAB2LT1T2P4T2,
            ],
            "PageAngleLineT1T2": [
                self.ui.CbALT1T2L1P1,
                self.ui.CbALT1T2L1P2,
                self.ui.CbALT1T2L2P3,
                self.ui.CbALT1T2L2P4,
            ],
            "PageDistance2Points": [self.ui.CbD2PP1, self.ui.CbD2PP2],
            "PageDistancePointLineT1T2": [
                self.ui.CbDPLT1T2P1T1,
                self.ui.CbDPLT1T2L1T1,
                self.ui.CbDPLT1T2L2T1,
                self.ui.CbDPLT1T2P1T2,
                self.ui.CbDPLT1T2L1T2,
                self.ui.CbDPLT1T2L2T2,
            ],
            "PageDistancePointLineT1": [
                self.ui.CbDPLT1L1,
                self.ui.CbDPLT1L2,
                self.ui.CbDPLT1P1,
            ],
        }

        dict_page2namemeasure_T1= {
            "PageDistancePointLineT1": ["Distance point line T1"],
            "PageAngleBetween2LinesT1": ["Angle between 2 lines T1"],
            "PageDistance2Points": ["Distance between 2 points T1"]
        }

        dict_page2namemeasure_T1T2 = {
            "PageDistancePointLineT1": ["Distance point line T1","Distance point line T2"],
            "PageAngleBetween2LinesT1": ["Angle between 2 lines T1","Angle between 2 lines T2"],
            "PageDistance2Points": ["Distance between 2 points T1","Distance between 2 points T2"]
        }

        dict_page2namemeasure_checkbox = {
            "PageDistancePointLineT1T2": ["Distance point line T1 T2"],
            "PageAngleBetween2LinesT1T2": ["Angle between 2 lines T1 T2"],
            "PageDistance2PointsT1T2": ["Distance between 2 points T1 T2"],
            "PageAngleLineT1T2": ["Angle line T1 and line T2"]
        }

        page = self.ui.StackedMeasure.currentWidget().name
        list_point = []
        for point in dict_page2combobox[page]:
            list_point.append(point.currentText)


        if page == "PageDistance2Points" and self.ui.CheckBoxT1T2.isChecked():
            page = "PageDistance2PointsT1T2"

        dict_page_to_namemeasure = dict_page2namemeasure_T1
        if self.ui.LineEditPathT2.text != '':
            dict_page_to_namemeasure = dict_page2namemeasure_T1T2
            if self.ui.CheckBoxT1T2.isChecked():
                dict_page_to_namemeasure = dict_page2namemeasure_checkbox
        out = self.logic.createMeasurement(dict_page_to_namemeasure[page], list_point)
        # print('out',out)

        for measure in out:
            self.addMeasurementToTabMeasurement(measure)

# ============================================================================================================================#
#    ___    _     _
#   / _ \  | |_  | |__     ___   _ __
#  | | | | | __| | '_ \   / _ \ | '__|
#  | |_| | | |_  | | | | |  __/ | |
#   \___/   \__| |_| |_|  \___| |_|
# ============================================================================================================================#

    def selectFolderT1Patient(self):
        """ Open window allow user to choose folder with T1 patients' information.
        """

        surface_folder = qt.QFileDialog.getExistingDirectory(
            self.parent, "Select a scan folder"
        )
        if surface_folder != "":
            self.ui.LineEditPathT1.setText(surface_folder)

            self.dict_patient_T1, self.dict_patient_extraction = self.logic.createDictPatient(surface_folder)

            (
                self.list_landmarks_exist,
                self.GROUPS_LANDMARKS,
            ) = self.logic.updateGroupLandmark(
                self.dict_patient_T1, self.GROUPS_LANDMARKS
            )
            self.manageTabLandmarks(self.list_landmarks_exist, self.GROUPS_LANDMARKS)


    def selectFolderT2Patients(self):
        """ Open window allow user to choose folder with T1 patients' information.
        """
        if self.dict_patient_T1 == None:
            self.warningMessage("Missing T1 folder")
        else:

            surface_folder = qt.QFileDialog.getExistingDirectory(
                self.parent, "Select a scan folder"
            )
            if surface_folder != "":
                self.ui.LineEditPathT2.setText(surface_folder)

                self.dict_patient_T2,x  = self.logic.createDictPatient(surface_folder)
                self.logic.compareT1T2(self.dict_patient_T1, self.dict_patient_T2)

                (self.list_landmarks_exist,
                    self.GROUPS_LANDMARKS,
                ) = self.logic.updateGroupLandmark(self.dict_patient_T2, self.GROUPS_LANDMARKS)
                self.manageTabLandmarks(self.list_landmarks_exist, self.GROUPS_LANDMARKS)

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


    def warningMessage(self, message):
        messageBox = ctk.ctkMessageBox()
        messageBox.setWindowTitle(" /!\\ WARNING /!\\ ")
        messageBox.setIcon(messageBox.Warning)
        messageBox.setText(message)
        messageBox.setStandardButtons(messageBox.Ok)
        messageBox.exec_()



# ============================================================================================================================#
#     _       ___    _____   ____     ____   _                       _
#    / \     / _ \  |___ /  |  _ \   / ___| | |       ___     __ _  (_)   ___
#   / _ \   | | | |   |_ \  | | | | | |     | |      / _ \   / _` | | |  / __|
#  / ___ \  | |_| |  ___) | | |_| | | |___  | |___  | (_) | | (_| | | | | (__
# /_/   \_\  \__\_\ |____/  |____/   \____| |_____|  \___/   \__, | |_|  \___|
#                                                            |___/
# ============================================================================================================================#


class AQ3DCLogic(ScriptedLoadableModuleLogic):
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

    def concatenateT1T2Patient(self, dict_patients_T1: dict, dict_patients_T2: dict):
        """ Concatenate dict patient T1 and dict patient T2
        Can concatenate if dict_patient_T2 is avoid

    Args:
        dict_patients_T1 (dict): patient Dict T1 with this organisation
          dict = {"001":{"A":[0,0,2],"B":[0,2,3],..},
                  ...,
                 '29':{"A":[0,3,5],"B":[3,6,2],...}
                }

        dict_patients_T2 (dict):patient dict T2 with organiation that dic_patient_T2

    Returns:
        dict: concatenate 2 dicts like this
        dict = {"001":
                    {"T1":{"A":[0,0,2],"B":[0,2,3],...},
                    "T2":{"A":[0,5,2],"B":[5,0,1],...}
                    },
                "029":
                    {"T1":{"A":[0,3,5],"B":[3,6,2],...},
                    "T2":{"A":[535,0,1],"B":[3,5,1],...}
                    }
              }
    """
        dict_patient = {}
        for patient, points in dict_patients_T1.items():
            try:
                dict_patient[patient] = {
                    "T1": {
                        landmark.upper(): value for landmark, value in points.items()
                    },
                    "T2": {
                        landmark.upper(): value
                        for landmark, value in dict_patients_T2[patient].items()
                    },
                }
            except KeyError:
                dict_patient[patient] = {
                    "T1": {
                        landmark.upper(): value for landmark, value in points.items()
                    }
                }
        return dict_patient

    def createDictPatient(self, folder_path: str) -> dict:

        """
        From Folder path create dictionnary with position landmarks for each patient

    Read each json file in the folder (recursively).
    CreateDictPatient can recognize multiple files belonging to the same patient, with the pattern at the beginning of the file name, it takes the pattern before the first '_' to search for another existing one.
        example:
                file name            patient
            - P1_hdgiopghod.json    ->  P1
            - P1_gjdpgjdgjo.json    ->  P1
                    .
                    .
                    .
            - Pn_nduoig.json        ->   Pn
            - Pn_nuifs.json         ->   Pn

    Print if patient have many times the same landmark
    Print if one landmark doesn't have postion, or NaN number

    Args:
        folder_path (str):  folder path containing json file

    Returns:
        dict : dict with num of patient and all landmark link patient
              dict = {patient 1 : {landmarks1 : [0 , 0, 0], landmarks2 : [0 , 0, 0], ... },
                                    .
                                    .
                                    .
                       patient n :{landmarks1 : [0 , 0, 0], landmarks2 : [0 , 0, 0], ...}}
    """
        patients_dict = {}
        dict_patient_extraction = {}
        patients_lst = []
        lst_files = []
        normpath = os.path.normpath("/".join([folder_path, "**", ""]))
        for jsonfile in sorted(glob.iglob(normpath, recursive=True)):
            if os.path.isfile(jsonfile) and ".json" in jsonfile:
                lst_files.append(jsonfile)
                patient = os.path.basename(jsonfile).split("_")[0].split('.mrk')[0]
                if patient not in patients_lst:
                    patients_lst.append(patient)
                if patient not in patients_dict:
                    patients_dict[patient] = {}
                # print(f'json file {jsonfile}')
                json_file = pd.read_json(jsonfile)
                markups = json_file.loc[0, "markups"]
                controlPoints = markups["controlPoints"]
                for i in range(len(controlPoints)):
                    landmark_name = controlPoints[i]["label"]
                    position = controlPoints[i]["position"]

                    # check the patient have many times the same landmark
                    if landmark_name in patients_dict[patient]:
                        print(
                            f"This patient {patient} have many times this landmark {landmark_name}"
                        )

                    patients_dict[patient][landmark_name] = position

                    # check if landmarks are useable
                    good = False
                    if isinstance(position, list):
                        if len(position) == 3:
                            if not False in [
                                isinstance(value, (int, float, np.ndarray))
                                for value in position
                            ] and not True in np.isnan(position):
                                good = True
                    if not good:
                        print(
                            f"For this file {jsonfile} this landmark {landmark_name} are not good "
                        )

                if "NE" in jsonfile or "Non_Extraction" in jsonfile or "Non Extraction" in jsonfile or "non_extraction" in jsonfile or "non extraction" in jsonfile :
                    dict_patient_extraction[patient]=0
                else :
                    dict_patient_extraction[patient]=1
                # print("*"*150)
                # print("json file : ",jsonfile)
                # print("*"*150)

        return patients_dict,dict_patient_extraction

    def compareT1T2(self, dict_patinetT1: dict, dict_patientT2: dict):
        """Check if patient T1 and T2 have the same landmark, and the same patient

    Display in the terminal difference between T1 and T2

    Args:
        dic_patinetT1 (dict): dict with all patients and landmarks at time T1
        dic_patientT2 (dict): dict with all patients and landmarks at time T2
        exemple of dic_patient :
            dico = {patient 1 : {landmarks1 : [0 , 0, 0], landmarks2 : [0 , 0, 0], ... },
                .
                .
                .
            patient n :{landmarks1 : [0 , 0, 0], landmarks2 : [0 , 0, 0], ...}}
    """

        # compare landmark patient T1 and T2
        dif_landmark = {}
        for patientT1, landmarks in dict_patinetT1.items():
            if patientT1 in dict_patientT2:
                if set(landmarks) != set(dict_patientT2[patientT1]):
                    dif = set(landmarks) - set(dict_patientT2[patientT1])
                    dif.union(set(dict_patientT2[patientT1]) - set(landmarks))
                    dif_landmark[patientT1] = dif
                    print(
                        f"T1 and T2 of this patient {patientT1} doesnt have the same landmark, landmark dif {dif}"
                    )

        # compare the name patient T1 and T2
        dif_patient = None
        if set(dict_patinetT1.keys()) != set(dict_patientT2.keys()):
            dif = set(dict_patinetT1.keys()) - set(dict_patientT2.keys())
            dif.union(set(dict_patientT2.keys()) - set(dict_patinetT1.keys()))
            dif_patient = dif_landmark
            print(f"T1 and T2 doesnt have the same patient, dif patient {dif}")
        return dif_landmark , dif_patient


    def updateGroupLandmark(self, dict_patient: dict, all_landmarks: Group_landmark) -> tuple[list,Group_landmark]:
        """
    Add in goup_landmark  midpoints and landmark not existing in group_landmark, but existing in dict_patient.
    Create landmark list with landmark existing in dic_patient.
    Print message if one patient miss landmark indicating the patient and landmark missing

    Args:
        dict_patient ( dict ): dict with patient, landmarks and position like this
                  dict_patient = {patient 1 : {landmarks1 : [0 , 0, 0], landmarks2 : [0 , 0, 0], ... },
                              .
                              .
                              .
                          patient n :{landmarks1 : [0 , 0, 0], landmarks2 : [0 , 0, 0], ...}}

        all_landmarks ( Group_Landmark ) : dict with all landmark use by the doctor


    Returns:
        list : list landmark content in dict_patient
        Group_landmark : dict with all landmarks use by the doctor and one more group with name other. This group is for landmark is not exist in all_patient
    """
        list_landmark = set()
        for patient, landmarks in dict_patient.items():
            list_landmark = list_landmark.union(set(landmarks.keys()))

        for patient, landamrks in dict_patient.items():
            dif = list_landmark.difference(set(landamrks.keys()))
            if len(dif) != 0:
                print(f"This patient {patient} doesn't have this landmark(s) {dif}")

        list_landmark = list(list_landmark)

        list_otherlandmarks = []
        list_midlandmarks = []

        for landmark in list_landmark:
            if landmark[:3].upper() == "Mid".upper():
                list_midlandmarks.append(landmark)
            elif not landmark in all_landmarks:
                list_otherlandmarks.append(landmark)
        if "Other" in all_landmarks.keys():
            all_landmarks["Other"] += list_otherlandmarks
        else :
            all_landmarks["Other"] = list_otherlandmarks

        if "Midpoint" in all_landmarks:
            existing = set(all_landmarks["Midpoint"])
            all_landmarks["Midpoint"] = list(existing.union(list_midlandmarks))
        else:
            all_landmarks["Midpoint"] = list(set(list_midlandmarks))

        return list_landmark, all_landmarks

    def saveMidpoint(
        self, patients_dict: dict, out_path: str, midpoints: list[list]
    ):
        """
    Write json file for each patient containing midpoint passed in argurmment
    All midpoint position are computed
    Printing warning message if computation of midpoints dont work or landmark missing for patient
    Args:
        patients_dict (dict): dictionnary of patient
            dic = {"001":{"A":[0,0,2],"B":[0,2,3],..},
                    ...,
                    '29':{"A":[0,3,5],"B":[3,6,2],...}
                  }
        out_path (str): the path folder where the midpoint will be save
        midpoints (list[list]): list midpoint
            list = [['A','UL6O'],['R2RM','LL6O']] -> Mid_A_UL6O, Mid_R2RM_LL6O
    """

        midpoint_dict = {}

        for patient in patients_dict.keys():
            lst_mid_point = []
            for mid_point in midpoints:
                P1_name = mid_point[0]
                P2_name = mid_point[1]
                if P1_name and P2_name in patients_dict[patient]:
                    try:
                        P1_pos = patients_dict[patient][P1_name]
                        P2_pos = patients_dict[patient][P2_name]
                        midpoint_position = self.computeMidPoint(
                            np.array(P1_pos), np.array(P2_pos)
                        )
                    except:
                        print(
                            f"Save Midpoint, Warning this patient : {patient}, landmark : {mid_point}, it s not save. Please verify your folder"
                        )
                        continue
                    controle_point = self.generateControlePoint(
                        f"Mid_{P1_name}_{P2_name}", midpoint_position
                    )
                    lst_mid_point.append(controle_point)
                    patients_dict[patient][controle_point["label"]] = controle_point[
                        "position"
                    ]

            if patient not in midpoint_dict.keys():
                midpoint_dict[patient] = lst_mid_point

        for patient, cp_lst in midpoint_dict.items():
            if os.path.exists(os.path.join(out_path,f"{patient}_Midpoint.json")) :
                cp_lst = self.mergeJsonControlePoint(os.path.join(out_path,f"{patient}_Midpoint.json"),cp_lst)

            self.writeJson(f"{patient}_Midpoint", cp_lst, out_path)
            
    def findOriginalJson(self, folderPath, patientId):
        for file in glob.glob(os.path.join(folderPath, "*.json")):
            if file.startswith(os.path.join(folderPath, patientId)):
                return file
        return None

    def appendMidpointsToJson(self, filePath, patientLandmarks, midpoints):
        with open(filePath, 'r') as f:
            data = json.load(f)

        controlPoints = data['markups'][0]['controlPoints']

        for P1, P2 in midpoints:
            if P1 in patientLandmarks and P2 in patientLandmarks:
                midpointPos = self.computeMidPoint(np.array(patientLandmarks[P1]), np.array(patientLandmarks[P2]))
                controlPoints.append(self.generateControlePoint(f"Mid_{P1}_{P2}", midpointPos))

        data['markups'][0]['controlPoints'] = controlPoints

        with open(filePath, 'w') as f:
            json.dump(data, f, indent=4)


    def mergeJsonControlePoint(self, file_name : str, list_controle_point : list ):
        with open(file_name) as f :
            data = json.load(f)
        list_controle_point += data['markups'][0]['controlPoints']

        return list_controle_point

    def writeJson(self, file_name: str, list_controle_point: list, folder: str):
        """Write json file containing landmarks

    Args:
        file_name (str): name of json file
        list_controle_point (list): list with controle point
                                  a controle point has this architectue :
                                      controle_point = {
                                      "id": str(1),
                                      "label": landmark name,
                                      "description": "",
                                      "associatedNodeID": "",
                                      "position": [0,0,0],
                                      "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                                      "selected": True,
                                      "locked": True,
                                      "visibility": True,
                                      "positionStatus": "defined"
                                    }
        folder (str): write json file
    """

        file = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
            "markups": [
                {
                    "type": "Fiducial",
                    "coordinateSystem": "LPS",
                    "locked": False,
                    "labelFormat": "%N-%d",
                    "controlPoints": list_controle_point,
                    "measurements": [],
                    "display": {
                        "visibility": False,
                        "opacity": 1.0,
                        "color": [0.4, 1.0, 0.0],
                        "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                        "activeColor": [0.4, 1.0, 0.0],
                        "propertiesLabelVisibility": False,
                        "pointLabelsVisibility": True,
                        "textScale": 3.0,
                        "glyphType": "Sphere3D",
                        "glyphScale": 1.0,
                        "glyphSize": 5.0,
                        "useGlyphScale": True,
                        "sliceProjection": False,
                        "sliceProjectionUseFiducialColor": True,
                        "sliceProjectionOutlinedBehindSlicePlane": False,
                        "sliceProjectionColor": [1.0, 1.0, 1.0],
                        "sliceProjectionOpacity": 0.6,
                        "lineThickness": 0.2,
                        "lineColorFadingStart": 1.0,
                        "lineColorFadingEnd": 10.0,
                        "lineColorFadingSaturation": 1.0,
                        "lineColorFadingHueOffset": 0.0,
                        "handlesInteractive": False,
                        "snapMode": "toVisibleSurface",
                    },
                }
            ],
        }
        if ".json" not in file_name:
            file_name = f"{file_name}.json"

        with open(os.path.join(folder, f"{file_name}"), "w", encoding="utf-8") as f:
            json.dump(file, f, ensure_ascii=False, indent=4)
        f.close

    def computeMidPoint(self, p1, p2):
        mp = (p1 + p2) / 2
        return mp

    def generateControlePoint(self, label: str, position: Union[list , np.ndarray]):
        """Generate ControlePoint readable by slicer

    Args:
        label (str): name of landmark
        position (list | np.ndarray): landmark's position

    Returns:
        dict: Controle point
    """
        controle_point = {
            "id": str(1),
            "label": label,
            "description": "",
            "associatedNodeID": "",
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": True,
            "visibility": True,
            "positionStatus": "defined",
        }
        return controle_point

    def exportMeasurement(self, path_file: str, measures: list):
        """ Write an excel with measures listed in measures (args)
        Create 3 pages in excel file: Distane between 2 points, Distance point line and Angle between 2 lines
        excel example:
            page Distane between 2 points:
               Type of measurement Point 1   Point 2 / Line
0     Distance between 2 points T1       S  MID_LR3RC_LR4RC
1     Distance between 2 points T2       S  MID_LR3RC_LR4RC
2  Distance between 2 points T1 T2       S  MID_LR3RC_LR4RC

            page Distnace point line
             Type of measurement            Point 1                       Point 2 / Line
0         Distance point line T1                  S                    N-MID_LR5RC_LR5RC
1         Distance point line T2    MID_LR5RC_LR4RC                    MID_LR3RC_LR5RC-S
2  Distance point line dif T1 T2  S/MID_LR5RC_LR4RC  N-MID_LR5RC_LR5RC/MID_LR3RC_LR5RC-S

            page Angle between 2 lines
               Type of measurement                                             Line 1                                             Line 2
0        Angle line T1 and line T2                    MID_LR3RC_LR4RC-MID_LR3RC_LR5RC                     MID_LR3RC_LR4RC-MID_LR5RC_LR3O
1         Angle between 2 lines T1                    MID_LR3RC_LR5RC-MID_LR3RC_LR4RC                     MID_LR3RC_LR4RC-MID_LR5RC_LR3O
2         Angle between 2 lines T2                     MID_LR3RC_LR4RC-MID_LR4O_LR5RC                    MID_LR5RC_LR4RC-MID_LR3RC_LR4RC
3  Angle between 2 lines dif T1 T2  MID_LR3RC_LR5RC-MID_LR3RC_LR4RC/MID_LR3RC_LR4R...  MID_LR3RC_LR4RC-MID_LR5RC_LR3O/MID_LR5RC_LR4RC...
4         Angle between 2 lines T1                                  MID_LR3RC_LR5RC-S                                   N-MID_LR5RC_LR3O
5         Angle between 2 lines T2                                  MID_LR3RC_LR4RC-S                                  N-MID_LR3RC_LR4RC
6  Angle between 2 lines dif T1 T2                MID_LR3RC_LR5RC-S/MID_LR3RC_LR4RC-S                 N-MID_LR5RC_LR3O/N-MID_LR3RC_LR4RC


    Args:
        path_file (str): path of the excel file
        measures (list): list = [measurement, measurement ,mesasurement] measurement is Measure class

    """
        dict_data_dist_pp = {
            "Type of measurement": [],
            "Point 1": [],
            "Point 2 / Line": [],
        }
        dict_data_dist_pl = {
            "Type of measurement": [],
            "Point 1": [],
            "Point 2 / Line": [],
        }
        dict_data_angl = {"Type of measurement": [], "Line 1": [], "Line 2": []}
        for __measure in measures:
            measure : Union[Diff2Measure,Distance,Angle] = __measure
            iterinformation = iter(measure.iterBasicInformation())
            if (
                measure["Type of measurement"] == "Distance between 2 points"
                or measure["Type of measurement"] == "Distance between 2 points T1 T2"
            ):
                dict_data_dist_pp["Type of measurement"].append(next(iterinformation))
                dict_data_dist_pp["Point 1"].append(next(iterinformation))
                dict_data_dist_pp["Point 2 / Line"].append(next(iterinformation))

            elif measure["Type of measurement"] == "Distance point line":
                dict_data_dist_pl["Type of measurement"].append(next(iterinformation))
                dict_data_dist_pl["Point 1"].append(next(iterinformation))
                dict_data_dist_pl["Point 2 / Line"].append(next(iterinformation))

            else:
                dict_data_angl["Type of measurement"].append(next(iterinformation))
                dict_data_angl["Line 1"].append(next(iterinformation))
                dict_data_angl["Line 2"].append(next(iterinformation))

        with pd.ExcelWriter(path_file) as writer:
            if len(dict_data_dist_pp["Type of measurement"]) > 0:
                df_dist_pp = pd.DataFrame(dict_data_dist_pp)
                df_dist_pp.to_excel(
                    writer, sheet_name="Distance between 2 points", index=False
                )
            if len(dict_data_dist_pl["Type of measurement"]) > 0:
                df_dist_pl = pd.DataFrame(dict_data_dist_pl)
                df_dist_pl.to_excel(
                    writer, sheet_name="Distance point line", index=False
                )
            if len(dict_data_angl["Type of measurement"]) > 0:
                df_angl = pd.DataFrame(dict_data_angl)
                df_angl.to_excel(
                    writer, sheet_name="Angle between 2 lines", index=False
                )

    def importMeasurement(self,path_file : str) :
        """From path of the excel file create list measures

        excel example:
            page Distane between 2 points:
               Type of measurement Point 1   Point 2 / Line
0     Distance between 2 points T1       S  MID_LR3RC_LR4RC
1     Distance between 2 points T2       S  MID_LR3RC_LR4RC
2  Distance between 2 points T1 T2       S  MID_LR3RC_LR4RC

            page Distnace point line
             Type of measurement            Point 1                       Point 2 / Line
0         Distance point line T1                  S                    N-MID_LR5RC_LR5RC
1         Distance point line T2    MID_LR5RC_LR4RC                    MID_LR3RC_LR5RC-S
2  Distance point line dif T1 T2  S/MID_LR5RC_LR4RC  N-MID_LR5RC_LR5RC/MID_LR3RC_LR5RC-S

            page Angle between 2 lines
               Type of measurement                                             Line 1                                             Line 2
0        Angle line T1 and line T2                    MID_LR3RC_LR4RC-MID_LR3RC_LR5RC                     MID_LR3RC_LR4RC-MID_LR5RC_LR3O
1         Angle between 2 lines T1                    MID_LR3RC_LR5RC-MID_LR3RC_LR4RC                     MID_LR3RC_LR4RC-MID_LR5RC_LR3O
2         Angle between 2 lines T2                     MID_LR3RC_LR4RC-MID_LR4O_LR5RC                    MID_LR5RC_LR4RC-MID_LR3RC_LR4RC
3  Angle between 2 lines dif T1 T2  MID_LR3RC_LR5RC-MID_LR3RC_LR4RC/MID_LR3RC_LR4R...  MID_LR3RC_LR4RC-MID_LR5RC_LR3O/MID_LR5RC_LR4RC...
4         Angle between 2 lines T1                                  MID_LR3RC_LR5RC-S                                   N-MID_LR5RC_LR3O
5         Angle between 2 lines T2                                  MID_LR3RC_LR4RC-S                                  N-MID_LR3RC_LR4RC
6  Angle between 2 lines dif T1 T2                MID_LR3RC_LR5RC-S/MID_LR3RC_LR4RC-S                 N-MID_LR5RC_LR3O/N-MID_LR3RC_LR4RC

    Args:
        path_file (str): path of the excel file

    Returns:
        list: list of measure
    """
        reader = pd.read_excel(path_file, sheet_name=None)
        newreader = {
            "Distance between 2 points": {
                "Type of measurement": [],
                "Point 1": [],
                "Point 2 / Line": [],
            },
            "Distance point line": {
                "Type of measurement": [],
                "Point 1": [],
                "Point 2 / Line": [],
            },
            "Angle between 2 lines": {
                "Type of measurement": [],
                "Line 1": [],
                "Line 2": [],
            },
        }

        list_measure = []
        for name_sheet in reader:
            for name_column in reader[name_sheet]:
                for i in reader[name_sheet][name_column]:
                    newreader[name_sheet][name_column].append(i)

        for name_sheet, sheet in newreader.items():
            list_point = []
            for i in range(len(sheet["Type of measurement"])):
                if name_sheet == "Distance between 2 points":
                    list_point = [sheet["Point 1"][i], sheet["Point 2 / Line"][i]]

                elif name_sheet == "Distance point line":
                    if "/" in sheet["Point 2 / Line"][i]:
                        point1 = sheet["Point 1"][i].split("/")
                        point2 = sheet["Point 2 / Line"][i].split("/")
                        list_point = [
                            point1[0],
                            point2[0].split("-")[0],
                            point2[0].split("-")[1],
                            point1[1],
                            point2[1].split("-")[0],
                            point2[1].split("-")[1],
                        ]
                    else:
                        list_point = [
                            sheet["Point 1"][i],
                            sheet["Point 2 / Line"][i].split("-")[0],
                            sheet["Point 2 / Line"][i].split("-")[1],
                        ]

                else:
                    if "/" in sheet["Line 1"][i]:
                        line1 = sheet["Line 1"][i].split("/")
                        line2 = sheet["Line 2"][i].split("/")
                        list_point = [
                            line1[0].split("-")[0],
                            line1[0].split("-")[1],
                            line1[1].split("-")[0],
                            line1[1].split("-")[1],
                            line2[0].split("-")[0],
                            line2[0].split("-")[1],
                            line2[1].split("-")[0],
                            line2[1].split("-")[1],
                        ]
                    else:
                        list_point = [
                            sheet["Line 1"][i].split("-")[0],
                            sheet["Line 1"][i].split("-")[1],
                            sheet["Line 2"][i].split("-")[0],
                            sheet["Line 2"][i].split("-")[1],
                        ]

                measures = self.createMeasurement(
                    [sheet["Type of measurement"][i]], list_point
                )
                for measure in measures:
                    list_measure.append(measure)

        return list_measure

    def createMeasurement(self, type_of_measure: list, list_landmark: list):
        """ Create Measure

    Args:
        type_of_measure (list): name of measure
        list_landmark (list): list landmark
            example:
                list = [ROr,LOr,LPo,C2] the size depend of the measure

    Returns:
        list: list = [measure , measure , measure] the size depend of measure is for T1 or T2
    """

        out = []

        if "Angle between 2 lines T1" in type_of_measure:

            L1 = Line(Point(list_landmark[0], "T1"), Point(list_landmark[1], "T1"))
            L2 = Line(Point(list_landmark[2], "T1"), Point(list_landmark[3], "T1"))
            measure = Angle(L1, L2, "Angle between 2 lines", "T1")
            out.append(measure)

        if "Angle between 2 lines T2" in type_of_measure:

            L1 = Line(Point(list_landmark[0], "T2"), Point(list_landmark[1], "T2"))
            L2 = Line(Point(list_landmark[2], "T2"), Point(list_landmark[3], "T2"))
            measure = Angle(L1, L2, "Angle between 2 lines", "T2")
            out.append(measure)

        if "Angle between 2 lines T1 T2" in type_of_measure :
            T1L1 = Line(Point(list_landmark[0], "T1"), Point(list_landmark[1], "T1"))
            T2L1 = Line(Point(list_landmark[2], "T2"), Point(list_landmark[3], "T2"))
            T1L2 = Line(Point(list_landmark[4], "T1"), Point(list_landmark[5], "T1"))
            T2L2 = Line(Point(list_landmark[6], "T2"), Point(list_landmark[7], "T2"))

            measure1 = Angle(T1L1, T1L2, "Angle between 2 lines", "T1")
            measure2 = Angle(T2L1, T2L2, "Angle between 2 lines", "T2")
            measure_dif = Diff2Measure(measure1, measure2)

            out.append(measure_dif)

        if "Angle line T1 and line T2" in type_of_measure :
            LT1 = Line(Point(list_landmark[0], "T1"), Point(list_landmark[1], "T1"))
            LT2 = Line(Point(list_landmark[2], "T2"), Point(list_landmark[3], "T2"))

            measure = Angle(LT1, LT2, "Angle line T1 and line T2")
            out.append(measure)

        if "Distance between 2 points T1" in type_of_measure:
            P1 = Point(list_landmark[0], "T1")
            P2 = Point(list_landmark[1], "T1")

            measure = Distance(P1, P2, "Distance between 2 points", time="T1")
            out.append(measure)

        if "Distance between 2 points T2" in type_of_measure:
            P1 = Point(list_landmark[0], "T2")
            P2 = Point(list_landmark[1], "T2")

            measure = Distance(P1, P2, "Distance between 2 points", time="T2")
            out.append(measure)

        if "Distance point line T1" in type_of_measure:
            P = Point(list_landmark[0], "T1")
            L = Line(Point(list_landmark[1], "T1"), Point(list_landmark[2], "T1"))

            measure = Distance(P, L, "Distance point line", time="T1")
            out.append(measure)

        if "Distance point line T2" in type_of_measure:
            P = Point(list_landmark[0], "T2")
            L = Line(Point(list_landmark[1], "T2"), Point(list_landmark[2], "T2"))

            measure = Distance(P, L, "Distance point line", time="T2")
            out.append(measure)

        if (
            "Distance point line T1 T2" in type_of_measure
        ):
            PT1 = Point(list_landmark[0], "T1")
            LT1 = Line(Point(list_landmark[1], "T1"), Point(list_landmark[2], "T1"))
            PT2 = Point(list_landmark[3], "T2")
            LT2 = Line(Point(list_landmark[4], "T2"), Point(list_landmark[5], "T2"))

            measure1 = Distance(PT1, LT1, "Distance point line", time="T1")
            measure2 = Distance(PT2, LT2, "Distance point line", time="T2")
            measure_dif = Diff2Measure(measure1, measure2)

            out.append(measure_dif)

        if "Distance between 2 points T1 T2" in type_of_measure:
            P1 = Point(list_landmark[0], "T1")
            P2 = Point(list_landmark[1], "T2")

            measure = Distance(P1, P2, "Distance between 2 points T1 T2")


            out.append(measure)


        return out

    def computeMeasurement(self, list_measure: list, dict_patient: dict):
        """Compute measure

            Printing warning, if there is error with computation or if landmark dont exist for one patient
            Printing message, if ComputeMeasurement remove one measure if there is useless (example : there are only zero in measure)
    Args:
        list_measure (list): list = [measure , measure , measure]
        dict_patient (dict):  contaneation of dict_patient
            dict = {"001":
                {"T1":{"A":[0,0,2],"B":[0,2,3],...},
                "T2":{"A":[0,5,2],"B":[5,0,1],...}
                },
            "029":
                {"T1":{"A":[0,3,5],"B":[3,6,2],...},
                "T2":{"A":[535,0,1],"B":[3,5,1],...}
                }
                }

    Returns:
        dict: return dict_patient_conmputation
    """
        dict_patient__computation = {
            "Patient": [],
            "Type of measurement": [],
            "Landmarks": [],
            "R-L Component": [],
            "R-L Meaning": [],
            "A-P Component": [],
            "A-P Meaning": [],
            "S-I Component": [],
            "S-I Meaning": [],
            "3D Distance": [],
            "Yaw Component": [],
            "Yaw Meaning": [],
            "Pitch Component": [],
            "Pitch Meaning": [],
            "Roll Component": [],
            "Roll Meaning": [],
        }
        # dic_with_all_measurement = {}
        # dic_short_cut = {'Distance point line T1':'DplT1',
        #                  "Distance point line T1 T2":'DplT12',
        #                  'Angle between 2 lines T1': 'Ab2lT1',
        #                  'Angle betwwen 2 lines T1 T2':'Ab2plT12',
        #                  "Distance between 2 points T1":"Db2pT1",
        #                  "Distance between 2 points T1 T2":"Db2pT12",
        #                  "Angle line T1 and line T2": "AlT1&lT2"}

        list_title = [
            "Landmarks",
            "R-L Component",
            "R-L Meaning",
            "A-P Component",
            "A-P Meaning",
            "S-I Component",
            "S-I Meaning",
            "3D Distance",
            "Yaw Component",
            "Yaw Meaning",
            "Pitch Component",
            "Pitch Meaning",
            "Roll Component",
            "Roll Meaning",
        ]

        for patient, point in dict_patient.items():
            print("patient : ",patient)
            for __measure in list_measure:
                measure : Union[Diff2Measure,Angle,Distance] = __measure
                try:
                    measure.setPosition(point)

                except KeyError as key:
                    print(f"this landmark {key} doesnt exist for this patient", patient)
                    continue

                try:
                    print("measure : ",measure)
                    measure.computation()
                except ZeroDivisionError as Zero:
                    print(
                        f"impossible to compute this measure {measure} for this patient {patient} a reason divide by 0 {Zero}"
                    )
                    continue

                measure.manageMeaningComponent()

                # if measure.isUtilMeasure():
                dict_patient__computation["Patient"].append(patient)
                dict_patient__computation["Type of measurement"].append(
                    measure["Type of measurement + time"]
                )
                for title in list_title:
                    dict_patient__computation[title].append(measure[title])

                # else:
                #     print(
                #         f"Dont write this measure {measure} for this patient {patient} because is useless measure"
                #     )
                #     continue

        # if all(value=="x" for value in dict_patient__computation["Lateral or medial-Left"]):
        #     del dict_patient__computation["Lateral or medial-Left"]

        # if all(value=="x" for value in dict_patient__computation["Lateral or medial-Right"]):
        #     del dict_patient__computation["Lateral or medial-Right"]

        # for measure in list_measure :
        #     dict_measurement_sheet = {
        #     "Patient": [],
        #     "Landmarks" : [],
        #     "R-L Component": [],
        #     "R-L Meaning": [],
        #     "A-P Component": [],
        #     "A-P Meaning": [],
        #     "S-I Component": [],
        #     "S-I Meaning": [],
        #     "3D Distance": [],
        #     "Yaw Component": [],
        #     "Yaw Meaning": [],
        #     "Pitch Component": [],
        #     "Pitch Meaning": [],
        #     "Roll Component": [],
        #     "Roll Meaning": [],
        # }
        #     add_in_name_sheet = ''
        #     if measure["Landmarks"].replace('/','-')[:3] == "Mid" :
        #         add_in_name_sheet = 'Mid'
        #     else :
        #         add_in_name_sheet = measure["Landmarks"].replace('/','-')
        #     type_measure = dict_short_cut[measure["Type of measurement + time"]] + add_in_name_sheet
        #     dict_with_all_measurement[type_measure] = dict_measurement_sheet
        #     for patient, point in dict_patient.items():
        #         try:
        #             measure["position"] = point
        #         except KeyError as key:
        #             print(f"this landmark {key} doesnt exist for this patient", patient)
        #             continue
        #         try:
        #             measure.computation()
        #         except ZeroDivisionError as Zero:
        #             print(
        #                 f"impossible to compute this measure {measure} for this patient {patient} a reason divide by 0 {Zero}"
        #             )
        #             continue

        #         measure.manageMeaningComponent()

        #         if measure.isUtilMeasure():
        #             dict_with_all_measurement[type_measure]["Patient"].append(patient)
        #             for title in list_title:
        #                 dict_with_all_measurement[type_measure][title].append(measure[title])

        #         else:
        #             print(
        #                 f"Dont write this measure {measure} for this patient {patient} because is useless measure"
        #             )
        #             continue


        return dict_patient__computation#, dic_with_all_measurement

    def writeMeasurementExcel(self, dict_patient__computation : dict, path : str, name_file : str):
        """Create excel file with result of computation

    Args:
        dict_patient__computation (dict):
        path (str): file's path
        name_file (str): file name
    """
        if "Patient" in dict_patient__computation:
            if len(dict_patient__computation["Patient"]) > 0:
                df = pd.DataFrame(dict_patient__computation)

                df.to_excel(os.path.join(path, name_file))

        if "ID" in dict_patient__computation:
            if len(dict_patient__computation["ID"]) > 0:
                df = pd.DataFrame(dict_patient__computation)

                df.to_excel(os.path.join(path, name_file))




    # def WriteMeasurementExcel2(self, dict_patient__computation : dict, path : str, name_file : str):
    #     """Create excel file with result of computation

    # Args:
    #     dict_patient__computation (dict):
    #     path (str): file's path
    #     name_file (str): file name
    # """
    #     dict_sheet = {}

    #     # if len(dict_patient__computation["Patient"]) > 0:
    #     print(dict_patient__computation)
    #     for key , value in dict_patient__computation.items():
    #         df = pd.DataFrame(value)
    #         dict_sheet[key] = df
    #     with pd.ExcelWriter(os.path.join(path, name_file)) as writer :
    #         for key , df in dict_sheet.items():
    #             df.to_excel(writer,sheet_name = key, index = False)

    def addMidpointToPatient(self, dict_patient: dict, landmark1: str, landmark2: str):
        """
    Add midpoint for each patient

    Args:
        dict_patient (dict): dict_patient
                            dict = {'patient1':{'landmark1':[0,1,0],'landmark2':[0,4,4],...},
                                        .
                                        .
                                    'patientn':{'landmark1':[1,1,0],'landmark2':[0,5,4],...}}
        landmark1 (str): landmark
        landmark2 (str): landmark

    Returns:
        dict: dict_patient
                  dict = {'patient1':{'landmark1':[0,1,0],'landmark2':[0,4,4],'Mid_landmark1_landmark2':[x,x,x],...},
                      .
                      .
                  'patientn':{'landmark1':[1,1,0],'landmark2':[0,5,4],'Mid_landmark1_landmark2':[x,x,x],...}}
    """
        for patient, landmark in dict_patient.items():
            try:
                p1 = landmark[landmark1]
                p2 = landmark[landmark2]
            except KeyError as key:
                print(
                    f"Warning midpoint, dont found landmark {key} for this patient {patient}"
                )
                continue

            try:
                p = list((np.array(p1) + np.array(p2)) / 2)
            except:
                print(
                    f"Warning compute midpoint error, patient : {patient}, landmarks : {landmark1} {landmark2}"
                )
                continue

            dict_patient[patient][f"Mid_{landmark1}_{landmark2}"] = p

        return dict_patient

    def getEnableLandmarks(self,markups_node : list,Group_Landmark : Group_landmark):
        """

        Args:
            markups_node (list): list label selected
            Group_Landmark (Group_landmark): Landmark can existe

        Returns:

        """

        labels = []


        for landmark1 in markups_node:
            if landmark1 in Group_Landmark:
               labels.append(landmark1)
            for landmark2 in markups_node:
                if landmark1+landmark2 in Group_Landmark:
                    labels.append(landmark1+landmark2)

        return labels

#
# AQ3DCTest
#

# ============================================================================================================================#
#    _       ___    _____   ____     ____   _____                _
#   / \     / _ \  |___ /  |  _ \   / ___| |_   _|   ___   ___  | |_
#  / _ \   | | | |   |_ \  | | | | | |       | |    / _ \ / __| | __|
# / ___ \  | |_| |  ___) | | |_| | | |___    | |   |  __/ \__ \ | |_
# /_/   \_\  \__\_\ |____/  |____/   \____|   |_|    \___| |___/  \__|
# ============================================================================================================================#


class AQ3DCTest(ScriptedLoadableModuleTest):
    """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setUp(self,path):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """

        if not os.path.exists(path):
            os.makedirs(path)

        print(f'path temp {path}')

        temp_path = os.path.join(path ,'temp.zip')
        import urllib, shutil, zipfile
        url = 'https://github.com/HUTIN1/Q3DCExtension/releases/download/v1.0.0/Testing.zip'
        with urllib.request.urlopen(url) as response, open(temp_path, 'wb') as out_file:
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
            shutil.copyfileobj(response, out_file)

        with zipfile.ZipFile(temp_path, "r") as zip:
            zip.extractall(path)



    def runTest(self):
        """Run as few or as many tests as needed here.
    """


        tmp_folder = os.path.join(slicer.util.tempDirectory())
        print(f'tmp folder {tmp_folder}')


        self.setUp(tmp_folder)
        # try :


        list_landmark_exist_groundthruth = ['RPCo', 'LOr', 'Ba', 'N', 'RAE', 'ROr', 'RACo', 'PNS', 'Mid_RPo_LPo', 'Mid_UR6R_UL6R', 'LLCo',
                                    'LR1R', 'LR6R', 'Mid_UR1R_UL1R', 'UL1O', 'LL6O', 'LPCo', 'Mid_UR1O_UL1O', 'RPo', 'UL6O',
                                    'Pog', 'Mid_RGo_LGo', 'Mid_LMCo_LLCo', 'Mid_UR4O_UL4O', 'RMCo', 'UR1O', 'Mid_ROr_LOr', 'UL1R',
                                    'RCo', 'LL1R', 'LL6R', 'LPo', 'RLCo', 'UR4O', 'Mid_LR6R_LL6R', 'Mid_UR6O_UL6O', 'LR6MB', 'ANS',
                                    'Mid_RCo_LCo', 'B', 'Mid_RACo_RPCo', 'UL6R', 'Mid_LACo_LPCo', 'RGo', 'LGo', 'LAF', 'LCo', 'UR6O',
                                    'LR6O', 'LACo', 'A', 'RAF', 'Mid_LR1R_LL1R','RSig', 'LL1O', 'UL4O', 'RPF',
                                    'Gn', 'LR1O', 'Mid_RMCo_RLCo', 'UR6MB', 'LPF', 'LAE', 'LSig', 'UR1R', 'S', 'Mid_LR1O_LL1O', 'Mid_LR6O_LL6O',
                                    'LL6MB', 'Me', 'UL6MB', 'UR6R', 'LMCo']

        self.delayDisplay(' Test Creation Dictionnary Patient')
        patient_T1 , patient_T2 = self.testCreateDictPatient(tmp_folder,list_landmark_exist_groundthruth)


        self.delayDisplay(' Test Create Measure')
        list_measure = self.testCreateMeasure()

        self.delayDisplay(' Test Compute Measure')
        compute = self.testComputeMeasure(patient_T1,patient_T2,list_measure)

        self.delayDisplay(' Test Write Measure')
        self.testWriteMeasure(compute,tmp_folder)

        self.delayDisplay(' Test Import Export')
        self.testImportExport(tmp_folder,list_measure)

        self.delayDisplay(' Test Midpoint')
        self.testMidpoint(tmp_folder,patient_T1,patient_T2,list_landmark_exist_groundthruth+['Mid_RPCo_LOr','Mid_Mid_RACo_RPCo_LMCo'])


        # except AssertionError :
        #     self.delayDisplay(f' Test Failed')
        #     return


        self.delayDisplay(' Tests Passed')


    def testCreateDictPatient(self,folder : str,landmark_exist : list) -> tuple[dict,dict]:

        widget = AQ3DCWidget()

        logic = AQ3DCLogic()
        group_landmark = Group_landmark(widget.resourcePath("name_landmark.xlsx"))
        patient_T1,x = logic.createDictPatient(os.path.join(folder,'T1'))
        patient_T2,x = logic.createDictPatient(os.path.join(folder,'T2'))

        dif_landmark , dif_patient = logic.compareT1T2(patient_T1,patient_T2)
        assert dif_landmark == {} and dif_patient == None

        list_landmark_exist, group_landmark = logic.updateGroupLandmark(patient_T1,group_landmark)
        list_landmark_exist, group_landmark = logic.updateGroupLandmark(patient_T2,group_landmark)

        list_landmark_exist.sort()
        landmark_exist.sort()

        assert list_landmark_exist== landmark_exist, f'ground truth : {landmark_exist} \n \n landmark list create : {list_landmark_exist} \n \n  difference : {list(set(landmark_exist).difference(set(list_landmark_exist)))+ list(set(list_landmark_exist).difference(set(landmark_exist))) }'


        return patient_T1, patient_T2

    def testCreateMeasure(self) -> list[Measure]:
        logic = AQ3DCLogic()
        measure_to_create = {'Angle between 2 lines T1' : ['RPCo', 'LOr','RAE', 'ROr'],
                            "Angle between 2 lines T2":['Ba', 'N', 'RACo', 'PNS'],
                            'Angle between 2 lines T1 T2':['RPCo', 'LOr', 'Ba', 'N', 'RAE', 'ROr', 'RACo', 'PNS'],
                            'Angle line T1 and line T2':['RPCo', 'LOr', 'Ba', 'N'],
                            "Distance point line T1" : ['RPCo', 'LOr', 'Ba'],
                            "Distance point line T2" : ['N', 'RAE', 'ROr'],
                            'Distance point line T1 T2':['RPCo', 'LOr', 'Ba', 'N', 'RAE', 'ROr'],
                            'Distance between 2 points T1 T2':['RPCo', 'LOr', 'Ba', 'N', 'RAE', 'ROr', 'RACo', 'PNS'],
                            "Distance between 2 points T1" : ['RPCo', 'LOr'],
                            "Distance between 2 points T2" : ['RPCo', 'LOr']}


        list_measure = []
        for measure , landmarks in measure_to_create.items():
            list_measure = list_measure + logic.createMeasurement([measure],landmarks)


        return list_measure

    def testComputeMeasure(self,patient_T1 : dict ,patient_T2 : dict,list_measure : list[Measure]):
        logic = AQ3DCLogic()

        cat_patient = logic.concatenateT1T2Patient(patient_T1,patient_T2)
        compute = logic.computeMeasurement(list_measure,cat_patient)

        return compute

    def testWriteMeasure(self,compute : list[Measure],folder : str ):
        logic = AQ3DCLogic()

        logic.writeMeasurementExcel(compute,folder,'computatation_test.xlsx')

        reader_computation_test = pd.read_excel(os.path.join(folder,'computatation_test.xlsx'), sheet_name=None)
        reader_grounthruth = pd.read_excel(os.path.join(folder,'computatation_groundtruth.xlsx'), sheet_name=None)
        print(f' file ground thruth : {os.path.join(folder,"computatation_groundtruth.xlsx")}, file test {os.path.join(folder,"computatation_test.xlsx")}')
        for key in reader_computation_test['Sheet1'].keys():
            assert reader_computation_test['Sheet1'][key].to_dict() == reader_grounthruth['Sheet1'][key].to_dict(), f'test : {reader_computation_test["Sheet1"][key].to_dict()} \n ground truth {reader_grounthruth["Sheet1"][key].to_dict()}'





    def testImportExport(self,folder : str ,list_measure : list[Measure]):
        logic = AQ3DCLogic()

        #Test Import Export Measure
        logic.exportMeasurement(os.path.join(folder,'ExportMeasure.xlsx'),list_measure)
        import_measure = logic.importMeasurement(os.path.join(folder,'ExportMeasure.xlsx'))

        #compare import measure and list measure
        for measure in import_measure :
            assert measure in list_measure, f'measure {measure} \n \n  list_measure {list_measure}'

        for measure in list_measure:
            assert measure in import_measure,  f'measure {measure} \n \n  import_measure {import_measure}'




    def testMidpoint(self,folder,patient_T1 : dict ,patient_T2 : dict,list_landmark_check : str):
        logic = AQ3DCLogic()
        # list_midpoint = ['Mid_RPCo_LOr','Mid_Mid_RACo_RPCo_LMCo']
        list_midpoint = [['RPCo','LOr'],['Mid_RACo_RPCo','LMCo']]
        for midpoint in list_midpoint :
            landmark1 = midpoint[0]
            landmark2 = midpoint[1]
            patient_T1 = logic.addMidpointToPatient(patient_T1,landmark1,landmark2)

            computation_midpoint_function = patient_T1['01'][f'Mid_{landmark1}_{landmark2}']
            compution_midpoint = list((np.array(patient_T1['01'][landmark1]) + np.array( patient_T1['01'][landmark2])) / 2)
            assert computation_midpoint_function == compution_midpoint, f'function : {computation_midpoint_function}, \n manually : {compution_midpoint}'

        logic.saveMidpoint(patient_T1,os.path.join(folder,'T1'),list_midpoint)
        logic.saveMidpoint(patient_T2,os.path.join(folder,'T2'),list_midpoint)


        patient_T1 , patient_T2 = self.testCreateDictPatient(folder,list_landmark_check)




