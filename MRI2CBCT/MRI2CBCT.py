import logging
import os
from typing import Annotated, Optional
from qt import QApplication, QWidget, QTableWidget, QDoubleSpinBox, QTableWidgetItem, QHeaderView,QSpinBox, QVBoxLayout, QLabel, QSizePolicy, QCheckBox, QFileDialog,QMessageBox, QApplication, QProgressDialog
import qt
from utils.Preprocess_CBCT import Process_CBCT
from utils.Preprocess_MRI import Process_MRI
from utils.Preprocess_CBCT_MRI import Preprocess_CBCT_MRI
from utils.Reg_MRI2CBCT import Registration_MRI2CBCT
import time 

import vtk
import shutil 
import urllib 
import zipfile

import slicer
from functools import partial
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# MRI2CBCT
#


class MRI2CBCT(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("MRI2CBCT")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#MRI2CBCT">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # MRI2CBCT1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="MRI2CBCT",
        sampleName="MRI2CBCT1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "MRI2CBCT1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="MRI2CBCT1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="MRI2CBCT1",
    )

    # MRI2CBCT2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="MRI2CBCT",
        sampleName="MRI2CBCT2",
        thumbnailFileName=os.path.join(iconsPath, "MRI2CBCT2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="MRI2CBCT2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="MRI2CBCT2",
    )


#
# MRI2CBCTParameterNode
#


@parameterNodeWrapper
class MRI2CBCTParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# MRI2CBCTWidget
#


class MRI2CBCTWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self.checked_cells = set() 
        self.minus_checked_rows = set()
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        
        

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MRI2CBCT.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MRI2CBCTLogic()
        
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documentsLocation)
        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
            "MRI2CBCT",
            "MRI2CBCT_" + "CBCT",
        )
        self.preprocess_cbct = Process_CBCT(self)
        self.preprocess_mri = Process_MRI(self)
        self.preprocess_mri_cbct = Preprocess_CBCT_MRI(self)
        self.registration_mri2cbct = Registration_MRI2CBCT(self)

        # Connections
        #        LineEditOutputReg
        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.registrationButton.connect("clicked(bool)", self.registration_MR2CBCT)
        self.ui.SearchButtonCBCT.connect("clicked(bool)",partial(self.openFinder,"InputCBCT"))
        self.ui.SearchButtonMRI.connect("clicked(bool)",partial(self.openFinder,"InputMRI"))
        self.ui.SearchButtonRegMRI.connect("clicked(bool)",partial(self.openFinder,"InputRegMRI"))
        self.ui.SearchButtonRegCBCT.connect("clicked(bool)",partial(self.openFinder,"InputRegCBCT"))
        self.ui.SearchButtonRegLabel.connect("clicked(bool)",partial(self.openFinder,"InputRegLabel"))
        self.ui.SearchOutputFolderOrientCBCT.connect("clicked(bool)",partial(self.openFinder,"OutputOrientCBCT"))
        self.ui.SearchOutputFolderOrientMRI.connect("clicked(bool)",partial(self.openFinder,"OutputOrientMRI"))
        self.ui.SearchOutputFolderResample.connect("clicked(bool)",partial(self.openFinder,"OutputOrientResample"))
        self.ui.SearchButtonOutput.connect("clicked(bool)",partial(self.openFinder,"OutputReg"))
        self.ui.pushButtonOrientCBCT.connect("clicked(bool)",self.orientCBCT)
        self.ui.pushButtonResample.connect("clicked(bool)",self.resampleMRICBCT)
        self.ui.pushButtonOrientMRI.connect("clicked(bool)",self.orientCenterMRI)
        self.ui.pushButtonDownloadOrientCBCT.connect("clicked(bool)",partial(self.downloadModel,self.ui.lineEditOrientCBCT, "Orientation", True))
        self.ui.pushButtonDownloadSegCBCT.connect("clicked(bool)",partial(self.downloadModel,self.ui.lineEditSegCBCT, "Segmentation", True))
        

        # Make sure parameter node is initialized (needed for module reload) 
        self.initializeParameterNode()
        self.ui.ComboBoxCBCT.setCurrentIndex(1)
        self.ui.ComboBoxCBCT.setEnabled(False)
        self.ui.ComboBoxMRI.setCurrentIndex(1)
        self.ui.ComboBoxMRI.setEnabled(False)
        self.ui.comboBoxRegMRI.setCurrentIndex(1)
        self.ui.comboBoxRegMRI.setEnabled(False)
        self.ui.comboBoxRegCBCT.setCurrentIndex(1)
        self.ui.comboBoxRegCBCT.setEnabled(False)
        self.ui.comboBoxRegLabel.setCurrentIndex(1)
        self.ui.comboBoxRegLabel.setEnabled(False)
        
        self.ui.label_time.setHidden(True)
        self.ui.label_info.setHidden(True)
        self.ui.progressBar.setHidden(True)
        
        self.ui.outputCollapsibleButton.setText("Registration")
        self.ui.inputsCollapsibleButton.setText("Preprocess")
        
        self.ui.outputCollapsibleButton.setChecked(True)  # True to expand, False to collapse
        self.ui.inputsCollapsibleButton.setChecked(False) 
        ##################################################################################################
        ### Orientation Table
        self.tableWidgetOrient = self.ui.tableWidgetOrient
        self.tableWidgetOrient.setRowCount(3)  # Rows for New Direction X, Y, Z
        self.tableWidgetOrient.setColumnCount(4)  # Columns for X, Y, Z, and Minus

        # Set the headers
        self.tableWidgetOrient.setHorizontalHeaderLabels(["X", "Y", "Z", "Negative"])
        self.tableWidgetOrient.setVerticalHeaderLabels(["New Direction X", "New Direction Y", "New Direction Z"])

        # Set the horizontal header to stretch and fill the available space
        header = self.tableWidgetOrient.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        # Set a fixed height for the table to avoid stretching
        self.tableWidgetOrient.setFixedHeight(self.tableWidgetOrient.horizontalHeader().height + 
                                            self.tableWidgetOrient.verticalHeader().sectionSize(0) * self.tableWidgetOrient.rowCount)

        # Add widgets for each cell
        for row in range(3):
            for col in range(4):  # Columns X, Y, Z, and Minus
                if col!=3 :
                    checkBox = QCheckBox('0')
                    checkBox.stateChanged.connect(lambda state, r=row, c=col: self.onCheckboxOrientClicked(r, c, state))
                    self.tableWidgetOrient.setCellWidget(row, col, checkBox)
                else :
                    checkBox = QCheckBox('No')
                    checkBox.stateChanged.connect(lambda state, r=row, c=col: self.onCheckboxOrientClicked(r, c, state))
                    self.tableWidgetOrient.setCellWidget(row, col, checkBox)

        self.ui.ButtonDefaultOrientMRI.connect("clicked(bool)",self.defaultOrientMRI)
        self.defaultOrientMRI()
        
        ##################################################################################################
        ### Normalization Table
        self.tableWidgetNorm = self.ui.tableWidgetNorm

        self.tableWidgetNorm.setRowCount(2)  # MRI and CBCT rows + header row
        self.tableWidgetNorm.setColumnCount(4)  # Min, Max for Normalization and Percentile
        
        # Set the horizontal header to stretch and fill the available space
        header = self.tableWidgetNorm.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        # Set a fixed height for the table to avoid stretching
        self.tableWidgetNorm.setFixedHeight(self.tableWidgetNorm.horizontalHeader().height + 
                                            self.tableWidgetNorm.verticalHeader().sectionSize(0) * self.tableWidgetNorm.rowCount)

        # Set the headers
        self.tableWidgetNorm.setHorizontalHeaderLabels(["Normalization Min", "Normalization Max", "Percentile Min", "Percentile Max"])
        self.tableWidgetNorm.setVerticalHeaderLabels([ "MRI", "CBCT"])


        for row in range(2):
            for col in range(4):
                spinBox = QSpinBox()
                if col in [2, 3]:  # Columns for Percentile Min and Percentile Max
                    spinBox.setMaximum(100)
                else:
                    spinBox.setMaximum(10000)
                self.tableWidgetNorm.setCellWidget(row, col, spinBox)
                
        self.ui.ButtonCheckBoxDefaultNorm1.connect("clicked(bool)",partial(self.DefaultNorm,"1"))
        self.ui.ButtonCheckBoxDefaultNorm2.connect("clicked(bool)",partial(self.DefaultNorm,"2"))
        
        self.DefaultNorm("1",_)
                
        ##################################################################################################
        # RESAMPLE TABLE
        self.tableWidgetResample = self.ui.tableWidgetResample
        
        # Increase the row and column count
        self.tableWidgetResample.setRowCount(2)  # Adding a second row
        self.tableWidgetResample.setColumnCount(4)  # Adding a new column

        # Set the horizontal header to stretch and fill the available space
        header = self.tableWidgetResample.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # Set a fixed height for the table to avoid stretching
        self.tableWidgetResample.setFixedHeight(
            self.tableWidgetResample.horizontalHeader().height + 
            self.tableWidgetResample.verticalHeader().sectionSize(0) * self.tableWidgetResample.rowCount
        )

        # Set the headers
        self.tableWidgetResample.setHorizontalHeaderLabels(["X", "Y", "Z", "Keep File "])
        self.tableWidgetResample.setVerticalHeaderLabels(["Number of slices", "Spacing"])

        # Add QSpinBoxes for the first row
        spinBox1 = QSpinBox()
        spinBox1.setMaximum(10000)
        spinBox1.setValue(119)
        self.tableWidgetResample.setCellWidget(0, 0, spinBox1)

        spinBox2 = QSpinBox()
        spinBox2.setMaximum(10000)
        spinBox2.setValue(443)
        self.tableWidgetResample.setCellWidget(0, 1, spinBox2)

        spinBox3 = QSpinBox()
        spinBox3.setMaximum(10000)
        spinBox3.setValue(443)
        self.tableWidgetResample.setCellWidget(0, 2, spinBox3)

        # Add QSpinBoxes for the new row
        spinBox4 = QDoubleSpinBox()
        spinBox4.setMaximum(10000)
        spinBox4.setSingleStep(0.1)
        spinBox4.setValue(0.3)
        self.tableWidgetResample.setCellWidget(1, 0, spinBox4)

        spinBox5 = QDoubleSpinBox()
        spinBox5.setMaximum(10000)
        spinBox5.setSingleStep(0.1)
        spinBox5.setValue(0.3)
        self.tableWidgetResample.setCellWidget(1, 1, spinBox5)

        spinBox6 = QDoubleSpinBox()
        spinBox6.setMaximum(10000)
        spinBox6.setSingleStep(0.1)
        spinBox6.setValue(0.3)
        self.tableWidgetResample.setCellWidget(1, 2, spinBox6)
        # Add QCheckBox for the "Keep File" column
        checkBox1 = QCheckBox("Keep the same size as the input scan")
        checkBox1.stateChanged.connect(lambda state: self.toggleSpinBoxes(state, [spinBox1, spinBox2, spinBox3]))
        self.tableWidgetResample.setCellWidget(0, 3, checkBox1)

        checkBox2 = QCheckBox("Keep the same spacing as the input scan")
        checkBox2.stateChanged.connect(lambda state: self.toggleSpinBoxes(state, [spinBox4, spinBox5, spinBox6]))
        self.tableWidgetResample.setCellWidget(1, 3, checkBox2)
        
    def toggleSpinBoxes(self, state, spinBoxes):
        """
        Enable or disable a list of QSpinBox widgets based on the provided state.

        Parameters:
        - state: An integer representing the state (2 for disabled, any other value for enabled).
        - spinBoxes: A list of QSpinBox widgets to be toggled.

        The function iterates through each QSpinBox in the provided list. If the state is 2,
        the QSpinBox is disabled and its text color is set to gray. Otherwise, the QSpinBox
        is enabled and its default stylesheet is restored.

        This function is connected to the "keep file" checkbox. When the checkbox is checked
        (state == 2), the spin boxes are disabled and shown in gray. If the checkbox is unchecked,
        the spin boxes are enabled and restored to their default style.
        """
        for spinBox in spinBoxes:
            if state == 2:
                spinBox.setEnabled(False)
                spinBox.setStyleSheet("color: gray;")
            else:
                spinBox.setEnabled(True)
                spinBox.setStyleSheet("")

        
    def get_resample_values(self):
        """
        Retrieves the resample values (X, Y, Z) from the QTableWidget.

        :return: A tuple of two lists representing the resample values for the two rows. 
                Each list contains three values (X, Y, Z) or None if the "Keep File" checkbox is checked.
                First output : number of slices.
                Second output : spacing 
        """
        resample_values_row1 = []
        resample_values_row2 = []

        # Check the "Keep File" checkbox for the first row
        if self.tableWidgetResample.cellWidget(0, 3).isChecked():
            resample_values_row1 = "None"
        else:
            resample_values_row1 = [
                self.tableWidgetResample.cellWidget(0, 0).value,
                self.tableWidgetResample.cellWidget(0, 1).value,
                self.tableWidgetResample.cellWidget(0, 2).value
            ]

        # Check the "Keep File" checkbox for the second row
        if self.tableWidgetResample.cellWidget(1, 3).isChecked():
            resample_values_row2 = "None"
        else:
            resample_values_row2 = [
                self.tableWidgetResample.cellWidget(1, 0).value,
                self.tableWidgetResample.cellWidget(1, 1).value,
                self.tableWidgetResample.cellWidget(1, 2).value
            ]

        return resample_values_row1, resample_values_row2

                
        
    def onCheckboxOrientClicked(self, row, col, state):
        """
        Handle the click event of the orientation checkboxes in the table.

        Parameters:
        - row: The row index of the clicked checkbox.
        - col: The column index of the clicked checkbox.
        - state: The state of the clicked checkbox (2 for checked, 0 for unchecked).

        This function updates the orientation checkboxes in the table based on the user's selection.
        It ensures that only one checkbox per row can be set to '1' (or '-1' if the "Minus" column is checked)
        and that the rest are set to '0'. Additionally, if the "Minus" column checkbox is checked, it sets
        the text to 'Yes' and updates related checkboxes in the same row accordingly. The function also handles
        unchecking a checkbox and updating the styles and texts of other checkboxes in the same row and column.

        This function is connected to the checkboxes for the orientation of the MRI. When a checkbox is clicked,
        it ensures the correct orientation is set, following the specified rules.
        """
        if col == 3:  # If the "Minus" column checkbox is clicked
            if state == 2:  # Checkbox is checked
                self.minus_checked_rows.add(row)
                checkBox = self.tableWidgetOrient.cellWidget(row, col)
                checkBox.setText('Yes')
                for c in range(3):
                    checkBox = self.tableWidgetOrient.cellWidget(row, c)
                    if checkBox.text=="1":
                        checkBox.setText('-1')
            else:  # Checkbox is unchecked
                self.minus_checked_rows.discard(row)
                checkBox = self.tableWidgetOrient.cellWidget(row, col)
                checkBox.setText('No')
                for c in range(3):
                    checkBox = self.tableWidgetOrient.cellWidget(row, c)
                    if checkBox.text=="-1":
                        checkBox.setText('1')
        else :   
            if state == 2:  # Checkbox is checked
                # Set the clicked checkbox to '1' and uncheck all others in the same row
                for c in range(3):
                    checkBox = self.tableWidgetOrient.cellWidget(row, c)
                    if checkBox:
                        if c == col:
                            if row in self.minus_checked_rows:
                                checkBox.setText('-1')
                            else :
                                checkBox.setText('1')
                            checkBox.setStyleSheet("color: black;")
                            checkBox.setStyleSheet("font-weight: bold;")
                            self.checked_cells.add((row, col))
                        else:
                            checkBox.setText('0')
                            checkBox.setChecked(False)
                            self.checked_cells.discard((row, c))

                # Check for other '1' in the same column and set them to '0'
                for r in range(3):
                    if r != row:
                        checkBox = self.tableWidgetOrient.cellWidget(r, col)
                        if checkBox and (checkBox.text == '1' or checkBox.text == '-1'):
                            checkBox.setText('0')
                            checkBox.setChecked(False)
                            checkBox.setStyleSheet("color: gray;")
                            checkBox.setStyleSheet("font-weight: normal;")
                            self.checked_cells.discard((r, col))
                            
                # Check if two checkboxes are checked in different rows, then check the third one
                if len(self.checked_cells) == 2:
                    all_rows = {0, 1, 2}
                    all_cols = {0, 1, 2}
                    checked_rows = {r for r, c in self.checked_cells}
                    unchecked_row = list(all_rows - checked_rows)[0]
                    
                    # Find the unchecked column
                    unchecked_cols = list(all_cols - {c for r, c in self.checked_cells})
                    # print("unchecked_cols : ",unchecked_cols)
                    for c in range(3):
                        checkBox = self.tableWidgetOrient.cellWidget(unchecked_row, c)
                        if c in unchecked_cols:
                            checkBox.setStyleSheet("color: black;")
                            checkBox.setStyleSheet("font-weight: bold;")
                            checkBox.setChecked(True)
                            if unchecked_row in self.minus_checked_rows:
                                checkBox.setText('-1')
                            else :
                                checkBox.setText('1')
                            self.checked_cells.add((unchecked_row, c))
                        else : 
                            checkBox.setText('0')
                            checkBox.setChecked(False)
                            self.checked_cells.discard((row, c))

            else:  # Checkbox is unchecked
                checkBox = self.tableWidgetOrient.cellWidget(row, col)
                if checkBox:
                    checkBox.setText('0')
                    checkBox.setStyleSheet("color: black;")
                    checkBox.setStyleSheet("font-weight: normal;")
                    self.checked_cells.discard((row, col))
                    
                # Reset the style of all checkboxes in the same row
                for c in range(3):
                    checkBox = self.tableWidgetOrient.cellWidget(row, c)
                    if checkBox:
                        checkBox.setStyleSheet("color: black;")
                        checkBox.setStyleSheet("font-weight: normal;")
                        
    def getCheckboxValuesOrient(self):
        """
        Retrieve the values of the orientation checkboxes in the table.

        This function iterates through each checkbox in a 3x3 grid within the tableWidgetOrient.
        It collects the integer value (text) of each checkbox and stores them in a list, which is
        then converted to a tuple and returned.

        Returns:
        - A tuple containing the integer values of the checkboxes, representing the orientation of the MRI.
        """
        values = []
        for row in range(3):
            for col in range(3):
                checkBox = self.tableWidgetOrient.cellWidget(row, col)
                if checkBox:
                    values.append(int(checkBox.text))
        return tuple(values)
    
    def defaultOrientMRI(self):
        """
        Set the default orientation values for the MRI checkboxes in the table.

        This function initializes the orientation of the MRI by setting specific checkboxes
        to predefined values. It iterates through a list of initial states, where each state
        is a tuple containing the row, column, and value to set. The value can be 1, -1, or 0.
        The corresponding checkbox is checked and its text is set accordingly. Additionally,
        the checkbox style is updated to make the checked state bold, and the respective sets
        (checked_cells and minus_checked_rows) are updated.

        The initial states are:
        - Row 0, Column 2: Set to -1
        - Row 1, Column 0: Set to 1
        - Row 2, Column 1: Set to -1
        """
        initial_states = [
            (0, 2, -1),
            (1, 0, 1),
            (2, 1, -1)
        ]
        for row, col, value in initial_states:
            checkBox = self.tableWidgetOrient.cellWidget(row, col)
            if checkBox:
                if value == 1:
                    checkBox.setChecked(True)
                    checkBox.setText('1')
                    checkBox.setStyleSheet("font-weight: bold;")
                    self.checked_cells.add((row, col))
                elif value == -1:
                    checkBox.setChecked(True)
                    checkBox.setText('-1')
                    checkBox.setStyleSheet("font-weight: bold;")
                    minus_checkBox = self.tableWidgetOrient.cellWidget(row, 3)
                    if minus_checkBox:
                        minus_checkBox.setChecked(True)
                        minus_checkBox.setText("Yes")
                    self.minus_checked_rows.add(row)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        pass

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        pass


    def _checkCanApply(self, caller=None, event=None) -> None:
        pass
    
    def getNormalization(self):
        """
        Retrieve the normalization values from the table.

        This function iterates through each cell in the tableWidgetNorm, collecting the values
        of QSpinBox widgets. It stores these values in a nested list, where each sublist represents
        a row of values. The collected values are then returned as a list of lists.

        Returns:
        - A list of lists containing the values of the QSpinBox widgets in the tableWidgetNorm.
        """
        values = []
        for row in range(self.tableWidgetNorm.rowCount):
            rowData = []
            for col in range(self.tableWidgetNorm.columnCount):
                widget = self.tableWidgetNorm.cellWidget(row, col)
                if isinstance(widget, QSpinBox):
                    rowData.append(widget.value)
            values.append(rowData)
        return(values)
    
    def DefaultNorm(self,num : str,_)->None:
        """
        Set default normalization values in the tableWidgetNorm based on the identifier 'num'.
        
        If 'num' is "1", set specific default values; otherwise, use another set of values.
        
        Parameters:
        - num: Identifier to select the set of default values.
        - _: Unused parameter.
        """
        # Define the default values for each cell
        if num=="1":
            default_values = [
                [0, 100, 0, 100],
                [0, 75, 10, 95]
            ]
        else :
            default_values = [
                [0, 100, 10, 95],
                [0, 100, 10, 95]
            ]
        
        for row in range(self.tableWidgetNorm.rowCount):
            for col in range(self.tableWidgetNorm.columnCount):
                spinBox = QSpinBox()
                spinBox.setMaximum(10000)
                spinBox.setValue(default_values[row][col])
                self.tableWidgetNorm.setCellWidget(row, col, spinBox)
                
    def openFinder(self,nom : str,_) -> None : 
        """
         Open finder to let the user choose is files or folder
        """
        if nom=="InputMRI":
            if self.ui.ComboBoxMRI.currentIndex==1:
                  print("oui")
                  surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            else :
                  surface_folder = QFileDialog.getOpenFileName(self.parent,'Open a file',)

            self.ui.LineEditMRI.setText(surface_folder)

        elif nom=="InputCBCT":
            if self.ui.ComboBoxCBCT.currentIndex==1:
                surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            else :
                surface_folder = QFileDialog.getOpenFileName(self.parent,'Open a file',)
            self.ui.LineEditCBCT.setText(surface_folder)
            
        elif nom=="InputRegCBCT":
            if self.ui.comboBoxRegCBCT.currentIndex==1:
                surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            else :
                surface_folder = QFileDialog.getOpenFileName(self.parent,'Open a file',)
            self.ui.lineEditRegCBCT.setText(surface_folder)
            
        elif nom=="InputRegMRI":
            if self.ui.comboBoxRegMRI.currentIndex==1:
                surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            else :
                surface_folder = QFileDialog.getOpenFileName(self.parent,'Open a file',)
            self.ui.lineEditRegMRI.setText(surface_folder)
            
        elif nom=="InputRegLabel":
            if self.ui.comboBoxRegLabel.currentIndex==1:
                surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            else :
                surface_folder = QFileDialog.getOpenFileName(self.parent,'Open a file',)
            self.ui.lineEditRegLabel.setText(surface_folder)
 

        elif nom=="OutputOrientCBCT":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditOutputOrientCBCT.setText(surface_folder)
            
        elif nom=="OutputOrientMRI":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditOutputOrientMRI.setText(surface_folder)
            
        elif nom=="OutputOrientResample":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditOuputResample.setText(surface_folder)
            
        elif nom=="OutputReg":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.LineEditOutput.setText(surface_folder)

        
        
    def downloadModel(self, lineEdit, name, test,_):
        """
        Download model files from the URL(s) provided by the getModelUrl function.

        Parameters:
        - lineEdit: The QLineEdit widget to update with the model folder path.
        - name: The name of the model to download.
        - test: A flag for testing purposes (unused in this function).
        - _: Unused parameter for compatibility.

        This function fetches the model URL(s) using getModelUrl, downloads the files,
        unzips them to the appropriate directory, and updates the lineEdit with the model
        folder path. It also runs a test on the downloaded model and shows a warning message
        if any errors occur.
        """

        # To select the reference files (CBCT Orientation and Registration mode only)
        listmodel = self.preprocess_cbct.getModelUrl()
        print("listmodel : ",listmodel)

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
            error = self.preprocess_cbct.TestModel(model_folder, lineEdit.name)

            if isinstance(error, str):
                QMessageBox.warning(self.parent, "Warning", error)

            else:
                lineEdit.setText(model_folder)

    def DownloadUnzip(
        self, url, directory, folder_name=None, num_downl=1, total_downloads=1
    ):
        """
        Download and unzip a file from a given URL to a specified directory.

        Parameters:
        - url: The URL of the zip file to download.
        - directory: The directory where the file should be downloaded and unzipped.
        - folder_name: The name of the folder to create and unzip the contents into.
        - num_downl: The current download number (for progress display).
        - total_downloads: The total number of downloads (for progress display).

        Returns:
        - out_path: The path to the unzipped folder.
        """
        
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
                progress = QProgressDialog(
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
                        QApplication.processEvents()
                shutil.copyfileobj(response, out_file)

            # Unzip the file
            with zipfile.ZipFile(temp_path, "r") as zip:
                zip.extractall(out_path)

            # Delete the zip file
            os.remove(temp_path)

        return out_path
    
    def orientCBCT(self)->None:
        """
        This function is called when the button "pushButtonOrientCBCT" is click.
        Orient CBCT images using specified parameters and initiate the processing pipeline.

        This function sets up the parameters for CBCT image orientation, tests the process and scan,
        and starts the processing pipeline if all checks pass. It handles the initial setup,
        parameter passing, and process initiation, including setting up observers for process updates.
        """
        
        param = {"input_t1_folder":self.ui.LineEditCBCT.text,
                "folder_output":self.ui.lineEditOutputOrientCBCT.text,
                "model_folder_1":self.ui.lineEditSegCBCT.text,
                "merge_seg":False,
                "isDCMInput":False,
                "slicerDownload":self.SlicerDownloadPath}
        
        ok,mess = self.preprocess_cbct.TestProcess(**param) 
        if not ok : 
            self.showMessage(mess)
            return
        ok,mess = self.preprocess_cbct.TestScan(param["input_t1_folder"])
        if not ok : 
            self.showMessage(mess)
            return
        
        self.list_Processes_Parameters = self.preprocess_cbct.Process(**param)
        
        self.onProcessStarted()
        
        # /!\ Launch of the first process /!\
        print("module name : ",self.list_Processes_Parameters[0]["Module"])
        print("Parameters : ",self.list_Processes_Parameters[0]["Parameter"])
        
        self.process = slicer.cli.run(
                self.list_Processes_Parameters[0]["Process"],
                None,
                self.list_Processes_Parameters[0]["Parameter"],
            )
        
        self.module_name = self.list_Processes_Parameters[0]["Module"]
        self.processObserver = self.process.AddObserver(
            "ModifiedEvent", self.onProcessUpdate
        )

        del self.list_Processes_Parameters[0]
    
    def orientCenterMRI(self):
        """
        This function is called when the button "pushButtonOrientMRI" is click.
        Orient and center MRI images using specified parameters and initiate the processing pipeline.

        This function sets up the parameters for MRI image orientation and centering, tests the process and scan,
        and starts the processing pipeline if all checks pass. It handles the initial setup, parameter passing,
        and process initiation, including setting up observers for process updates.
        """
        
        param = {"input_folder":self.ui.LineEditMRI.text,
                "direction":self.getCheckboxValuesOrient(),
                "output_folder":self.ui.lineEditOutputOrientMRI.text}
        
        ok,mess = self.preprocess_mri.TestProcess(**param) 
        if not ok : 
            self.showMessage(mess)
            return
        ok,mess = self.preprocess_mri.TestScan(param["input_folder"])
        if not ok : 
            self.showMessage(mess)
            return
        
        self.list_Processes_Parameters = self.preprocess_mri.Process(**param)
        
        self.onProcessStarted()
        
        # /!\ Launch of the first process /!\
        print("module name : ",self.list_Processes_Parameters[0]["Module"])
        print("Parameters : ",self.list_Processes_Parameters[0]["Parameter"])
        
        self.process = slicer.cli.run(
                self.list_Processes_Parameters[0]["Process"],
                None,
                self.list_Processes_Parameters[0]["Parameter"],
            )
        
        self.module_name = self.list_Processes_Parameters[0]["Module"]
        self.processObserver = self.process.AddObserver(
            "ModifiedEvent", self.onProcessUpdate
        )

        del self.list_Processes_Parameters[0]
        
    def resampleMRICBCT(self):
        """
        Resample MRI and/or CBCT images based on the selected options and initiate the processing pipeline.

        This function determines which input folders (MRI, CBCT, or both) to use based on the user's selection
        in the comboBoxResample widget. It sets up the resampling parameters, tests the process and scans,
        and starts the processing pipeline if all checks pass. The function handles the initial setup, parameter
        passing, and process initiation, including setting up observers for process updates.
        """
    
        if self.ui.comboBoxResample.currentText=="CBCT":
            LineEditMRI = "None"
            LineEditCBCT = self.ui.LineEditCBCT.text
        elif self.ui.comboBoxResample.currentText=="MRI":
            LineEditMRI = self.ui.LineEditMRI.text
            LineEditCBCT = "None"
        else : 
            LineEditMRI = self.ui.LineEditMRI.text
            LineEditCBCT = self.ui.LineEditCBCT.text
            
        param = {"input_folder_MRI": LineEditMRI,
            "input_folder_CBCT": LineEditCBCT,
            "output_folder": self.ui.lineEditOuputResample.text,
            "resample_size": self.get_resample_values()[0],
            "spacing" : self.get_resample_values()[1]
        }
            
        ok,mess = self.preprocess_mri_cbct.TestProcess(**param) 
        if not ok : 
            self.showMessage(mess)
            return
        
        ok,mess = self.preprocess_mri_cbct.TestScan(param["input_folder_MRI"])

        if not ok : 
            mess = mess + "MRI folder"
            self.showMessage(mess)
            return
        
        ok,mess = self.preprocess_mri_cbct.TestScan(param["input_folder_CBCT"])
        if not ok : 
            mess = mess + "CBCT folder"
            self.showMessage(mess)
            return
            
            
        self.list_Processes_Parameters = self.preprocess_mri_cbct.Process(**param)
        
        self.onProcessStarted()
        
        # /!\ Launch of the first process /!\
        print("module name : ",self.list_Processes_Parameters[0]["Module"])
        print("Parameters : ",self.list_Processes_Parameters[0]["Parameter"])
        
        self.process = slicer.cli.run(
                self.list_Processes_Parameters[0]["Process"],
                None,
                self.list_Processes_Parameters[0]["Parameter"],
            )
        
        self.module_name = self.list_Processes_Parameters[0]["Module"]
        self.processObserver = self.process.AddObserver(
            "ModifiedEvent", self.onProcessUpdate
        )

        del self.list_Processes_Parameters[0]
        
        
    def registration_MR2CBCT(self) -> None:
        """
        Register MRI images to CBCT images using specified parameters and initiate the processing pipeline.

        This function sets up the parameters for MRI to CBCT registration, tests the process and scans,
        and starts the processing pipeline if all checks pass. It handles the initial setup, parameter passing,
        and process initiation, including setting up observers for process updates. The function also checks
        for normalization parameters and validates input folders for the presence of necessary files.
        """
        
        param = {"folder_general": self.ui.LineEditOutput.text,
            "mri_folder": self.ui.lineEditRegMRI.text,
            "cbct_folder": self.ui.lineEditRegCBCT.text,
            "cbct_label2": self.ui.lineEditRegLabel.text,
            "normalization" : [self.getNormalization()],
            "tempo_fold" : self.ui.checkBoxTompraryFold.isChecked()}
        
        ok,mess = self.registration_mri2cbct.TestProcess(**param) 
        if not ok : 
            self.showMessage(mess)
            return
        
        ok1,mess = self.registration_mri2cbct.TestScan(param["mri_folder"])
        ok2,mess2 = self.registration_mri2cbct.TestScan(param["cbct_folder"])
        ok3,mess3 = self.registration_mri2cbct.TestScan(param["cbct_label2"])
        
        error_messages = []

        if not ok1:
            error_messages.append("MRI folder")
        if not ok2:
            error_messages.append("CBCT folder")
        if not ok3:
            error_messages.append("CBCT label2 folder")

        if error_messages:
            error_message = "No files to run has been found in the following folders: " + ", ".join(error_messages)
            self.showMessage(error_message)
            return
        
        ok,mess = self.registration_mri2cbct.CheckNormalization(param["normalization"])
        if not ok : 
            self.showMessage(mess)
            return 
        
        self.list_Processes_Parameters = self.registration_mri2cbct.Process(**param)
        
        self.onProcessStarted()
        
        # /!\ Launch of the first process /!\
        print("module name : ",self.list_Processes_Parameters[0]["Module"])
        print("Parameters : ",self.list_Processes_Parameters[0]["Parameter"])
        
        self.process = slicer.cli.run(
                self.list_Processes_Parameters[0]["Process"],
                None,
                self.list_Processes_Parameters[0]["Parameter"],
            )
        
        self.module_name = self.list_Processes_Parameters[0]["Module"]
        self.processObserver = self.process.AddObserver(
            "ModifiedEvent", self.onProcessUpdate
        )

        del self.list_Processes_Parameters[0]
        
        
        
    def onProcessStarted(self):
        """
        Initialize and update the UI components when a process starts.

        This function sets the start time, initializes the progress bar and related UI elements,
        and updates the process-related attributes such as the number of extensions and modules.
        It also enables the running state UI to reflect that a process is in progress.
        """
        self.startTime = time.time()

        # self.ui.progressBar.setMaximum(self.nb_patient)
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setTextVisible(True)
        self.ui.progressBar.setFormat("0%")

        self.ui.label_info.setText(f"Starting process")
        
        self.nb_extnesion_did = 0
        self.nb_extension_launch = len(self.list_Processes_Parameters)

        self.module_name_before = 0
        self.nb_change_bystep = 0

        self.RunningUI(True)

    def onProcessUpdate(self, caller, event):
        """
        Update the UI components during the process execution and handle process completion.

        This function updates the progress bar, time label, and information label during the process execution.
        It handles the completion of each process step, manages errors, and initiates the next process if available.
        
        Parameters:
        - caller: The process that triggered the update.
        - event: The event that triggered the update.
        """
        
        self.ui.progressBar.setVisible(False)
        # timer = f"Time : {time.time()-self.startTime:.2f}s"
        currentTime = time.time() - self.startTime
        if currentTime < 60:
            timer = f"Time : {int(currentTime)}s"
        elif currentTime < 3600:
            timer = f"Time : {int(currentTime/60)}min and {int(currentTime%60)}s"
        else:
            timer = f"Time : {int(currentTime/3600)}h, {int(currentTime%3600/60)}min and {int(currentTime%60)}s"

        self.ui.label_time.setText(timer)
        # self.module_name = caller.GetModuleTitle() if self.module_name_bis is None else self.module_name_bis
        self.ui.label_info.setText(f"Extension {self.module_name} is running. \nNumber of extension runned : {self.nb_extnesion_did} / {self.nb_extension_launch}")
        # self.displayModule = self.displayModule_bis if self.displayModule_bis is not None else self.display[self.module_name.split(' ')[0]]

        if self.module_name_before != self.module_name:
            print("Valeur progress barre : ",100*self.nb_extnesion_did/self.nb_extension_launch)
            self.ui.progressBar.setValue(self.nb_extnesion_did/self.nb_extension_launch)
            self.ui.progressBar.setFormat(f"{100*self.nb_extnesion_did/self.nb_extension_launch}%")
            self.nb_extnesion_did += 1
            self.ui.label_info.setText(
                f"Extension {self.module_name} is running. \nNumber of extension runned : {self.nb_extnesion_did} / {self.nb_extension_launch}"
            )
            

            self.module_name_before = self.module_name
            self.nb_change_bystep = 0


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
                    print("name process : ",self.list_Processes_Parameters[0]["Process"])
                    self.process = slicer.cli.run(
                        self.list_Processes_Parameters[0]["Process"],
                        None,
                        self.list_Processes_Parameters[0]["Parameter"],
                    )
                    self.module_name = self.list_Processes_Parameters[0]["Module"]
                    self.processObserver = self.process.AddObserver(
                        "ModifiedEvent", self.onProcessUpdate
                    )
                    del self.list_Processes_Parameters[0]
                    # self.displayModule.progress = 0
                except IndexError:
                    self.OnEndProcess()

    def OnEndProcess(self):
        """
        Finalize the process execution and update the UI components accordingly.

        This function increments the number of completed extensions, updates the information label,
        resets the progress bar, calculates the total time taken, and displays a message box indicating
        the completion of the process. It also disables the running state UI.
        """
        
        self.nb_extnesion_did += 1
        self.ui.label_info.setText(
            f"Process end"
        )
        self.ui.progressBar.setValue(0)

        self.module_name_before = self.module_name
        self.nb_change_bystep = 0
        total_time = time.time() - self.startTime
        

        print("PROCESS DONE.")
        print(
            "Done in {} min and {} sec".format(
                int(total_time / 60), int(total_time % 60)
            )
        )

        self.RunningUI(False)

        stopTime = time.time()

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        # setting message for Message Box
        msg.setText(f"Processing completed in {int(total_time / 60)} min and {int(total_time % 60)} sec")

        # setting Message box window title
        msg.setWindowTitle("Information")

        # declaring buttons on Message Box
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
       
        
    def onCancel(self):
        self.process.Cancel()

        self.RunningUI(False)
        
    def RunningUI(self, run=False):

        self.ui.progressBar.setVisible(run)
        self.ui.label_time.setVisible(run)
        self.ui.label_info.setVisible(run)
        
    def showMessage(self,mess):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        # setting message for Message Box
        msg.setText(mess)

        # setting Message box window title
        msg.setWindowTitle("Information")

        # declaring buttons on Message Box
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


    
    


#
# MRI2CBCTLogic
#


class MRI2CBCTLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return MRI2CBCTParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
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
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# MRI2CBCTTest
#


class MRI2CBCTTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_MRI2CBCT1()

    def test_MRI2CBCT1(self):
        """Ideally you should have several levels of tests.  At the lowest level
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
        inputVolume = SampleData.downloadSample("MRI2CBCT1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = MRI2CBCTLogic()

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

        self.delayDisplay("Test passed")
