import logging
import os
import sys
import sys
from typing import Annotated, Optional
from qt import QApplication, QWidget, QTableWidget, QDoubleSpinBox, QTableWidgetItem, QHeaderView,QSpinBox, QVBoxLayout, QLabel, QSizePolicy, QCheckBox, QFileDialog,QMessageBox, QApplication, QProgressDialog
import qt
from MRI2CBCT_utils.Preprocess_MRI import Process_MRI
from MRI2CBCT_utils.Preprocess_CBCT_MRI import Preprocess_CBCT_MRI
from MRI2CBCT_utils.Reg_MRI2CBCT import Registration_MRI2CBCT
from MRI2CBCT_utils.Approx_MRI2CBCT import Approximation_MRI2CBCT
from MRI2CBCT_utils.LR_crop import LR_CROP_MRI2CBCT
from MRI2CBCT_utils.TMJ_crop import TMJ_CROP_MRI2CBCT

from SlicerNNUNetLib import InstallLogic

import time 



import slicer
from functools import partial
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

import shutil
import urllib
import zipfile
import importlib.metadata
from pathlib import Path

from packaging.version import Version
from packaging.specifiers import SpecifierSet
import platform
from slicer import vtkMRMLScalarVolumeNode

# def check_lib_installed(lib_name, required_version=None):
#     try:
#         installed_version = importlib.metadata.version(lib_name)
#         if required_version and installed_version != required_version:
#             return False
#         return True
#     except importlib.metadata.PackageNotFoundError:
#         return False

# # import csv
    
# def install_function():
#     libs = [('itk',None),('monai','0.7.0'),('einops',None),('dicom2nifti', '2.3.0'),('pydicom', '2.2.2'),('nibabel',None),('itk-elastix',None),('connected-components-3d','3.9.1'),("pandas",None),("scikit-learn",None),("torch",None),("torchreg",None),("SimpleITK",None)]
#     libs_to_install = []
#     for lib, version in libs:
#         if not check_lib_installed(lib, version):
#             libs_to_install.append((lib, version))

#     if libs_to_install:
#         message = "The following libraries are not installed or need updating:\n"
#         message += "\n".join([f"{lib}=={version}" if version else lib for lib, version in libs_to_install])
#         message += "\n\nDo you want to install/update these libraries?\n Doing it could break other modules"
#         user_choice = slicer.util.confirmYesNoDisplay(message)

#         if user_choice:
#             for lib, version in libs_to_install:
#                 lib_version = f'{lib}=={version}' if version else lib
#                 pip_install(lib_version)
#         else :
#           return False
#     import vtk
#     import itk
#     return True
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
        self.parent.dependencies = ["SlicerNNUNet"]  # TODO: add here list of module names that this module requires
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
        self.processWasCanceled = False
        self.observerTags = []
        self.all_installed = True
        
        

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

        qt.QTimer.singleShot(100, self.onCheckRequirements)
        
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documentsLocation)
        self.SlicerDownloadPath = os.path.join(
            self.documents,
            slicer.app.applicationName + "Downloads",
            "MRI2CBCT",
            "MRI2CBCT_" + "CBCT",
        )
        self.preprocess_mri = Process_MRI(self)
        self.preprocess_mri_cbct = Preprocess_CBCT_MRI(self)
        self.registration_mri2cbct = Registration_MRI2CBCT(self)
        self.approximate_mri2cbct = Approximation_MRI2CBCT(self)
        self.lr_crop_mri2cbct = LR_CROP_MRI2CBCT(self)
        self.tmj_crop_mri2cbct = TMJ_CROP_MRI2CBCT(self)

        # Connections
        #        LineEditOutputReg
        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        
        ### Frames ###
        self.ui.resampleCollapsibleButton.toggled.connect(partial(self.onCollapsibleToggled, "Resample"))
        self.ui.inputsCollapsibleButton.toggled.connect(partial(self.onCollapsibleToggled, "Inputs"))
        self.ui.approxCollapsibleButton.toggled.connect(partial(self.onCollapsibleToggled, "Approx"))
        
        
        
        ### Approximation ###
        self.ui.SearchButtonApproxCBCT.connect("clicked(bool)",partial(self.openFinder,"InputCBCTApprox"))
        self.ui.SearchButtonApproxMRI.connect("clicked(bool)",partial(self.openFinder,"InputMRIApprox"))
        self.ui.SearchButtonOutputApprox.connect("clicked(bool)",partial(self.openFinder,"OutputApprox"))
        self.ui.pushButtonApproximateMRI.connect("clicked(bool)", self.approximateMRI)
        
        
        
        ### L/R Cropping ###
        self.ui.SearchButtonSepCBCT.connect("clicked(bool)",partial(self.openFinder,"InputCBCTSep"))
        self.ui.SearchButtonSepMRI.connect("clicked(bool)",partial(self.openFinder,"InputMRISep"))
        self.ui.SearchButtonSepSeg.connect("clicked(bool)",partial(self.openFinder,"InputSegSep"))
        self.ui.SearchButtonSepOut.connect("clicked(bool)",partial(self.openFinder,"OutputSep"))
        self.ui.lineEditSepCBCT.textChanged.connect(self.updateSepLabel)
        self.ui.lineEditSepMRI.textChanged.connect(self.updateSepLabel)
        self.ui.lineEditSepSeg.textChanged.connect(self.updateSepLabel)
        self.ui.pushButtonCropLR.connect("clicked(bool)", self.lrCropMRI2CBCT)
        
        
        ### TMJ Cropping ###
        self.ui.SearchButtonTMJCBCT.connect("clicked(bool)",partial(self.openFinder,"InputCBCTTMJ"))
        self.ui.SearchButtonTMJMRI.connect("clicked(bool)",partial(self.openFinder,"InputMRITMJ"))
        self.ui.SearchButtonTMJSeg.connect("clicked(bool)",partial(self.openFinder,"InputSegTMJ"))
        self.ui.SearchButtonTMJOut.connect("clicked(bool)",partial(self.openFinder,"OutputTMJ"))
        self.ui.pushButtonDownloadModelTMJ.pressed.connect(
            lambda: self.downloadTMJModel(
                self.ui.lineEditTMJModel
            )
        )
        self.ui.pushButtonSearchModelTMJ.connect("clicked(bool)",partial(self.openFinder,"InputTMJModel"))
        self.ui.pushButtonCropTMJ.connect("clicked(bool)", self.tmjCropMRI2CBCT)
        
        
        
        ### Resampling ###
        self.ui.SearchButtonResampleCBCT.connect("clicked(bool)",partial(self.openFinder,"InputResampleCBCT"))
        self.ui.SearchButtonResampleMRI.connect("clicked(bool)",partial(self.openFinder,"InputResampleMRI"))
        self.ui.SearchButtonResampleSeg.connect("clicked(bool)",partial(self.openFinder,"InputResampleSeg"))
        self.ui.SearchButtonResampleT2CBCT.connect("clicked(bool)",partial(self.openFinder,"InputResampleT2CBCT"))
        self.ui.SearchButtonResampleT2MRI.connect("clicked(bool)",partial(self.openFinder,"InputResampleT2MRI"))
        self.ui.SearchButtonResampleT2Seg.connect("clicked(bool)",partial(self.openFinder,"InputResampleT2Seg"))
        self.ui.SearchOutputFolderResample.connect("clicked(bool)",partial(self.openFinder,"OutputOrientResample"))
        self.ui.lineEditResampleCBCT.textChanged.connect(self.updateResamplingLabel)
        self.ui.lineEditResampleT2CBCT.textChanged.connect(self.updateResamplingLabel)
        self.ui.lineEditResampleMRI.textChanged.connect(self.updateResamplingLabel)
        self.ui.lineEditResampleT2MRI.textChanged.connect(self.updateResamplingLabel)
        self.ui.lineEditResampleSeg.textChanged.connect(self.updateResamplingLabel)
        self.ui.lineEditResampleT2Seg.textChanged.connect(self.updateResamplingLabel)
        
        self.ui.CheckBoxT2CBCT.connect("clicked(bool)",self.toggleT2)
        self.ui.CheckBoxT2MRI.connect("clicked(bool)",self.toggleT2)
        self.ui.CheckBoxT2Seg.connect("clicked(bool)",self.toggleT2)
        
        self.ui.pushButtonResample.connect("clicked(bool)",self.resampleMRICBCT)
        
        
        
        ### MRI Orientation ###
        self.ui.comboBoxDICOMVolumes.currentIndexChanged.connect(self.onDICOMSelectionChanged)
        self.addSceneObservers()
        self.updateDICOMComboBox()
        self.ui.checkBoxBilateralMRI.connect("clicked(bool)",self.toggleBilateral)
        self.toggleBilateral()
        
        self.ui.SearchButtonMRI.connect("clicked(bool)",partial(self.openFinder,"InputMRI"))
        self.ui.SearchOutputFolderOrientMRI.connect("clicked(bool)",partial(self.openFinder,"OutputOrientMRI"))
        self.ui.pushButtonOrientMRI.connect("clicked(bool)",self.orientCenterMRI)
        
        
        
        
        ### Registration ###
        self.ui.SearchButtonOutput.connect("clicked(bool)",partial(self.openFinder,"OutputReg"))
        self.ui.pushButtonTestFilePreMRI.connect("clicked(bool)",partial(self.downloadModel,self.ui.LineEditMRI, "MRI2CBCT", True))
        self.ui.pushButtonTestFileRegMRI.connect("clicked(bool)",partial(self.downloadModel,self.ui.lineEditRegMRI, "MRI2CBCT", True))
        self.ui.pushButtonTestFileRegCBCT.connect("clicked(bool)",partial(self.downloadModel,self.ui.lineEditRegCBCT, "MRI2CBCT", True))
        self.ui.pushButtonTestFileRegSeg.connect("clicked(bool)",partial(self.downloadModel,self.ui.lineEditRegLabel, "MRI2CBCT", True))
        self.ui.SearchButtonRegMRI.connect("clicked(bool)",partial(self.openFinder,"InputRegMRI"))
        self.ui.SearchButtonRegCBCT.connect("clicked(bool)",partial(self.openFinder,"InputRegCBCT"))
        self.ui.SearchButtonRegLabel.connect("clicked(bool)",partial(self.openFinder,"InputRegLabel"))
        self.ui.registrationButton.connect("clicked(bool)", self.registration_MR2CBCT)
        
        
        
        
        self.ui.pushButtonCancelProcess.connect("clicked(bool)", self.onCancel)

        # Make sure parameter node is initialized (needed for module reload) 
        self.initializeParameterNode()
        self.ui.ComboBoxMRI.setCurrentIndex(1)
        self.ui.ComboBoxMRI.setEnabled(False)
        
        self.ui.labelT2CBCT.setVisible(False)
        self.ui.lineEditResampleT2CBCT.setVisible(False)
        self.ui.SearchButtonResampleT2CBCT.setVisible(False)
        self.ui.labelT2MRI.setVisible(False)
        self.ui.lineEditResampleT2MRI.setVisible(False)
        self.ui.SearchButtonResampleT2MRI.setVisible(False)
        self.ui.labelT2Seg.setVisible(False)
        self.ui.lineEditResampleT2Seg.setVisible(False)
        self.ui.SearchButtonResampleT2Seg.setVisible(False)
        self.ui.labelBarNotWorking.setVisible(False)
        
        self.ui.comboBoxRegMRI.setCurrentIndex(1)
        self.ui.comboBoxRegMRI.setEnabled(False)
        self.ui.comboBoxRegCBCT.setCurrentIndex(1)
        self.ui.comboBoxRegCBCT.setEnabled(False)
        self.ui.comboBoxRegLabel.setCurrentIndex(1)
        self.ui.comboBoxRegLabel.setEnabled(False)
        
        self.ui.label_time.setHidden(True)
        self.ui.label_info.setHidden(True)
        self.ui.progressBar.setHidden(True)
        
        self.ui.ComboBoxMRI.setHidden(True)
        self.ui.comboBoxRegMRI.setHidden(True)
        self.ui.comboBoxRegCBCT.setHidden(True)
        self.ui.comboBoxRegLabel.setHidden(True)
        
        self.ui.outputCollapsibleButton.setText("Registration")
        self.ui.inputsCollapsibleButton.setText("Preprocess")
        self.ui.approxCollapsibleButton.setText("Approximate")
        self.ui.resampleCollapsibleButton.setText("Resample")
        
        self.ui.outputCollapsibleButton.setChecked(True)  # True to expand, False to collapse
        self.ui.inputsCollapsibleButton.setChecked(False)
        self.ui.approxCollapsibleButton.setChecked(False)
        self.ui.resampleCollapsibleButton.setChecked(True)
        
        self.ui.pushButtonCancelProcess.setVisible(False)
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
        spinBox1.setValue(443)
        self.tableWidgetResample.setCellWidget(0, 0, spinBox1)

        spinBox2 = QSpinBox()
        spinBox2.setMaximum(10000)
        spinBox2.setValue(443)
        self.tableWidgetResample.setCellWidget(0, 1, spinBox2)

        spinBox3 = QSpinBox()
        spinBox3.setMaximum(10000)
        spinBox3.setValue(119)
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
    
    def addSceneObservers(self):
        # Remove any existing observers first
        self.removeSceneObservers()
        self.observerTags.append(slicer.mrmlScene.AddObserver(slicer.mrmlScene.NodeAddedEvent, self.onSceneNodeChanged))
        self.observerTags.append(slicer.mrmlScene.AddObserver(slicer.mrmlScene.NodeRemovedEvent, self.onSceneNodeChanged))
        
    def removeSceneObservers(self):
        # Remove all observers and clear the list
        for tag in self.observerTags:
            slicer.mrmlScene.RemoveObserver(tag)
        self.observerTags = []

    def onSceneNodeChanged(self, caller=None, event=None):
        self.updateDICOMComboBox()

    def updateDICOMComboBox(self):
        currentID = self.ui.comboBoxDICOMVolumes.itemData(self.ui.comboBoxDICOMVolumes.currentIndex) if self.ui.comboBoxDICOMVolumes.currentIndex > 0 else None
        self.ui.comboBoxDICOMVolumes.blockSignals(True)
        self.ui.comboBoxDICOMVolumes.clear()
        self.ui.comboBoxDICOMVolumes.addItem("Select DICOM node")
        
        volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        for volume in volumes:
            if volume.GetAttribute("DICOM.instanceUIDs"):
                self.ui.comboBoxDICOMVolumes.addItem(volume.GetName(), volume.GetID())
        
        if currentID:
            index = self.ui.comboBoxDICOMVolumes.findData(currentID)
            if index != -1:
                self.ui.comboBoxDICOMVolumes.setCurrentIndex(index)
        self.ui.comboBoxDICOMVolumes.blockSignals(False)

    def onDICOMSelectionChanged(self):
        if self.ui.comboBoxDICOMVolumes.currentIndex == 0:
            self.ui.labelDICOMSpacing.text = "Acquisition Spacing: None"
            return
        
        volumeID = self.ui.comboBoxDICOMVolumes.itemData(self.ui.comboBoxDICOMVolumes.currentIndex)
        volumeNode = slicer.mrmlScene.GetNodeByID(volumeID)
        if not volumeNode:
            return
        
        instanceUIDs = volumeNode.GetAttribute("DICOM.instanceUIDs").split()
        if not instanceUIDs:
            self.ui.labelDICOMSpacing.text = "Acquisition Spacing: N/A (No DICOM metadata)"
            return
        
        firstInstanceUID = instanceUIDs[0]
        db = slicer.dicomDatabase
        if not db:
            self.ui.labelDICOMSpacing.text = "DICOM database not available"
            return
        
        # Get spacing values from DICOM tags
        spacing = db.instanceValue(firstInstanceUID, "0018,0088")  # Spacing Between Slices
        if not spacing:
            spacing = db.instanceValue(firstInstanceUID, "0018,0050")  # Slice Thickness
        
        if spacing:
            self.ui.labelDICOMSpacing.text = f"Acquisition Spacing: {float(spacing):.2f} mm"
        else:
            self.ui.labelDICOMSpacing.text = "Acquisition Spacing: Not found in DICOM tags"

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
                
    def onCollapsibleToggled(self, name: str, expanded: bool) -> None:
        if name == "Resample":
            self.ui.frame.setVisible(expanded)
        elif name == "Inputs":
            self.ui.frame_4.setVisible(expanded)
        elif name == "Approx":
            self.ui.frame_5.setVisible(expanded)
            
    def downloadTMJModel(self, lineEdit):
        """
        Download a model from the Slicer Models repository and set the path in the specified line edit.

        Parameters:
        - lineEdit: The QLineEdit widget where the model path will be set.
        """
        foldPath, is_installed = self.install_nnunet()
        if is_installed:
            self.ui.lineEditTMJModel.setText(foldPath)
        else:
            slicer.util.errorDisplay("Failed to download TMJ model.")
            
    def download_file_with_progress(self, url, dest_path, label="Downloading..."):
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
            progress = qt.QProgressDialog(
                f"{label}: {dest_path.name}", "Cancel", 0, 100, self.parent)
            progress.setCancelButton(None)
            progress.setWindowModality(qt.Qt.WindowModal)
            progress.setWindowTitle("Downloading model file...")
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
            else:
                out_file.write(response.read())  # No progress bar fallback
                
    def install_nnunet(self) -> bool:
        # Set up base and fold paths
        basePath = Path(self.SlicerDownloadPath).joinpath("ML", "Dataset001_myseg", "nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres").resolve()
        foldPath = basePath.joinpath("fold_0")
        foldPath.mkdir(parents=True, exist_ok=True)

        # Define destination paths
        checkpoint_path = foldPath.joinpath("checkpoint_final.pth")
        dataset_json_path = basePath.joinpath("dataset.json")
        plans_json_path = basePath.joinpath("plans.json")

        # Define URLs
        url_checkpoint = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/TMJ_CROP_MODEL/checkpoint_final.pth"
        url_dataset = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/TMJ_CROP_MODEL/dataset.json"
        url_plans = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/TMJ_CROP_MODEL/plans.json"

        # Download files if missing
        if not checkpoint_path.exists():
            self.download_file_with_progress(url_checkpoint, checkpoint_path, label="Downloading checkpoint")

        if not dataset_json_path.exists():
            self.download_file_with_progress(url_dataset, dataset_json_path, label="Downloading dataset.json")

        if not plans_json_path.exists():
            self.download_file_with_progress(url_plans, plans_json_path, label="Downloading plans.json")

        # If everything exists, return True
        return basePath, (checkpoint_path.exists() and dataset_json_path.exists() and plans_json_path.exists())

    def openFinder(self,nom : str,_) -> None : 
        """
         Open finder to let the user choose is files or folder
        """
        if nom=="InputMRI":
            if self.ui.ComboBoxMRI.currentIndex==1:
                  surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            else :
                  surface_folder = QFileDialog.getOpenFileName(self.parent,'Open a file',)

            self.ui.LineEditMRI.setText(surface_folder)
            
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
            
        elif nom=="InputResampleCBCT":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditResampleCBCT.setText(surface_folder)
            
        elif nom=="InputResampleT2CBCT":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditResampleT2CBCT.setText(surface_folder)
            
        elif nom=="InputResampleMRI":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditResampleMRI.setText(surface_folder)
            
        elif nom=="InputResampleT2MRI":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditResampleT2MRI.setText(surface_folder)
            
        elif nom=="InputResampleSeg":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditResampleSeg.setText(surface_folder)
            
        elif nom=="InputResampleT2Seg":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditResampleT2Seg.setText(surface_folder)
            
        elif nom=="OutputOrientMRI":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditOutputOrientMRI.setText(surface_folder)
            
        elif nom=="OutputOrientResample":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditOuputResample.setText(surface_folder)
            
        elif nom=="OutputReg":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.LineEditOutput.setText(surface_folder)
            
        elif nom=="InputCBCTApprox":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditApproxCBCT.setText(surface_folder)
            
        elif nom=="InputMRIApprox":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditApproxMRI.setText(surface_folder)
            
        elif nom=="OutputApprox":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditOutputApprox.setText(surface_folder)
            
        elif nom=="InputCBCTSep":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditSepCBCT.setText(surface_folder)
        
        elif nom=="InputMRISep":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditSepMRI.setText(surface_folder)
            
        elif nom=="InputSegSep":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditSepSeg.setText(surface_folder)
        
        elif nom=="OutputSep":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditSepOut.setText(surface_folder)
            
        elif nom=="InputCBCTTMJ":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditCropTMJCBCT.setText(surface_folder)
        
        elif nom=="InputMRITMJ":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditCropTMJMRI.setText(surface_folder)
            
        elif nom=="InputSegTMJ":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditCropTMJSeg.setText(surface_folder)
            
        elif nom=="OutputTMJ":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
            self.ui.lineEditCropTMJOut.setText(surface_folder)

        elif nom=="InputTMJModel":
            surface_folder = QFileDialog.getExistingDirectory(self.parent, "Select a model folder")
            self.ui.lineEditTMJModel.setText(surface_folder)
        
        
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
        install_function()

        # # To select the reference files (CBCT Orientation and Registration mode only)
        # if name=="Segmentation" or name=="Orientation" :
        #     listmodel = self.preprocess_cbct.getModelUrl()
        #     print("listmodel : ",listmodel)

        #     urls = listmodel[name]
        #     if isinstance(urls, str):
        #         url = urls
        #         _ = self.DownloadUnzip(
        #             url=url,
        #             directory=os.path.join(self.SlicerDownloadPath),
        #             folder_name=os.path.join("Models", name),
        #             num_downl=1,
        #             total_downloads=1,
        #         )
        #         model_folder = os.path.join(self.SlicerDownloadPath, "Models", name)

        #     elif isinstance(urls, dict):
        #         for i, (name_bis, url) in enumerate(urls.items()):
        #             _ = self.DownloadUnzip(
        #                 url=url,
        #                 directory=os.path.join(self.SlicerDownloadPath),
        #                 folder_name=os.path.join("Models", name, name_bis),
        #                 num_downl=i + 1,
        #                 total_downloads=len(urls),
        #             )
        #         model_folder = os.path.join(self.SlicerDownloadPath, "Models", name)

        #     if not model_folder == "":
        #         error = self.preprocess_cbct.TestModel(model_folder, lineEdit.name)

        #         if isinstance(error, str):
        #             QMessageBox.warning(self.parent, "Warning", error)

        #         else:
        #             lineEdit.setText(model_folder)
                    
        # elif name == "MeanCBCT" or name == "ROI":
        #     listmodel = self.approximate_mri2cbct.getModelUrl()
        #     print("listmodel : ", listmodel)

        #     urls = listmodel[name]
        #     if isinstance(urls, str):
        #         url = urls
        #         _ = self.DownloadUnzip(
        #             url=url,
        #             directory=os.path.join(self.SlicerDownloadPath),
        #             folder_name=os.path.join("Models", name),
        #             num_downl=1,
        #             total_downloads=1,
        #         )
        #         model_folder = os.path.join(self.SlicerDownloadPath, "Models", name)
               
        # else :
        url = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/test_files/TestFile.zip"

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
        
        scan_folder = os.path.join(scan_folder,"TestFile")
        print("scan folder : ",scan_folder)
        print("name : ",name)
        if lineEdit.objectName=="LineEditMRI":
            lineEdit.setText(os.path.join(scan_folder,"MRI_ori"))
        elif lineEdit.objectName=="lineEditRegMRI":
            lineEdit.setText(os.path.join(scan_folder,"REG","MRI"))
        elif lineEdit.objectName=="lineEditRegCBCT":
            lineEdit.setText(os.path.join(scan_folder,"REG","CBCT"))
        elif lineEdit.objectName=="lineEditRegLabel":
            lineEdit.setText(os.path.join(scan_folder,"REG","Seg"))

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
    
    def run_conda_tool(self):
        module=self.list_Processes_Parameters[0]['Module']
        print(f"in conda tool: {module} wants to run", )

        args = self.list_Processes_Parameters[0]["Parameter"]
        self.logic.check_cli_script(f"{module}")

        conda_exe = self.logic.conda.getCondaExecutable()
        command = [conda_exe, "run", "-n", self.logic.name_env, "python" ,"-m", f"{module}"]

        for key, value in args.items():
            print("key : ",key)
            if isinstance(value, str) and ("\\" in value or (len(value) > 1 and value[1] == ":")):
                value = self.logic.windows_to_linux_path(value)
            command.append(f"\"{value}\"")
        print("command : ",command)
        return command

        # running in // to not block Slicer
        process = threading.Thread(target=self.logic.conda.condaRunCommand, args=(command,))
        process.start()
        self.ui.LabelTimer.setHidden(False)
        self.ui.LabelTimer.setText(f"Time : 0.00s")
        previous_time = self.startTime
        while process.is_alive():
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
                
                self.ui.LabelTimer.setText(timer)

        del self.list_Processes_Parameters[0]



    def tmjCropMRI2CBCT(self)->None:
        """
        This function is called when the button "pushButtonCropTMJ" is clicked.
        It crops the TMJ on both the Right and Left of MRI, CBCT, and segmentation images and initiates the processing pipeline.
        This function sets up the parameters for MRI and CBCT separation, tests the process and scan,
        and starts the processing pipeline if all checks pass. It handles the initial setup,
        parameter passing, and process initiation, including setting up observers for process updates.
        """
        
        install_function()
        # LinEditMRISep = "None"
        # LinEditCBCTSep = "None"
        
        # if self.ui.lineEditSepMRI.text != "":
        #     LinEditMRISep = self.ui.lineEditSepMRI.text
        # if self.ui.lineEditSepCBCT.text != "":
        #     LinEditCBCTSep = self.ui.lineEditSepCBCT.text
        param = {
            "cbct_folder": self.ui.lineEditCropTMJCBCT.text,
            "mri_folder": self.ui.lineEditCropTMJMRI.text,
            "seg_folder": self.ui.lineEditCropTMJSeg.text,
            "output_folder": self.ui.lineEditCropTMJOut.text,
            "model_folder": self.ui.lineEditTMJModel.text,
        }
        
        ok,mess = self.tmj_crop_mri2cbct.TestProcess(**param)
        if not ok :
            self.showMessage(mess)
            return
        
        ok,mess = self.tmj_crop_mri2cbct.TestScan(param["cbct_folder"])
        if not ok :
            self.showMessage(mess)
            return
        ok,mess = self.tmj_crop_mri2cbct.TestScan(param["mri_folder"])
        if not ok :
            self.showMessage(mess)
            return
        ok,mess = self.tmj_crop_mri2cbct.TestScan(param["seg_folder"])
        if not ok :
            self.showMessage(mess)
            return
        ok,mess = self.tmj_crop_mri2cbct.TestModel(param["model_folder"])
        if not ok :
            self.showMessage(mess)
            return
        
        self.list_Processes_Parameters = self.tmj_crop_mri2cbct.Process(**param)
        
        self.onProcessStarted()
        
        # /!\ Launch of the first process /!\
        print("module name : ",self.list_Processes_Parameters[0]["Module"])
        print("Parameters TMJ Crop: ",self.list_Processes_Parameters[0]["Parameter"])
        
        self.process = self.run_conda_tool()

    def lrCropMRI2CBCT(self)->None:
        """
        This function is called when the button "pushButtonCropLR" is clicked.
        It crops the Right and Left of MRI and CBCT images and initiates the processing pipeline.
        This function sets up the parameters for MRI and CBCT separation, tests the process and scan,
        and starts the processing pipeline if all checks pass. It handles the initial setup,
        parameter passing, and process initiation, including setting up observers for process updates.
        """
        
        install_function()
        LinEditMRISep = "None"
        LinEditCBCTSep = "None"
        LineEditSegSep = "None"
        
        if self.ui.lineEditSepMRI.text != "":
            LinEditMRISep = self.ui.lineEditSepMRI.text
        if self.ui.lineEditSepCBCT.text != "":
            LinEditCBCTSep = self.ui.lineEditSepCBCT.text
        if self.ui.lineEditSepSeg.text != "":
            LineEditSegSep = self.ui.lineEditSepSeg.text
            
        param = {
            "input_folder_CBCT": LinEditCBCTSep,
            "input_folder_MRI": LinEditMRISep,
            "input_folder_Seg": LineEditSegSep,
            "output_folder": self.ui.lineEditSepOut.text,
        }
        
        ok,mess = self.lr_crop_mri2cbct.TestProcess(**param)
        if not ok :
            self.showMessage(mess)
            return
        
        ok,mess = self.lr_crop_mri2cbct.TestScan(param["input_folder_CBCT"])
        if not ok :
            self.showMessage(mess)
            return
        ok,mess = self.lr_crop_mri2cbct.TestScan(param["input_folder_MRI"])
        if not ok :
            self.showMessage(mess)
            return
        ok,mess = self.lr_crop_mri2cbct.TestScan(param["input_folder_Seg"])
        if not ok :
            self.showMessage(mess)
            return
        
        self.list_Processes_Parameters = self.lr_crop_mri2cbct.Process(**param)
        
        self.onProcessStarted()
        
        # /!\ Launch of the first process /!\
        print("module name : ",self.list_Processes_Parameters[0]["Module"])
        print("Parameters : ",self.list_Processes_Parameters[0]["Parameter"])
        
        self.process = self.run_conda_tool()
    
    def orientCenterMRI(self):
        """
        This function is called when the button "pushButtonOrientMRI" is click.
        Orient and center MRI images using specified parameters and initiate the processing pipeline.

        This function sets up the parameters for MRI image orientation and centering, tests the process and scan,
        and starts the processing pipeline if all checks pass. It handles the initial setup, parameter passing,
        and process initiation, including setting up observers for process updates.
        """
        if self.ui.checkBoxBilateralMRI.isChecked():
            z_spacing = self.ui.AcquisitionSpacing.value
        else:
            z_spacing = "None"
            
        param = {
            "input_folder":self.ui.LineEditMRI.text,
            "direction":self.getCheckboxValuesOrient(),
            "output_folder":self.ui.lineEditOutputOrientMRI.text,
            "acquisition_z_spacing": z_spacing,
        }
        
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
        
        self.process = self.run_conda_tool()
        
    def resampleMRICBCT(self):
        """
        Resample MRI and/or CBCT images based on the selected options and initiate the processing pipeline.

        This function determines which input folders (MRI, CBCT, or both) to use based on the user's selection
        in the comboBoxResample widget. It sets up the resampling parameters, tests the process and scans,
        and starts the processing pipeline if all checks pass. The function handles the initial setup, parameter
        passing, and process initiation, including setting up observers for process updates.
        """
        LineEditMRI = "None"
        LineEditT2MRI = "None"
        LineEditCBCT = "None"
        LineEditT2CBCT = "None"
        LineEditSeg = "None"
        LineEditT2Seg = "None"
        if self.ui.lineEditResampleMRI.text != "":
            LineEditMRI = self.ui.lineEditResampleMRI.text
        if self.ui.lineEditResampleT2MRI.text != "" and self.ui.CheckBoxT2MRI.isChecked():
            LineEditT2MRI = self.ui.lineEditResampleT2MRI.text
        if self.ui.lineEditResampleCBCT.text != "":
            LineEditCBCT = self.ui.lineEditResampleCBCT.text
        if self.ui.lineEditResampleT2CBCT.text != "" and self.ui.CheckBoxT2CBCT.isChecked():
            LineEditT2CBCT = self.ui.lineEditResampleT2CBCT.text
        if self.ui.lineEditResampleSeg.text != "":
            LineEditSeg = self.ui.lineEditResampleSeg.text
        if self.ui.lineEditResampleT2Seg.text != "" and self.ui.CheckBoxT2Seg.isChecked():
            LineEditT2Seg = self.ui.lineEditResampleT2Seg.text
            
        param = {"input_folder_MRI": LineEditMRI,
            "input_folder_T2_MRI": LineEditT2MRI,
            "input_folder_CBCT": LineEditCBCT,
            "input_folder_T2_CBCT": LineEditT2CBCT,
            "input_folder_Seg": LineEditSeg,
            "input_folder_T2_Seg": LineEditT2Seg,
            "output_folder": self.ui.lineEditOuputResample.text,
            "resample_size": self.get_resample_values()[0],
            "spacing": self.get_resample_values()[1],
            "center": str(self.ui.checkBoxCenterImage.isChecked()),
        }
            
        ok,mess = self.preprocess_mri_cbct.TestProcess(**param) 
        if not ok : 
            self.showMessage(mess)
            return
        
        ok,mess = self.preprocess_mri_cbct.TestScan(param["input_folder_MRI"])
        if not ok : 
            if self.ui.CheckBoxT2MRI.isChecked():
                mess = mess + "MRI T1 folder"
            else:
                mess = mess + "MRI folder"
            self.showMessage(mess)
            return
        
        ok,mess = self.preprocess_mri_cbct.TestScan(param["input_folder_T2_MRI"])
        if not ok :
            mess = mess + "MRI T2 folder"
            self.showMessage(mess)
            return
        
        ok,mess = self.preprocess_mri_cbct.TestScan(param["input_folder_CBCT"])
        if not ok : 
            if self.ui.CheckBoxT2CBCT.isChecked():
                mess = mess + "CBCT T1 folder"
            else:
                mess = mess + "CBCT folder"
            self.showMessage(mess)
            return
        
        ok,mess = self.preprocess_mri_cbct.TestScan(param["input_folder_T2_CBCT"])
        if not ok :
            mess = mess + "CBCT T2 folder"
            self.showMessage(mess)
            return
            
        ok,mess = self.preprocess_mri_cbct.TestScan(param["input_folder_Seg"])
        if not ok :
            if self.ui.CheckBoxT2Seg.isChecked():
                mess = mess + "Seg T1 folder"
            else: 
                mess = mess + "Seg folder"
            self.showMessage(mess)
            return

        ok,mess = self.preprocess_mri_cbct.TestScan(param["input_folder_T2_Seg"])
        if not ok :
            mess = mess + "Seg T2 folder"
            self.showMessage(mess)
            return
        
            
        self.list_Processes_Parameters = self.preprocess_mri_cbct.Process(**param)
        
        self.onProcessStarted()
        
        # /!\ Launch of the first process /!\
        print("module name : ",self.list_Processes_Parameters[0]["Module"])
        print("Parameters : ",self.list_Processes_Parameters[0]["Parameter"])
        
        self.process = self.run_conda_tool()
        
    def updateSepLabel(self):
        """
        Updates the 'labelCropLR' text dynamically based on which input folders are set.
        """
        selected = []
        if self.ui.lineEditSepCBCT.text.strip():
            selected.append("CBCT")
        if self.ui.lineEditSepMRI.text.strip():
            selected.append("MRI")
        if self.ui.lineEditSepSeg.text.strip():
            selected.append("Seg")
                
        if selected:
            self.ui.labelCropLR.setText(f"<b>Running cropping for: {', '.join(selected)}</b>")
        else:
            self.ui.labelCropLR.setText("No folder selected")
        
        
    def updateResamplingLabel(self):
        """
        Updates the 'labelResampling' text dynamically based on which input folders are set.
        """
        selected = []
        if self.ui.lineEditResampleCBCT.text.strip():
            if self.ui.lineEditResampleT2CBCT.text.strip():
                selected.append("CBCT T1&T2")
            else:
                selected.append("CBCT")
        if self.ui.lineEditResampleMRI.text.strip():
            if self.ui.lineEditResampleT2MRI.text.strip():
                selected.append("MRI T1&T2")
            else:
                selected.append("MRI")
                
        if self.ui.lineEditResampleSeg.text.strip():
            if self.ui.lineEditResampleT2Seg.text.strip():
                selected.append("Seg T1&T2")
            else:
                selected.append("Seg")
                
        if selected:
            self.ui.labelResampling.setText(f"<b>Running resampling for: {', '.join(selected)}</b>")
        else:
            self.ui.labelResampling.setText("No resampling selected")

    def toggleT2(self):
        if self.ui.CheckBoxT2CBCT.text == "T1 and T2 CBCT":
            is_visible = False
            if self.ui.CheckBoxT2CBCT.isChecked():
                is_visible = True
                self.ui.labelT1CBCT.setText("Input CBCT T1 folder:")
            else:
                self.ui.lineEditResampleT2CBCT.setText("")
                self.ui.labelT1CBCT.setText("Input CBCT folder:")
                
            self.ui.labelT2CBCT.setVisible(is_visible)
            self.ui.lineEditResampleT2CBCT.setVisible(is_visible)
            self.ui.SearchButtonResampleT2CBCT.setVisible(is_visible)
            
        if self.ui.CheckBoxT2MRI.text == "T1 and T2 MRI":
            is_visible = False
            if self.ui.CheckBoxT2MRI.isChecked():
                is_visible = True
                self.ui.labelT1MRI.setText("Input MRI T1 folder:")
            else:
                self.ui.lineEditResampleT2MRI.setText("")
                self.ui.labelT1MRI.setText("Input MRI folder:")
                
            self.ui.labelT2MRI.setVisible(is_visible)
            self.ui.lineEditResampleT2MRI.setVisible(is_visible)
            self.ui.SearchButtonResampleT2MRI.setVisible(is_visible)
            
        if self.ui.CheckBoxT2Seg.text == "T1 and T2 Seg":
            is_visible = False
            if self.ui.CheckBoxT2Seg.isChecked():
                is_visible = True
                self.ui.labelT1Seg.setText("Input Seg T1 folder:")
            else:
                self.ui.lineEditResampleT2Seg.setText("")
                self.ui.labelT1Seg.setText("Input Seg folder:")
                
            self.ui.labelT2Seg.setVisible(is_visible)
            self.ui.lineEditResampleT2Seg.setVisible(is_visible)
            self.ui.SearchButtonResampleT2Seg.setVisible(is_visible)
            
    def toggleBilateral(self):
        """
        Toggle the visibility of the bilateral resampling options based on the state of the checkbox.
        """
        is_visible = self.ui.checkBoxBilateralMRI.isChecked()
        self.ui.comboBoxDICOMVolumes.setVisible(is_visible)
        self.ui.labelDICOMSpacing.setVisible(is_visible)
        self.ui.labelOutSpacing.setVisible(is_visible)
        self.ui.AcquisitionSpacing.setVisible(is_visible)
        self.ui.labelmm.setVisible(is_visible)
        
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
        print()
        print()
        
        self.process = self.run_conda_tool()
        
    def approximateMRI(self) -> None:
        """
        Approximates MRI images to CBCT images using specified parameters and initiate the processing pipeline.

        This function sets up the parameters for MRI to CBCT registration, tests the process and scans,
        and starts the processing pipeline if all checks pass. It handles the initial setup, parameter passing,
        and process initiation, including setting up observers for process updates. The function also checks
        for normalization parameters and validates input folders for the presence of necessary files.
        """
        
        param = {"cbct_folder": self.ui.lineEditApproxCBCT.text,
            "mri_folder": self.ui.lineEditApproxMRI.text,
            "output_folder" : self.ui.lineEditOutputApprox.text}
        
        ok,mess = self.approximate_mri2cbct.TestProcess(**param) 
        if not ok : 
            self.showMessage(mess)
            return
        
        ok1,mess = self.approximate_mri2cbct.TestScan(param["cbct_folder"])
        ok2,mess2 = self.approximate_mri2cbct.TestScan(param["mri_folder"])
        
        error_messages = []

        if not ok1:
            error_messages.append("CBCT folder")
        if not ok2:
            error_messages.append("MRI folder")

        if error_messages:
            error_message = "No files to run has been found in the following folders: " + ", ".join(error_messages)
            self.showMessage(error_message)
            return
        
        self.list_Processes_Parameters = self.approximate_mri2cbct.Process(**param)
        
        self.onProcessStarted()
        
        # /!\ Launch of the first process /!\
        print("module name : ",self.list_Processes_Parameters[0]["Module"])
        print("Parameters : ",self.list_Processes_Parameters[0]["Parameter"])
        
        self.process = self.run_conda_tool()
        
    def onProcessStarted(self):
        """
        Initialize and update the UI components when a process starts.

        This function sets the start time, initializes the progress bar and related UI elements,
        and updates the process-related attributes such as the number of extensions and modules.
        It also enables the running state UI to reflect that a process is in progress.
        """
        self.startTime = time.time()

        self.ui.progressBar.setHidden(False)
        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setMaximum(100)
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setTextVisible(True)
        self.ui.progressBar.setFormat("%p%")

        self.ui.label_info.setHidden(False)
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
        
        if not self.processWasCanceled:
            self.ui.pushButtonCancelProcess.setVisible(True)
        
        currentTime = time.time() - self.startTime
        if currentTime < 60:
            timer = f"Time: {int(currentTime)}s"
        elif currentTime < 3600:
            timer = f"Time: {int(currentTime/60)}min and {int(currentTime%60)}s"
        else:
            timer = f"Time: {int(currentTime/3600)}h, {int(currentTime%3600/60)}min and {int(currentTime%60)}s"

        self.ui.label_time.setText(timer)
        # self.module_name = caller.GetModuleTitle() if self.module_name_bis is None else self.module_name_bis
        self.ui.label_info.setText(f"Extension {self.module_name} is running. \nNumber of extension runned: {self.nb_extnesion_did} / {self.nb_extension_launch}")
        # self.displayModule = self.displayModule_bis if self.displayModule_bis is not None else self.display[self.module_name.split(' ')[0]]
        
        progress_value = caller.GetProgress()
        self.ui.progressBar.setValue(progress_value)
        self.ui.progressBar.setFormat(f"{progress_value}%")
        
        if self.module_name_before != self.module_name:
            self.nb_extnesion_did += 1
            self.module_name_before = self.module_name
            self.nb_change_bystep = 0

        if caller.GetStatus() & caller.Completed:
            self.ui.pushButtonCancelProcess.setVisible(False)
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
                    self.process = self.run_conda_tool()
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
        self.processWasCanceled = True
        self.process.Cancel()
        print("\n\n ========= PROCESS CANCELED ========= \n")
        self.ui.label_info.setText("Process was canceled.")
        self.RunningUI(False)
        
    def RunningUI(self, run=False):

        self.ui.progressBar.setVisible(run)
        self.ui.label_time.setVisible(run)
        self.ui.label_info.setVisible(run)
        self.ui.pushButtonCancelProcess.setVisible(run)
        self.ui.labelBarNotWorking.setVisible(False)
        
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
            self.ui.label_time.setHidden(False)
            self.ui.label_time.setText(f"Checking if wsl is installed, this task may take a moments")
            
            if self.logic.testWslAvailable():
                self.ui.label_time.setText(f"WSL installed")
                if not self.logic.check_lib_wsl():
                    self.ui.label_time.setText(f"Checking if the required librairies are installed, this task may take a moments")
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
        self.ui.label_time.setText(f"Checking if miniconda is installed")
        if "Error" in self.logic.conda.condaRunCommand([self.logic.conda.getCondaExecutable(),"--version"]):
            messageBox = qt.QMessageBox()
            text = textwrap.dedent("""
            Code can't be launch. \nConda is not setup. 
            Please go the extension CondaSetUp in SlicerConda to do it.""").strip()
            messageBox.information(None, "Information", text)
            return False
        
        
        ## shapeAXI
        self.ui.label_time.setText(f"Checking if environnement exists")
        if not self.logic.conda.condaTestEnv(self.logic.name_env) : # check is environnement exist, if not ask user the permission to do it
            userResponse = slicer.util.confirmYesNoDisplay("The environnement to run the classification doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
            if userResponse :
                start_time = time.time()
                previous_time = start_time
                formatted_time = self.format_time(0)
                self.ui.label_time.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
                process = self.logic.install_shapeaxi()
                
                while self.logic.process.is_alive():
                    slicer.app.processEvents()
                    formatted_time = self.update_ui_time(start_time, previous_time)
                    self.ui.label_time.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
            
                start_time = time.time()
                previous_time = start_time
                formatted_time = self.format_time(0)
                text = textwrap.dedent(f"""
                Installation of librairies into the new environnement. 
                This task may take a few minutes.\ntime: {formatted_time}""").strip()
                self.ui.label_time.setText(text)
            else:
                return False
        else:
            self.ui.label_time.setText(f"Ennvironnement already exists")
            
    


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
        self.isCondaSetUp = False
        self.conda = self.init_conda()
        self.name_env = "shapeaxi"
        self.cliNode = None
        
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
        print("output CHECK python path: ", result)
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
        print("output GIVE python path: ", results)
        
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
    
    def check_cli_script(self,file):
        if not self.check_pythonpath_windows(f"{file}"): 
            self.give_pythonpath_windows()
            results = self.check_pythonpath_windows(f"{file}")
                        
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