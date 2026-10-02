import os
import sys
from typing import Annotated
from qt import QDoubleSpinBox, QHeaderView,QSpinBox, QCheckBox, QFileDialog,QMessageBox
import qt
# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
# This has to run before the first import of anything local, not just before
# the ADTLib ones: ALI reaches ADTLib through ALI_Method.IOS.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

from ADTLib.logging_setup import get_logger
from ADTLib.env.deps import check_lib_installed as lib_satisfies
from ADTLib.env.cuda import torch_install_arguments

from MRI2CBCT_utils.Preprocess_MRI import Process_MRI
from MRI2CBCT_utils.Preprocess_CBCT_MRI import Preprocess_CBCT_MRI
from MRI2CBCT_utils.Reg_MRI2CBCT import Registration_MRI2CBCT
from MRI2CBCT_utils.Approx_MRI2CBCT import Approximation_MRI2CBCT
from MRI2CBCT_utils.ManualApprox_MRI2CBCT import ManualApproximation_MRI2CBCT
from MRI2CBCT_utils.LR_crop import LR_CROP_MRI2CBCT
from MRI2CBCT_utils.TMJ_crop import TMJ_CROP_MRI2CBCT

import time

import slicer
from functools import partial
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

import urllib
from pathlib import Path


from slicer import vtkMRMLScalarVolumeNode

from ADTLib.theming import apply_dark_mode, update_line_edit_and_combo_box
from ADTLib import testdata
from ADTLib.model_registry import (
    MRI2CBCT_TEST_FILES,
    SLICER_TESTING_DATA,
    TMJ_CROP_MODEL,
)

# ===== Logging Configuration =====
logger = get_logger("MRI2CBCT")

#: The folder, under the Slicer downloads, where the test set is dropped,
#: and the one where the test outputs are written.
TEST_FILES_DIRECTORY = "MRI2CBCT_TestFiles"
TEST_OUTPUT_DIRECTORY = "MRI2CBCT_TestFiles_output"

#: What the published test set feeds, step by step.
#:
#: `inputs` gives, for each input field of the step, the subfolder of the
#: archive that fills it; `output` gives the output field and the name of the
#: folder it receives; `model` names the model field, when the step asks for
#: one. The step is chosen by the button, not guessed by comparing
#: `objectName`: adding a field is done here, in a single place.
#:
#: The archive does not carry everything: it has neither an original
#: segmentation -- its own is already preprocessed -- nor a second timepoint,
#: so `lineEditResampleSeg` and the three T2 fields of the resampling stay
#: empty. Neither is required: `resampleMRICBCT` passes "None" to the CLI for
#: an empty field.
TEST_FILE_STEPS = {
    "Resample": {
        "inputs": (("lineEditResampleMRI", ("MRI_ori",)),
                   ("lineEditResampleCBCT", ("CBCT_ori",))),
        "output": ("lineEditOuputResample", "Resample"),
    },
    "Orient": {
        "inputs": (("LineEditMRI", ("MRI_ori",)),),
        "output": ("lineEditOutputOrientMRI", "Orient"),
    },
    "LRCrop": {
        "inputs": (("lineEditSepMRI", ("REG", "MRI")),
                   ("lineEditSepCBCT", ("REG", "CBCT")),
                   ("lineEditSepSeg", ("REG", "Seg"))),
        "output": ("lineEditSepOut", "LR_crop"),
    },
    "Approx": {
        "inputs": (("lineEditApproxMRI", ("REG", "MRI")),
                   ("lineEditApproxCBCT", ("REG", "CBCT"))),
        "output": ("lineEditOutputApprox", "Approx"),
    },
    "TMJCrop": {
        "inputs": (("lineEditCropTMJMRI", ("REG", "MRI")),
                   ("lineEditCropTMJCBCT", ("REG", "CBCT")),
                   ("lineEditCropTMJSeg", ("REG", "Seg"))),
        "output": ("lineEditCropTMJOut", "TMJ_crop"),
        "model": "lineEditTMJModel",
    },
    "Registration": {
        "inputs": (("lineEditRegMRI", ("REG", "MRI")),
                   ("lineEditRegCBCT", ("REG", "CBCT")),
                   ("lineEditRegLabel", ("REG", "Seg"))),
        "output": ("LineEditOutput", "Registration"),
    },
}


def pathFromVolumeNode(node):
    """Returns the on-disk file path a volume node was loaded from, or None
    if it has none (e.g. created in-session and never saved) - same idea as
    AMASSS's PathFromNode helper for its single-file input mode."""
    if node is None:
        return None
    storage_node = node.GetStorageNode()
    if storage_node is None:
        return None
    return storage_node.GetFullNameFromFileName()

def check_lib_installed(lib_name, required_version=None):
    """Whether the library is installed and satisfies the constraint."""
    return lib_satisfies(lib_name, required_version)

def install_function():
    libs = [
        ('itk', None),
        ('einops', None),
        ('dicom2nifti', '>=2.6.2'),
        # pydicom is kept on the version Slicer ships: downgrading it to 2.x breaks
        # dicomweb-client and highdicom, hence every DICOM module of Slicer.
        ('pydicom', '==3.0.2'),
        ('nibabel', None),
        ('itk-elastix', None),
        ('pandas', None),
        ('scikit-learn', None),
        ('torch', None),  # Special case
        ('torchreg', None),
        ('SimpleITK', None),
        ('numpy', '==1.26.4'),
        ('numexpr', '>=2.9.0'),
        ('psutil', None),
        ('nnunetv2', '>=2.8.0')  # 'nnunet_version' is published nowhere; AREG already names it nnunetv2
    ]

    libs_to_install = []
    for lib, version_spec in libs:
        if not check_lib_installed(lib, version_spec):
            libs_to_install.append((lib, version_spec))

    if libs_to_install:
        message = "The following libraries are not installed or need updating:\n"
        message += "\n".join([
            f"{lib}{version if version else ''}" for lib, version in libs_to_install
        ])
        message += "\n\nDo you want to install/update these libraries?\nDoing it could break other modules"
        user_choice = slicer.util.confirmYesNoDisplay(message)

        if user_choice:
            for lib, version_spec in libs_to_install:
                try:
                    if lib == "torch":
                        # cu118 was hardcoded here, and its kernels stop at
                        # sm_90: on an RTX 50 series it installed cleanly and
                        # then failed every launch with "no kernel image is
                        # available". Which build this machine needs is a
                        # question about its GPU, so it is asked of the GPU.
                        arguments = torch_install_arguments()
                        if arguments is None:
                            logger.info("the installed torch already serves this GPU")
                        else:
                            logger.info("Installing torch: %s", arguments)
                            pip_install(arguments)
                    else:
                        pip_target = f"{lib}{version_spec}" if version_spec else lib
                        pip_install(pip_target)
                except Exception as e:
                    slicer.util.errorDisplay(f"Failed to install {lib}: {str(e)}")
                    return False
        else:
            return False

    try:
        import vtk  # noqa: F401  (sonde de disponibilite)
        import itk  # noqa: F401  (sonde de disponibilite)
    except ImportError as e:
        slicer.util.errorDisplay(f"Final import check failed: {e}")
        return False

    return True
#
# MRI2CBCT
#


class MRI2CBCT(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("MRI2CBCT")
      
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["SlicerNNUNet"]
        self.parent.contributors = ["Gaelle Leroux (UoM), Alexandre Buisson (UoM), Raphael Barret (UoM)"]
        
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools#MRI2CBCT">module documentation</a>.
""")
        
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

    icons_path = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # MRI2CBCT1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="MRI2CBCT",
        sampleName="MRI2CBCT1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(icons_path, "MRI2CBCT1.png"),
        # Download URL and target file name
        uris=f"{SLICER_TESTING_DATA}/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="MRI2CBCT1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="MRI2CBCT1",
    )

    # MRI2CBCT2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="MRI2CBCT",
        sampleName="MRI2CBCT2",
        thumbnailFileName=os.path.join(icons_path, "MRI2CBCT2.png"),
        # Download URL and target file name
        uris=f"{SLICER_TESTING_DATA}/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
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
        
        

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        ui_widget = slicer.util.loadUI(self.resourcePath("UI/MRI2CBCT.ui"))
        self.layout.addWidget(ui_widget)
        self.uiWidget = ui_widget  # Store reference for styling
        self.ui = slicer.util.childWidgetVariables(ui_widget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        ui_widget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MRI2CBCTLogic()
        
        documents_location = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documents_location)
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
        self.manual_approx_mri2cbct = ManualApproximation_MRI2CBCT(self)
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
        self.ui.pushButtonTestFileApprox.connect(
            "clicked(bool)", partial(self.fillWithTestFiles, "Approx"))
        self._setupApproxSceneInputs()
        self.manual_approx_mri2cbct.injectUI(self.ui.approxCollapsibleButton)

        
        
        ### L/R Cropping ###
        self.ui.SearchButtonSepCBCT.connect("clicked(bool)",partial(self.openFinder,"InputCBCTSep"))
        self.ui.SearchButtonSepMRI.connect("clicked(bool)",partial(self.openFinder,"InputMRISep"))
        self.ui.SearchButtonSepSeg.connect("clicked(bool)",partial(self.openFinder,"InputSegSep"))
        self.ui.SearchButtonSepOut.connect("clicked(bool)",partial(self.openFinder,"OutputSep"))
        self.ui.lineEditSepCBCT.textChanged.connect(self.updateSepLabel)
        self.ui.lineEditSepMRI.textChanged.connect(self.updateSepLabel)
        self.ui.lineEditSepSeg.textChanged.connect(self.updateSepLabel)
        self.ui.pushButtonCropLR.connect("clicked(bool)", self.lrCropMRI2CBCT)
        self.ui.pushButtonTestFileSep.connect(
            "clicked(bool)", partial(self.fillWithTestFiles, "LRCrop"))

        
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
        self.ui.pushButtonTestFileTMJ.connect(
            "clicked(bool)", partial(self.fillWithTestFiles, "TMJCrop"))
        
        
        
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
        self.ui.pushButtonTestFileResample.connect(
            "clicked(bool)", partial(self.fillWithTestFiles, "Resample"))
        
        
        
        ### MRI Orientation ###
        self.ui.comboBoxDICOMVolumes.currentIndexChanged.connect(self.onDICOMSelectionChanged)
        self.addSceneObservers()
        self.updateDICOMComboBox()
        self.ui.checkBoxBilateralMRI.connect("clicked(bool)",self.toggleBilateral)
        self.toggleBilateral()
        
        self.ui.SearchButtonMRI.connect("clicked(bool)",partial(self.openFinder,"InputMRI"))
        self.ui.SearchOutputFolderOrientMRI.connect("clicked(bool)",partial(self.openFinder,"OutputOrientMRI"))
        self.ui.pushButtonOrientMRI.connect("clicked(bool)",self.orientCenterMRI)
        self.ui.pushButtonTestFilePreMRI.connect(
            "clicked(bool)", partial(self.fillWithTestFiles, "Orient"))
        
        
        
        
        ### Registration ###
        self.ui.SearchButtonOutput.connect("clicked(bool)",partial(self.openFinder,"OutputReg"))
        # The three buttons of the step fill the same thing: the whole step.
        # Which one is pressed changes nothing -- a test set is not chosen
        # field by field.
        for button in (self.ui.pushButtonTestFileRegMRI,
                       self.ui.pushButtonTestFileRegCBCT,
                       self.ui.pushButtonTestFileRegSeg):
            button.connect("clicked(bool)",
                           partial(self.fillWithTestFiles, "Registration"))
        self.ui.SearchButtonRegMRI.connect("clicked(bool)",partial(self.openFinder,"InputRegMRI"))
        self.ui.SearchButtonRegCBCT.connect("clicked(bool)",partial(self.openFinder,"InputRegCBCT"))
        self.ui.SearchButtonRegLabel.connect("clicked(bool)",partial(self.openFinder,"InputRegLabel"))
        self.ui.registrationButton.connect("clicked(bool)", self.registration_MR2CBCT)
        
        
        
        
        self.ui.pushButtonCancelProcess.connect("clicked(bool)", self.onCancel)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.ui.ComboBoxMRI.setCurrentIndex(1)
        self.ui.ComboBoxMRI.setEnabled(False)
        
        # Apply dark mode styling
        self.applyDarkModeStyles()
        
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
                    check_box = QCheckBox('0')
                    check_box.stateChanged.connect(lambda state, r=row, c=col: self.onCheckboxOrientClicked(r, c, state))
                    self.tableWidgetOrient.setCellWidget(row, col, check_box)
                else :
                    check_box = QCheckBox('No')
                    check_box.stateChanged.connect(lambda state, r=row, c=col: self.onCheckboxOrientClicked(r, c, state))
                    self.tableWidgetOrient.setCellWidget(row, col, check_box)

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
                spin_box = QSpinBox()
                if col in [2, 3]:  # Columns for Percentile Min and Percentile Max
                    spin_box.setMaximum(100)
                else:
                    spin_box.setMaximum(10000)
                self.tableWidgetNorm.setCellWidget(row, col, spin_box)
                
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
        spin_box1 = QSpinBox()
        spin_box1.setMaximum(10000)
        spin_box1.setValue(443)
        self.tableWidgetResample.setCellWidget(0, 0, spin_box1)

        spin_box2 = QSpinBox()
        spin_box2.setMaximum(10000)
        spin_box2.setValue(443)
        self.tableWidgetResample.setCellWidget(0, 1, spin_box2)

        spin_box3 = QSpinBox()
        spin_box3.setMaximum(10000)
        spin_box3.setValue(119)
        self.tableWidgetResample.setCellWidget(0, 2, spin_box3)

        # Add QSpinBoxes for the new row
        spin_box4 = QDoubleSpinBox()
        spin_box4.setMaximum(10000)
        spin_box4.setSingleStep(0.1)
        spin_box4.setValue(0.3)
        self.tableWidgetResample.setCellWidget(1, 0, spin_box4)

        spin_box5 = QDoubleSpinBox()
        spin_box5.setMaximum(10000)
        spin_box5.setSingleStep(0.1)
        spin_box5.setValue(0.3)
        self.tableWidgetResample.setCellWidget(1, 1, spin_box5)

        spin_box6 = QDoubleSpinBox()
        spin_box6.setMaximum(10000)
        spin_box6.setSingleStep(0.1)
        spin_box6.setValue(0.3)
        self.tableWidgetResample.setCellWidget(1, 2, spin_box6)
        # Add QCheckBox for the "Keep File" column
        # Check if dark mode
        app = qt.QApplication.instance()
        palette = app.palette()
        bg_color = palette.color(qt.QPalette.Window)
        is_dark_mode = bg_color.lightness() < 128
        
        checkbox_stylesheet = """
          QCheckBox {
            color: #ffffff;
            background-color: transparent;
            font-weight: 500;
          }
          QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555555;
            border-radius: 3px;
            background-color: #3c3c3c;
          }
          QCheckBox::indicator:hover {
            border: 1px solid #5dade2;
          }
          QCheckBox::indicator:checked {
            border: 1px solid #5dade2;
            background-color: #5dade2;
            image: url(:/Icons/SmallCheckMark.png);
          }
          QCheckBox::indicator:checked:hover {
            border: 1px solid #7bbcef;
            background-color: #7bbcef;
          }
        """
        
        check_box1 = QCheckBox("Keep the same size as the input scan")
        if is_dark_mode:
            check_box1.setStyleSheet(checkbox_stylesheet)
        check_box1.stateChanged.connect(lambda state: self.toggleSpinBoxes(state, [spin_box1, spin_box2, spin_box3]))
        self.tableWidgetResample.setCellWidget(0, 3, check_box1)

        check_box2 = QCheckBox("Keep the same spacing as the input scan")
        if is_dark_mode:
            check_box2.setStyleSheet(checkbox_stylesheet)
        check_box2.stateChanged.connect(lambda state: self.toggleSpinBoxes(state, [spin_box4, spin_box5, spin_box6]))
        self.tableWidgetResample.setCellWidget(1, 3, check_box2)
        
    def toggleSpinBoxes(self, state, spin_boxes):
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
        for spin_box in spin_boxes:
            if state == 2:
                spin_box.setEnabled(False)
                spin_box.setStyleSheet("color: gray;")
            else:
                spin_box.setEnabled(True)
                spin_box.setStyleSheet("")

        
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
        current_id = self.ui.comboBoxDICOMVolumes.itemData(self.ui.comboBoxDICOMVolumes.currentIndex) if self.ui.comboBoxDICOMVolumes.currentIndex > 0 else None
        self.ui.comboBoxDICOMVolumes.blockSignals(True)
        self.ui.comboBoxDICOMVolumes.clear()
        self.ui.comboBoxDICOMVolumes.addItem("Select DICOM node")
        
        volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        for volume in volumes:
            if volume.GetAttribute("DICOM.instanceUIDs"):
                self.ui.comboBoxDICOMVolumes.addItem(volume.GetName(), volume.GetID())
        
        if current_id:
            index = self.ui.comboBoxDICOMVolumes.findData(current_id)
            if index != -1:
                self.ui.comboBoxDICOMVolumes.setCurrentIndex(index)
        self.ui.comboBoxDICOMVolumes.blockSignals(False)

    def onDICOMSelectionChanged(self):
        if self.ui.comboBoxDICOMVolumes.currentIndex == 0:
            self.ui.labelDICOMSpacing.text = "Acquisition Spacing: None"
            return
        
        volume_id = self.ui.comboBoxDICOMVolumes.itemData(self.ui.comboBoxDICOMVolumes.currentIndex)
        volume_node = slicer.mrmlScene.GetNodeByID(volume_id)
        if not volume_node:
            return
        
        instance_ui_ds = volume_node.GetAttribute("DICOM.instanceUIDs").split()
        if not instance_ui_ds:
            self.ui.labelDICOMSpacing.text = "Acquisition Spacing: N/A (No DICOM metadata)"
            return
        
        first_instance_uid = instance_ui_ds[0]
        db = slicer.dicomDatabase
        if not db:
            self.ui.labelDICOMSpacing.text = "DICOM database not available"
            return
        
        # Get spacing values from DICOM tags
        spacing = db.instanceValue(first_instance_uid, "0018,0088")  # Spacing Between Slices
        if not spacing:
            spacing = db.instanceValue(first_instance_uid, "0018,0050")  # Slice Thickness
        
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
                check_box = self.tableWidgetOrient.cellWidget(row, col)
                check_box.setText('Yes')
                for c in range(3):
                    check_box = self.tableWidgetOrient.cellWidget(row, c)
                    if check_box.text=="1":
                        check_box.setText('-1')
            else:  # Checkbox is unchecked
                self.minus_checked_rows.discard(row)
                check_box = self.tableWidgetOrient.cellWidget(row, col)
                check_box.setText('No')
                for c in range(3):
                    check_box = self.tableWidgetOrient.cellWidget(row, c)
                    if check_box.text=="-1":
                        check_box.setText('1')
        else :
            if state == 2:  # Checkbox is checked
                # Set the clicked checkbox to '1' and uncheck all others in the same row
                for c in range(3):
                    check_box = self.tableWidgetOrient.cellWidget(row, c)
                    if check_box:
                        if c == col:
                            if row in self.minus_checked_rows:
                                check_box.setText('-1')
                            else :
                                check_box.setText('1')
                            check_box.setStyleSheet("color: black;")
                            check_box.setStyleSheet("font-weight: bold;")
                            self.checked_cells.add((row, col))
                        else:
                            check_box.setText('0')
                            check_box.setChecked(False)
                            self.checked_cells.discard((row, c))

                # Check for other '1' in the same column and set them to '0'
                for r in range(3):
                    if r != row:
                        check_box = self.tableWidgetOrient.cellWidget(r, col)
                        if check_box and (check_box.text == '1' or check_box.text == '-1'):
                            check_box.setText('0')
                            check_box.setChecked(False)
                            check_box.setStyleSheet("color: gray;")
                            check_box.setStyleSheet("font-weight: normal;")
                            self.checked_cells.discard((r, col))
                            
                # Check if two checkboxes are checked in different rows, then check the third one
                if len(self.checked_cells) == 2:
                    all_rows = {0, 1, 2}
                    all_cols = {0, 1, 2}
                    checked_rows = {r for r, c in self.checked_cells}
                    unchecked_row = list(all_rows - checked_rows)[0]
                    
                    # Find the unchecked column
                    unchecked_cols = list(all_cols - {c for r, c in self.checked_cells})
                    for c in range(3):
                        check_box = self.tableWidgetOrient.cellWidget(unchecked_row, c)
                        if c in unchecked_cols:
                            check_box.setStyleSheet("color: black;")
                            check_box.setStyleSheet("font-weight: bold;")
                            check_box.setChecked(True)
                            if unchecked_row in self.minus_checked_rows:
                                check_box.setText('-1')
                            else :
                                check_box.setText('1')
                            self.checked_cells.add((unchecked_row, c))
                        else :
                            check_box.setText('0')
                            check_box.setChecked(False)
                            self.checked_cells.discard((row, c))

            else:  # Checkbox is unchecked
                check_box = self.tableWidgetOrient.cellWidget(row, col)
                if check_box:
                    check_box.setText('0')
                    check_box.setStyleSheet("color: black;")
                    check_box.setStyleSheet("font-weight: normal;")
                    self.checked_cells.discard((row, col))
                    
                # Reset the style of all checkboxes in the same row
                for c in range(3):
                    check_box = self.tableWidgetOrient.cellWidget(row, c)
                    if check_box:
                        check_box.setStyleSheet("color: black;")
                        check_box.setStyleSheet("font-weight: normal;")
                        
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
                check_box = self.tableWidgetOrient.cellWidget(row, col)
                if check_box:
                    values.append(int(check_box.text))
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
            check_box = self.tableWidgetOrient.cellWidget(row, col)
            if check_box:
                if value == 1:
                    check_box.setChecked(True)
                    check_box.setText('1')
                    check_box.setStyleSheet("font-weight: bold;")
                    self.checked_cells.add((row, col))
                elif value == -1:
                    check_box.setChecked(True)
                    check_box.setText('-1')
                    check_box.setStyleSheet("font-weight: bold;")
                    minus_check_box = self.tableWidgetOrient.cellWidget(row, 3)
                    if minus_check_box:
                        minus_check_box.setChecked(True)
                        minus_check_box.setText("Yes")
                    self.minus_checked_rows.add(row)

    def applyDarkModeStyles(self):
        """Give this module's widget the palette shared by the extension."""
        apply_dark_mode(self.uiWidget)

    def _updateLineEditAndComboBoxDarkMode(self, parent):
        """Shared recursive pass, kept as a method for the existing call sites."""
        update_line_edit_and_combo_box(parent)

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
            row_data = []
            for col in range(self.tableWidgetNorm.columnCount):
                widget = self.tableWidgetNorm.cellWidget(row, col)
                if isinstance(widget, QSpinBox):
                    row_data.append(widget.value)
            values.append(row_data)
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
                spin_box = QSpinBox()
                spin_box.setMaximum(10000)
                spin_box.setValue(default_values[row][col])
                self.tableWidgetNorm.setCellWidget(row, col, spin_box)
                
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
        fold_path, is_installed = self.install_nnunet()
        if is_installed:
            # install_nnunet answers a pathlib.Path; setText wants a string.
            self.ui.lineEditTMJModel.setText(str(fold_path))
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
        base_path = Path(self.SlicerDownloadPath).joinpath("ML", "Dataset001_myseg", "nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres").resolve()
        fold_path = base_path.joinpath("fold_0")
        fold_path.mkdir(parents=True, exist_ok=True)

        # Define destination paths
        checkpoint_path = fold_path.joinpath("checkpoint_final.pth")
        dataset_json_path = base_path.joinpath("dataset.json")
        plans_json_path = base_path.joinpath("plans.json")

        # Define URLs
        url_checkpoint = f"{TMJ_CROP_MODEL}/checkpoint_final.pth"
        url_dataset = f"{TMJ_CROP_MODEL}/dataset.json"
        url_plans = f"{TMJ_CROP_MODEL}/plans.json"

        # Download files if missing
        if not checkpoint_path.exists():
            self.download_file_with_progress(url_checkpoint, checkpoint_path, label="Downloading checkpoint")

        if not dataset_json_path.exists():
            self.download_file_with_progress(url_dataset, dataset_json_path, label="Downloading dataset.json")

        if not plans_json_path.exists():
            self.download_file_with_progress(url_plans, plans_json_path, label="Downloading plans.json")

        # If everything exists, return True
        return base_path, (checkpoint_path.exists() and dataset_json_path.exists() and plans_json_path.exists())

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
        
        
    def testFilesRoot(self):
        """Where the test dataset is kept: Slicer's download directory."""
        documents_location = qt.QStandardPaths.DocumentsLocation
        self.documents = qt.QStandardPaths.writableLocation(documents_location)
        return os.path.join(self.documents, slicer.app.applicationName + "Downloads")

    def fillWithTestFiles(self, step, _=None):
        """Fill every field of one step of the module with the test dataset.

        `step` is a key of `TEST_FILE_STEPS`, given by the button that was
        pressed -- the fields to fill are named there, not guessed from the
        `objectName` of a widget handed over one at a time.

        The dataset is downloaded only when it is missing, and the download is
        checked: an interrupted one does not count as present, and a release
        link that answers with a web page is named as such instead of failing
        later on "not a zip file". Input fields are always rewritten, that
        being what the button is for; the output folder is only filled when
        the user has not chosen one.
        """
        fields = TEST_FILE_STEPS[step]
        root = self.testFilesRoot()

        try:
            dataset = testdata.ensure_with_progress(
                MRI2CBCT_TEST_FILES, root, TEST_FILES_DIRECTORY,
                parent=self.parent,
                title="Downloading the MRI2CBCT test files...")
        except testdata.TestDataError as error:
            self.showMessage(str(error))
            return
        except OSError as error:
            self.showMessage(
                "The MRI2CBCT test files could not be downloaded from\n%s\n\n%s"
                % (MRI2CBCT_TEST_FILES, error))
            return

        scans = os.path.join(dataset, "TestFile")
        for name, parts in fields["inputs"]:
            getattr(self.ui, name).setText(os.path.join(scans, *parts))

        if "model" in fields:
            model_folder, is_installed = self.install_nnunet()
            if not is_installed:
                self.showMessage(
                    "The test scans are in place, but the nnU-Net model this "
                    "step needs could not be downloaded.")
                return
            getattr(self.ui, fields["model"]).setText(str(model_folder))

        output_name, folder = fields["output"]
        output_field = getattr(self.ui, output_name)
        if not output_field.text:
            destination = os.path.join(root, TEST_OUTPUT_DIRECTORY, folder)
            os.makedirs(destination, exist_ok=True)
            output_field.setText(destination)

    def tmjCropMRI2CBCT(self)->None:
        """
        This function is called when the button "pushButtonCropTMJ" is clicked.
        It crops the TMJ on both the Right and Left of MRI, CBCT, and segmentation images and initiates the processing pipeline.
        This function sets up the parameters for MRI and CBCT separation, tests the process and scan,
        and starts the processing pipeline if all checks pass. It handles the initial setup,
        parameter passing, and process initiation, including setting up observers for process updates.
        """
        
        install_function()
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
        logger.info(f"module name : {self.list_Processes_Parameters[0]["Module"]}")
        logger.info(f"Parameters TMJ Crop: {self.list_Processes_Parameters[0]["Parameter"]}")
        
        self.process = slicer.cli.run(
                self.list_Processes_Parameters[0]["Process"],
                None,
                self.list_Processes_Parameters[0]["Parameter"],
            )
        
        self.ui.labelBarNotWorking.setVisible(True)
        
        self.module_name = self.list_Processes_Parameters[0]["Module"]
        self.processObserver = self.process.AddObserver(
            "ModifiedEvent", self.onProcessUpdate
        )
        del self.list_Processes_Parameters[0]
    
    def lrCropMRI2CBCT(self)->None:
        """
        This function is called when the button "pushButtonCropLR" is clicked.
        It crops the Right and Left of MRI and CBCT images and initiates the processing pipeline.
        This function sets up the parameters for MRI and CBCT separation, tests the process and scan,
        and starts the processing pipeline if all checks pass. It handles the initial setup,
        parameter passing, and process initiation, including setting up observers for process updates.
        """
        
        install_function()
        lin_edit_mri_sep = "None"
        lin_edit_cbct_sep = "None"
        line_edit_seg_sep = "None"
        
        if self.ui.lineEditSepMRI.text != "":
            lin_edit_mri_sep = self.ui.lineEditSepMRI.text
        if self.ui.lineEditSepCBCT.text != "":
            lin_edit_cbct_sep = self.ui.lineEditSepCBCT.text
        if self.ui.lineEditSepSeg.text != "":
            line_edit_seg_sep = self.ui.lineEditSepSeg.text
            
        param = {
            "input_folder_CBCT": lin_edit_cbct_sep,
            "input_folder_MRI": lin_edit_mri_sep,
            "input_folder_Seg": line_edit_seg_sep,
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
        logger.info(f"module name : {self.list_Processes_Parameters[0]["Module"]}")
        logger.info(f"Parameters : {self.list_Processes_Parameters[0]["Parameter"]}")
        
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
        install_function()
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
        logger.info(f"module name : {self.list_Processes_Parameters[0]["Module"]}")
        logger.info(f"Parameters : {self.list_Processes_Parameters[0]["Parameter"]}")
        
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
        install_function()
        line_edit_mri = "None"
        line_edit_t2_mri = "None"
        line_edit_cbct = "None"
        line_edit_t2_cbct = "None"
        line_edit_seg = "None"
        line_edit_t2_seg = "None"
        if self.ui.lineEditResampleMRI.text != "":
            line_edit_mri = self.ui.lineEditResampleMRI.text
        if self.ui.lineEditResampleT2MRI.text != "" and self.ui.CheckBoxT2MRI.isChecked():
            line_edit_t2_mri = self.ui.lineEditResampleT2MRI.text
        if self.ui.lineEditResampleCBCT.text != "":
            line_edit_cbct = self.ui.lineEditResampleCBCT.text
        if self.ui.lineEditResampleT2CBCT.text != "" and self.ui.CheckBoxT2CBCT.isChecked():
            line_edit_t2_cbct = self.ui.lineEditResampleT2CBCT.text
        if self.ui.lineEditResampleSeg.text != "":
            line_edit_seg = self.ui.lineEditResampleSeg.text
        if self.ui.lineEditResampleT2Seg.text != "" and self.ui.CheckBoxT2Seg.isChecked():
            line_edit_t2_seg = self.ui.lineEditResampleT2Seg.text
            
        param = {"input_folder_MRI": line_edit_mri,
            "input_folder_T2_MRI": line_edit_t2_mri,
            "input_folder_CBCT": line_edit_cbct,
            "input_folder_T2_CBCT": line_edit_t2_cbct,
            "input_folder_Seg": line_edit_seg,
            "input_folder_T2_Seg": line_edit_t2_seg,
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
        logger.info(f"module name : {self.list_Processes_Parameters[0]["Module"]}")
        logger.info(f"Parameters : {self.list_Processes_Parameters[0]["Parameter"]}")
        
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
        install_function()
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
        logger.info(f"module name : {self.list_Processes_Parameters[0]["Module"]}")
        logger.info(f"Parameters : {self.list_Processes_Parameters[0]["Parameter"]}")
        
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

    def _setupApproxSceneInputs(self) -> None:
        """
        Adds the option to pick the CBCT/MRI inputs for Approximate directly
        from volumes already loaded in the scene, instead of always pointing
        at a folder on disk - mirrors the Folder/File input-type toggle AMASSS
        already uses for its own scan input (input_type_select +
        MRMLNodeComboBox_file).
        """
        grid_layout = self.ui.gridLayout_4

        self.labelApproxInputType = qt.QLabel("Input type:")
        grid_layout.addWidget(self.labelApproxInputType, 3, 0)
        self.comboBoxApproxInputType = qt.QComboBox()
        self.comboBoxApproxInputType.addItems(["Folder", "Scene Volume"])
        grid_layout.addWidget(self.comboBoxApproxInputType, 3, 1)

        self.approxSceneCBCTSelector = slicer.qMRMLNodeComboBox()
        self.approxSceneCBCTSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.approxSceneCBCTSelector.setMRMLScene(slicer.mrmlScene)
        self.approxSceneCBCTSelector.noneEnabled = True
        self.approxSceneCBCTSelector.addEnabled = False
        self.approxSceneCBCTSelector.removeEnabled = False
        self.approxSceneCBCTSelector.setToolTip("CBCT volume already loaded in the scene")
        grid_layout.addWidget(self.approxSceneCBCTSelector, 0, 1, 1, 3)

        self.approxSceneMRISelector = slicer.qMRMLNodeComboBox()
        self.approxSceneMRISelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.approxSceneMRISelector.setMRMLScene(slicer.mrmlScene)
        self.approxSceneMRISelector.noneEnabled = True
        self.approxSceneMRISelector.addEnabled = False
        self.approxSceneMRISelector.removeEnabled = False
        self.approxSceneMRISelector.setToolTip("MRI volume already loaded in the scene")
        grid_layout.addWidget(self.approxSceneMRISelector, 1, 1, 1, 3)

        def onInputTypeChanged(index):
            use_scene = (index == 1)
            self.ui.label_17.setText("CBCT volume:" if use_scene else "Input CBCT folder:")
            self.ui.lineEditApproxCBCT.setVisible(not use_scene)
            self.ui.SearchButtonApproxCBCT.setVisible(not use_scene)
            self.approxSceneCBCTSelector.setVisible(use_scene)

            self.ui.label_16.setText("MRI volume:" if use_scene else "Input MRI folder:")
            self.ui.lineEditApproxMRI.setVisible(not use_scene)
            self.ui.SearchButtonApproxMRI.setVisible(not use_scene)
            self.approxSceneMRISelector.setVisible(use_scene)

        self.comboBoxApproxInputType.currentIndexChanged.connect(onInputTypeChanged)
        onInputTypeChanged(0)

    def approximateMRI(self) -> None:
        """
        Approximates MRI images to CBCT images using specified parameters and initiate the processing pipeline.

        This function sets up the parameters for MRI to CBCT registration, tests the process and scans,
        and starts the processing pipeline if all checks pass. It handles the initial setup, parameter passing,
        and process initiation, including setting up observers for process updates. The function also checks
        for normalization parameters and validates input folders for the presence of necessary files.
        """
        install_function()

        model_folder, model_ok = self.install_nnunet()
        if not model_ok:
            self.showMessage("Failed to download the condyle segmentation model required for Approximate.")
            return

        use_scene_volumes = self.comboBoxApproxInputType.currentIndex == 1
        cbct_folder = self.ui.lineEditApproxCBCT.text
        mri_folder = self.ui.lineEditApproxMRI.text

        if use_scene_volumes:
            cbct_node = self.approxSceneCBCTSelector.currentNode()
            mri_node = self.approxSceneMRISelector.currentNode()
            if not cbct_node or not mri_node:
                self.showMessage("Please select a CBCT and an MRI volume from the scene.")
                return

            # Volumes already loaded from a file already have that file's path
            # on their storage node, same as AMASSS's single-file input mode -
            # no need to export/copy anything.
            cbct_folder = pathFromVolumeNode(cbct_node)
            mri_folder = pathFromVolumeNode(mri_node)
            if not cbct_folder or not mri_folder:
                self.showMessage(
                    "The selected volume(s) don't have a file on disk yet. "
                    "Save them first, or switch Input type to Folder.")
                return

        param = {"cbct_folder": cbct_folder,
            "mri_folder": mri_folder,
            "output_folder" : self.ui.lineEditOutputApprox.text,
            "model_folder": str(model_folder),
            "use_scene_volumes": use_scene_volumes}

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
        logger.info(f"module name : {self.list_Processes_Parameters[0]["Module"]}")
        logger.info(f"Parameters : {self.list_Processes_Parameters[0]["Parameter"]}")
        
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
        
        current_time = time.time() - self.startTime
        if current_time < 60:
            timer = f"Time: {int(current_time)}s"
        elif current_time < 3600:
            timer = f"Time: {int(current_time/60)}min and {int(current_time%60)}s"
        else:
            timer = f"Time: {int(current_time/3600)}h, {int(current_time%3600/60)}min and {int(current_time%60)}s"

        self.ui.label_time.setText(timer)
        self.ui.label_info.setText(f"Extension {self.module_name} is running. \nNumber of extension runned: {self.nb_extnesion_did} / {self.nb_extension_launch}")
        
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
                logger.info("\n\n ========= PROCESSED ========= \n")

                logger.info(self.process.GetOutputText())
                logger.error("\n\n ========= ERROR ========= \n")
                error_text = self.process.GetErrorText()
                logger.error("CLI execution failed: \n \n" + error_text)

                self.onCancel()

            else:
                logger.info("\n\n ========= PROCESSED ========= \n")

                logger.info(self.process.GetOutputText())

                if self.module_name == "MRI2CBCT approximation":
                    self.approximate_mri2cbct.finalizeApproximation()

                try:
                    logger.info(f"name process : {self.list_Processes_Parameters[0]["Process"]}")
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
        

        logger.info("PROCESS DONE.")
        logger.info(
            "Done in {} min and {} sec".format(
                int(total_time / 60), int(total_time % 60)
            )
        )

        self.RunningUI(False)

        stop_time = time.time()

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
        logger.info("\n\n ========= PROCESS CANCELED ========= \n")
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
                image_threshold: float,
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

        start_time = time.time()
        logger.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cli_params = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": image_threshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cli_node = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cli_params, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cli_node)

        stop_time = time.time()
        logger.info(f"Processing completed in {stop_time-start_time:.2f} seconds")


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
        input_volume = SampleData.downloadSample("MRI2CBCT1")
        self.delayDisplay("Loaded test data set")

        input_scalar_range = input_volume.GetImageData().GetScalarRange()
        self.assertEqual(input_scalar_range[0], 0)
        self.assertEqual(input_scalar_range[1], 695)

        output_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = MRI2CBCTLogic()

        # Test algorithm with non-inverted threshold
        logic.process(input_volume, output_volume, threshold, True)
        output_scalar_range = output_volume.GetImageData().GetScalarRange()
        self.assertEqual(output_scalar_range[0], input_scalar_range[0])
        self.assertEqual(output_scalar_range[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(input_volume, output_volume, threshold, False)
        output_scalar_range = output_volume.GetImageData().GetScalarRange()
        self.assertEqual(output_scalar_range[0], input_scalar_range[0])
        self.assertEqual(output_scalar_range[1], input_scalar_range[1])

        self.delayDisplay("Test passed")