from enum import Flag, auto
from pathlib import Path
import vtk
import SegmentEditorEffects
import ctk
import numpy as np
import qt
import subprocess
import sys
import slicer
import os
from .IconPath import icon, iconPath
from .PythonDependencyChecker import PythonDependencyChecker, hasInternetConnection
from .Queue import (
    SegmentationQueue,
    listVolumes,
    volumeStem,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
)
from .Utils import (
    createButton,
    addInCollapsibleLayout,
    set3DViewBackgroundColors,
    setConventionalWideScreenView,
    setBoxAndTextVisibilityOnThreeDViews,
)
from collections import deque

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
import gc
import json
from ADTLib.model_registry import NASOMAXILLA_DENT_SEG, PEDIATRIC_DENTAL_SEG, UNIVERSAL_LAB
from ADTLib.testdata import TestDataError

logger = get_logger("BatchDentalSeg_SegmentationWidget")

# The test set is the "CBCTDentalSurgery" CBCT published by Slicer, the very
# one `Testing/Utils.load_test_CT_volume` uses. The TEST_FILES_BATCHDENTALSEG
# release only carries *segmentations*: those are the expected outputs of the
# tests, not inputs -- this module segments scans, so its input folder must
# hold some. The two files of that set are `.gipl.gz`, an extension
# `listVolumes` already recognises.
TEST_FILES_SAMPLE_NAME = "CBCTDentalSurgery"


vtk.vtkObject.GlobalWarningDisplayOff()

MODEL_DESCRIPTIONS = {
    "DentalSegmentator": (
        "<b>DentalSegmentator</b><br>"
        "Segments: Upper Skull (includes Maxilla), Mandible, Mandibular Canal, Upper Teeth, Lower Teeth<br>"
        "Designed for <b>permanent dentition</b>."
    ),
    "PediatricDentalsegmentator": (
        "<b>PediatricDentalsegmentator</b><br>"
        "Segments: Upper Skull (includes Maxilla), Mandible, Mandibular Canal, Upper Teeth, Lower Teeth<br>"
        "Designed for <b>mixed dentition</b> (baby and permanent teeth)."
    ),
    "NasoMaxillaDentSeg": (
        "<b>NasoMaxillaDentSeg</b><br>"
        "Segments: Upper Skull, <u>separate</u> Maxilla, Mandible, Mandibular Canal, Upper Teeth, Lower Teeth<br>"
        "Designed for <b>permanent dentition</b> ."
    ),
    "UniversalLabDentalsegmentator": (
        "<b>UniversalLabDentalsegmentator</b><br>"
        "Segments: Upper Skull, Mandibular Canal,All teeth<br>"
        "Designed for <b>mixed and Permanent dentition</b> ."
    ),
}

# ─── Export formats enumeration ───────────────────────────────────────────────

class ExportFormat(Flag):
    OBJ = auto()
    STL = auto()
    NIFTI = auto()
    GLTF = auto()
    VTK = auto()
    VTK_MERGED = auto()

# ─── Segmentation Widget Class ────────────────────────────────────────────────


class PipRunner(qt.QObject):
    """
    Run "pip install ..."
    """
    def __init__(self, packages, onLine, onFinished, parent=None):
        super().__init__(parent)
        self._onLine     = onLine
        self._onFinished = onFinished
        self._proc       = qt.QProcess(self)           # lifetime = that of the runner

        # — configuration process —
        self._proc.setProgram(sys.executable)          # PythonSlicer
        self._proc.setArguments(["-m", "pip", "install"] + packages)
        self._proc.setProcessChannelMode(qt.QProcess.MergedChannels)

        # — connect signals —
        self._proc.readyReadStandardOutput.connect(self._readLines)
        self._proc.readyReadStandardError.connect(self._readLines)
        self._proc.finished.connect(self._procFinished)

        self._proc.start()

    # ---------- slots internes ----------
    def _readLines(self):
        while self._proc.canReadLine():
            # Qt → QByteArray → bytes → str
            line_ba  = self._proc.readLine()           # QByteArray
            line_str = line_ba.data().decode("utf-8", "ignore").rstrip()
            self._onLine(line_str)


    def _procFinished(self, exit_code, *args):
        """
        Slot called at the end of QProcess.
        Qt5 : finished(int)
        Qt6 : finished(int, QProcess.ExitStatus)
        """
        self._onFinished(exit_code == 0)
        self.deleteLater()

class SegmentationWidget(qt.QWidget):

    # ─── Initialization ─────────────────────────────────────────────────────────
    def __init__(self, logic=None, parent=None):
        super().__init__(parent)

        # ----------------------------------------------------------------- state
        self.logic                    = logic or self._createSlicerSegmentationLogic()
        self._prevSegmentationNode    = None
        self._minimumIslandSize_mm3   = 60
        self.folderPath               = ""
        self.outputFolderPath         = ""
        self.folderFiles              = []
        self.currentFileIndex         = 0
        self.currentVolumeNode        = None
        self.fullInfoLogs             = deque(maxlen=200_000)   # message log (bounded)

        # ------------------------------------------------------------ queue state
        self.queue                    = SegmentationQueue()
        self._queueRunning            = False
        self._itemFinalized           = True        # guard against double advance
        self._itemStartTime           = None
        self._setupDone               = False       # pip / weights: once per session
        self._deviceFallbackAccepted  = None        # CPU answer remembered for the queue

        # --------------------------------------------------- buffered log output
        self._logBuffer               = []
        self._logFlushTimer           = qt.QTimer(self)
        self._logFlushTimer.setSingleShot(True)
        self._logFlushTimer.setInterval(200)
        self._logFlushTimer.timeout.connect(self._flushLogBuffer)

        # ========================================================================
        # 1)  INPUT / OUTPUT FOLDERS
        # ========================================================================
        self.folderPathLineEdit   = qt.QLineEdit(self);  self.folderPathLineEdit.setReadOnly(True)
        self.outputFolderLineEdit = qt.QLineEdit(self);  self.outputFolderLineEdit.setReadOnly(True)

        folder_btn = createButton("Select Folder",        callback=self.selectFolder)
        out_btn    = createButton("Select Output Folder", callback=self.selectOutputFolder)
        test_btn   = createButton(
            "Test Files", callback=self.onTestFiles,
            toolTip="Fill both folders with Slicer's CBCTDentalSurgery sample "
                    "(two adult CBCT scans, about 70 MB, downloaded once).",
            parent=self)

        self.inputWidget = qt.QWidget(self)
        input_layout      = qt.QFormLayout(self.inputWidget); input_layout.setContentsMargins(0,0,0,0)
        input_layout.addRow("Input Folder:",  self.folderPathLineEdit)
        input_layout.addRow("",               folder_btn)
        input_layout.addRow("Output Folder:", self.outputFolderLineEdit)
        input_layout.addRow("",               out_btn)
        input_layout.addRow("",               test_btn)

        # ========================================================================
        # 2)  EXPORT FORMATS
        # ========================================================================
        export_widget = qt.QWidget()
        export_layout = qt.QFormLayout(export_widget)

        self.stlCheckBox       = qt.QCheckBox(export_widget); self.stlCheckBox.setChecked(True)
        self.objCheckBox       = qt.QCheckBox(export_widget)
        self.niftiCheckBox     = qt.QCheckBox(export_widget)
        self.gltfCheckBox      = qt.QCheckBox(export_widget)
        self.vtkCheckBox       = qt.QCheckBox(export_widget)
        self.vtkmergedCheckBox = qt.QCheckBox(export_widget)

        self.reductionFactorSlider = ctk.ctkSliderWidget()
        self.reductionFactorSlider.maximum     = 1.0
        self.reductionFactorSlider.value       = 0.9
        self.reductionFactorSlider.singleStep  = 0.01
        self.reductionFactorSlider.toolTip     = "Decimation factor for glTF export."

        export_layout.addRow("Export STL",           self.stlCheckBox)
        export_layout.addRow("Export OBJ",           self.objCheckBox)
        export_layout.addRow("Export NIFTI",         self.niftiCheckBox)
        export_layout.addRow("Export glTF",          self.gltfCheckBox)
        export_layout.addRow("Export VTK",           self.vtkCheckBox)
        export_layout.addRow("Export VTK (merged)",  self.vtkmergedCheckBox)
        export_layout.addRow("glTF reduction factor:", self.reductionFactorSlider)

        # Add to the layout the export formats widget
        input_layout.addRow("Export formats :", export_widget)

        # ========================================================================
        # 3)  DEVICE & MODEL
        # ========================================================================
        self.deviceComboBox = qt.QComboBox(); self.deviceComboBox.addItems(["cuda","cpu","mps"])
        # The order IS the default: nothing stores the last choice, so the
        # combo opens on index 0 every time. The universal model is first
        # because that is the one to hand somebody who has not been told
        # which to pick -- not whichever happened to be typed first.
        self.modelComboBox  = qt.QComboBox(); self.modelComboBox.addItems([
            "UniversalLabDentalsegmentator","DentalSegmentator","PediatricDentalsegmentator","NasoMaxillaDentSeg"])

        # Resolve-mirroring
        self.resolveMirroringButton = createButton(
            "Resolve Mirroring", callback=self.onResolveMirroring,
            toolTip="Automatically mirrors labeled segments", parent=self)
        self.resolveMirroringButton.setVisible(False)
        self.modelComboBox.currentTextChanged.connect(self._updateResolveButtonVisibility)
        self._updateResolveButtonVisibility(self.modelComboBox.currentText)

        # ========================================================================
        # 4)  SEGMENTATION NODE SELECTOR & EDITOR
        # ========================================================================
        self.segmentationNodeSelector = slicer.qMRMLNodeComboBox(self)
        self.segmentationNodeSelector.nodeTypes  = ["vtkMRMLSegmentationNode"]
        self.segmentationNodeSelector.selectNodeUponCreation = True
        self.segmentationNodeSelector.addEnabled = True
        self.segmentationNodeSelector.removeEnabled = True
        self.segmentationNodeSelector.showHidden = False
        self.segmentationNodeSelector.renameEnabled = True
        self.segmentationNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentationNodeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateSegmentEditorWidget)
        self.segmentationNodeSelector.findChild("ctkComboBox").defaultText = "Create new Segmentation on Apply"

        self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget(self)
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorWidget.setSegmentationNodeSelectorVisible(False)
        self.segmentEditorWidget.setSourceVolumeNodeSelectorVisible(False)
        self.segmentEditorWidget.layout().setContentsMargins(0,0,0,0)
        self.segmentEditorNode = None

        # surface smoothing slider with Show-3D
        self.show3DButton = slicer.util.findChild(self.segmentEditorWidget, "Show3DButton")
        smoothing_slider = self.show3DButton.findChild("ctkSliderWidget")

        self.surfaceSmoothingSlider = ctk.ctkSliderWidget(self)
        self.surfaceSmoothingSlider.decimals   = 2
        self.surfaceSmoothingSlider.maximum    = 1
        self.surfaceSmoothingSlider.singleStep = 0.1
        self.surfaceSmoothingSlider.setValue(smoothing_slider.value)
        self.surfaceSmoothingSlider.tracking   = False
        self.surfaceSmoothingSlider.valueChanged.connect(smoothing_slider.setValue)

        # ========================================================================
        # 5)  MAIN LAYOUT
        # ========================================================================
        layout = qt.QVBoxLayout(self)

        # bloc haut : dossiers + formats + device/model
        self.mainInputWidget = qt.QWidget(self)
        main_input_layout = qt.QFormLayout(self.mainInputWidget); main_input_layout.setContentsMargins(0,0,0,0)
        main_input_layout.addRow(self.inputWidget)
        main_input_layout.addRow(self.segmentationNodeSelector)
        main_input_layout.addRow("Device:", self.deviceComboBox)
        main_input_layout.addRow("Model:",  self.modelComboBox)
        layout.addWidget(self.mainInputWidget)

        self._addModelScopeDescription()

        # ========================================================================
        # 5b)  PROCESSING QUEUE
        # ========================================================================
        self._buildQueueUi(layout)

        # Apply / Stop widgets
        self.applyButton = createButton(
            "Apply", callback=self.onApplyClicked,
            toolTip="Run the segmentation.", icon=icon("start_icon.png"))

        self.currentInfoTextEdit = qt.QTextEdit(); self.currentInfoTextEdit.setReadOnly(True)
        self.currentInfoTextEdit.setLineWrapMode(qt.QTextEdit.NoWrap)
        # Rolling window: a multi-hour run would otherwise grow the Qt document
        # without bound and slow every insertion down. Full history stays in
        # fullInfoLogs, reachable through the "info" button.
        # PythonQt exposes Qt getters as properties: document, not document().
        self.currentInfoTextEdit.document.setMaximumBlockCount(5000)

        self.stopButton = createButton("Stop", callback=self.onStopClicked, toolTip="Stop the segmentation.")
        self.loading    = qt.QMovie(iconPath("loading.gif")); self.loading.setScaledSize(qt.QSize(24,24))
        self.loading.frameChanged.connect(self._updateStopIcon); self.loading.start()

        self.applyWidget = qt.QWidget(self)
        apply_layout = qt.QHBoxLayout(self.applyWidget); apply_layout.setContentsMargins(0,0,0,0)
        apply_layout.addWidget(self.applyButton, 1)
        apply_layout.addWidget(createButton("", callback=self.showInfoLogs,
                                        icon=icon("info.png"), toolTip="Show logs."))

        self.stopWidgetContainer = qt.QWidget(self)
        stop_layout = qt.QVBoxLayout(self.stopWidgetContainer); stop_layout.setContentsMargins(0,0,0,0)
        stop_layout.addWidget(self.stopButton); stop_layout.addWidget(self.currentInfoTextEdit)
        self.stopWidgetContainer.setVisible(False)

        layout.addWidget(self.applyWidget)
                # --- Batch scan counter (Scan i/N) ------------------------------------
        self.batchCounterLabel = qt.QLabel("", self)
        self.batchCounterLabel.setAlignment(qt.Qt.AlignCenter)
        self.batchCounterLabel.setStyleSheet("color: #666; font-style: italic; margin-top:2px;")
        self.batchCounterLabel.setVisible(False)  # visible only during a batch
        layout.addWidget(self.batchCounterLabel)
        layout.addWidget(self.stopWidgetContainer)
        layout.addWidget(self.resolveMirroringButton)

        # progress bar mirroring
        self.mirroringProgressBar = qt.QProgressBar(); self.mirroringProgressBar.setMinimum(0); self.mirroringProgressBar.setMaximum(100)
        self.mirroringProgressBar.setVisible(False); layout.addWidget(self.mirroringProgressBar)

        # 3-D + smoothing slider
        layout.addWidget(self.segmentEditorWidget)
        surf_layout = qt.QFormLayout(); surf_layout.setContentsMargins(0,0,0,0)
        surf_layout.addRow("Surface smoothing :", self.surfaceSmoothingSlider)
        layout.addLayout(surf_layout)

        layout.addStretch()

        # ========================================================================
        # 6)  INTERNAL SETUP
        # ========================================================================
        self.isStopping         = False
        self._dependencyChecker = PythonDependencyChecker()
        self.processedVolumes   = {}

        # Initialize display
        self.onInputChangedForLoadedVolume(None)
        self.updateSegmentEditorWidget()

        # Add observer to the scene
        self.sceneCloseObserver = slicer.mrmlScene.AddObserver(
            slicer.mrmlScene.EndCloseEvent, self.onSceneChanged)
        self.onSceneChanged(doStopInference=False)

        # connect logic NNUNet
        self._connectSegmentationLogic()
        self._last_save_state = {}

        # Per-scan watchdog: covers the inference itself, so a hung scan can never
        # freeze the queue. Started right before startSegmentation, stopped by the
        # single exit point _finishCurrentItem.
        self._itemWatchdog = qt.QTimer(self)
        self._itemWatchdog.setSingleShot(True)
        self._itemWatchdog.timeout.connect(self._onItemTimeout)

        # RAM guard: nnUNet keeps a (numClasses, Z, Y, X) float32 array in memory,
        # so a single wide field of view can ask for tens of GB and take the whole
        # machine down. Sampled while a scan runs; see _onMemCheck.
        self._memWatchdog = qt.QTimer(self)
        self._memWatchdog.timeout.connect(self._onMemCheck)
        self._ramHits = 0
        self._ramWarned = False
        self._ramPeakPercent = 0.0

        # Automatic crop state, set only for a scan re-queued with autoCrop.
        self._uncroppedVolumeNode = None
        self._cropOffsetIJK = None

        self._inferenceFinalized = False
        self._doneVolumeSeen = False
        self._fallbackCheckAttempts = 0
        self._fallbackLastOutputSize = None

        self._rebuildQueueTable()

    def _checkpoint(self, name):
        """Print progress for debug"""
        logger.debug(f"CHECKPOINT: {name}")
        logger.debug(f"[DEBUG] Checkpoint: {name}")
        slicer.app.processEvents()

    def _save_state_before_crash(self):
        """Save status before crash"""
        item = self.queue.current()
        self._last_save_state = {
            "current_file": item.inputPath if item else None,
            "queue_index": self.queue.index,
            "queue_summary": self.queue.summary(),
            "memory_usage": self._get_memory_usage(),
        }
        logger.critical(f"CRASH STATE DUMP: {self._last_save_state}")

    def _get_memory_usage(self):
        """Return current memory usage"""
        try:
            import psutil
            return f"{psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB"
        except Exception:
            return "n/a"

    # ══════════════════════════════════════════════════════════════════════════
    #  PROCESSING QUEUE
    # ══════════════════════════════════════════════════════════════════════════

    def _buildQueueUi(self, layout):
        self.queueTable = qt.QTableWidget(0, 4, self)
        self.queueTable.setHorizontalHeaderLabels(["Scan", "Model", "Status", "Detail"])
        self.queueTable.horizontalHeader().setStretchLastSection(True)
        self.queueTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.queueTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.queueTable.verticalHeader().setVisible(False)
        self.queueTable.setMinimumHeight(170)

        self.queueSummaryLabel = qt.QLabel("Queue empty", self)
        self.queueSummaryLabel.setStyleSheet("color:#666; font-style:italic;")

        self.chunkSizeSpinBox = qt.QSpinBox(self)
        self.chunkSizeSpinBox.setRange(1, 1000)
        self.chunkSizeSpinBox.setValue(5)
        self.chunkSizeSpinBox.setToolTip(
            "Deep cleanup (orphan nodes, GPU cache, GC) every N scans.")

        self.itemTimeoutSpinBox = qt.QSpinBox(self)
        self.itemTimeoutSpinBox.setRange(1, 600)
        self.itemTimeoutSpinBox.setValue(60)
        self.itemTimeoutSpinBox.setSuffix(" min")
        self.itemTimeoutSpinBox.setToolTip(
            "A scan exceeding this delay is marked failed and the queue moves on.")

        self.ramLimitSpinBox = qt.QSpinBox(self)
        self.ramLimitSpinBox.setRange(30, 99)
        self.ramLimitSpinBox.setValue(85)
        self.ramLimitSpinBox.setSuffix(" % of system RAM")
        self.ramLimitSpinBox.setToolTip(
            "RAM guard. A scan whose estimated peak does not fit in this budget is "
            "skipped before nnUNet starts, and a running scan crossing the limit is "
            "killed instead of letting the system swap or the OOM killer strike.")

        self.ramPreflightCheckBox = qt.QCheckBox("Skip scans too large for the free RAM", self)
        self.ramPreflightCheckBox.setChecked(True)
        self.ramPreflightCheckBox.setToolTip(
            "Estimate the peak memory of a scan from its field of view, the model "
            "target spacing and its number of labels, and skip it when it cannot fit. "
            "Uncheck to only rely on the runtime guard.")

        self.skipExistingCheckBox = qt.QCheckBox("Skip scans already segmented", self)
        self.skipExistingCheckBox.setChecked(True)
        self.skipExistingCheckBox.setToolTip(
            "An input scan whose *_Segmentation.nii.gz already exists in the output "
            "folder is not queued again.")

        self.unattendedCheckBox = qt.QCheckBox("Unattended (no pop-up)", self)
        self.unattendedCheckBox.setChecked(True)
        self.unattendedCheckBox.setToolTip(
            "Errors and export confirmations go to the log instead of a modal dialog, "
            "so the queue never waits for a click.")

        buttons_widget = qt.QWidget(self)
        buttons_layout = qt.QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.addWidget(createButton(
            "Add input folder", callback=self.onAddFolderToQueue,
            toolTip="Queue every scan of the selected input folder with the current "
                    "model / device / output folder.", parent=self))
        buttons_layout.addWidget(createButton(
            "Remove selected", callback=self.onRemoveSelectedFromQueue,
            toolTip="Remove the selected pending scans.", parent=self))
        buttons_layout.addWidget(createButton(
            "Retry failed", callback=self.onRetryFailed,
            toolTip="Append every failed scan back at the end of the queue.", parent=self))
        buttons_layout.addWidget(createButton(
            "Clear", callback=self.onClearQueue,
            toolTip="Empty the queue.", parent=self))
        buttons_layout.addWidget(createButton(
            "Free memory", callback=self.onFreeMemoryClicked,
            toolTip="Kill nnUNet processes left behind by a crashed scan and run a "
                    "deep cleanup. Use it when the RAM stays full after a failure.",
            parent=self))

        queue_widget = qt.QWidget(self)
        queue_layout = qt.QFormLayout(queue_widget)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        queue_layout.addRow(buttons_widget)
        queue_layout.addRow(self.queueTable)
        queue_layout.addRow(self.queueSummaryLabel)
        queue_layout.addRow("Deep cleanup every:", self.chunkSizeSpinBox)
        queue_layout.addRow("Timeout per scan:", self.itemTimeoutSpinBox)
        queue_layout.addRow("RAM limit:", self.ramLimitSpinBox)
        queue_layout.addRow(self.ramPreflightCheckBox)
        queue_layout.addRow(self.skipExistingCheckBox)
        queue_layout.addRow(self.unattendedCheckBox)

        addInCollapsibleLayout(queue_widget, layout, "Processing queue", isCollapsed=False)

    # ─── Queue edition ─────────────────────────────────────────────────────────

    def onAddFolderToQueue(self):
        if not self.folderPath:
            slicer.util.errorDisplay("Please select an input folder first.")
            return
        if not self.outputFolderPath:
            slicer.util.errorDisplay("Please select an output folder first.")
            return

        self.queue.setStatePath(self.outputFolderPath)
        added, skipped = self.queue.addFolder(
            self.folderPath,
            self.outputFolderPath,
            self.modelComboBox.currentText,
            self.deviceComboBox.currentText,
            skipExisting=self.skipExistingCheckBox.isChecked(),
        )
        self._rebuildQueueTable()
        self.onProgressInfo(f"Queue: {added} scan(s) added, {skipped} skipped.")

    def onRemoveSelectedFromQueue(self):
        rows = {index.row() for index in self.queueTable.selectionModel().selectedRows()}
        removed = self.queue.removeAt(rows)
        self._rebuildQueueTable()
        self.onProgressInfo(f"Queue: {removed} pending scan(s) removed.")

    @staticmethod
    def _isRamFailure(error):
        """True for the two reasons the RAM guard writes in the Detail column."""
        text = (error or "").lower()
        return "ram" in text or "out of memory" in text

    def _askAutoCropConfirmation(self, ramFailures):
        """
        Offer the automatic crop for the scans the RAM guard refused.

        Deliberately modal: this one is a decision on the clinical data, taken
        by someone sitting in front of the module, not during an unattended run.
        """
        names = "\n".join(f"  • {item.name}" for item in ramFailures[:10])
        if len(ramFailures) > 10:
            names += f"\n  … and {len(ramFailures) - 10} more"

        answer = qt.QMessageBox.question(
            self,
            "Crop before retrying?",
            f"{len(ramFailures)} scan(s) failed because their field of view does not "
            f"fit in the RAM budget:\n\n{names}\n\n"
            "Retry them with an automatic crop?\n\n"
            "The scan is cropped to the bounding box of the patient — only "
            "surrounding air is removed. It is a plain voxel selection: no "
            "resampling, no interpolation, and the intensities nnUNet sees are "
            "unchanged (CT normalization uses fixed dataset statistics). The "
            "segmentation is written back on the original grid, so the output "
            "still matches the scan as acquired.\n\n"
            "A field of view that is genuinely too wide (a full head CT) will "
            "still not fit and will fail again: those need a manual crop on the "
            "dento-maxillo-facial region.\n\n"
            "Yes: crop and retry.        No: retry unchanged.",
            qt.QMessageBox.Yes | qt.QMessageBox.No,
            qt.QMessageBox.Yes)
        return answer == qt.QMessageBox.Yes

    def onRetryFailed(self):
        ram_failures = [item for item in self.queue.items
                       if item.status == STATUS_FAILED and self._isRamFailure(item.error)]
        auto_crop = bool(ram_failures) and self._askAutoCropConfirmation(ram_failures)

        requeued = self.queue.retryFailed(
            shouldAutoCrop=(lambda item: self._isRamFailure(item.error)) if auto_crop else None)
        self._rebuildQueueTable()
        if auto_crop:
            self.onProgressInfo(
                f"Queue: {requeued} failed scan(s) re-queued, "
                f"{len(ram_failures)} of them with automatic crop.")
        else:
            self.onProgressInfo(f"Queue: {requeued} failed scan(s) re-queued.")

    def onClearQueue(self):
        self.queue.clear()
        self._rebuildQueueTable()

    def _restoreQueueFromDisk(self):
        """Offer to resume the run recorded in the output folder, if any."""
        if not self.outputFolderPath:
            return
        candidate = SegmentationQueue()
        candidate.setStatePath(self.outputFolderPath)
        if not candidate.load() or candidate.isEmpty() or candidate.isFinished():
            self.queue.setStatePath(self.outputFolderPath)
            return

        remaining = len(candidate.items) - candidate.index
        answer = qt.QMessageBox.question(
            self, "Resume previous run",
            f"An interrupted run was found in this output folder "
            f"({candidate.summary()}).\n\nResume it? ({remaining} scan(s) left)"
        )
        if answer == qt.QMessageBox.Yes:
            self.queue = candidate
            self.chunkSizeSpinBox.setValue(self.queue.chunkSize)
            self.onProgressInfo(f"Queue restored: {self.queue.summary()}")
        else:
            self.queue.setStatePath(self.outputFolderPath)
        self._rebuildQueueTable()

    # ─── Queue display ─────────────────────────────────────────────────────────

    _STATUS_COLORS = {
        STATUS_PENDING: "#666666",
        STATUS_RUNNING: "#0a6ebd",
        STATUS_DONE: "#1a7f37",
        STATUS_FAILED: "#b42318",
    }

    def _rebuildQueueTable(self):
        """Full rebuild — only on structural changes, never per processed scan."""
        self.queueTable.setRowCount(len(self.queue.items))
        for row in range(len(self.queue.items)):
            self._updateQueueRow(row, rebuild=True)
        self.queueTable.resizeColumnsToContents()
        self._updateQueueSummary()

    def _updateQueueRow(self, row, rebuild=False):
        if not 0 <= row < len(self.queue.items):
            return
        item = self.queue.items[row]
        detail = item.error if item.error else (
            f"{item.durationSec:.0f}s" if item.durationSec else "")
        values = [item.name, item.model, item.status, detail]
        for column, value in enumerate(values):
            cell = None if rebuild else self.queueTable.item(row, column)
            if cell is None:
                cell = qt.QTableWidgetItem()
                self.queueTable.setItem(row, column, cell)
            cell.setText(value)
            cell.setToolTip(item.inputPath if column == 0 else value)
        status_cell = self.queueTable.item(row, 2)
        status_cell.setForeground(qt.QBrush(qt.QColor(self._STATUS_COLORS.get(item.status, "#666666"))))

    def _updateQueueSummary(self):
        if self.queue.isEmpty():
            self.queueSummaryLabel.setText("Queue empty — Apply will queue the input folder.")
        else:
            self.queueSummaryLabel.setText(self.queue.summary())

    # ─── Queue execution ───────────────────────────────────────────────────────

    def _isUnattended(self):
        return self.unattendedCheckBox.isChecked()

    def _notify(self, message, is_error=False):
        """Log; only interrupt the user when not running unattended."""
        self.onProgressInfo(message)
        if self._isUnattended():
            return
        if is_error:
            slicer.util.errorDisplay(message)
        else:
            slicer.util.infoDisplay(message)

    def _startQueue(self):
        if self.queue.isEmpty():
            slicer.util.errorDisplay("The queue is empty. Add an input folder first.")
            self._setApplyVisible(True)
            return
        if self.queue.isFinished():
            slicer.util.errorDisplay(
                "Every scan of the queue has already been processed.\n"
                'Use "Retry failed" or "Clear" to start over.')
            self._setApplyVisible(True)
            return

        self.queue.chunkSize = self.chunkSizeSpinBox.value
        self._queueRunning = True
        self._deviceFallbackAccepted = None
        self.onProgressInfo(f"=== Starting queue: {self.queue.summary()} ===")
        self._startNextItem()

    def _startNextItem(self):
        if not self._queueRunning or self.isStopping:
            return

        item = self.queue.current()
        if item is None:
            self._onQueueFinished()
            return

        if self.queue.isChunkBoundary():
            self._coolDown()

        try:
            item.status = STATUS_RUNNING
            self._updateQueueRow(self.queue.index)
            self._updateQueueSummary()
            self._itemFinalized = False
            self._itemStartTime = qt.QDateTime.currentDateTime()
            self.outputFolderPath = item.outputDir
            Path(item.outputDir).mkdir(parents=True, exist_ok=True)
            self._selectComboItem(self.modelComboBox, item.model)
            self._selectComboItem(self.deviceComboBox, item.device)

            self.currentFileIndex = self.queue.index
            self._updateBatchCounter(show_file_name=True)
            self.onProgressInfo(
                f"--- Scan {self.queue.index + 1}/{len(self.queue.items)}: {item.name} "
                f"[{item.model} / {item.device}] ---")

            self._itemWatchdog.start(self.itemTimeoutSpinBox.value * 60_000)

            loaded_volume = slicer.util.loadVolume(item.inputPath)
            self._releaseCropNodes()
            if getattr(item, "autoCrop", False):
                loaded_volume = self._applyAutoCrop(loaded_volume)
            self.currentVolumeNode = loaded_volume
            self.onInputChangedForLoadedVolume(loaded_volume)

            if not self._ramPreflightOk(loaded_volume, item):
                return
            self._memWatchdogStart()

            self.onApplyClickedForVolume(loaded_volume)

        except Exception as e:
            logger.error(f"Failed to start {item.inputPath}: {e}", exc_info=True)
            self._save_state_before_crash()
            self._finishCurrentItem(STATUS_FAILED, f"start failed: {e}")

    @staticmethod
    def _selectComboItem(combo_box, text):
        index = combo_box.findText(text)
        if index >= 0 and index != combo_box.currentIndex:
            combo_box.setCurrentIndex(index)

    def _finishCurrentItem(self, status, error=""):
        """Single exit point for a scan: records the result and schedules the next."""
        if self._itemFinalized:
            return
        self._itemFinalized = True
        self._itemWatchdog.stop()
        self._memWatchdogStop()

        # nnUNet workers can outlive their parent — a scan that died silently
        # would otherwise keep its memory for the rest of the run.
        self._reclaimStrayProcesses()
        self._releaseCropNodes()
        if self._ramPeakPercent:
            self.onProgressInfo(f"[RAM] Peak during this scan: {self._ramPeakPercent:.0f}%")

        duration = 0.0
        if self._itemStartTime is not None:
            duration = self._itemStartTime.msecsTo(qt.QDateTime.currentDateTime()) / 1000.0
        item = self.queue.advance(status, error, duration)
        if item is not None:
            self._updateQueueRow(self.queue.index - 1)
        self._updateQueueSummary()

        if not self._queueRunning or self.isStopping:
            self._setApplyVisible(True)
            return
        qt.QTimer.singleShot(150, self._startNextItem)

    def _onItemTimeout(self):
        item = self.queue.current()
        name = item.name if item else "unknown"
        self.onProgressInfo(
            f"[TIMEOUT] {name} exceeded {self.itemTimeoutSpinBox.value} min — skipping.")
        logger.error(f"Timeout on {name}")

        # Same exit as the RAM guard: kill the whole process tree, clean up,
        # and move on. A hung scan usually holds memory too.
        self._abortCurrentItem(f"timeout after {self.itemTimeoutSpinBox.value} min")

    def _onQueueFinished(self):
        self._queueRunning = False
        self._setApplyVisible(True)
        self._updateBatchCounter(show_file_name=False)
        summary = self.queue.summary()
        self.onProgressInfo(f"=== Queue finished: {summary} ===")

        failed = [i for i in self.queue.items if i.status == STATUS_FAILED]
        if failed:
            details = "\n".join(f"  • {i.name}: {i.error}" for i in failed[:20])
            if len(failed) > 20:
                details += f"\n  … and {len(failed) - 20} more"
            self.onProgressInfo(f"Failed scans:\n{details}")
        self._notify(f"Queue finished — {summary}")

    # ─── RAM guard ─────────────────────────────────────────────────────────────
    #
    # nnUNet resamples the scan to the model target spacing, then holds a
    # (numClasses, Z, Y, X) float32 logits array. A wide field of view on a
    # many-label model therefore needs tens of GB: one such scan filled 114 GB
    # of a 125 GB machine, was killed by the OOM killer, and left its
    # multiprocessing workers behind still holding that memory.
    #
    # Three lines of defence:
    #   1. _ramPreflightOk   — estimate before starting, skip what cannot fit
    #   2. _onMemCheck       — sample while running, abort before the system dies
    #   3. _killInferenceTree / _reclaimStrayProcesses — always release the RAM

    _RAM_SAMPLE_MS = 3000
    _RAM_CONSECUTIVE_HITS = 2       # a transient spike must not kill a good scan
    _RAM_FIXED_OVERHEAD_GB = 4.0    # torch + weights + workers, constant per scan
    _NNUNET_PROCESS_MARKERS = ("nnunetv2_predict", "nnunetv2/inference", "nnunet")

    @staticmethod
    def _psutil():
        try:
            import psutil
            return psutil
        except ImportError:
            return None

    def _virtualMemory(self):
        psutil = self._psutil()
        if psutil is None:
            return None
        try:
            return psutil.virtual_memory()
        except Exception:
            return None

    def _memWatchdogStart(self):
        self._ramHits = 0
        self._ramWarned = False
        self._ramPeakPercent = 0.0
        if self._virtualMemory() is not None:
            self._memWatchdog.start(self._RAM_SAMPLE_MS)

    def _memWatchdogStop(self):
        self._memWatchdog.stop()

    def _onMemCheck(self):
        vm = self._virtualMemory()
        if vm is None or self._itemFinalized:
            return

        percent = vm.percent
        limit = self.ramLimitSpinBox.value
        self._ramPeakPercent = max(self._ramPeakPercent, percent)

        if percent >= limit - 10 and not self._ramWarned:
            self._ramWarned = True
            self.onProgressInfo(
                f"[RAM] {percent:.0f}% used, {vm.available / 2 ** 30:.1f} GB free — "
                f"approaching the {limit}% limit.")

        if percent < limit:
            self._ramHits = 0
            return

        # Require several samples in a row: a short peak at the end of a scan is
        # normal, a runaway allocation is not.
        self._ramHits += 1
        self.onProgressInfo(
            f"[RAM] {percent:.0f}% used — over the {limit}% limit "
            f"({self._ramHits}/{self._RAM_CONSECUTIVE_HITS} samples)")
        if self._ramHits < self._RAM_CONSECUTIVE_HITS:
            return

        item = self.queue.current()
        name = item.name if item else "unknown"
        logger.error(f"RAM guard triggered on {name}: {percent:.1f}% used")
        self.onProgressInfo(
            f"[RAM] Aborting {name}: {percent:.0f}% of system RAM used "
            f"({vm.available / 2 ** 30:.1f} GB free).")
        self._abortCurrentItem(f"out of memory ({percent:.0f}% RAM used)")

    def _abortCurrentItem(self, reason):
        """Kill the inference, release its memory, and let the queue move on."""
        self._memWatchdogStop()
        self._itemWatchdog.stop()
        # A killed process still emits inferenceFinished / errorOccurred:
        # neutralize that path so a dead scan is not processed anyway.
        self._inferenceFinalized = True

        killed = self._killInferenceTree()
        if killed:
            self.onProgressInfo(f"Inference process tree killed ({killed} process(es)).")
        try:
            self._cleanupAfterCase(self.currentVolumeNode, self.getCurrentSegmentationNode())
        except Exception as e:
            # Never let a cleanup failure keep the queue from moving on.
            logger.error(f"Cleanup after abort failed: {e}", exc_info=True)
        self._coolDown()
        self._finishCurrentItem(STATUS_FAILED, reason)

    def _inferenceProcessId(self):
        try:
            return int(self.logic.inferenceProcess.process.processId())
        except Exception:
            return 0

    def _isInferenceProcessRunning(self):
        try:
            return self.logic.inferenceProcess.process.state() != qt.QProcess.NotRunning
        except Exception:
            return False

    def _killInferenceTree(self, timeout_sec=10):
        """
        Kill nnUNet *and every descendant*.

        QProcess.kill() only reaches the direct child. nnUNet spawns
        multiprocessing workers for preprocessing and export: those survive,
        keep their share of the RAM, and are why a run stays wedged even after
        the offending scan is removed from the queue.
        """
        killed = 0
        psutil = self._psutil()
        pid = self._inferenceProcessId()

        if psutil is not None and pid:
            try:
                parent = psutil.Process(pid)
                victims = parent.children(recursive=True) + [parent]
                for proc in victims:
                    try:
                        proc.kill()
                        killed += 1
                    except psutil.NoSuchProcess:
                        pass
                    except Exception as e:
                        logger.error(f"Could not kill pid {proc.pid}: {e}")
                psutil.wait_procs(victims, timeout=timeout_sec)
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                logger.error(f"Killing the inference tree failed: {e}", exc_info=True)

        try:
            self.logic.stopSegmentation()
            self.logic.waitForSegmentationFinished()
        except Exception:
            logger.debug("Could not stop the segmentation after the kill", exc_info=True)

        return killed + self._reclaimStrayProcesses()

    def _reclaimStrayProcesses(self):
        """
        Kill nnUNet processes nobody owns any more.

        Only touches our own descendants and orphans re-parented to init, so a
        second Slicer instance segmenting on the same machine is left alone.
        Never runs while our own inference is alive.
        """
        psutil = self._psutil()
        if psutil is None or self._isInferenceProcessRunning():
            return 0

        try:
            me = psutil.Process()
            own_pids = {me.pid} | {child.pid for child in me.children(recursive=True)}
        except Exception:
            return 0

        killed = 0
        for proc in psutil.process_iter(["pid", "ppid", "cmdline", "username"]):
            try:
                if proc.pid == me.pid:
                    continue
                info = proc.info
                cmdline = " ".join(info.get("cmdline") or []).lower()
                if not any(marker in cmdline for marker in self._NNUNET_PROCESS_MARKERS):
                    continue
                is_ours = proc.pid in own_pids
                is_orphan = info.get("ppid") == 1 and info.get("username") == me.username()
                if not (is_ours or is_orphan):
                    continue
                proc.kill()
                killed += 1
                logger.info(f"Killed stray nnUNet process {proc.pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception as e:
                logger.error(f"Stray process sweep failed on a process: {e}")
        if killed:
            self.onProgressInfo(f"[RAM] {killed} stray nnUNet process(es) killed.")
        return killed

    def onFreeMemoryClicked(self):
        """Manual recovery: release whatever a crashed scan left behind."""
        if self._queueRunning:
            slicer.util.warningDisplay(
                "Stop the queue before freeing memory: this kills the running inference.")
            return
        before = self._virtualMemory()
        killed = self._killInferenceTree()
        self._removeOrphanNodes()
        self._coolDown()
        after = self._virtualMemory()
        if before is not None and after is not None:
            freed = (after.available - before.available) / 2 ** 30
            self.onProgressInfo(
                f"[RAM] {killed} process(es) killed, {freed:+.1f} GB reclaimed "
                f"({after.available / 2 ** 30:.1f} GB free now).")
        else:
            self.onProgressInfo(f"[RAM] {killed} process(es) killed.")

    # ─── Automatic crop (RAM retry) ────────────────────────────────────────────
    #
    # Opt-in, per scan, set by "Retry failed" after an explicit confirmation.
    # The crop is an axis-aligned voxel subset: same spacing, same axes, only the
    # origin moves. Nothing is interpolated, so the label map is pasted back on
    # the original grid with a plain index offset and the exported NIfTI still
    # matches the scan as acquired.

    _CROP_MARGIN_MM = 15.0
    _CROP_MIN_GAIN = 0.85       # below 15% removed, cropping is not worth it

    @staticmethod
    def _airThreshold(array):
        """
        Air / tissue split, tolerant to the intensity scale.

        A calibrated CT has air near -1000 HU. A CBCT can be shifted or rescaled,
        so fall back to a low fraction of the intensity range — the 15 mm margin
        covers what such a rough threshold may clip.
        """
        minimum = float(array.min())
        if minimum < -800:                       # Hounsfield units
            return -500.0
        maximum = float(np.percentile(array, 99.5))
        return minimum + 0.15 * (maximum - minimum)

    def _applyAutoCrop(self, volumeNode):
        """
        Return the node to segment: cropped to the patient, or the original one
        when cropping would not help. Sets the state needed to paste the result
        back on the original grid.
        """
        try:
            array = slicer.util.arrayFromVolume(volumeNode)          # (K, J, I)
            mask = array > self._airThreshold(array)
            if not mask.any():
                self.onProgressInfo("[CROP] Nothing above the air threshold — crop skipped.")
                return volumeNode

            spacing = volumeNode.GetSpacing()                          # (I, J, K)
            margin_vox = [max(int(round(self._CROP_MARGIN_MM / s)), 1) for s in spacing]

            bounds = []
            for axis, axis_margin in enumerate((margin_vox[2], margin_vox[1], margin_vox[0])):
                projected = mask.any(axis=tuple(a for a in (0, 1, 2) if a != axis))
                indices = np.where(projected)[0]
                low = max(int(indices[0]) - axis_margin, 0)
                high = min(int(indices[-1]) + 1 + axis_margin, mask.shape[axis])
                bounds.append((low, high))

            kept = 1.0
            for (low, high), size in zip(bounds, mask.shape):
                kept *= (high - low) / float(size)
            if kept > self._CROP_MIN_GAIN:
                self.onProgressInfo(
                    f"[CROP] Field of view already tight ({kept * 100:.0f}% kept) — "
                    "crop skipped, this scan needs a manual crop.")
                return volumeNode

            (k0, k1), (j0, j1), (i0, i1) = bounds
            cropped_array = array[k0:k1, j0:j1, i0:i1]

            cropped = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", f"{volumeNode.GetName()}_cropped")
            slicer.util.updateVolumeFromArray(cropped, cropped_array)

            # Keep spacing and axes, move the origin to the new first voxel.
            ijk_to_ras = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijk_to_ras)
            new_origin = [0.0, 0.0, 0.0, 1.0]
            ijk_to_ras.MultiplyPoint([i0, j0, k0, 1.0], new_origin)
            cropped_ijk_to_ras = vtk.vtkMatrix4x4()
            cropped_ijk_to_ras.DeepCopy(ijk_to_ras)
            for row in range(3):
                cropped_ijk_to_ras.SetElement(row, 3, new_origin[row])
            cropped.SetIJKToRASMatrix(cropped_ijk_to_ras)

            self._uncroppedVolumeNode = volumeNode
            self._cropOffsetIJK = (i0, j0, k0)
            self.onProgressInfo(
                f"[CROP] {array.shape[2]}x{array.shape[1]}x{array.shape[0]} -> "
                f"{cropped_array.shape[2]}x{cropped_array.shape[1]}x{cropped_array.shape[0]} "
                f"({kept * 100:.0f}% of the voxels kept, {self._CROP_MARGIN_MM:.0f} mm margin)")
            return cropped

        except Exception as e:
            logger.error(f"Automatic crop failed: {e}", exc_info=True)
            self.onProgressInfo(f"[CROP] Failed ({e}) — segmenting the scan unchanged.")
            self._uncroppedVolumeNode = None
            self._cropOffsetIJK = None
            return volumeNode

    def _restoreCropToOriginalGrid(self, labelArray, cropped_node):
        """
        Paste a label array computed on the cropped grid back into the original.

        Returns (array, node whose geometry the output must use).
        """
        original = self._uncroppedVolumeNode
        offset = self._cropOffsetIJK
        if original is None or offset is None:
            return labelArray, cropped_node

        try:
            dims = original.GetImageData().GetDimensions()             # (I, J, K)
            full = np.zeros((dims[2], dims[1], dims[0]), dtype=labelArray.dtype)
            i0, j0, k0 = offset
            full[k0:k0 + labelArray.shape[0],
                 j0:j0 + labelArray.shape[1],
                 i0:i0 + labelArray.shape[2]] = labelArray
            self.onProgressInfo("[CROP] Result pasted back on the original grid.")
            return full, original
        except Exception as e:
            # Better a segmentation on the cropped grid than none at all.
            logger.error(f"Could not restore the crop: {e}", exc_info=True)
            self.onProgressInfo(f"[CROP][WARN] Output kept on the cropped grid: {e}")
            return labelArray, cropped_node

    def _releaseCropNodes(self):
        node = self._uncroppedVolumeNode
        self._uncroppedVolumeNode = None
        self._cropOffsetIJK = None
        if node is None:
            return
        try:
            if slicer.mrmlScene.GetNodeByID(node.GetID()):
                slicer.mrmlScene.RemoveNode(node)
        except (AttributeError, RuntimeError):
            # The node has already been removed from the scene elsewhere.
            logger.debug("Node already released", exc_info=True)

    # ─── RAM pre-flight estimate ───────────────────────────────────────────────

    def _modelBasePath(self, modelName):
        """Folder holding dataset.json / plans.json for the given model."""
        resources = Path(__file__).parent.joinpath("..", "Resources", "ML").resolve()
        datasets = {
            "PediatricDentalsegmentator": "Dataset001_380CT",
            "NasoMaxillaDentSeg": "Dataset001_max4",
            "UniversalLabDentalsegmentator": "Dataset002_380CT",
        }
        if modelName in datasets:
            return resources.joinpath(
                datasets[modelName], "nnUNetTrainer__nnUNetPlans__3d_fullres")
        # DentalSegmentator ships its own dataset folder under Resources/ML.
        return resources

    @staticmethod
    def _configurationFolder(basePath, max_depth=3):
        """
        Folder holding dataset.json, searched the way nnUNet itself does it:
        shallowest match wins (see Parameter._getFirstFolderWithDatasetFile).
        """
        pattern = "dataset.json"
        for _ in range(max_depth):
            match = next(Path(basePath).glob(pattern), None)
            if match is not None:
                return match.parent
            pattern = f"*/{pattern}"
        return None

    @staticmethod
    def _readJson(path):
        try:
            with open(path, "r") as handle:
                return json.load(handle)
        except Exception as e:
            logger.debug(f"Could not read {path}: {e}")
            return None

    def _estimatePeakRamGb(self, volumeNode, modelName):
        """
        Rough peak RAM for one nnUNet inference, in GB, or None if unknown.

        Dominant term: the logits array, (numClasses, Z, Y, X) float32 on the
        grid resampled to the model target spacing. The physical volume of the
        field of view is spacing-independent, so the resampled voxel count is
        simply fovVolume / targetVoxelVolume.
        """
        try:
            image_data = volumeNode.GetImageData()
            if image_data is None:
                return None
            dims = image_data.GetDimensions()
            spacing = volumeNode.GetSpacing()
            fov_mm3 = (dims[0] * spacing[0]) * (dims[1] * spacing[1]) * (dims[2] * spacing[2])

            # Both files must come from the same configuration folder, otherwise
            # a spacing and a label count from two different models get mixed.
            config_folder = self._configurationFolder(self._modelBasePath(modelName))
            if config_folder is None:
                return None
            plans = self._readJson(config_folder.joinpath("plans.json"))
            dataset = self._readJson(config_folder.joinpath("dataset.json"))
            if not plans or not dataset:
                return None

            target_spacing = plans.get("configurations", {}).get("3d_fullres", {}).get("spacing")
            labels = dataset.get("labels") or {}
            if not target_spacing or not labels:
                return None

            target_voxel_mm3 = float(target_spacing[0]) * float(target_spacing[1]) * float(target_spacing[2])
            if target_voxel_mm3 <= 0:
                return None

            original_voxels = float(dims[0]) * dims[1] * dims[2]
            resampled_voxels = fov_mm3 / target_voxel_mm3
            num_classes = len(labels)

            # Peak is reached in resample_data_or_seg, called on the way out
            # through convert_predicted_logits_to_segmentation_with_correct_shape.
            # Live at that moment, per class:
            #   - the sliding-window accumulator, torch.half on the resampled
            #     grid, still referenced by the caller          -> 2 bytes
            #   - its float64 copy: `data = data.astype(float)` casts the whole
            #     4D array at once, before the per-class loop   -> 8 bytes
            #   - reshaped_final, torch.half on the original grid -> 2 bytes
            logits_gb = (10.0 * num_classes * resampled_voxels
                        + 2.0 * num_classes * original_voxels) / 2 ** 30
            # Per-class transients of skimage resize (spline coefficients and
            # output buffer, float64).
            images_gb = 8.0 * (original_voxels + resampled_voxels) / 2 ** 30
            # torch, the weights and the worker processes cost the same on every
            # scan; the arrays are what makes one scan explode.
            return logits_gb + images_gb + self._RAM_FIXED_OVERHEAD_GB
        except Exception as e:
            logger.debug(f"RAM estimate unavailable: {e}")
            return None

    def _ramPreflightOk(self, volumeNode, item):
        """
        False when the scan is skipped: it cannot fit in the RAM budget.

        Skipping costs seconds; letting it run costs the machine.
        """
        estimate = self._estimatePeakRamGb(volumeNode, item.model)
        vm = self._virtualMemory()
        if estimate is None or vm is None:
            return True

        available_gb = vm.available / 2 ** 30
        budget_gb = available_gb * (self.ramLimitSpinBox.value / 100.0)
        self.onProgressInfo(
            f"[RAM] Estimated peak: {estimate:.1f} GB — free: {available_gb:.1f} GB, "
            f"budget: {budget_gb:.1f} GB")
        if estimate <= budget_gb or not self.ramPreflightCheckBox.isChecked():
            if estimate > budget_gb:
                self.onProgressInfo(
                    "[RAM] Over budget, but the pre-flight skip is disabled — "
                    "the runtime guard stays armed.")
            return True

        self.onProgressInfo(
            f"[RAM] Skipping {item.name}: needs ~{estimate:.0f} GB, "
            f"only {budget_gb:.0f} GB usable. Crop the field of view or free memory "
            f'and use "Retry failed".')
        self._memWatchdogStop()
        self._inferenceFinalized = True
        try:
            self._cleanupAfterCase(volumeNode, None)
        except Exception as e:
            logger.error(f"Cleanup after pre-flight skip failed: {e}", exc_info=True)
        self._finishCurrentItem(
            STATUS_FAILED,
            f"skipped: needs ~{estimate:.0f} GB RAM, {available_gb:.0f} GB free")
        return False

    def _coolDown(self):
        """Deep cleanup at a chunk boundary, to keep a long run from drifting."""
        self.onProgressInfo("--- Chunk boundary: deep cleanup ---")
        removed = self._removeOrphanNodes()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            # torch may be missing, and empty_cache fails if the CUDA context
            # has already been destroyed.
            logger.debug("CUDA cache not emptied", exc_info=True)
        gc.collect()
        self.onProgressInfo(
            f"Deep cleanup done ({removed} orphan node(s) removed). "
            f"Memory: {self._get_memory_usage()}")
        slicer.app.processEvents()

    def _removeOrphanNodes(self):
        """Drop volume / segmentation nodes left behind by an interrupted scan."""
        keep = {
            id(node) for node in (
                self.currentVolumeNode,
                self.getCurrentSegmentationNode(),
                self._prevSegmentationNode,
            ) if node is not None
        }
        removed = 0
        for class_name in ("vtkMRMLSegmentationNode",
                          "vtkMRMLLabelMapVolumeNode",
                          "vtkMRMLScalarVolumeNode"):
            for node in slicer.util.getNodesByClass(class_name):
                if id(node) in keep:
                    continue
                try:
                    slicer.mrmlScene.RemoveNode(node)
                    removed += 1
                except (AttributeError, RuntimeError):
                    logger.debug("Noeud orphelin non retire", exc_info=True)
        self.processedVolumes = {}
        return removed

    # ─── Resolve Mirroring Button Visibility ────────────────────────────────────

    def _updateResolveButtonVisibility(self, model_name):
        self.resolveMirroringButton.setVisible(model_name == "UniversalLabDentalsegmentator")

    # ─── Resolve Mirroring Function ─────────────────────────────────────────────

    def onResolveMirroring(self):
        """
        Detects and corrects mirrored segments while preserving
        the Mandible (53), Maxilla (54), and Mandibular Canal (55).

        The function first reconstructs a label map containing the official
        values, then applies the mirror correction to these same values.
        """
        import numpy as np, vtk, slicer

        # ─── UI Pre-settings ────────────────────────────────────────────────
        self.mirroringProgressBar.setVisible(True)
        self.mirroringProgressBar.setValue(0)
        slicer.app.processEvents()

        segmentation_node = self.getCurrentSegmentationNode()
        volume_node       = self.getCurrentVolumeNode()
        if not segmentation_node or not volume_node:
            slicer.util.warningDisplay("Missing volume or segmentation.")
            return

        logic = slicer.modules.segmentations.logic()

        # ─── Official label map dictionary (values ↔ names) ───────────────

        full_label_map = {
            "Upper-right third molar": 1, "Upper-right second molar": 2, "Upper-right first molar": 3,
            "Upper-right second premolar": 4, "Upper-right first premolar": 5, "Upper-right canine": 6,
            "Upper-right lateral incisor": 7, "Upper-right central incisor": 8, "Upper-left central incisor": 9,
            "Upper-left lateral incisor": 10, "Upper-left canine": 11, "Upper-left first premolar": 12,
            "Upper-left second premolar": 13, "Upper-left first molar": 14, "Upper-left second molar": 15,
            "Upper-left third molar": 16, "Lower-left third molar": 17, "Lower-left second molar": 18,
            "Lower-left first molar": 19, "Lower-left second premolar": 20, "Lower-left first premolar": 21,
            "Lower-left canine": 22, "Lower-left lateral incisor": 23, "Lower-left central incisor": 24,
            "Lower-right central incisor": 25, "Lower-right lateral incisor": 26, "Lower-right canine": 27,
            "Lower-right first premolar": 28, "Lower-right second premolar": 29, "Lower-right first molar": 30,
            "Lower-right second molar": 31, "Lower-right third molar": 32, "Upper-right second molar (baby)": 33,
            "Upper-right first molar (baby)": 34, "Upper-right canine (baby)": 35,
            "Upper-right lateral incisor (baby)": 36, "Upper-right central incisor (baby)": 37,
            "Upper-left central incisor (baby)": 38, "Upper-left lateral incisor (baby)": 39,
            "Upper-left canine (baby)": 40, "Upper-left first molar (baby)": 41,
            "Upper-left second molar (baby)": 42, "Lower-left second molar (baby)": 43,
            "Lower-left first molar (baby)": 44, "Lower-left canine (baby)": 45,
            "Lower-left lateral incisor (baby)": 46, "Lower-left central incisor (baby)": 47,
            "Lower-right central incisor (baby)": 48, "Lower-right lateral incisor (baby)": 49,
            "Lower-right canine (baby)": 50, "Lower-right first molar (baby)": 51,
            "Lower-right second molar (baby)": 52,
            "Mandible": 53, "Maxilla": 54, "Mandibular canal": 55
        }
        reverse_full_map = {v: k for k, v in full_label_map.items()}

        # ─── 1-2. Rebuild the label map with the official values ────────────
        # Single export + LUT remap (see _buildLabelArray), instead of one
        # full-extent export per segment. The array is rasterized on the volume
        # grid, so the geometry is taken from the volume itself.
        label_array = self._buildLabelArray(segmentation_node, volume_node, full_label_map)

        ijk_to_ras = vtk.vtkMatrix4x4(); volume_node.GetIJKToRASMatrix(ijk_to_ras)
        spacing, origin = volume_node.GetSpacing(), volume_node.GetOrigin()

        # ─── 3. Protected mask & mirror table ───────────────────────────────
        protected_vals = {53, 54, 55}

        mirror_label_map = {}
        for name, val in full_label_map.items():
            if val in protected_vals:
                continue
            if "left" in name.lower():
                mirror_name = name.replace("Left", "Right").replace("left", "right")
            elif "right" in name.lower():
                mirror_name = name.replace("Right", "Left").replace("right", "left")
            else:
                continue
            mirror_val = full_label_map.get(mirror_name)
            if mirror_val:
                mirror_label_map[val] = mirror_val

        # ─── 4. Mirror plane based on incisors ──────────────────────────────
        # Everything below works on the foreground voxels only, and computes the
        # RAS "R" coordinate with numpy. The previous version called
        # vtkMatrix4x4.MultiplyPoint once per voxel from Python, for every label:
        # tens of millions of VTK calls on a full-mouth CBCT.
        fg_mask   = label_array > 0
        fg_coords = np.argwhere(fg_mask)            # (M, 3) as (z, y, x)
        fg_values = label_array[fg_mask]             # (M,)

        if fg_coords.size == 0:
            slicer.util.warningDisplay("Segmentation is empty.")
            self.mirroringProgressBar.setVisible(False)
            return

        # RAS_R = m00*x + m01*y + m02*z + m03
        m00 = ijk_to_ras.GetElement(0, 0)
        m01 = ijk_to_ras.GetElement(0, 1)
        m02 = ijk_to_ras.GetElement(0, 2)
        m03 = ijk_to_ras.GetElement(0, 3)
        fg_ras_x = (m00 * fg_coords[:, 2] + m01 * fg_coords[:, 1] + m02 * fg_coords[:, 0] + m03)

        incisive_vals = (8, 9, 24, 25)
        inc_centroids = []
        for val in incisive_vals:
            selected = fg_ras_x[fg_values == val]
            if selected.size == 0:
                slicer.util.warningDisplay("Missing central incisors, unable to calculate mirror plane.")
                self.mirroringProgressBar.setVisible(False)
                return
            inc_centroids.append(selected.mean())
        mirror_x_ras = float(np.mean(inc_centroids))

        # ─── 5. Perform mirror correction ────────────────────────────────────
        changed = []
        fg_protected = np.isin(fg_values, list(protected_vals))
        unique_vals = np.unique(fg_values)
        for i, val in enumerate(unique_vals):
            self.mirroringProgressBar.setValue(int(100 * (i + 1) / len(unique_vals)))
            slicer.app.processEvents()

            val = int(val)
            if val == 0 or val in protected_vals or val not in mirror_label_map:
                continue

            name        = reverse_full_map.get(val, f"label_{val}")
            mirror_val  = mirror_label_map[val]
            is_left     = "left" in name.lower()

            indices = np.flatnonzero((fg_values == val) & ~fg_protected)
            if indices.size == 0:
                continue

            ras_x    = fg_ras_x[indices]
            wrong_side = ras_x > mirror_x_ras if is_left else ras_x < mirror_x_ras
            indices = indices[wrong_side]
            if indices.size == 0:
                continue

            coords = fg_coords[indices]
            label_array[coords[:, 0], coords[:, 1], coords[:, 2]] = mirror_val
            # Keep the working copy in sync, so a later label sees the same state
            # the original per-voxel loop would have seen.
            fg_values[indices] = mirror_val
            changed.append(
                f"{name} → {reverse_full_map.get(mirror_val, mirror_val)} ({indices.size} vox)")

        self.mirroringProgressBar.setValue(100)

        # ─── 6. Rebuild corrected segmentation ──────────────────────────────
        corrected_lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(corrected_lm, label_array)
        corrected_lm.SetSpacing(spacing)
        corrected_lm.SetOrigin(origin)
        corrected_lm.SetIJKToRASMatrix(ijk_to_ras)

        # New name: original segmentation name + suffix
        base_name = segmentation_node.GetName() if segmentation_node else "Segmentation"
        suffix   = "_Mirrored"                           # choose your suffix here
        corrected_seg = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            base_name + suffix
        )

        corrected_seg.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
        logic.ImportLabelmapToSegmentationNode(corrected_lm, corrected_seg)
        corrected_seg.CreateClosedSurfaceRepresentation()

        # (Optional) Automatically select corrected node
        self.segmentationNodeSelector.setCurrentNode(corrected_seg)

        # ─── 7. Rename + tag segments (by creation order) ──────────────────
        final_values      = [int(v) for v in np.unique(fg_values)]              # [1,2,…,55]
        seg_ids_sorted    = list(corrected_seg.GetSegmentation().GetSegmentIDs())

        if len(final_values) != len(seg_ids_sorted):
            self.onProgressInfo("[WARN] Number of values \u200b\u200b≠ number of segments — check import.")

        for val, seg_id in zip(final_values, seg_ids_sorted):
            segment = corrected_seg.GetSegmentation().GetSegment(seg_id)
            segment.SetName(reverse_full_map.get(val, f"label_{val}"))
            segment.SetTag("LabelValue", str(val))

        self.onProgressInfo(f"Unique labels AFTER correction: {final_values}")

        # Cleanup
        slicer.mrmlScene.RemoveNode(corrected_lm)
        self.mirroringProgressBar.setVisible(False)

        msg = ("Corrected voxels:\n" + "\n".join(changed)) if changed else "No mirrored voxels detected."
        slicer.util.infoDisplay(msg)

    # ─── Model scope description ──────────────────────────────────────────────

    def _addModelScopeDescription(self):
        self.modelDescriptionLabel = qt.QLabel(self)
        self.modelDescriptionLabel.setTextFormat(qt.Qt.RichText)
        self.modelDescriptionLabel.setWordWrap(True)
        self.modelComboBox.currentTextChanged.connect(self._updateModelDescription)
        self._updateModelDescription(self.modelComboBox.currentText)
        self.mainInputWidget.layout().addRow("Model Scope:", self.modelDescriptionLabel)

    def _updateModelDescription(self, model_name):
        self.modelDescriptionLabel.setText(MODEL_DESCRIPTIONS.get(model_name, "No description available."))

    # ─── Folder and output selection ───────────────────────────────────────────

    def selectOutputFolder(self):
        folder_path = qt.QFileDialog.getExistingDirectory(self, "Select Folder to Save Segmentations")
        if folder_path:
            self.setOutputFolder(folder_path)

    def setOutputFolder(self, folder_path):
        """Take a folder as the output, as picking it in the dialog would."""
        self.outputFolderPath = folder_path
        self.outputFolderLineEdit.setText(folder_path)
        self._restoreQueueFromDisk()

    # ──────────────────────────────────────────────────────────────────────────────
    # 3)  _saveSegmentationAsNifti
    # ──────────────────────────────────────────────────────────────────────────────
    def _saveSegmentationAsNifti(self, segmentationNode, volumeNode):
        self.onProgressInfo("=== Start of saving the segmentation in NIfTI ===")
        if not segmentationNode:
            self.onProgressInfo("ERROR: segmentationNode is invalid or not provided.")
            return

        if volumeNode:
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

        labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        success = slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            segmentationNode, labelmap_volume_node, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)

        if not success:
            self.onProgressInfo("ERROR: Exporting segments to the labelmap failed.")
            return

        output_path = os.path.join(self.outputFolderPath, segmentationNode.GetName() + ".nii.gz")
        saved = slicer.util.saveNode(labelmap_volume_node, output_path)
        if saved:
            self.onProgressInfo(f"Segmentation saved in {output_path}")
        else:
            self.onProgressInfo(f"Failed to save segmentation in {output_path}")

        # Clean
        slicer.mrmlScene.RemoveNode(labelmap_volume_node)


    def __del__(self):
        slicer.mrmlScene.RemoveObserver(self.sceneCloseObserver)
        super().__del__()

    def selectFolder(self):
        folder_path = qt.QFileDialog.getExistingDirectory(self, "Select Folder Containing Volumes")
        if folder_path:
            self.setInputFolder(folder_path)

    def setInputFolder(self, folder_path):
        """Take a folder as the input, as picking it in the dialog would."""
        self.folderPath = folder_path
        self.folderPathLineEdit.text = folder_path
        self.folderFiles = listVolumes(folder_path)
        self.currentFileIndex = 0
        self.onProgressInfo(f"Found {len(self.folderFiles)} file(s) in the folder.")

    # ─── Test files ────────────────────────────────────────────────────────────

    def testFilesRoot(self):
        """Where the sample data set lives, once and for all models."""
        documents = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation)
        return os.path.join(documents, slicer.app.applicationName + "Downloads", "BATCHDENTALSEG")

    def _sampleDataLog(self, message, log_level=None):
        """Show SampleData's download progress in this module's own log."""
        document = qt.QTextDocument()
        document.setHtml(message)      # SampleData formats its messages in HTML
        self.onProgressInfo(document.toPlainText())
        slicer.app.processEvents()

    def downloadTestScans(self, scans_dir):
        """The folder holding the sample CBCTs, fetched only if they are missing.

        Slicer's own fetcher is used rather than the extension's: it is what
        publishes this data set, it knows its URLs and its SHA256, and it keeps
        the published file names -- which `listVolumes` needs, since it matches
        on the extension. A file already there whose checksum matches is reused;
        one left half-written by an interrupted download does not match, and is
        fetched again.
        """
        import SampleData
        logic = SampleData.SampleDataLogic(logMessage=self._sampleDataLog)
        source = logic.sourceForSampleName(TEST_FILES_SAMPLE_NAME)
        if source is None:
            raise TestDataError(
                "This Slicer installation does not publish the %s sample data set."
                % TEST_FILES_SAMPLE_NAME)

        os.makedirs(scans_dir, exist_ok=True)
        for uri, name, checksum in zip(source.uris, source.fileNames, source.checksums):
            try:
                path = logic.downloadFile(uri, scans_dir, name, checksum)
            except (OSError, ValueError) as error:
                raise TestDataError("%s could not be downloaded from %s: %s"
                                    % (name, uri, error))
            # Checksum refused: downloadFile deletes the file and returns its
            # path all the same. Without this check, the input folder would be
            # filled with a scan that does not exist.
            if not os.path.isfile(path):
                raise TestDataError(
                    "%s was downloaded from %s but its checksum did not match."
                    % (name, uri))
        return scans_dir

    def onTestFiles(self):
        """Fill the input and output folders with a data set ready to segment.

        Every model in the list segments a CBCT, so the same scans serve them
        all; the weights are downloaded on their own when the run starts.
        """
        root = self.testFilesRoot()
        scans_dir = os.path.join(root, "Scans")
        failure = None
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.downloadTestScans(scans_dir)
        except TestDataError as error:
            failure = str(error)
        except OSError as error:
            failure = ("The sample data set could not be written into %s: %s"
                       % (scans_dir, error))
        finally:
            qt.QApplication.restoreOverrideCursor()
        if failure:
            qt.QMessageBox.warning(self, "Test files", failure)
            return

        self.setInputFolder(scans_dir)
        if not self.outputFolderLineEdit.text:
            output_dir = os.path.join(root, "Output")
            os.makedirs(output_dir, exist_ok=True)
            self.setOutputFolder(output_dir)

    # ──────────────────────────────────────────────────────────────────────────────
    # 2)  onSceneChanged
    # ──────────────────────────────────────────────────────────────────────────────
    def onSceneChanged(self, *_, doStopInference=True):
        if doStopInference:
            self.onStopClicked()

        # Keep just one SegmentEditorNode
        if not hasattr(self, "segmentEditorNode") or self.segmentEditorNode is None \
        or not slicer.mrmlScene.IsNodePresent(self.segmentEditorNode):
            self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentEditorNode")

        self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)

        self.processedVolumes   = {}
        self._prevSegmentationNode = None
        self._initSlicerDisplay()


    @staticmethod
    def _initSlicerDisplay():
        set3DViewBackgroundColors([1, 1, 1], [1, 1, 1])
        setConventionalWideScreenView()
        setBoxAndTextVisibilityOnThreeDViews(False)

    # ─── UI helpers ────────────────────────────────────────────────────────────

    def _updateStopIcon(self):
        self.stopButton.setIcon(qt.QIcon(self.loading.currentPixmap()))

    def onStopClicked(self):
        self.isStopping = True
        self._queueRunning = False
        watchdog = getattr(self, "_itemWatchdog", None)
        if watchdog is not None:
            watchdog.stop()
        if getattr(self, "_memWatchdog", None) is not None:
            self._memWatchdogStop()
        if self.logic is not None:
            # Kill the descendants too, otherwise stopping the queue leaves the
            # nnUNet workers running and the RAM taken.
            self._killInferenceTree()
        slicer.app.processEvents()
        self.isStopping = False
        self._setApplyVisible(True)

        if not self.queue.isEmpty() and not self.queue.isFinished():
            # The current scan stays "running" in the state file, so a resume
            # restarts it rather than silently skipping it.
            self.onProgressInfo(
                f"Queue paused at scan {self.queue.index + 1}/{len(self.queue.items)}. "
                "Press Apply to resume.")

    # ─── Apply segmentation ─────────────────────────────────────────────────────
    
    def onApplyClicked(self, *_):
        # --- quick validation ---
        if not self.outputFolderPath:
            slicer.util.errorDisplay("Please select an output folder.")
            return

        if self.queue.isEmpty() or self.queue.isFinished():
            # No explicit queue: Apply keeps its original meaning and enqueues
            # the whole input folder with the current model / device.
            if not self.folderPath:
                slicer.util.errorDisplay("Please select a folder containing volumes.")
                return
            if not self.folderFiles:
                slicer.util.errorDisplay("No valid volume file found in the folder.")
                return
            self.onAddFolderToQueue()
            if self.queue.isFinished():
                slicer.util.errorDisplay(
                    "Every scan of the input folder is already segmented in the output folder.\n"
                    'Uncheck "Skip scans already segmented" to process them again.'
                )
                return

        self.currentInfoTextEdit.clear()
        self._logBuffer = []
        self._setApplyVisible(False)

        # Environment setup is done once per session, not once per scan.
        if self._setupDone:
            self._startQueue()
        else:
            self._runSetupThenStartQueue()

    def _runSetupThenStartQueue(self):
        """Install the Python / nnUNet dependencies, then start the queue."""
        slicer.util.pip_install("light-the-torch")
        subprocess.check_call([sys.executable, "-m", "light_the_torch", "install", "torch", "torchvision"])
        slicer.util.pip_install("numexpr>=2.10.2")
        packages = ["numpy<2.0", "numexpr>=2.10.2","psutil"]

        def _onLine(line: str):
            self.onProgressInfo(line)

        def _onFinished(ok: bool):
            if not ok:
                qt.QMessageBox.critical(
                    self, "Installation error",
                    "Some Python library couldn't have been install.\n"
                    "Please check your connexion or restart slicer."
                )
                self._setApplyVisible(True)
                return

            # ---------- Step 2 : Internal dependencies ----------
            if not self.isNNUNetModuleInstalled() or self.logic is None:
                slicer.util.errorDisplay(
                    "This module depends on the NNUNet module. "
                    "Please install the NNUNet module and restart to proceed."
                )
                self._setApplyVisible(True)
                return

            if not self._installNNUNetIfNeeded():
                self._setApplyVisible(True)
                return

            if not self._dependencyChecker.downloadWeightsIfNeeded(_onLine):
                self._setApplyVisible(True)
                return

            # ---------- Step 3 : Process the queue ----------
            self._setupDone = True
            self._startQueue()

        self._pipRunner = PipRunner(packages, _onLine, _onFinished, parent=self)

    def _updateBatchCounter(self, show_file_name: bool = False):
        """
        Update label 'Scan i/N'.
        show_file_name : True to show name of the scan being processed.
        """
        total = len(self.queue.items)
        if total == 0:
            self.batchCounterLabel.clear()
            return

        index = min(self.queue.index, total - 1)
        counts = self.queue.counts()
        text = f"Scan {min(self.queue.index + 1, total)}/{total}"
        if show_file_name:
            text += f"  –  {self.queue.items[index].name}"
        if counts[STATUS_FAILED]:
            text += f"   ({counts[STATUS_FAILED]} failed)"

        self.batchCounterLabel.setText(text)

# ─── Volume input change handling ──────────────────────────────────────────


    def onInputChangedForLoadedVolume(self, volumeNode):
        if volumeNode:
            slicer.util.setSliceViewerLayers(background=volumeNode)
            slicer.util.resetSliceViews()
            self._restoreProcessedSegmentationForVolume(volumeNode)

    def _restoreProcessedSegmentationForVolume(self, volumeNode):
        segmentation_node = self.processedVolumes.get(volumeNode)
        self.segmentationNodeSelector.setCurrentNode(segmentation_node)

# ─── Apply segmentation for a given volume ────────────────────────────────

    def onApplyClickedForVolume(self, volumeNode):
        from SlicerNNUNetLib import Parameter
        self._inferenceFinalized = False
        self._doneVolumeSeen = False
        self._fallbackCheckAttempts = 0
        self._fallbackLastOutputSize = None
        selected_model = self.modelComboBox.currentText
        if selected_model == "PediatricDentalsegmentator":
            self.onProgressInfo(f"Selected Model: {selected_model}")

            # Base path where full model must be installed
            base_path = Path(__file__).parent.joinpath("..", "Resources", "ML", "Dataset001_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            # Choose fold_0 (you can adapt for fold_1 if needed)
            fold_path = base_path.joinpath("fold_0")
            if not fold_path.exists():
                fold_path.mkdir(parents=True, exist_ok=True)
            # Checkpoint path inside fold_0
            pediatric_checkpoint = fold_path.joinpath("checkpoint_final.pth")
            # If checkpoint doesn't exist, download checkpoint and dataset.json and plans.json inside basePath
            if not pediatric_checkpoint.exists():
                url_checkpoint = f"{PEDIATRIC_DENTAL_SEG}/checkpoint_final.pth"
                url_dataset = f"{PEDIATRIC_DENTAL_SEG}/dataset.json"
                url_plans = f"{PEDIATRIC_DENTAL_SEG}/plans.json"
                self.onProgressInfo("Downloading pediatricdentalseg model...")
                # Download checkpoint; convert Path to string for downloadFile
                slicer.util.downloadFile(url_checkpoint, str(pediatric_checkpoint))
                # Download dataset.json and plans.json in basePath
                slicer.util.downloadFile(url_dataset, str(base_path.joinpath("dataset.json")))
                slicer.util.downloadFile(url_plans, str(base_path.joinpath("plans.json")))
            # For nnUNet, modelPath must point to folder containing dataset.json and fold_x
            parameter = Parameter(folds="0", modelPath=base_path, device=self.deviceComboBox.currentText)

        elif selected_model == "NasoMaxillaDentSeg":
            self.onProgressInfo(f"Selected Model: {selected_model}")

            # Base path where full model must be installed
            base_path = Path(__file__).parent.joinpath("..", "Resources", "ML", "Dataset001_max4", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            # Choose fold_0 (you can adapt for fold_1 if needed)
            fold_path = base_path.joinpath("fold_0")
            if not fold_path.exists():
                fold_path.mkdir(parents=True, exist_ok=True)
            # Checkpoint path inside fold_0
            naso_maxilla_dent_seg_checkpoint = fold_path.joinpath("checkpoint_final.pth")
            # If checkpoint doesn't exist, download checkpoint and dataset.json and plans.json inside basePath
            if not naso_maxilla_dent_seg_checkpoint .exists():
                url_checkpoint = f"{NASOMAXILLA_DENT_SEG}/checkpoint_final.pth"
                url_dataset = f"{NASOMAXILLA_DENT_SEG}/dataset.json"
                url_plans = f"{NASOMAXILLA_DENT_SEG}/plans.json"
                self.onProgressInfo("Downloading NasoMaxillaDentSeg model...")
                # Download checkpoint; convert Path to string for downloadFile
                slicer.util.downloadFile(url_checkpoint, str(naso_maxilla_dent_seg_checkpoint))
                # Download dataset.json and plans.json in basePath
                slicer.util.downloadFile(url_dataset, str(base_path.joinpath("dataset.json")))
                slicer.util.downloadFile(url_plans, str(base_path.joinpath("plans.json")))
            # For nnUNet, modelPath must point to folder containing dataset.json and fold_x
            parameter = Parameter(folds="0", modelPath=base_path, device=self.deviceComboBox.currentText)


        elif selected_model == "UniversalLabDentalsegmentator":
            self.onProgressInfo(f"Selected Model: {selected_model}")

            # Base path where full model must be installed
            base_path = Path(__file__).parent.joinpath("..", "Resources", "ML", "Dataset002_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            # Choose fold_0 (you can adapt for fold_1 if needed)
            fold_path = base_path.joinpath("fold_0")
            if not fold_path.exists():
                fold_path.mkdir(parents=True, exist_ok=True)
            # Checkpoint path inside fold_0
            pediatric_checkpoint = fold_path.joinpath("checkpoint_final.pth")
            # If checkpoint doesn't exist, download checkpoint and dataset.json and plans.json inside basePath
            if not pediatric_checkpoint.exists():
                url_checkpoint = f"{UNIVERSAL_LAB}/checkpoint_final.pth"
                url_dataset = f"{UNIVERSAL_LAB}/dataset.json"
                url_plans = f"{UNIVERSAL_LAB}/plans.json"
                self.onProgressInfo("Downloading pediatricdentalseg model...")
                # Download checkpoint; convert Path to string for downloadFile
                slicer.util.downloadFile(url_checkpoint, str(pediatric_checkpoint))
                # Download dataset.json and plans.json in basePath
                slicer.util.downloadFile(url_dataset, str(base_path.joinpath("dataset.json")))
                slicer.util.downloadFile(url_plans, str(base_path.joinpath("plans.json")))
            # For nnUNet, modelPath must point to folder containing dataset.json and fold_x
            parameter = Parameter(folds="0", modelPath=base_path, device=self.deviceComboBox.currentText)




        else:
            self.onProgressInfo(f"Selected Model: {selected_model}")

            parameter = Parameter(folds="0", modelPath=self.nnUnetFolder(), device=self.deviceComboBox.currentText)
                
        if not parameter.isSelectedDeviceAvailable():
            device_name = parameter.device.upper()
            # Asked once for the whole queue — never once per scan.
            if self._deviceFallbackAccepted is None:
                if self._isUnattended():
                    self._deviceFallbackAccepted = True
                    self.onProgressInfo(
                        f"[WARN] {device_name} not available — falling back to CPU for the whole queue.")
                else:
                    ret = qt.QMessageBox.question(
                        self,
                        f"{device_name} device not available",
                        f"Selected device ({device_name}) is not available and will default to CPU.\n"
                        "Running the segmentation may take up to 1 hour per scan.\n"
                        "Would you like to proceed with the whole queue?"
                    )
                    self._deviceFallbackAccepted = (ret == qt.QMessageBox.Yes)
            if not self._deviceFallbackAccepted:
                self._queueRunning = False
                self._finishCurrentItem(STATUS_FAILED, f"{device_name} unavailable, aborted by user")
                self._setApplyVisible(True)
                return
        slicer.app.processEvents()
        self.logic.setParameter(parameter)
        self.logic.startSegmentation(volumeNode)

    # ─── Inference finished callback ──────────────────────────────────────────
    def _get_active_label_map(self):

        model = self.modelComboBox.currentText

        # === Universal: 55 labels (adulte + dents temporaires + mandibule/maxilla/canal)
        if model == "UniversalLabDentalsegmentator":
            return {
                "Upper-right third molar": 1, "Upper-right second molar": 2, "Upper-right first molar": 3,
                "Upper-right second premolar": 4, "Upper-right first premolar": 5, "Upper-right canine": 6,
                "Upper-right lateral incisor": 7, "Upper-right central incisor": 8, "Upper-left central incisor": 9,
                "Upper-left lateral incisor": 10, "Upper-left canine": 11, "Upper-left first premolar": 12,
                "Upper-left second premolar": 13, "Upper-left first molar": 14, "Upper-left second molar": 15,
                "Upper-left third molar": 16, "Lower-left third molar": 17, "Lower-left second molar": 18,
                "Lower-left first molar": 19, "Lower-left second premolar": 20, "Lower-left first premolar": 21,
                "Lower-left canine": 22, "Lower-left lateral incisor": 23, "Lower-left central incisor": 24,
                "Lower-right central incisor": 25, "Lower-right lateral incisor": 26, "Lower-right canine": 27,
                "Lower-right first premolar": 28, "Lower-right second premolar": 29, "Lower-right first molar": 30,
                "Lower-right second molar": 31, "Lower-right third molar": 32, "Upper-right second molar (baby)": 33,
                "Upper-right first molar (baby)": 34, "Upper-right canine (baby)": 35,
                "Upper-right lateral incisor (baby)": 36, "Upper-right central incisor (baby)": 37,
                "Upper-left central incisor (baby)": 38, "Upper-left lateral incisor (baby)": 39,
                "Upper-left canine (baby)": 40, "Upper-left first molar (baby)": 41,
                "Upper-left second molar (baby)": 42, "Lower-left second molar (baby)": 43,
                "Lower-left first molar (baby)": 44, "Lower-left canine (baby)": 45,
                "Lower-left lateral incisor (baby)": 46, "Lower-left central incisor (baby)": 47,
                "Lower-right central incisor (baby)": 48, "Lower-right lateral incisor (baby)": 49,
                "Lower-right canine (baby)": 50, "Lower-right first molar (baby)": 51,
                "Lower-right second molar (baby)": 52, "Mandible": 53, "Maxilla": 54, "Mandibular canal": 55
            }

        # === NasoMaxillaDentSeg: 6 labels
        if model == "NasoMaxillaDentSeg":
            # Warning: The order have to be the same as the training.
            return {
                "Upper Skull": 1,
                "Mandible": 2,
                "Maxilla": 3,
                "Upper Teeth": 4,
                "Lower Teeth": 5,
                "Mandibular canal": 6,
            }

        # === DentalSegmentator & PediatricDentalsegmentator: 5 labels
        # (Maxilla include in Upper Skull)
        return {
            "Upper Skull": 1,
            "Mandible": 2,
            "Upper Teeth": 3,
            "Lower Teeth": 4,
            "Mandibular canal": 5,
        }


    @staticmethod
    def _segmentLabelValue(segment, full_label_map):
        """Official scalar value of a segment: 'LabelValue' tag first, then the active map."""
        import vtk
        tag_val = vtk.mutable("")
        if segment.GetTag("LabelValue", tag_val) and tag_val.get():
            try:
                return int(tag_val.get())
            except ValueError:
                pass
        return full_label_map.get(segment.GetName())

    def _buildLabelArray(self, segNode, volNode, full_label_map):
        """
        Rebuild the multi-label array carrying the official label values.

        One ExportSegmentsToLabelmapNode call for all segments at once, followed by
        a lookup-table remap. The previous implementation rasterized the full volume
        extent once per segment, so the cost scaled with the number of segments
        (55 for UniversalLab) — here it no longer does.
        """
        import numpy as np
        import vtk

        segmentation = segNode.GetSegmentation()
        seg_ids = list(segmentation.GetSegmentIDs())
        if not seg_ids:
            raise RuntimeError("Segmentation has no segment")

        # Passing the IDs explicitly pins the mapping: exported value i+1 <-> segIds[i].
        ids = vtk.vtkStringArray()
        for seg_id in seg_ids:
            ids.InsertNextValue(seg_id)

        tmp_lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        try:
            success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segNode, ids, tmp_lm, volNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
            )
            if not success:
                raise RuntimeError("ExportSegmentsToLabelmapNode failed")
            exported = slicer.util.arrayFromVolume(tmp_lm)
        finally:
            slicer.mrmlScene.RemoveNode(tmp_lm)

        max_exported = int(exported.max()) if exported.size else 0
        lut = np.zeros(max(len(seg_ids), max_exported) + 1, dtype=np.uint16)
        for exported_value, seg_id in enumerate(seg_ids, start=1):
            segment = segmentation.GetSegment(seg_id)
            value = self._segmentLabelValue(segment, full_label_map)
            if value is None:
                self.onProgressInfo(f"[WARN] Unknown label for segment \"{segment.GetName()}\" - skipped")
                continue
            lut[exported_value] = value

        return lut[exported]

    def _currentCaseName(self):
        """Deterministic case name, so output files match what the queue expects."""
        item = self.queue.current()
        if item is not None:
            return volumeStem(item.inputPath)
        if self.currentVolumeNode is not None:
            return self.currentVolumeNode.GetName()
        return "Segmentation"

    def onInferenceFinished(self, *_):
        """End inference handling"""
        if self._inferenceFinalized:
            self.onProgressInfo("[DEBUG][SegWidget] onInferenceFinished ignored (already finalized)")
            return
        self._inferenceFinalized = True
        logger.debug(f"[DEBUG][SegWidget] onInferenceFinished called. isStopping={self.isStopping}")
        self.onProgressInfo(f"[DEBUG][SegWidget] onInferenceFinished received (isStopping={self.isStopping})")
        if self.isStopping:
            self.onProgressInfo("stop requested")
            self._setApplyVisible(True)
            return

        seg_node = vol_node = None
        status, error_detail = STATUS_DONE, ""
        try:
            # === Step 1: Initialization ===
            self.onProgressInfo("Processing results in progress...")

            # === Step 2: Load results ===
            try:
                self._loadSegmentationResults()
                seg_node = self.getCurrentSegmentationNode()
                vol_node = self.getCurrentVolumeNode()
                if not seg_node:
                    raise RuntimeError("No segmentation node found")
                if not vol_node:
                    raise RuntimeError("No volume node found")

                segmentation = seg_node.GetSegmentation()
                full_label_map = self._get_active_label_map()

                # Normalize the LabelValue tags once (cheap: one pass over segments).
                raw_values = []
                for seg_id in segmentation.GetSegmentIDs():
                    segment = segmentation.GetSegment(seg_id)
                    value = self._segmentLabelValue(segment, full_label_map)
                    if value is None:
                        self.onProgressInfo(f"[WARN] unexpected segment \"{segment.GetName()}\" - ignored")
                        continue
                    segment.SetTag("LabelValue", str(value))
                    raw_values.append(value)

                self.onProgressInfo(f"Predicted label values (raw): {sorted(set(raw_values))}")

            except Exception as e:
                raise RuntimeError(f"Failed to load results: {str(e)}")

            # === PHASE 3: NIfTI export ===
            import vtk as _vtk

            label_arr = self._buildLabelArray(seg_node, vol_node, full_label_map)
            # After an automatic crop the result goes back on the grid of the
            # scan as acquired, so the output matches what the clinician sent.
            label_arr, geometry_node = self._restoreCropToOriginalGrid(label_arr, vol_node)

            tmp_out = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            try:
                slicer.util.updateVolumeFromArray(tmp_out, label_arr)
                tmp_out.SetSpacing(geometry_node.GetSpacing())
                tmp_out.SetOrigin(geometry_node.GetOrigin())
                ijk2ras = _vtk.vtkMatrix4x4()
                geometry_node.GetIJKToRASMatrix(ijk2ras)
                tmp_out.SetIJKToRASMatrix(ijk2ras)

                output_path = str(Path(self.outputFolderPath).joinpath(
                    f"{self._currentCaseName()}_Segmentation.nii.gz"))
                saved = slicer.util.saveNode(tmp_out, output_path)
            finally:
                slicer.mrmlScene.RemoveNode(tmp_out)

            if saved:
                self.onProgressInfo(f"Segmentation saved in {output_path}")
            else:
                raise RuntimeError(f"saveNode failed for {output_path}")

            # Other formats (STL / OBJ / VTK / glTF), without any modal dialog.
            error_detail = self._exportSegmentation(seg_node, silent=True)

            # === Step 4: Success ===
            self.onProgressInfo("Processing completed successfully")
            logger.info(f"Volume processed: {vol_node.GetName() if vol_node else 'unknown'}")

        except Exception as e:
            # === Error handling ===
            status, error_detail = STATUS_FAILED, str(e)
            error_msg = f"ERROR: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            self.onProgressInfo(f"PROCESSING FAILURE:\n{error_msg}")
            self._save_state_before_crash()
            if not self._isUnattended():
                slicer.util.errorDisplay(f"Critical error:\n{error_msg}")

        finally:
            # === PHASE 5: cleanup, then hand over to the queue ===
            try:
                self._cleanupAfterCase(vol_node, seg_node)
            except Exception as cleanup_error:
                logger.critical(f"Final cleaning failure: {cleanup_error}", exc_info=True)
                self.onProgressInfo(f"CLEANING ERROR: {cleanup_error}")

            self._finishCurrentItem(status, error_detail)



    def _cleanupAfterCase(self, volumeNode, segmentationNode):

        self.onProgressInfo("Starting cleanup")
        try:
            def is_node_in_scene(node):
                if not node:
                    return False
                try:
                    return bool(slicer.mrmlScene.GetNodeByID(node.GetID()))
                except Exception:
                    return False

            try:
                self.segmentEditorWidget.blockSignals(True)
                # also neutralise the internal MRML node
                if hasattr(self, 'segmentEditorNode'):
                    self.segmentEditorWidget.setSegmentationNode(None)
                    self.segmentEditorWidget.setSourceVolumeNode(None)
            except (AttributeError, RuntimeError):
                logger.debug("Segment editor not neutralised", exc_info=True)

            # 2) Remove the display node of the segmentation
            if segmentationNode and is_node_in_scene(segmentationNode):
                seg_disp = segmentationNode.GetDisplayNode()
                if seg_disp and is_node_in_scene(seg_disp):
                    slicer.mrmlScene.RemoveNode(seg_disp)

            # 3) Remove the subject hierarchy entry THEN the node itself.
            #    RemoveNode used to be indented inside the "except" block, so it
            #    never ran: every scan left its segmentation in the scene.
            if segmentationNode:
                try:
                    sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
                    if sh_node and sh_node.GetScene():
                        item_id = sh_node.GetItemByDataNode(segmentationNode)
                        if item_id and item_id != sh_node.GetInvalidItemID():
                            sh_node.RemoveItem(item_id)
                except (AttributeError, RuntimeError):
                    logger.debug("Hierarchy item not removed", exc_info=True)

                if is_node_in_scene(segmentationNode):
                    slicer.mrmlScene.RemoveNode(segmentationNode)

            if self._prevSegmentationNode is segmentationNode:
                self._prevSegmentationNode = None

            try:
                self.segmentEditorWidget.blockSignals(False)
            except (AttributeError, RuntimeError):
                logger.debug("Editor signals not restored", exc_info=True)

            if volumeNode and is_node_in_scene(volumeNode):
                vol_disp = volumeNode.GetDisplayNode()
                if vol_disp and is_node_in_scene(vol_disp):
                    slicer.mrmlScene.RemoveNode(vol_disp)
                slicer.mrmlScene.RemoveNode(volumeNode)

            if self.currentVolumeNode is volumeNode:
                self.currentVolumeNode = None

            # 6) Keep no Python reference on deleted nodes, otherwise the
            #    gc.collect() below can free nothing.
            self.processedVolumes = {}

            # 7) CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.onProgressInfo("CUDA cache cleared")
            except ImportError:
                pass

            # 8) GC et memory
            gc.collect()
            self.onProgressInfo(f"Cleanup complete. Memory: {self._get_memory_usage()}")

        except Exception as e:
            logger.error(f"Cleanup crashed: {str(e)}", exc_info=True)
            raise

    # ─── Load segmentation results ────────────────────────────────────────────

    def _loadSegmentationResults(self):
        current_segmentation = self.getCurrentSegmentationNode()
        segmentation_node = self.logic.loadSegmentation()
        segmentation_node.SetName(self._currentCaseName() + "_Segmentation")
        if current_segmentation is not None:
            self._copySegmentationResultsToExistingNode(current_segmentation, segmentation_node)
        else:
            self.segmentationNodeSelector.setCurrentNode(segmentation_node)
        slicer.app.processEvents()
        self._updateSegmentationDisplay()
        self._storeProcessedSegmentation()

    # ─── Helper to copy segmentation results ──────────────────────────────────

    @staticmethod
    def _copySegmentationResultsToExistingNode(currentSegmentation, segmentationNode):
        current_name = currentSegmentation.GetName()
        currentSegmentation.Copy(segmentationNode)
        currentSegmentation.SetName(current_name)
        slicer.mrmlScene.RemoveNode(segmentationNode)

    @staticmethod
    def toRGB(color_string):
        color = qt.QColor(color_string)
        return color.redF(), color.greenF(), color.blueF()

    def _updateSegmentationDisplay(self):
        segmentation_node = self.getCurrentSegmentationNode()
        if not segmentation_node:
            return
        self._initializeSegmentationNodeDisplay(segmentation_node)
        segmentation = segmentation_node.GetSegmentation()
        selected_model = self.modelComboBox.currentText
       
        if selected_model == "UniversalLabDentalsegmentator":
            # For UniversalLabDentalsegmentator model,
            # we consider 55 labels (ignore "background")
            UNIVERSAL_LABELS = [
                "Upper-right third molar",
                "Upper-right second molar",
                "Upper-right first molar",
                "Upper-right second premolar",
                "Upper-right first premolar",
                "Upper-right canine",
                "Upper-right lateral incisor",
                "Upper-right central incisor",
                "Upper-left central incisor",
                "Upper-left lateral incisor",
                "Upper-left canine",
                "Upper-left first premolar",
                "Upper-left second premolar",
                "Upper-left first molar",
                "Upper-left second molar",
                "Upper-left third molar",
                "Lower-left third molar",
                "Lower-left second molar",
                "Lower-left first molar",
                "Lower-left second premolar",
                "Lower-left first premolar",
                "Lower-left canine",
                "Lower-left lateral incisor",
                "Lower-left central incisor",
                "Lower-right central incisor",
                "Lower-right lateral incisor",
                "Lower-right canine",
                "Lower-right first premolar",
                "Lower-right second premolar",
                "Lower-right first molar",
                "Lower-right second molar",
                "Lower-right third molar",
                "Upper-right second molar (baby)",
                "Upper-right first molar (baby)",
                "Upper-right canine (baby)",
                "Upper-right lateral incisor (baby)",
                "Upper-right central incisor (baby)",
                "Upper-left central incisor (baby)",
                "Upper-left lateral incisor (baby)",
                "Upper-left canine (baby)",
                "Upper-left first molar (baby)",
                "Upper-left second molar (baby)",
                "Lower-left second molar (baby)",
                "Lower-left first molar (baby)",
                "Lower-left canine (baby)",
                "Lower-left lateral incisor (baby)",
                "Lower-left central incisor (baby)",
                "Lower-right central incisor (baby)",
                "Lower-right lateral incisor (baby)",
                "Lower-right canine (baby)",
                "Lower-right first molar (baby)",
                "Lower-right second molar (baby)",
                "Mandible",
                "Maxilla",
                "Mandibular canal"
            ]

            # A palette of 55 hex colors (you can adapt the codes)
            UNIVERSAL_COLORS = [
                "#FF0000",  # Upper-right third molar
                "#00FF00",  # Upper-right second molar
                "#0000FF",  # Upper-right first molar
                "#FFFF00",  # Upper-right second premolar
                "#FF00FF",  # Upper-right first premolar
                "#00FFFF",  # Upper-right canine
                "#800000",  # Upper-right lateral incisor
                "#008000",  # Upper-right central incisor
                "#000080",  # Upper-left central incisor
                "#808000",  # Upper-left lateral incisor
                "#800080",  # Upper-left canine
                "#008080",  # Upper-left first premolar
                "#C0C0C0",  # Upper-left second premolar
                "#808080",  # Upper-left first molar
                "#FFA500",  # Upper-left second molar
                "#F0E68C",  # Upper-left third molar
                "#B22222",  # Lower-left third molar
                "#8FBC8F",  # Lower-left second molar
                "#483D8B",  # Lower-left first molar
                "#2F4F4F",  # Lower-left second premolar
                "#00CED1",  # Lower-left first premolar
                "#9400D3",  # Lower-left canine
                "#FF1493",  # Lower-left lateral incisor
                "#7FFF00",  # Lower-left central incisor
                "#1E90FF",  # Lower-right central incisor
                "#FF4500",  # Lower-right lateral incisor
                "#DA70D6",  # Lower-right canine
                "#EEE8AA",  # Lower-right first premolar
                "#98FB98",  # Lower-right second premolar
                "#AFEEEE",  # Lower-right first molar
                "#DB7093",  # Lower-right second molar
                "#FFE4E1",  # Lower-right third molar
                "#FFDAB9",  # Upper-right second molar (baby)
                "#CD5C5C",  # Upper-right first molar (baby)
                "#F08080",  # Upper-right canine (baby)
                "#E9967A",  # Upper-right lateral incisor (baby)
                "#FA8072",  # Upper-right central incisor (baby)
                "#FF7F50",  # Upper-left central incisor (baby)
                "#FF6347",  # Upper-left lateral incisor (baby)
                "#00FA9A",  # Upper-left canine (baby)
                "#00FF7F",  # Upper-left first molar (baby)
                "#4682B4",  # Upper-left second molar (baby)
                "#87CEEB",  # Lower-left second molar (baby)
                "#6A5ACD",  # Lower-left first molar (baby)
                "#7B68EE",  # Lower-left canine (baby)
                "#4169E1",  # Lower-left lateral incisor (baby)
                "#6495ED",  # Lower-left central incisor (baby)
                "#B0C4DE",  # Lower-right central incisor (baby)
                "#008080",  # Lower-right lateral incisor (baby)
                "#ADFF2F",  # Lower-right canine (baby)
                "#FF69B4",  # Lower-right first molar (baby)
                "#CD853F",  # Lower-right second molar (baby)
                "#D2691E",  # Mandible
                "#B8860B",  # Maxilla
                "#A0522D"   # Mandibular canal
            ]

            # Uniform opacity, for example 1.0 for each segment
            UNIVERSAL_OPACITIES = [
                1.0,  # Upper-right third molar
                1.0,  # Upper-right second molar
                1.0,  # Upper-right first molar
                1.0,  # Upper-right second premolar
                1.0,  # Upper-right first premolar
                1.0,  # Upper-right canine
                1.0,  # Upper-right lateral incisor
                1.0,  # Upper-right central incisor
                1.0,  # Upper-left central incisor
                1.0,  # Upper-left lateral incisor
                1.0,  # Upper-left canine
                1.0,  # Upper-left first premolar
                1.0,  # Upper-left second premolar
                1.0,  # Upper-left first molar
                1.0,  # Upper-left second molar
                1.0,  # Upper-left third molar
                1.0,  # Lower-left third molar
                1.0,  # Lower-left second molar
                1.0,  # Lower-left first molar
                1.0,  # Lower-left second premolar
                1.0,  # Lower-left first premolar
                1.0,  # Lower-left canine
                1.0,  # Lower-left lateral incisor
                1.0,  # Lower-left central incisor
                1.0,  # Lower-right central incisor
                1.0,  # Lower-right lateral incisor
                1.0,  # Lower-right canine
                1.0,  # Lower-right first premolar
                1.0,  # Lower-right second premolar
                1.0,  # Lower-right first molar
                1.0,  # Lower-right second molar
                1.0,  # Lower-right third molar
                1.0,  # Upper-right second molar (baby)
                1.0,  # Upper-right first molar (baby)
                1.0,  # Upper-right canine (baby)
                1.0,  # Upper-right lateral incisor (baby)
                1.0,  # Upper-right central incisor (baby)
                1.0,  # Upper-left central incisor (baby)
                1.0,  # Upper-left lateral incisor (baby)
                1.0,  # Upper-left canine (baby)
                1.0,  # Upper-left first molar (baby)
                1.0,  # Upper-left second molar (baby)
                1.0,  # Lower-left second molar (baby)
                1.0,  # Lower-left first molar (baby)
                1.0,  # Lower-left canine (baby)
                1.0,  # Lower-left lateral incisor (baby)
                1.0,  # Lower-left central incisor (baby)
                1.0,  # Lower-right central incisor (baby)
                1.0,  # Lower-right lateral incisor (baby)
                1.0,  # Lower-right canine (baby)
                1.0,  # Lower-right first molar (baby)
                1.0,  # Lower-right second molar (baby)
                0.45,  # Mandible
                0.45,  # Maxilla
                0.45   # Mandibular canal
            ]
            labels = UNIVERSAL_LABELS
            colors = UNIVERSAL_COLORS
            opacities = UNIVERSAL_OPACITIES
            # Create segment IDs as before, e.g. "Segment_1", "Segment_2", ...
            segment_ids = [f"Segment_{i+1}" for i in range(len(labels))]
            segmentation_display_node = segmentation_node.GetDisplayNode()
            for segment_id, label, color, opacity in zip(segment_ids, labels, colors, opacities):
                segment = segmentation.GetSegment(segment_id)
                if segment is None:
                    continue
                segment.SetName(label)
                segment.SetColor(*self.toRGB(color))
                segmentation_display_node.SetSegmentOpacity3D(segment_id, opacity)

            self.show3DButton.setChecked(True)
            slicer.util.resetThreeDViews()

        elif selected_model == "NasoMaxillaDentSeg":
            labels = ["Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal","Maxilla "]
            colors = [self.toRGB(c) for c in ["#E3DD90", "#D4A1E6","#DC9565", "#EBDFB4", "#D8654F", "#6AC4A4"]]
            opacities = [0.65, 0.65,1.0, 1.0, 1.0, 0.65]
            segment_ids = [f"Segment_{i + 1}" for i in range(len(labels))]
            segmentation_display_node = self.getCurrentSegmentationNode().GetDisplayNode()
            for segment_id, label, color, opacity in zip(segment_ids, labels, colors, opacities):
                segment = segmentation.GetSegment(segment_id)
                if segment is None:
                    continue
                segment.SetName(label)
                segment.SetColor(*color)
                segmentation_display_node.SetSegmentOpacity3D(segment_id, opacity)
            self.show3DButton.setChecked(True)
            slicer.util.resetThreeDViews()

        else:
            labels = ["Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal"]
            colors = [self.toRGB(c) for c in ["#E3DD90", "#D4A1E6","#DC9565", "#EBDFB4", "#D8654F"]]
            opacities = [0.65, 0.65,1.0, 1.0, 1.0]
            segment_ids = [f"Segment_{i + 1}" for i in range(len(labels))]
            segmentation_display_node = self.getCurrentSegmentationNode().GetDisplayNode()
            for segment_id, label, color, opacity in zip(segment_ids, labels, colors, opacities):
                segment = segmentation.GetSegment(segment_id)
                if segment is None:
                    continue
                segment.SetName(label)
                segment.SetColor(*color)
                segmentation_display_node.SetSegmentOpacity3D(segment_id, opacity)
            self.show3DButton.setChecked(True)
            slicer.util.resetThreeDViews()

    def _initializeSegmentationNodeDisplay(self, segmentationNode):
        if not segmentationNode:
            return
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self.getCurrentVolumeNode())
        if not segmentationNode.GetDisplayNode():
            segmentationNode.CreateDefaultDisplayNodes()
            slicer.app.processEvents()
        segmentationNode.SetDisplayVisibility(True)
        layout_manager = slicer.app.layoutManager()
        three_d_widget = layout_manager.threeDWidget(0)
        three_d_widget.threeDView().rotateToViewAxis(3)
        slicer.util.resetThreeDViews()

    def _postProcessSegments(self):
        self.onProgressInfo("Post processing results...")
        self.onProgressInfo("Post processing done.")

    def _keepLargestIsland(self, segmentId):
        segment = self._getSegment(segmentId)
        if not segment:
            return
        self.onProgressInfo(f"Keep largest region for {segment.GetName()}...")
        self.segmentEditorWidget.setCurrentSegmentID(segmentId)
        effect = self.segmentEditorWidget.effectByName("Islands")
        effect.setParameter("Operation", SegmentEditorEffects.KEEP_LARGEST_ISLAND)
        effect.self().onApply()

    def _removeSmallIsland(self, segmentId):
        segment = self._getSegment(segmentId)
        if not segment:
            return
        self.onProgressInfo(f"Remove small voxels for {segment.GetName()}...")
        self.segmentEditorWidget.setCurrentSegmentID(segmentId)
        voxel_size_mm3 = np.cumprod(self.getCurrentVolumeNode().GetSpacing())[-1]
        minimum_island_size = int(np.ceil(self._minimumIslandSize_mm3 / voxel_size_mm3))
        effect = self.segmentEditorWidget.effectByName("Islands")
        effect.setParameter("Operation", SegmentEditorEffects.REMOVE_SMALL_ISLANDS)
        effect.setParameter("MinimumSize", minimum_island_size)
        effect.self().onApply()

    def _getSegment(self, segmentId):
        segmentation_node = self.getCurrentSegmentationNode()
        if not segmentation_node:
            return
        return segmentation_node.GetSegmentation().GetSegment(segmentId)

    def onInferenceError(self, errorMsg):
        logger.error(f"[SegWidget] onInferenceError: {errorMsg}")
        self.onProgressInfo(f"[ERROR] Inference failed: {errorMsg}")
        if self.isStopping:
            return

        if not self._queueRunning:
            self._setApplyVisible(True)
            slicer.util.errorDisplay("Encountered error during inference :\n" + str(errorMsg))
            return

        # During a queue run a bad scan must never stop the batch: clean up and move on.
        # inferenceFinished may still be emitted afterwards — neutralize it.
        self._inferenceFinalized = True
        try:
            self._cleanupAfterCase(self.getCurrentVolumeNode(), self.getCurrentSegmentationNode())
        except Exception as e:
            logger.error(f"Cleanup after inference error failed: {e}", exc_info=True)
        self._finishCurrentItem(STATUS_FAILED, f"inference error: {errorMsg}")

    def onProgressInfo(self, info_msg):
        info_msg = self.removeImageIOError(info_msg)
        if not info_msg:
            return
        self._appendLog(info_msg)
        if "done with volume" in info_msg.lower():
            self._doneVolumeSeen = True
            self._fallbackCheckAttempts = 0
            self._fallbackLastOutputSize = None
            self._appendLog(
                "[DEBUG][SegWidget] 'done with volume' detected, starting fallback completion check")
            qt.QTimer.singleShot(1500, self._checkInferenceCompletionFallback)

    def _appendLog(self, message):
        """
        Queue a log line. The text widget is refreshed at most every 200 ms instead
        of once per line: nnUNet emits thousands of lines per run, and an
        insertPlainText + processEvents on each of them dominated the UI thread.
        """
        self._logBuffer.append(message)
        self.insertDatedInfoLogs(message)
        if not self._logFlushTimer.isActive():
            self._logFlushTimer.start()

    def _flushLogBuffer(self):
        if not self._logBuffer:
            return
        pending, self._logBuffer = self._logBuffer, []
        self.currentInfoTextEdit.insertPlainText("\n".join(pending) + "\n")
        self.moveTextEditToEnd(self.currentInfoTextEdit)
        slicer.app.processEvents()

    def _checkInferenceCompletionFallback(self):
        if self._inferenceFinalized or not self._doneVolumeSeen:
            return

        self._fallbackCheckAttempts += 1

        out_file_path = None
        out_file_size = None
        try:
            out_file_path = self.logic._outFile
            out_file_size = Path(out_file_path).stat().st_size
        except Exception:
            out_file_path = None

        process_state = None
        try:
            process_state = self.logic.inferenceProcess.process.state()
        except Exception:
            process_state = None

        self.onProgressInfo(
            f"[DEBUG][SegWidget] Fallback check #{self._fallbackCheckAttempts}: "
            f"state={process_state}, outFile={out_file_path}, size={out_file_size}"
        )

        if out_file_path and out_file_size is not None and out_file_size > 0:
            if self._fallbackLastOutputSize == out_file_size:
                self.onProgressInfo("[DEBUG][SegWidget] Output file stable, forcing finalization")
                qt.QTimer.singleShot(0, self.onInferenceFinished)
                return
            self._fallbackLastOutputSize = out_file_size

        if self._fallbackCheckAttempts < 40:
            qt.QTimer.singleShot(1500, self._checkInferenceCompletionFallback)
        else:
            self.onProgressInfo("[DEBUG][SegWidget] Fallback completion check timeout (no stable output file)")

    @staticmethod
    def removeImageIOError(info_msg):
        return "\n".join([msg for msg in info_msg.strip().splitlines() if "Error ImageIO factory" not in msg])

    def insertDatedInfoLogs(self, info_msg):
        now = qt.QDateTime.currentDateTime().toString("yyyy/MM/dd hh:mm:ss.zzz")
        self.fullInfoLogs.extend([f"{now} :: {msg_line}" for msg_line in info_msg.splitlines()])

    def showInfoLogs(self):
        dialog = qt.QDialog()
        layout = qt.QVBoxLayout(dialog)
        text_edit = qt.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.append("\n".join(self.fullInfoLogs))
        text_edit.setLineWrapMode(qt.QTextEdit.NoWrap)
        self.moveTextEditToEnd(text_edit)
        layout.addWidget(text_edit)
        dialog.setWindowFlags(qt.Qt.WindowCloseButtonHint)
        dialog.resize(slicer.util.mainWindow().size * 0.7)
        dialog.exec()


    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)

    def _setApplyVisible(self, is_visible):
        self.applyWidget.setVisible(is_visible)
        self.stopWidgetContainer.setVisible(not is_visible)
        self.inputWidget.setEnabled(is_visible)

        self.batchCounterLabel.setVisible(not is_visible)
        if not is_visible:
            self._updateBatchCounter(show_file_name=True)


    def getCurrentVolumeNode(self):
        return self.currentVolumeNode

    def getCurrentSegmentationNode(self):
        return self.segmentationNodeSelector.currentNode()

    def _storeProcessedSegmentation(self):
        volume_node = self.getCurrentVolumeNode()
        segmentation_node = self.getCurrentSegmentationNode()
        if volume_node and segmentation_node:
            self.processedVolumes[volume_node] = segmentation_node
    def updateSegmentEditorWidget(self, *_):

        # Hide previous node
        if self._prevSegmentationNode:
            try:
                self._prevSegmentationNode.SetDisplayVisibility(False)
            except (AttributeError, RuntimeError):
                logger.debug("Visibility of the previous segmentation unchanged", exc_info=True)

        segmentation_node = self.getCurrentSegmentationNode()

        # If no segmentation or deleted node, we stop here
        if not segmentation_node or not slicer.mrmlScene.IsNodePresent(segmentation_node):
            return

        # Initialization and display
        self._initializeSegmentationNodeDisplay(segmentation_node)
        self.segmentEditorWidget.setSegmentationNode(segmentation_node)
        slicer.app.processEvents()

        volume_node = self.getCurrentVolumeNode()
        if volume_node and slicer.mrmlScene.IsNodePresent(volume_node):
            self.segmentEditorWidget.setSourceVolumeNode(volume_node)
            slicer.app.processEvents()

        self._prevSegmentationNode = segmentation_node


    def getSelectedExportFormats(self):
        selected_formats = ExportFormat(0)
        check_boxes = {
            self.objCheckBox: ExportFormat.OBJ,
            self.stlCheckBox: ExportFormat.STL,
            self.niftiCheckBox: ExportFormat.NIFTI,
            self.gltfCheckBox: ExportFormat.GLTF,
            self.vtkCheckBox: ExportFormat.VTK,
            self.vtkmergedCheckBox  : ExportFormat.VTK_MERGED

        }
        for check_box, export_format in check_boxes.items():
            if check_box.isChecked():
                selected_formats |= export_format
        return selected_formats

    def onExportClicked(self):
        self._exportSegmentation(silent=False)

    def _exportSegmentation(self, segmentationNode=None, silent=False):
        """
        Export the extra formats (STL / OBJ / VTK / glTF).

        In silent mode no modal dialog is ever raised — a confirmation pop-up per
        scan would block the queue until someone clicks. Returns a warning string
        when the export failed, "" otherwise.
        """
        segmentationNode = segmentationNode or self.getCurrentSegmentationNode()
        if not segmentationNode:
            message = "Please select a valid segmentation before exporting."
            if silent:
                self.onProgressInfo(f"[WARN] {message}")
                return f"export warning: {message}"
            slicer.util.warningDisplay(message)
            return ""

        selected_formats = self.getSelectedExportFormats()
        if selected_formats == ExportFormat(0):
            if silent:
                self.onProgressInfo("No additional export format selected — NIfTI only.")
                return ""
            slicer.util.warningDisplay("Please select at least one export format before exporting.")
            return ""

        if silent:
            try:
                self.exportSegmentation(segmentationNode, self.outputFolderPath, selected_formats)
                self.onProgressInfo(f"Export successful to {self.outputFolderPath}.")
                return ""
            except Exception as e:
                # The NIfTI is already written: a mesh export failure downgrades the
                # scan to a warning, it does not invalidate the segmentation.
                logger.error(f"Additional format export failed: {e}", exc_info=True)
                self.onProgressInfo(f"[WARN] Additional format export failed: {e}")
                return f"export warning: {e}"

        with slicer.util.tryWithErrorDisplay(f"Export to {self.outputFolderPath} failed.", waitCursor=True):
            self.exportSegmentation(segmentationNode, self.outputFolderPath, selected_formats)
            slicer.util.infoDisplay(f"Export successful to {self.outputFolderPath}.")
        return ""

    def exportSegmentation(self, segNode, folderPath, selectedFormats):

        # ------------------------------------------------------------------ STL/OBJ
        for fmt in (ExportFormat.STL, ExportFormat.OBJ):
            if selectedFormats & fmt:
                slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(
                    folderPath, segNode, None, fmt.name, True, 1.0, False
                )

        # ----------------------------------------------------------------- VTK(s)
        if selectedFormats & ExportFormat.VTK_MERGED:
            self._exportMergedVTK(segNode, folderPath)

        if selectedFormats & ExportFormat.VTK:
            self._exportVTKPerLabel(segNode, folderPath)

        # -------------------------------------------------------------------- NIfTI
        if selectedFormats & ExportFormat.NIFTI:
            slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsBinaryLabelmapRepresentationToFiles(
                folderPath, segNode, None, "nii.gz"
            )

        # --------------------------------------------------------------------- glTF
        if selectedFormats & ExportFormat.GLTF:
            self._exportToGLTF(segNode, folderPath)

    # ─── 4. Pipelines helpers ──────────────────────────────────────────────────
    def _exportMergedVTK(self, segNode, folderPath):

        import vtk, os, numpy as np
        from vtk.util.numpy_support import vtk_to_numpy
        vtk.vtkObject.GlobalWarningDisplayOff()
        self.onProgressInfo("MergedVTK: Start")
        ref_vol = self.getCurrentVolumeNode()
        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segNode, labelmap)
        img = labelmap.GetImageData()

        # Marching Cubes
        # SetValue takes a contour *index*, not the label value. Passing the label
        # as index left index 0 unset (contour value 0.0), so the background was
        # contoured too and its surface went through cleaning, smoothing and
        # normals before being thrown away by the per-label thresholding below.
        self.onProgressInfo("MergedVTK: MarchingCubes")
        mc = vtk.vtkDiscreteMarchingCubes(); mc.SetInputData(img)
        foreground_labels = [int(l) for l in np.unique(vtk_to_numpy(img.GetPointData().GetScalars())) if l]
        mc.SetNumberOfContours(len(foreground_labels))
        for contour_index, label_value in enumerate(foreground_labels):
            mc.SetValue(contour_index, label_value)
        mc.Update()

        # Clean + smooth
        self.onProgressInfo("MergedVTK: Cleaning + smoothing")
        clean = vtk.vtkCleanPolyData(); clean.SetInputConnection(mc.GetOutputPort()); clean.Update()
        ws = vtk.vtkWindowedSincPolyDataFilter(); ws.SetInputConnection(clean.GetOutputPort())
        ws.SetNumberOfIterations(60); ws.SetPassBand(0.05)
        ws.BoundarySmoothingOn(); ws.FeatureEdgeSmoothingOn()
        ws.NonManifoldSmoothingOn(); ws.NormalizeCoordinatesOn(); ws.Update()

        # Normales
        self.onProgressInfo("MergedVTK: Computing normals")
        flat_n = vtk.vtkPolyDataNormals(); flat_n.SetInputConnection(ws.GetOutputPort())
        flat_n.ComputePointNormalsOff(); flat_n.ComputeCellNormalsOn()
        flat_n.SplittingOff(); flat_n.AutoOrientNormalsOn()
        flat_n.ConsistencyOn(); flat_n.SetFeatureAngle(180); flat_n.Update()

        raw_poly   = flat_n.GetOutput()
        label_array = raw_poly.GetCellData().GetScalars()
        labels     = np.unique(vtk_to_numpy(label_array))
        append     = vtk.vtkAppendPolyData()

        # Walk the labels
        for i, label_value in enumerate(labels, start=1):
            if label_value == 0:
                continue
            self.onProgressInfo(f"MergedVTK: Processing label {int(label_value)} ({i}/{len(labels)})")

            thresh = vtk.vtkThreshold()
            thresh.SetInputData(raw_poly)
            thresh.SetInputArrayToProcess(0,0,0,
                vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                label_array.GetName())
            thresh.SetLowerThreshold(label_value)
            thresh.SetUpperThreshold(label_value)
            thresh.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
            thresh.Update()

            surf = vtk.vtkDataSetSurfaceFilter(); surf.SetInputConnection(thresh.GetOutputPort()); surf.Update()

            dec = vtk.vtkQuadricDecimation()
            dec.SetInputConnection(surf.GetOutputPort())
            dec.SetTargetReduction(0.4)
            dec.Update()

            out = dec.GetOutput()
            from vtk.util.numpy_support import numpy_to_vtk
            const_label = numpy_to_vtk(
                np.full(out.GetNumberOfCells(), int(label_value), dtype=np.int32), deep=True)
            const_label.SetName("Label")
            out.GetCellData().AddArray(const_label)
            out.GetCellData().SetScalars(const_label)

            append.AddInputData(out)

        append.Update()
        self.onProgressInfo("MergedVTK: AppendPolyData done")

        # Transform + Write
        self.onProgressInfo("MergedVTK: Transform & Write")
        ijk2ras = vtk.vtkMatrix4x4(); labelmap.GetIJKToRASMatrix(ijk2ras)
        parent_mat = vtk.vtkMatrix4x4(); parent_mat.Identity()
        if ref_vol and ref_vol.GetParentTransformNode():
            ref_vol.GetParentTransformNode().GetMatrixTransformToWorld(parent_mat)
        ras_mat = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(parent_mat, ijk2ras, ras_mat)

        ras_t = vtk.vtkTransform(); ras_t.SetMatrix(ras_mat)
        ras_f = vtk.vtkTransformPolyDataFilter()
        ras_f.SetTransform(ras_t); ras_f.SetInputConnection(append.GetOutputPort()); ras_f.Update()
        lps_t = vtk.vtkTransform(); lps_t.Scale(-1,-1,1)
        lps_f = vtk.vtkTransformPolyDataFilter(); lps_f.SetTransform(lps_t)
        lps_f.SetInputConnection(ras_f.GetOutputPort()); lps_f.Update()

        out_path = os.path.join(folderPath, f"{segNode.GetName()}_merged.vtk")
        w = vtk.vtkPolyDataWriter(); w.SetFileName(out_path)
        w.SetInputData(lps_f.GetOutput()); w.SetFileTypeToBinary(); w.Write()
        slicer.mrmlScene.RemoveNode(labelmap)

        self.onProgressInfo("MergedVTK: Done")


    def _exportVTKPerLabel(self, segNode, folderPath):
        """Export one VTK file per segment, logging through onProgressInfo."""
        import vtk, os, re
        vtk.vtkObject.GlobalWarningDisplayOff()
        segNode.CreateClosedSurfaceRepresentation()
        seg       = segNode.GetSegmentation()
        seg_safe   = re.sub(r"[^0-9A-Za-z_-]+","_", segNode.GetName())
        tr        = segNode.GetParentTransformNode()
        parent_mat = vtk.vtkMatrix4x4(); parent_mat.Identity()
        if tr:
            tr.GetMatrixTransformToWorld(parent_mat)

        segment_i_ds = seg.GetSegmentIDs()
        total = len(segment_i_ds)
        for idx, seg_id in enumerate(segment_i_ds, start=1):
            self.onProgressInfo(f"PerLabelVTK: Segment {idx}/{total}")

            s    = seg.GetSegment(seg_id)
            poly = s.GetRepresentation("Closed surface")
            if not poly or poly.GetNumberOfPoints()==0:
                continue

            # Clean + smooth
            clean = vtk.vtkCleanPolyData(); clean.SetInputData(poly); clean.Update()
            ws    = vtk.vtkWindowedSincPolyDataFilter(); ws.SetInputConnection(clean.GetOutputPort())
            ws.SetNumberOfIterations(60); ws.SetPassBand(0.05)
            ws.BoundarySmoothingOn(); ws.FeatureEdgeSmoothingOn()
            ws.NonManifoldSmoothingOn(); ws.NormalizeCoordinatesOn(); ws.Update()

            # Normales
            flat_n = vtk.vtkPolyDataNormals(); flat_n.SetInputConnection(ws.GetOutputPort())
            flat_n.ComputePointNormalsOff(); flat_n.ComputeCellNormalsOn()
            flat_n.SplittingOff(); flat_n.AutoOrientNormalsOn()
            flat_n.ConsistencyOn(); flat_n.SetFeatureAngle(180); flat_n.Update()

            # Decimation
            self.onProgressInfo(f"PerLabelVTK: Decimating {s.GetName()}")
            dec = vtk.vtkQuadricDecimation()
            dec.SetInputConnection(flat_n.GetOutputPort()); dec.SetTargetReduction(0.4); dec.Update()

            # Transform & Write
            ras_t = vtk.vtkTransform(); ras_t.SetMatrix(parent_mat)
            ras_f = vtk.vtkTransformPolyDataFilter(); ras_f.SetTransform(ras_t)
            ras_f.SetInputConnection(dec.GetOutputPort()); ras_f.Update()
            lps_t = vtk.vtkTransform(); lps_t.Scale(-1,-1,1)
            lps_f = vtk.vtkTransformPolyDataFilter(); lps_f.SetTransform(lps_t)
            lps_f.SetInputConnection(ras_f.GetOutputPort()); lps_f.Update()

            label_safe = re.sub(r"[^0-9A-Za-z_-]+","_", s.GetName())
            out_path   = os.path.join(folderPath, f"{seg_safe}_{label_safe}.vtk")
            self.onProgressInfo(f"PerLabelVTK: Writing {label_safe}.vtk")
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(out_path); writer.SetInputData(lps_f.GetOutput())
            writer.SetFileTypeToBinary(); writer.Write()

        self.onProgressInfo("PerLabelVTK: Done")


    def _exportToGLTF(self, segmentationNode, folderPath, tryInstall=True):
        try:
            from OpenAnatomyExport import OpenAnatomyExportLogic
            logic = OpenAnatomyExportLogic()
            sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            segmentation_item = sh_node.GetItemByDataNode(self.segmentationNodeSelector.currentNode())
            logic.exportModel(segmentation_item, folderPath, self.reductionFactorSlider.value, "glTF")
        except ImportError:
            if not tryInstall or not hasInternetConnection():
                slicer.util.errorDisplay(
                    "Failed to export to glTF. Try installing the SlicerOpenAnatomy extension manually to continue."
                )
                return
            self._installOpenAnatomyExtension()
            self._exportToGLTF(segmentationNode, folderPath, tryInstall=False)

    @classmethod
    def _installOpenAnatomyExtension(cls):
        extension_manager = slicer.app.extensionsManagerModel()
        extension_manager.setInteractive(False)
        ext_name = "SlicerOpenAnatomy"
        if extension_manager.isExtensionInstalled(ext_name):
            return

        success = extension_manager.installExtensionFromServer(ext_name, False, False)
        if not success:
            return

        module_name = "OpenAnatomyExport"
        module_path = extension_manager.extensionModulePaths(ext_name)[0] + f"/{module_name}.py"
        factory = slicer.app.moduleManager().factoryManager()
        factory.registerModule(qt.QFileInfo(module_path))
        factory.loadModules([module_name])

    @staticmethod
    def isNNUNetModuleInstalled():
        try:
            import SlicerNNUNetLib  # noqa: F401  (sonde de disponibilite)
            return True
        except ImportError:
            return False

    def _installNNUNetIfNeeded(self) -> bool:
        from SlicerNNUNetLib import InstallLogic
        logic = InstallLogic()
        logic.progressInfo.connect(self.onProgressInfo)
        return logic.setupPythonRequirements()

    def _createSlicerSegmentationLogic(self):
        if not self.isNNUNetModuleInstalled():
            return None
        from SlicerNNUNetLib import SegmentationLogic
        return SegmentationLogic()

    def _connectSegmentationLogic(self):
        if self.logic is None:
            logger.debug("[DEBUG][SegWidget] _connectSegmentationLogic skipped: logic is None")
            return
        self.logic.progressInfo.connect(self.onProgressInfo)
        self.logic.errorOccurred.connect(self.onInferenceError)
        self.logic.inferenceFinished.connect(self.onInferenceFinished)

    @classmethod
    def nnUnetFolder(cls) -> Path:
        file_dir = Path(__file__).parent
        return file_dir.joinpath("..", "Resources", "ML").resolve()
