from enum import Flag, auto
from pathlib import Path
import vtk
import SegmentEditorEffects
import ctk
import numpy as np
import qt
import subprocess
import sys
import sys
import slicer
import logging
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
import sys

# ===== Logging Configuration =====
logger = logging.getLogger("BatchDentalSeg_SegmentationWidget")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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
    Run « pip install … »
    """
    def __init__(self, packages, onLine, onFinished, parent=None):
        super().__init__(parent)
        self._onLine     = onLine
        self._onFinished = onFinished
        self._proc       = qt.QProcess(self)           # vie = celle du runner

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
            lineBA  = self._proc.readLine()           # QByteArray
            lineStr = lineBA.data().decode("utf-8", "ignore").rstrip()
            self._onLine(lineStr)


    def _procFinished(self, exitCode, *args):
        """
        Slot called at the end of QProcess.
        Qt5 : finished(int)
        Qt6 : finished(int, QProcess.ExitStatus)
        """
        self._onFinished(exitCode == 0)
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
        self.fullInfoLogs             = deque(maxlen=200_000)   # journal des messages (borné)

        # ------------------------------------------------------------ queue state
        self.queue                    = SegmentationQueue()
        self._queueRunning            = False
        self._itemFinalized           = True        # garde anti double-avancement
        self._itemStartTime           = None
        self._setupDone               = False       # pip / poids : une fois par session
        self._deviceFallbackAccepted  = None        # réponse CPU mémorisée pour la file

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

        folderBtn = createButton("Select Folder",        callback=self.selectFolder)
        outBtn    = createButton("Select Output Folder", callback=self.selectOutputFolder)

        self.inputWidget = qt.QWidget(self)
        inputLayout      = qt.QFormLayout(self.inputWidget); inputLayout.setContentsMargins(0,0,0,0)
        inputLayout.addRow("Input Folder:",  self.folderPathLineEdit)
        inputLayout.addRow("",               folderBtn)
        inputLayout.addRow("Output Folder:", self.outputFolderLineEdit)
        inputLayout.addRow("",               outBtn)

        # ========================================================================
        # 2)  EXPORT FORMATS
        # ========================================================================
        exportWidget = qt.QWidget()
        exportLayout = qt.QFormLayout(exportWidget)

        self.stlCheckBox       = qt.QCheckBox(exportWidget); self.stlCheckBox.setChecked(True)
        self.objCheckBox       = qt.QCheckBox(exportWidget)
        self.niftiCheckBox     = qt.QCheckBox(exportWidget)
        self.gltfCheckBox      = qt.QCheckBox(exportWidget)
        self.vtkCheckBox       = qt.QCheckBox(exportWidget)
        self.vtkmergedCheckBox = qt.QCheckBox(exportWidget)

        self.reductionFactorSlider = ctk.ctkSliderWidget()
        self.reductionFactorSlider.maximum     = 1.0
        self.reductionFactorSlider.value       = 0.9
        self.reductionFactorSlider.singleStep  = 0.01
        self.reductionFactorSlider.toolTip     = "Decimation factor for glTF export."

        exportLayout.addRow("Export STL",           self.stlCheckBox)
        exportLayout.addRow("Export OBJ",           self.objCheckBox)
        exportLayout.addRow("Export NIFTI",         self.niftiCheckBox)
        exportLayout.addRow("Export glTF",          self.gltfCheckBox)
        exportLayout.addRow("Export VTK",           self.vtkCheckBox)
        exportLayout.addRow("Export VTK (merged)",  self.vtkmergedCheckBox)
        exportLayout.addRow("glTF reduction factor:", self.reductionFactorSlider)

        # Add to the layout the export formats widget
        inputLayout.addRow("Export formats :", exportWidget)

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
        smoothingSlider = self.show3DButton.findChild("ctkSliderWidget")

        self.surfaceSmoothingSlider = ctk.ctkSliderWidget(self)
        self.surfaceSmoothingSlider.decimals   = 2
        self.surfaceSmoothingSlider.maximum    = 1
        self.surfaceSmoothingSlider.singleStep = 0.1
        self.surfaceSmoothingSlider.setValue(smoothingSlider.value)
        self.surfaceSmoothingSlider.tracking   = False
        self.surfaceSmoothingSlider.valueChanged.connect(smoothingSlider.setValue)

        # ========================================================================
        # 5)  MAIN LAYOUT
        # ========================================================================
        layout = qt.QVBoxLayout(self)

        # bloc haut : dossiers + formats + device/model
        self.mainInputWidget = qt.QWidget(self)
        mainInputLayout = qt.QFormLayout(self.mainInputWidget); mainInputLayout.setContentsMargins(0,0,0,0)
        mainInputLayout.addRow(self.inputWidget)
        mainInputLayout.addRow(self.segmentationNodeSelector)
        mainInputLayout.addRow("Device:", self.deviceComboBox)
        mainInputLayout.addRow("Model:",  self.modelComboBox)
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
        # fullInfoLogs, reachable through the « info » button.
        # PythonQt exposes Qt getters as properties: document, not document().
        self.currentInfoTextEdit.document.setMaximumBlockCount(5000)

        self.stopButton = createButton("Stop", callback=self.onStopClicked, toolTip="Stop the segmentation.")
        self.loading    = qt.QMovie(iconPath("loading.gif")); self.loading.setScaledSize(qt.QSize(24,24))
        self.loading.frameChanged.connect(self._updateStopIcon); self.loading.start()

        self.applyWidget = qt.QWidget(self)
        applyLayout = qt.QHBoxLayout(self.applyWidget); applyLayout.setContentsMargins(0,0,0,0)
        applyLayout.addWidget(self.applyButton, 1)
        applyLayout.addWidget(createButton("", callback=self.showInfoLogs,
                                        icon=icon("info.png"), toolTip="Show logs."))

        self.stopWidgetContainer = qt.QWidget(self)
        stopLayout = qt.QVBoxLayout(self.stopWidgetContainer); stopLayout.setContentsMargins(0,0,0,0)
        stopLayout.addWidget(self.stopButton); stopLayout.addWidget(self.currentInfoTextEdit)
        self.stopWidgetContainer.setVisible(False)

        layout.addWidget(self.applyWidget)
                # --- Batch scan counter (Scan i/N) ------------------------------------
        self.batchCounterLabel = qt.QLabel("", self)
        self.batchCounterLabel.setAlignment(qt.Qt.AlignCenter)
        self.batchCounterLabel.setStyleSheet("color: #666; font-style: italic; margin-top:2px;")
        self.batchCounterLabel.setVisible(False)  # visible seulement pendant batch
        layout.addWidget(self.batchCounterLabel)
        layout.addWidget(self.stopWidgetContainer)
        layout.addWidget(self.resolveMirroringButton)

        # progress bar mirroring
        self.mirroringProgressBar = qt.QProgressBar(); self.mirroringProgressBar.setMinimum(0); self.mirroringProgressBar.setMaximum(100)
        self.mirroringProgressBar.setVisible(False); layout.addWidget(self.mirroringProgressBar)

        # 3-D + smoothing slider
        layout.addWidget(self.segmentEditorWidget)
        surfLayout = qt.QFormLayout(); surfLayout.setContentsMargins(0,0,0,0)
        surfLayout.addRow("Surface smoothing :", self.surfaceSmoothingSlider)
        layout.addLayout(surfLayout)

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

        buttonsWidget = qt.QWidget(self)
        buttonsLayout = qt.QHBoxLayout(buttonsWidget)
        buttonsLayout.setContentsMargins(0, 0, 0, 0)
        buttonsLayout.addWidget(createButton(
            "Add input folder", callback=self.onAddFolderToQueue,
            toolTip="Queue every scan of the selected input folder with the current "
                    "model / device / output folder.", parent=self))
        buttonsLayout.addWidget(createButton(
            "Remove selected", callback=self.onRemoveSelectedFromQueue,
            toolTip="Remove the selected pending scans.", parent=self))
        buttonsLayout.addWidget(createButton(
            "Retry failed", callback=self.onRetryFailed,
            toolTip="Append every failed scan back at the end of the queue.", parent=self))
        buttonsLayout.addWidget(createButton(
            "Clear", callback=self.onClearQueue,
            toolTip="Empty the queue.", parent=self))
        buttonsLayout.addWidget(createButton(
            "Free memory", callback=self.onFreeMemoryClicked,
            toolTip="Kill nnUNet processes left behind by a crashed scan and run a "
                    "deep cleanup. Use it when the RAM stays full after a failure.",
            parent=self))

        queueWidget = qt.QWidget(self)
        queueLayout = qt.QFormLayout(queueWidget)
        queueLayout.setContentsMargins(0, 0, 0, 0)
        queueLayout.addRow(buttonsWidget)
        queueLayout.addRow(self.queueTable)
        queueLayout.addRow(self.queueSummaryLabel)
        queueLayout.addRow("Deep cleanup every:", self.chunkSizeSpinBox)
        queueLayout.addRow("Timeout per scan:", self.itemTimeoutSpinBox)
        queueLayout.addRow("RAM limit:", self.ramLimitSpinBox)
        queueLayout.addRow(self.ramPreflightCheckBox)
        queueLayout.addRow(self.skipExistingCheckBox)
        queueLayout.addRow(self.unattendedCheckBox)

        addInCollapsibleLayout(queueWidget, layout, "Processing queue", isCollapsed=False)

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
        ramFailures = [item for item in self.queue.items
                       if item.status == STATUS_FAILED and self._isRamFailure(item.error)]
        autoCrop = bool(ramFailures) and self._askAutoCropConfirmation(ramFailures)

        requeued = self.queue.retryFailed(
            shouldAutoCrop=(lambda item: self._isRamFailure(item.error)) if autoCrop else None)
        self._rebuildQueueTable()
        if autoCrop:
            self.onProgressInfo(
                f"Queue: {requeued} failed scan(s) re-queued, "
                f"{len(ramFailures)} of them with automatic crop.")
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
        statusCell = self.queueTable.item(row, 2)
        statusCell.setForeground(qt.QBrush(qt.QColor(self._STATUS_COLORS.get(item.status, "#666666"))))

    def _updateQueueSummary(self):
        if self.queue.isEmpty():
            self.queueSummaryLabel.setText("Queue empty — Apply will queue the input folder.")
        else:
            self.queueSummaryLabel.setText(self.queue.summary())

    # ─── Queue execution ───────────────────────────────────────────────────────

    def _isUnattended(self):
        return self.unattendedCheckBox.isChecked()

    def _notify(self, message, isError=False):
        """Log; only interrupt the user when not running unattended."""
        self.onProgressInfo(message)
        if self._isUnattended():
            return
        if isError:
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
                "Use « Retry failed » or « Clear » to start over.")
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

            loadedVolume = slicer.util.loadVolume(item.inputPath)
            self._releaseCropNodes()
            if getattr(item, "autoCrop", False):
                loadedVolume = self._applyAutoCrop(loadedVolume)
            self.currentVolumeNode = loadedVolume
            self.onInputChangedForLoadedVolume(loadedVolume)

            if not self._ramPreflightOk(loadedVolume, item):
                return
            self._memWatchdogStart()

            self.onApplyClickedForVolume(loadedVolume)

        except Exception as e:
            logger.error(f"Failed to start {item.inputPath}: {e}", exc_info=True)
            self._save_state_before_crash()
            self._finishCurrentItem(STATUS_FAILED, f"start failed: {e}")

    @staticmethod
    def _selectComboItem(comboBox, text):
        index = comboBox.findText(text)
        if index >= 0 and index != comboBox.currentIndex:
            comboBox.setCurrentIndex(index)

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

    def _killInferenceTree(self, timeoutSec=10):
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
                psutil.wait_procs(victims, timeout=timeoutSec)
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                logger.error(f"Killing the inference tree failed: {e}", exc_info=True)

        try:
            self.logic.stopSegmentation()
            self.logic.waitForSegmentationFinished()
        except Exception:
            pass

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
            ownPids = {me.pid} | {child.pid for child in me.children(recursive=True)}
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
                isOurs = proc.pid in ownPids
                isOrphan = info.get("ppid") == 1 and info.get("username") == me.username()
                if not (isOurs or isOrphan):
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
    # Opt-in, per scan, set by « Retry failed » after an explicit confirmation.
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
            marginVox = [max(int(round(self._CROP_MARGIN_MM / s)), 1) for s in spacing]

            bounds = []
            for axis, axisMargin in enumerate((marginVox[2], marginVox[1], marginVox[0])):
                projected = mask.any(axis=tuple(a for a in (0, 1, 2) if a != axis))
                indices = np.where(projected)[0]
                low = max(int(indices[0]) - axisMargin, 0)
                high = min(int(indices[-1]) + 1 + axisMargin, mask.shape[axis])
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
            croppedArray = array[k0:k1, j0:j1, i0:i1]

            cropped = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", f"{volumeNode.GetName()}_cropped")
            slicer.util.updateVolumeFromArray(cropped, croppedArray)

            # Keep spacing and axes, move the origin to the new first voxel.
            ijkToRas = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRas)
            newOrigin = [0.0, 0.0, 0.0, 1.0]
            ijkToRas.MultiplyPoint([i0, j0, k0, 1.0], newOrigin)
            croppedIjkToRas = vtk.vtkMatrix4x4()
            croppedIjkToRas.DeepCopy(ijkToRas)
            for row in range(3):
                croppedIjkToRas.SetElement(row, 3, newOrigin[row])
            cropped.SetIJKToRASMatrix(croppedIjkToRas)

            self._uncroppedVolumeNode = volumeNode
            self._cropOffsetIJK = (i0, j0, k0)
            self.onProgressInfo(
                f"[CROP] {array.shape[2]}x{array.shape[1]}x{array.shape[0]} -> "
                f"{croppedArray.shape[2]}x{croppedArray.shape[1]}x{croppedArray.shape[0]} "
                f"({kept * 100:.0f}% of the voxels kept, {self._CROP_MARGIN_MM:.0f} mm margin)")
            return cropped

        except Exception as e:
            logger.error(f"Automatic crop failed: {e}", exc_info=True)
            self.onProgressInfo(f"[CROP] Failed ({e}) — segmenting the scan unchanged.")
            self._uncroppedVolumeNode = None
            self._cropOffsetIJK = None
            return volumeNode

    def _restoreCropToOriginalGrid(self, labelArray, croppedNode):
        """
        Paste a label array computed on the cropped grid back into the original.

        Returns (array, node whose geometry the output must use).
        """
        original = self._uncroppedVolumeNode
        offset = self._cropOffsetIJK
        if original is None or offset is None:
            return labelArray, croppedNode

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
            return labelArray, croppedNode

    def _releaseCropNodes(self):
        node = self._uncroppedVolumeNode
        self._uncroppedVolumeNode = None
        self._cropOffsetIJK = None
        if node is None:
            return
        try:
            if slicer.mrmlScene.GetNodeByID(node.GetID()):
                slicer.mrmlScene.RemoveNode(node)
        except Exception:
            pass

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
    def _configurationFolder(basePath, maxDepth=3):
        """
        Folder holding dataset.json, searched the way nnUNet itself does it:
        shallowest match wins (see Parameter._getFirstFolderWithDatasetFile).
        """
        pattern = "dataset.json"
        for _ in range(maxDepth):
            match = next(Path(basePath).glob(pattern), None)
            if match is not None:
                return match.parent
            pattern = f"*/{pattern}"
        return None

    @staticmethod
    def _readJson(path):
        import json
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
            imageData = volumeNode.GetImageData()
            if imageData is None:
                return None
            dims = imageData.GetDimensions()
            spacing = volumeNode.GetSpacing()
            fovMm3 = (dims[0] * spacing[0]) * (dims[1] * spacing[1]) * (dims[2] * spacing[2])

            # Both files must come from the same configuration folder, otherwise
            # a spacing and a label count from two different models get mixed.
            configFolder = self._configurationFolder(self._modelBasePath(modelName))
            if configFolder is None:
                return None
            plans = self._readJson(configFolder.joinpath("plans.json"))
            dataset = self._readJson(configFolder.joinpath("dataset.json"))
            if not plans or not dataset:
                return None

            targetSpacing = plans.get("configurations", {}).get("3d_fullres", {}).get("spacing")
            labels = dataset.get("labels") or {}
            if not targetSpacing or not labels:
                return None

            targetVoxelMm3 = float(targetSpacing[0]) * float(targetSpacing[1]) * float(targetSpacing[2])
            if targetVoxelMm3 <= 0:
                return None

            originalVoxels = float(dims[0]) * dims[1] * dims[2]
            resampledVoxels = fovMm3 / targetVoxelMm3
            numClasses = len(labels)

            # Peak is reached in resample_data_or_seg, called on the way out
            # through convert_predicted_logits_to_segmentation_with_correct_shape.
            # Live at that moment, per class:
            #   - the sliding-window accumulator, torch.half on the resampled
            #     grid, still referenced by the caller          -> 2 bytes
            #   - its float64 copy: `data = data.astype(float)` casts the whole
            #     4D array at once, before the per-class loop   -> 8 bytes
            #   - reshaped_final, torch.half on the original grid -> 2 bytes
            logitsGb = (10.0 * numClasses * resampledVoxels
                        + 2.0 * numClasses * originalVoxels) / 2 ** 30
            # Per-class transients of skimage resize (spline coefficients and
            # output buffer, float64).
            imagesGb = 8.0 * (originalVoxels + resampledVoxels) / 2 ** 30
            # torch, the weights and the worker processes cost the same on every
            # scan; the arrays are what makes one scan explode.
            return logitsGb + imagesGb + self._RAM_FIXED_OVERHEAD_GB
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

        availableGb = vm.available / 2 ** 30
        budgetGb = availableGb * (self.ramLimitSpinBox.value / 100.0)
        self.onProgressInfo(
            f"[RAM] Estimated peak: {estimate:.1f} GB — free: {availableGb:.1f} GB, "
            f"budget: {budgetGb:.1f} GB")
        if estimate <= budgetGb or not self.ramPreflightCheckBox.isChecked():
            if estimate > budgetGb:
                self.onProgressInfo(
                    "[RAM] Over budget, but the pre-flight skip is disabled — "
                    "the runtime guard stays armed.")
            return True

        self.onProgressInfo(
            f"[RAM] Skipping {item.name}: needs ~{estimate:.0f} GB, "
            f"only {budgetGb:.0f} GB usable. Crop the field of view or free memory "
            f"and use « Retry failed ».")
        self._memWatchdogStop()
        self._inferenceFinalized = True
        try:
            self._cleanupAfterCase(volumeNode, None)
        except Exception as e:
            logger.error(f"Cleanup after pre-flight skip failed: {e}", exc_info=True)
        self._finishCurrentItem(
            STATUS_FAILED,
            f"skipped: needs ~{estimate:.0f} GB RAM, {availableGb:.0f} GB free")
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
        except Exception:
            pass
        import gc
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
        for className in ("vtkMRMLSegmentationNode",
                          "vtkMRMLLabelMapVolumeNode",
                          "vtkMRMLScalarVolumeNode"):
            for node in slicer.util.getNodesByClass(className):
                if id(node) in keep:
                    continue
                try:
                    slicer.mrmlScene.RemoveNode(node)
                    removed += 1
                except Exception:
                    pass
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

        segmentationNode = self.getCurrentSegmentationNode()
        volumeNode       = self.getCurrentVolumeNode()
        if not segmentationNode or not volumeNode:
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
        labelArray = self._buildLabelArray(segmentationNode, volumeNode, full_label_map)

        ijkToRAS = vtk.vtkMatrix4x4(); volumeNode.GetIJKToRASMatrix(ijkToRAS)
        spacing, origin = volumeNode.GetSpacing(), volumeNode.GetOrigin()

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
        fgMask   = labelArray > 0
        fgCoords = np.argwhere(fgMask)            # (M, 3) as (z, y, x)
        fgValues = labelArray[fgMask]             # (M,)

        if fgCoords.size == 0:
            slicer.util.warningDisplay("Segmentation is empty.")
            self.mirroringProgressBar.setVisible(False)
            return

        # RAS_R = m00*x + m01*y + m02*z + m03
        m00 = ijkToRAS.GetElement(0, 0)
        m01 = ijkToRAS.GetElement(0, 1)
        m02 = ijkToRAS.GetElement(0, 2)
        m03 = ijkToRAS.GetElement(0, 3)
        fgRasX = (m00 * fgCoords[:, 2] + m01 * fgCoords[:, 1] + m02 * fgCoords[:, 0] + m03)

        incisive_vals = (8, 9, 24, 25)
        inc_centroids = []
        for val in incisive_vals:
            selected = fgRasX[fgValues == val]
            if selected.size == 0:
                slicer.util.warningDisplay("Missing central incisors, unable to calculate mirror plane.")
                self.mirroringProgressBar.setVisible(False)
                return
            inc_centroids.append(selected.mean())
        mirror_x_ras = float(np.mean(inc_centroids))

        # ─── 5. Perform mirror correction ────────────────────────────────────
        changed = []
        fgProtected = np.isin(fgValues, list(protected_vals))
        unique_vals = np.unique(fgValues)
        for i, val in enumerate(unique_vals):
            self.mirroringProgressBar.setValue(int(100 * (i + 1) / len(unique_vals)))
            slicer.app.processEvents()

            val = int(val)
            if val == 0 or val in protected_vals or val not in mirror_label_map:
                continue

            name        = reverse_full_map.get(val, f"label_{val}")
            mirror_val  = mirror_label_map[val]
            is_left     = "left" in name.lower()

            indices = np.flatnonzero((fgValues == val) & ~fgProtected)
            if indices.size == 0:
                continue

            rasX    = fgRasX[indices]
            wrongSide = rasX > mirror_x_ras if is_left else rasX < mirror_x_ras
            indices = indices[wrongSide]
            if indices.size == 0:
                continue

            coords = fgCoords[indices]
            labelArray[coords[:, 0], coords[:, 1], coords[:, 2]] = mirror_val
            # Keep the working copy in sync, so a later label sees the same state
            # the original per-voxel loop would have seen.
            fgValues[indices] = mirror_val
            changed.append(
                f"{name} → {reverse_full_map.get(mirror_val, mirror_val)} ({indices.size} vox)")

        self.mirroringProgressBar.setValue(100)

        # ─── 6. Rebuild corrected segmentation ──────────────────────────────
        correctedLM = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(correctedLM, labelArray)
        correctedLM.SetSpacing(spacing)
        correctedLM.SetOrigin(origin)
        correctedLM.SetIJKToRASMatrix(ijkToRAS)

        # New name: original segmentation name + suffix
        baseName = segmentationNode.GetName() if segmentationNode else "Segmentation"
        suffix   = "_Mirrored"                           # choose your suffix here
        correctedSeg = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            baseName + suffix
        )

        correctedSeg.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        logic.ImportLabelmapToSegmentationNode(correctedLM, correctedSeg)
        correctedSeg.CreateClosedSurfaceRepresentation()

        # (Optional) Automatically select corrected node
        self.segmentationNodeSelector.setCurrentNode(correctedSeg)

        # ─── 7. Rename + tag segments (by creation order) ──────────────────
        finalValues      = [int(v) for v in np.unique(fgValues)]              # [1,2,…,55]
        segIds_sorted    = list(correctedSeg.GetSegmentation().GetSegmentIDs())

        if len(finalValues) != len(segIds_sorted):
            self.onProgressInfo("[WARN] Number of values ​​≠ number of segments — check import.")

        for val, segId in zip(finalValues, segIds_sorted):
            segment = correctedSeg.GetSegmentation().GetSegment(segId)
            segment.SetName(reverse_full_map.get(val, f"label_{val}"))
            segment.SetTag("LabelValue", str(val))

        self.onProgressInfo(f"Unique labels AFTER correction: {finalValues}")

        # Cleanup
        slicer.mrmlScene.RemoveNode(correctedLM)
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
        folderPath = qt.QFileDialog.getExistingDirectory(self, "Select Folder to Save Segmentations")
        if folderPath:
            self.outputFolderPath = folderPath
            self.outputFolderLineEdit.setText(folderPath)
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

        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        success = slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            segmentationNode, labelmapVolumeNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)

        if not success:
            self.onProgressInfo("ERROR: Exporting segments to the labelmap failed.")
            return

        output_path = os.path.join(self.outputFolderPath, segmentationNode.GetName() + ".nii.gz")
        saved = slicer.util.saveNode(labelmapVolumeNode, output_path)
        if saved:
            self.onProgressInfo(f"Segmentation saved in {output_path}")
        else:
            self.onProgressInfo(f"Failed to save segmentation in {output_path}")

        # Clean
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)


    def __del__(self):
        slicer.mrmlScene.RemoveObserver(self.sceneCloseObserver)
        super().__del__()

    def selectFolder(self):
        folderPath = qt.QFileDialog.getExistingDirectory(self, "Select Folder Containing Volumes")
        if folderPath:
            self.folderPath = folderPath
            self.folderPathLineEdit.text = folderPath
            self.folderFiles = listVolumes(folderPath)
            self.currentFileIndex = 0
            self.onProgressInfo(f"Found {len(self.folderFiles)} file(s) in the folder.")

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
                    "Uncheck « Skip scans already segmented » to process them again."
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
        segmentationNode = self.processedVolumes.get(volumeNode)
        self.segmentationNodeSelector.setCurrentNode(segmentationNode)

# ─── Apply segmentation for a given volume ────────────────────────────────

    def onApplyClickedForVolume(self, volumeNode):
        from SlicerNNUNetLib import Parameter
        self._inferenceFinalized = False
        self._doneVolumeSeen = False
        self._fallbackCheckAttempts = 0
        self._fallbackLastOutputSize = None
        selectedModel = self.modelComboBox.currentText
        if selectedModel == "PediatricDentalsegmentator":
            self.onProgressInfo(f"Selected Model: {selectedModel}")

            # Base path where full model must be installed
            basePath = Path(__file__).parent.joinpath("..", "Resources", "ML", "Dataset001_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            # Choose fold_0 (you can adapt for fold_1 if needed)
            fold_path = basePath.joinpath("fold_0")
            if not fold_path.exists():
                fold_path.mkdir(parents=True, exist_ok=True)
            # Checkpoint path inside fold_0
            pediatricCheckpoint = fold_path.joinpath("checkpoint_final.pth")
            # If checkpoint doesn't exist, download checkpoint and dataset.json and plans.json inside basePath
            if not pediatricCheckpoint.exists():
                url_checkpoint = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/PEDIATRICDENTALSEG_MODEL/checkpoint_final.pth"
                url_dataset = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/PEDIATRICDENTALSEG_MODEL/dataset.json"
                url_plans = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/PEDIATRICDENTALSEG_MODEL/plans.json"
                self.onProgressInfo("Downloading pediatricdentalseg model...")
                # Download checkpoint; convert Path to string for downloadFile
                slicer.util.downloadFile(url_checkpoint, str(pediatricCheckpoint))
                # Download dataset.json and plans.json in basePath
                slicer.util.downloadFile(url_dataset, str(basePath.joinpath("dataset.json")))
                slicer.util.downloadFile(url_plans, str(basePath.joinpath("plans.json")))
            # For nnUNet, modelPath must point to folder containing dataset.json and fold_x
            parameter = Parameter(folds="0", modelPath=basePath, device=self.deviceComboBox.currentText)

        elif selectedModel == "NasoMaxillaDentSeg":
            self.onProgressInfo(f"Selected Model: {selectedModel}")

            # Base path where full model must be installed
            basePath = Path(__file__).parent.joinpath("..", "Resources", "ML", "Dataset001_max4", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            # Choose fold_0 (you can adapt for fold_1 if needed)
            fold_path = basePath.joinpath("fold_0")
            if not fold_path.exists():
                fold_path.mkdir(parents=True, exist_ok=True)
            # Checkpoint path inside fold_0
            NasoMaxillaDentSegCheckpoint = fold_path.joinpath("checkpoint_final.pth")
            # If checkpoint doesn't exist, download checkpoint and dataset.json and plans.json inside basePath
            if not NasoMaxillaDentSegCheckpoint .exists():
                url_checkpoint = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/NASOMAXILLADENTSEG_MODEL/checkpoint_final.pth"
                url_dataset = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/NASOMAXILLADENTSEG_MODEL/dataset.json"
                url_plans = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/NASOMAXILLADENTSEG_MODEL/plans.json"
                self.onProgressInfo("Downloading NasoMaxillaDentSeg model...")
                # Download checkpoint; convert Path to string for downloadFile
                slicer.util.downloadFile(url_checkpoint, str(NasoMaxillaDentSegCheckpoint))
                # Download dataset.json and plans.json in basePath
                slicer.util.downloadFile(url_dataset, str(basePath.joinpath("dataset.json")))
                slicer.util.downloadFile(url_plans, str(basePath.joinpath("plans.json")))
            # For nnUNet, modelPath must point to folder containing dataset.json and fold_x
            parameter = Parameter(folds="0", modelPath=basePath, device=self.deviceComboBox.currentText)


        elif selectedModel == "UniversalLabDentalsegmentator":
            self.onProgressInfo(f"Selected Model: {selectedModel}")

            # Base path where full model must be installed
            basePath = Path(__file__).parent.joinpath("..", "Resources", "ML", "Dataset002_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            # Choose fold_0 (you can adapt for fold_1 if needed)
            fold_path = basePath.joinpath("fold_0")
            if not fold_path.exists():
                fold_path.mkdir(parents=True, exist_ok=True)
            # Checkpoint path inside fold_0
            pediatricCheckpoint = fold_path.joinpath("checkpoint_final.pth")
            # If checkpoint doesn't exist, download checkpoint and dataset.json and plans.json inside basePath
            if not pediatricCheckpoint.exists():
                url_checkpoint = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/UNIVERSALLAB_MODEL/checkpoint_final.pth"
                url_dataset = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/UNIVERSALLAB_MODEL/dataset.json"
                url_plans = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/UNIVERSALLAB_MODEL/plans.json"
                self.onProgressInfo("Downloading pediatricdentalseg model...")
                # Download checkpoint; convert Path to string for downloadFile
                slicer.util.downloadFile(url_checkpoint, str(pediatricCheckpoint))
                # Download dataset.json and plans.json in basePath
                slicer.util.downloadFile(url_dataset, str(basePath.joinpath("dataset.json")))
                slicer.util.downloadFile(url_plans, str(basePath.joinpath("plans.json")))
            # For nnUNet, modelPath must point to folder containing dataset.json and fold_x
            parameter = Parameter(folds="0", modelPath=basePath, device=self.deviceComboBox.currentText)




        else:
            self.onProgressInfo(f"Selected Model: {selectedModel}")

            parameter = Parameter(folds="0", modelPath=self.nnUnetFolder(), device=self.deviceComboBox.currentText)
                
        if not parameter.isSelectedDeviceAvailable():
            deviceName = parameter.device.upper()
            # Asked once for the whole queue — never once per scan.
            if self._deviceFallbackAccepted is None:
                if self._isUnattended():
                    self._deviceFallbackAccepted = True
                    self.onProgressInfo(
                        f"[WARN] {deviceName} not available — falling back to CPU for the whole queue.")
                else:
                    ret = qt.QMessageBox.question(
                        self,
                        f"{deviceName} device not available",
                        f"Selected device ({deviceName}) is not available and will default to CPU.\n"
                        "Running the segmentation may take up to 1 hour per scan.\n"
                        "Would you like to proceed with the whole queue?"
                    )
                    self._deviceFallbackAccepted = (ret == qt.QMessageBox.Yes)
            if not self._deviceFallbackAccepted:
                self._queueRunning = False
                self._finishCurrentItem(STATUS_FAILED, f"{deviceName} unavailable, aborted by user")
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
        segIds = list(segmentation.GetSegmentIDs())
        if not segIds:
            raise RuntimeError("Segmentation has no segment")

        # Passing the IDs explicitly pins the mapping: exported value i+1 <-> segIds[i].
        ids = vtk.vtkStringArray()
        for segId in segIds:
            ids.InsertNextValue(segId)

        tmpLM = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        try:
            success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segNode, ids, tmpLM, volNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
            )
            if not success:
                raise RuntimeError("ExportSegmentsToLabelmapNode failed")
            exported = slicer.util.arrayFromVolume(tmpLM)
        finally:
            slicer.mrmlScene.RemoveNode(tmpLM)

        maxExported = int(exported.max()) if exported.size else 0
        lut = np.zeros(max(len(segIds), maxExported) + 1, dtype=np.uint16)
        for exportedValue, segId in enumerate(segIds, start=1):
            segment = segmentation.GetSegment(segId)
            value = self._segmentLabelValue(segment, full_label_map)
            if value is None:
                self.onProgressInfo(f"[WARN] Unknown label for segment «{segment.GetName()}» — skipped")
                continue
            lut[exportedValue] = value

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

        segNode = volNode = None
        status, errorDetail = STATUS_DONE, ""
        try:
            # === Step 1: Initialization ===
            self.onProgressInfo("Processing results in progress...")

            # === Step 2: Load results ===
            try:
                self._loadSegmentationResults()
                segNode = self.getCurrentSegmentationNode()
                volNode = self.getCurrentVolumeNode()
                if not segNode:
                    raise RuntimeError("No segmentation node found")
                if not volNode:
                    raise RuntimeError("No volume node found")

                segmentation = segNode.GetSegmentation()
                full_label_map = self._get_active_label_map()

                # Normalize the LabelValue tags once (cheap: one pass over segments).
                raw_values = []
                for segId in segmentation.GetSegmentIDs():
                    segment = segmentation.GetSegment(segId)
                    value = self._segmentLabelValue(segment, full_label_map)
                    if value is None:
                        self.onProgressInfo(f"[WARN] unexpected segment «{segment.GetName()}» — ignored")
                        continue
                    segment.SetTag("LabelValue", str(value))
                    raw_values.append(value)

                self.onProgressInfo(f"Predicted label values (raw): {sorted(set(raw_values))}")

            except Exception as e:
                raise RuntimeError(f"Failed to load results: {str(e)}")

            # === PHASE 3: NIfTI export ===
            import vtk as _vtk

            label_arr = self._buildLabelArray(segNode, volNode, full_label_map)
            # After an automatic crop the result goes back on the grid of the
            # scan as acquired, so the output matches what the clinician sent.
            label_arr, geometryNode = self._restoreCropToOriginalGrid(label_arr, volNode)

            tmpOut = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            try:
                slicer.util.updateVolumeFromArray(tmpOut, label_arr)
                tmpOut.SetSpacing(geometryNode.GetSpacing())
                tmpOut.SetOrigin(geometryNode.GetOrigin())
                ijk2ras = _vtk.vtkMatrix4x4()
                geometryNode.GetIJKToRASMatrix(ijk2ras)
                tmpOut.SetIJKToRASMatrix(ijk2ras)

                output_path = str(Path(self.outputFolderPath).joinpath(
                    f"{self._currentCaseName()}_Segmentation.nii.gz"))
                saved = slicer.util.saveNode(tmpOut, output_path)
            finally:
                slicer.mrmlScene.RemoveNode(tmpOut)

            if saved:
                self.onProgressInfo(f"Segmentation saved in {output_path}")
            else:
                raise RuntimeError(f"saveNode failed for {output_path}")

            # Other formats (STL / OBJ / VTK / glTF), without any modal dialog.
            errorDetail = self._exportSegmentation(segNode, silent=True)

            # === Step 4: Success ===
            self.onProgressInfo("Processing completed successfully")
            logger.info(f"Volume processed: {volNode.GetName() if volNode else 'unknown'}")

        except Exception as e:
            # === Error handling ===
            status, errorDetail = STATUS_FAILED, str(e)
            error_msg = f"ERROR: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            self.onProgressInfo(f"PROCESSING FAILURE:\n{error_msg}")
            self._save_state_before_crash()
            if not self._isUnattended():
                slicer.util.errorDisplay(f"Critical error:\n{error_msg}")

        finally:
            # === PHASE 5: cleanup, then hand over to the queue ===
            try:
                self._cleanupAfterCase(volNode, segNode)
            except Exception as cleanup_error:
                logger.critical(f"Final cleaning failure: {cleanup_error}", exc_info=True)
                self.onProgressInfo(f"CLEANING ERROR: {cleanup_error}")

            self._finishCurrentItem(status, errorDetail)



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
                # aussi neutraliser le MRML node interne
                if hasattr(self, 'segmentEditorNode'):
                    self.segmentEditorWidget.setSegmentationNode(None)
                    self.segmentEditorWidget.setSourceVolumeNode(None)
            except Exception:
                pass

            # 2) Supprimer le display-node de la segmentation
            if segmentationNode and is_node_in_scene(segmentationNode):
                segDisp = segmentationNode.GetDisplayNode()
                if segDisp and is_node_in_scene(segDisp):
                    slicer.mrmlScene.RemoveNode(segDisp)

            # 3) Retirer l'entrée de la subject hierarchy PUIS le nœud lui-même.
            #    Le RemoveNode était auparavant indenté dans le bloc « except », donc
            #    jamais exécuté : chaque scan laissait sa segmentation dans la scène.
            if segmentationNode:
                try:
                    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
                    if shNode and shNode.GetScene():
                        itemID = shNode.GetItemByDataNode(segmentationNode)
                        if itemID and itemID != shNode.GetInvalidItemID():
                            shNode.RemoveItem(itemID)
                except Exception:
                    pass

                if is_node_in_scene(segmentationNode):
                    slicer.mrmlScene.RemoveNode(segmentationNode)

            if self._prevSegmentationNode is segmentationNode:
                self._prevSegmentationNode = None

            try:
                self.segmentEditorWidget.blockSignals(False)
            except Exception:
                pass

            if volumeNode and is_node_in_scene(volumeNode):
                volDisp = volumeNode.GetDisplayNode()
                if volDisp and is_node_in_scene(volDisp):
                    slicer.mrmlScene.RemoveNode(volDisp)
                slicer.mrmlScene.RemoveNode(volumeNode)

            if self.currentVolumeNode is volumeNode:
                self.currentVolumeNode = None

            # 6) Ne pas garder de référence Python sur des nœuds supprimés, sinon
            #    le gc.collect() ci-dessous ne peut rien libérer.
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
            import gc
            gc.collect()
            self.onProgressInfo(f"Cleanup complete. Memory: {self._get_memory_usage()}")

        except Exception as e:
            logger.error(f"Cleanup crashed: {str(e)}", exc_info=True)
            raise

    # ─── Load segmentation results ────────────────────────────────────────────

    def _loadSegmentationResults(self):
        currentSegmentation = self.getCurrentSegmentationNode()
        segmentationNode = self.logic.loadSegmentation()
        segmentationNode.SetName(self._currentCaseName() + "_Segmentation")
        if currentSegmentation is not None:
            self._copySegmentationResultsToExistingNode(currentSegmentation, segmentationNode)
        else:
            self.segmentationNodeSelector.setCurrentNode(segmentationNode)
        slicer.app.processEvents()
        self._updateSegmentationDisplay()
        self._storeProcessedSegmentation()

    # ─── Helper to copy segmentation results ──────────────────────────────────

    @staticmethod
    def _copySegmentationResultsToExistingNode(currentSegmentation, segmentationNode):
        currentName = currentSegmentation.GetName()
        currentSegmentation.Copy(segmentationNode)
        currentSegmentation.SetName(currentName)
        slicer.mrmlScene.RemoveNode(segmentationNode)

    @staticmethod
    def toRGB(colorString):
        color = qt.QColor(colorString)
        return color.redF(), color.greenF(), color.blueF()

    def _updateSegmentationDisplay(self):
        segmentationNode = self.getCurrentSegmentationNode()
        if not segmentationNode:
            return
        self._initializeSegmentationNodeDisplay(segmentationNode)
        segmentation = segmentationNode.GetSegmentation()
        selectedModel = self.modelComboBox.currentText
       
        if selectedModel == "UniversalLabDentalsegmentator":
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
            segmentIds = [f"Segment_{i+1}" for i in range(len(labels))]
            segmentationDisplayNode = segmentationNode.GetDisplayNode()
            for segmentId, label, color, opacity in zip(segmentIds, labels, colors, opacities):
                segment = segmentation.GetSegment(segmentId)
                if segment is None:
                    continue
                segment.SetName(label)
                segment.SetColor(*self.toRGB(color))
                segmentationDisplayNode.SetSegmentOpacity3D(segmentId, opacity)

            self.show3DButton.setChecked(True)
            slicer.util.resetThreeDViews()

        elif selectedModel == "NasoMaxillaDentSeg":
            labels = ["Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal","Maxilla "]
            colors = [self.toRGB(c) for c in ["#E3DD90", "#D4A1E6","#DC9565", "#EBDFB4", "#D8654F", "#6AC4A4"]]
            opacities = [0.65, 0.65,1.0, 1.0, 1.0, 0.65]
            segmentIds = [f"Segment_{i + 1}" for i in range(len(labels))]
            segmentationDisplayNode = self.getCurrentSegmentationNode().GetDisplayNode()
            for segmentId, label, color, opacity in zip(segmentIds, labels, colors, opacities):
                segment = segmentation.GetSegment(segmentId)
                if segment is None:
                    continue
                segment.SetName(label)
                segment.SetColor(*color)
                segmentationDisplayNode.SetSegmentOpacity3D(segmentId, opacity)
            self.show3DButton.setChecked(True)
            slicer.util.resetThreeDViews()

        else:
            labels = ["Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal"]
            colors = [self.toRGB(c) for c in ["#E3DD90", "#D4A1E6","#DC9565", "#EBDFB4", "#D8654F"]]
            opacities = [0.65, 0.65,1.0, 1.0, 1.0]
            segmentIds = [f"Segment_{i + 1}" for i in range(len(labels))]
            segmentationDisplayNode = self.getCurrentSegmentationNode().GetDisplayNode()
            for segmentId, label, color, opacity in zip(segmentIds, labels, colors, opacities):
                segment = segmentation.GetSegment(segmentId)
                if segment is None:
                    continue
                segment.SetName(label)
                segment.SetColor(*color)
                segmentationDisplayNode.SetSegmentOpacity3D(segmentId, opacity)
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
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDWidget.threeDView().rotateToViewAxis(3)
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
        voxelSize_mm3 = np.cumprod(self.getCurrentVolumeNode().GetSpacing())[-1]
        minimumIslandSize = int(np.ceil(self._minimumIslandSize_mm3 / voxelSize_mm3))
        effect = self.segmentEditorWidget.effectByName("Islands")
        effect.setParameter("Operation", SegmentEditorEffects.REMOVE_SMALL_ISLANDS)
        effect.setParameter("MinimumSize", minimumIslandSize)
        effect.self().onApply()

    def _getSegment(self, segmentId):
        segmentationNode = self.getCurrentSegmentationNode()
        if not segmentationNode:
            return
        return segmentationNode.GetSegmentation().GetSegment(segmentId)

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

    def onProgressInfo(self, infoMsg):
        infoMsg = self.removeImageIOError(infoMsg)
        if not infoMsg:
            return
        self._appendLog(infoMsg)
        if "done with volume" in infoMsg.lower():
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

        outFilePath = None
        outFileSize = None
        try:
            outFilePath = self.logic._outFile
            outFileSize = Path(outFilePath).stat().st_size
        except Exception:
            outFilePath = None

        processState = None
        try:
            processState = self.logic.inferenceProcess.process.state()
        except Exception:
            processState = None

        self.onProgressInfo(
            f"[DEBUG][SegWidget] Fallback check #{self._fallbackCheckAttempts}: "
            f"state={processState}, outFile={outFilePath}, size={outFileSize}"
        )

        if outFilePath and outFileSize is not None and outFileSize > 0:
            if self._fallbackLastOutputSize == outFileSize:
                self.onProgressInfo("[DEBUG][SegWidget] Output file stable, forcing finalization")
                qt.QTimer.singleShot(0, self.onInferenceFinished)
                return
            self._fallbackLastOutputSize = outFileSize

        if self._fallbackCheckAttempts < 40:
            qt.QTimer.singleShot(1500, self._checkInferenceCompletionFallback)
        else:
            self.onProgressInfo("[DEBUG][SegWidget] Fallback completion check timeout (no stable output file)")

    @staticmethod
    def removeImageIOError(infoMsg):
        return "\n".join([msg for msg in infoMsg.strip().splitlines() if "Error ImageIO factory" not in msg])

    def insertDatedInfoLogs(self, infoMsg):
        now = qt.QDateTime.currentDateTime().toString("yyyy/MM/dd hh:mm:ss.zzz")
        self.fullInfoLogs.extend([f"{now} :: {msgLine}" for msgLine in infoMsg.splitlines()])

    def showInfoLogs(self):
        dialog = qt.QDialog()
        layout = qt.QVBoxLayout(dialog)
        textEdit = qt.QTextEdit()
        textEdit.setReadOnly(True)
        textEdit.append("\n".join(self.fullInfoLogs))
        textEdit.setLineWrapMode(qt.QTextEdit.NoWrap)
        self.moveTextEditToEnd(textEdit)
        layout.addWidget(textEdit)
        dialog.setWindowFlags(qt.Qt.WindowCloseButtonHint)
        dialog.resize(slicer.util.mainWindow().size * 0.7)
        dialog.exec()


    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)

    def _setApplyVisible(self, isVisible):
        self.applyWidget.setVisible(isVisible)
        self.stopWidgetContainer.setVisible(not isVisible)
        self.inputWidget.setEnabled(isVisible)

        self.batchCounterLabel.setVisible(not isVisible)
        if not isVisible:
            self._updateBatchCounter(show_file_name=True)


    def getCurrentVolumeNode(self):
        return self.currentVolumeNode

    def getCurrentSegmentationNode(self):
        return self.segmentationNodeSelector.currentNode()

    def _storeProcessedSegmentation(self):
        volumeNode = self.getCurrentVolumeNode()
        segmentationNode = self.getCurrentSegmentationNode()
        if volumeNode and segmentationNode:
            self.processedVolumes[volumeNode] = segmentationNode
    def updateSegmentEditorWidget(self, *_):

        # Hide previous node
        if self._prevSegmentationNode:
            try:
                self._prevSegmentationNode.SetDisplayVisibility(False)
            except Exception:
                pass

        segmentationNode = self.getCurrentSegmentationNode()

        # If no segmentation or deleted node, we stop here
        if not segmentationNode or not slicer.mrmlScene.IsNodePresent(segmentationNode):
            return

        # Initialization and display
        self._initializeSegmentationNodeDisplay(segmentationNode)
        self.segmentEditorWidget.setSegmentationNode(segmentationNode)
        slicer.app.processEvents()

        volumeNode = self.getCurrentVolumeNode()
        if volumeNode and slicer.mrmlScene.IsNodePresent(volumeNode):
            self.segmentEditorWidget.setSourceVolumeNode(volumeNode)
            slicer.app.processEvents()

        self._prevSegmentationNode = segmentationNode


    def getSelectedExportFormats(self):
        selectedFormats = ExportFormat(0)
        checkBoxes = {
            self.objCheckBox: ExportFormat.OBJ,
            self.stlCheckBox: ExportFormat.STL,
            self.niftiCheckBox: ExportFormat.NIFTI,
            self.gltfCheckBox: ExportFormat.GLTF,
            self.vtkCheckBox: ExportFormat.VTK,
            self.vtkmergedCheckBox  : ExportFormat.VTK_MERGED

        }
        for checkBox, exportFormat in checkBoxes.items():
            if checkBox.isChecked():
                selectedFormats |= exportFormat
        return selectedFormats

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

        selectedFormats = self.getSelectedExportFormats()
        if selectedFormats == ExportFormat(0):
            if silent:
                self.onProgressInfo("No additional export format selected — NIfTI only.")
                return ""
            slicer.util.warningDisplay("Please select at least one export format before exporting.")
            return ""

        if silent:
            try:
                self.exportSegmentation(segmentationNode, self.outputFolderPath, selectedFormats)
                self.onProgressInfo(f"Export successful to {self.outputFolderPath}.")
                return ""
            except Exception as e:
                # The NIfTI is already written: a mesh export failure downgrades the
                # scan to a warning, it does not invalidate the segmentation.
                logger.error(f"Additional format export failed: {e}", exc_info=True)
                self.onProgressInfo(f"[WARN] Additional format export failed: {e}")
                return f"export warning: {e}"

        with slicer.util.tryWithErrorDisplay(f"Export to {self.outputFolderPath} failed.", waitCursor=True):
            self.exportSegmentation(segmentationNode, self.outputFolderPath, selectedFormats)
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
        refVol = self.getCurrentVolumeNode()
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
        foregroundLabels = [int(l) for l in np.unique(vtk_to_numpy(img.GetPointData().GetScalars())) if l]
        mc.SetNumberOfContours(len(foregroundLabels))
        for contourIndex, labelValue in enumerate(foregroundLabels):
            mc.SetValue(contourIndex, labelValue)
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
        flatN = vtk.vtkPolyDataNormals(); flatN.SetInputConnection(ws.GetOutputPort())
        flatN.ComputePointNormalsOff(); flatN.ComputeCellNormalsOn()
        flatN.SplittingOff(); flatN.AutoOrientNormalsOn()
        flatN.ConsistencyOn(); flatN.SetFeatureAngle(180); flatN.Update()

        rawPoly   = flatN.GetOutput()
        labelArray = rawPoly.GetCellData().GetScalars()
        labels     = np.unique(vtk_to_numpy(labelArray))
        append     = vtk.vtkAppendPolyData()

        # Parcours des labels
        for i, labelValue in enumerate(labels, start=1):
            if labelValue == 0:
                continue
            self.onProgressInfo(f"MergedVTK: Processing label {int(labelValue)} ({i}/{len(labels)})")

            thresh = vtk.vtkThreshold()
            thresh.SetInputData(rawPoly)
            thresh.SetInputArrayToProcess(0,0,0,
                vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                labelArray.GetName())
            thresh.SetLowerThreshold(labelValue)
            thresh.SetUpperThreshold(labelValue)
            thresh.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
            thresh.Update()

            surf = vtk.vtkDataSetSurfaceFilter(); surf.SetInputConnection(thresh.GetOutputPort()); surf.Update()

            dec = vtk.vtkQuadricDecimation()
            dec.SetInputConnection(surf.GetOutputPort())
            dec.SetTargetReduction(0.4)
            dec.Update()

            out = dec.GetOutput()
            from vtk.util.numpy_support import numpy_to_vtk
            constLabel = numpy_to_vtk(
                np.full(out.GetNumberOfCells(), int(labelValue), dtype=np.int32), deep=True)
            constLabel.SetName("Label")
            out.GetCellData().AddArray(constLabel)
            out.GetCellData().SetScalars(constLabel)

            append.AddInputData(out)

        append.Update()
        self.onProgressInfo("MergedVTK: AppendPolyData done")

        # Transform + Write
        self.onProgressInfo("MergedVTK: Transform & Write")
        ijk2ras = vtk.vtkMatrix4x4(); labelmap.GetIJKToRASMatrix(ijk2ras)
        parentMat = vtk.vtkMatrix4x4(); parentMat.Identity()
        if refVol and refVol.GetParentTransformNode():
            refVol.GetParentTransformNode().GetMatrixTransformToWorld(parentMat)
        rasMat = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(parentMat, ijk2ras, rasMat)

        rasT = vtk.vtkTransform(); rasT.SetMatrix(rasMat)
        rasF = vtk.vtkTransformPolyDataFilter()
        rasF.SetTransform(rasT); rasF.SetInputConnection(append.GetOutputPort()); rasF.Update()
        lpsT = vtk.vtkTransform(); lpsT.Scale(-1,-1,1)
        lpsF = vtk.vtkTransformPolyDataFilter(); lpsF.SetTransform(lpsT)
        lpsF.SetInputConnection(rasF.GetOutputPort()); lpsF.Update()

        outPath = os.path.join(folderPath, f"{segNode.GetName()}_merged.vtk")
        w = vtk.vtkPolyDataWriter(); w.SetFileName(outPath)
        w.SetInputData(lpsF.GetOutput()); w.SetFileTypeToBinary(); w.Write()
        slicer.mrmlScene.RemoveNode(labelmap)

        self.onProgressInfo("MergedVTK: Done")


    def _exportVTKPerLabel(self, segNode, folderPath):
        """export un fichier VTK par segment + log via onProgressInfo."""
        import vtk, os, re
        vtk.vtkObject.GlobalWarningDisplayOff()
        segNode.CreateClosedSurfaceRepresentation()
        seg       = segNode.GetSegmentation()
        segSafe   = re.sub(r"[^0-9A-Za-z_-]+","_", segNode.GetName())
        tr        = segNode.GetParentTransformNode()
        parentMat = vtk.vtkMatrix4x4(); parentMat.Identity()
        if tr:
            tr.GetMatrixTransformToWorld(parentMat)

        segmentIDs = seg.GetSegmentIDs()
        total = len(segmentIDs)
        for idx, segId in enumerate(segmentIDs, start=1):
            self.onProgressInfo(f"PerLabelVTK: Segment {idx}/{total}")

            s    = seg.GetSegment(segId)
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
            flatN = vtk.vtkPolyDataNormals(); flatN.SetInputConnection(ws.GetOutputPort())
            flatN.ComputePointNormalsOff(); flatN.ComputeCellNormalsOn()
            flatN.SplittingOff(); flatN.AutoOrientNormalsOn()
            flatN.ConsistencyOn(); flatN.SetFeatureAngle(180); flatN.Update()

            # Decimation
            self.onProgressInfo(f"PerLabelVTK: Decimating {s.GetName()}")
            dec = vtk.vtkQuadricDecimation()
            dec.SetInputConnection(flatN.GetOutputPort()); dec.SetTargetReduction(0.4); dec.Update()

            # Transform & Write
            rasT = vtk.vtkTransform(); rasT.SetMatrix(parentMat)
            rasF = vtk.vtkTransformPolyDataFilter(); rasF.SetTransform(rasT)
            rasF.SetInputConnection(dec.GetOutputPort()); rasF.Update()
            lpsT = vtk.vtkTransform(); lpsT.Scale(-1,-1,1)
            lpsF = vtk.vtkTransformPolyDataFilter(); lpsF.SetTransform(lpsT)
            lpsF.SetInputConnection(rasF.GetOutputPort()); lpsF.Update()

            labelSafe = re.sub(r"[^0-9A-Za-z_-]+","_", s.GetName())
            outPath   = os.path.join(folderPath, f"{segSafe}_{labelSafe}.vtk")
            self.onProgressInfo(f"PerLabelVTK: Writing {labelSafe}.vtk")
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(outPath); writer.SetInputData(lpsF.GetOutput())
            writer.SetFileTypeToBinary(); writer.Write()

        self.onProgressInfo("PerLabelVTK: Done")


    def _exportToGLTF(self, segmentationNode, folderPath, tryInstall=True):
        try:
            from OpenAnatomyExport import OpenAnatomyExportLogic
            logic = OpenAnatomyExportLogic()
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            segmentationItem = shNode.GetItemByDataNode(self.segmentationNodeSelector.currentNode())
            logic.exportModel(segmentationItem, folderPath, self.reductionFactorSlider.value, "glTF")
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
        extensionManager = slicer.app.extensionsManagerModel()
        extensionManager.setInteractive(False)
        extName = "SlicerOpenAnatomy"
        if extensionManager.isExtensionInstalled(extName):
            return

        success = extensionManager.installExtensionFromServer(extName, False, False)
        if not success:
            return

        moduleName = "OpenAnatomyExport"
        modulePath = extensionManager.extensionModulePaths(extName)[0] + f"/{moduleName}.py"
        factory = slicer.app.moduleManager().factoryManager()
        factory.registerModule(qt.QFileInfo(modulePath))
        factory.loadModules([moduleName])

    @staticmethod
    def isNNUNetModuleInstalled():
        try:
            import SlicerNNUNetLib
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
        fileDir = Path(__file__).parent
        return fileDir.joinpath("..", "Resources", "ML").resolve()
