from enum import Flag, auto
from pathlib import Path
import vtk
import SegmentEditorEffects
import ctk
import numpy as np
import qt
import sys
import slicer
import logging
import os
from .IconPath import icon, iconPath
from .PythonDependencyChecker import PythonDependencyChecker, hasInternetConnection
from .Utils import (
    createButton,
    addInCollapsibleLayout,
    set3DViewBackgroundColors,
    setConventionalWideScreenView,
    setBoxAndTextVisibilityOnThreeDViews,
)

# ─── Model descriptions ──────────────────────────────────────────────────────

MODEL_DESCRIPTIONS = {
    "DentalSegmentator": (
        "<b>DentalSegmentator</b><br>"
        "→ Segments: Upper Skull (includes Maxilla), Mandible, Mandibular Canal, Upper Teeth, Lower Teeth<br>"
        "→ Designed for <b>permanent dentition</b>."
    ),
    "PediatricDentalsegmentator": (
        "<b>PediatricDentalsegmentator</b><br>"
        "→ Segments: Upper Skull (includes Maxilla), Mandible, Mandibular Canal, Upper Teeth, Lower Teeth<br>"
        "→ Designed for <b>mixed dentition</b> (baby and permanent teeth)."
    ),
    "NasoMaxillaDentSeg": (
        "<b>NasoMaxillaDentSeg</b><br>"
        "→ Segments: Upper Skull, <u>separate</u> Maxilla, Mandible, Mandibular Canal, Upper Teeth, Lower Teeth<br>"
        "→ Designed for <b>permanent dentition</b> ."
    ),
    "UniversalLabDentalsegmentator": (
        "<b>UniversalLabDentalsegmentator</b><br>"
        "→ Segments: Upper Skull, Mandibular Canal,All teeth<br>"
        "→ Designed for <b>mixed and Permanent dentition</b> ."
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
    Lance « pip install … » de façon non bloquante grâce à qt.QProcess.
    • onLine(str)     : sortie temps réel (stdout + stderr fusionnés)
    • onFinished(ok)  : bool → True si retour 0
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

        # — connexion signaux —
        self._proc.readyReadStandardOutput.connect(self._readLines)
        self._proc.readyReadStandardError.connect(self._readLines)  # fusionné, par sécurité
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
        Slot appelé à la fin du QProcess.
        Qt5 : finished(int)
        Qt6 : finished(int, QProcess.ExitStatus)
        -> *args absorbe éventuellement le 2ᵉ paramètre.
        """
        self._onFinished(exitCode == 0)
        self.deleteLater()          # auto-nettoyage de l’objet
            # auto-nettoyage

class SegmentationWidget(qt.QWidget):

    # ─── Initialization ─────────────────────────────────────────────────────────
    # ─── Initialization ─────────────────────────────────────────────────────────
    def __init__(self, logic=None, parent=None):
        super().__init__(parent)

        # ----------------------------------------------------------------- state
        self.logic                    = logic or self._createSlicerSegmentationLogic()
        self._prevSegmentationNode    = None
        self._minimumIslandSize_mm3   = 60
        self.folderPath               = ""
        self.folderFiles              = []
        self.currentFileIndex         = 0
        self.currentVolumeNode        = None
        self.fullInfoLogs             = []          # journal des messages

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
        # 2)  EXPORT FORMATS  (placé juste sous le dossier de sortie)
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
        # exportLayout.addRow(createButton("Export", callback=self.onExportClicked, parent=exportWidget))

        # ↳ on insère le widget d’export sous les dossiers
        inputLayout.addRow("Export formats :", exportWidget)

        # ========================================================================
        # 3)  DEVICE & MODEL
        # ========================================================================
        self.deviceComboBox = qt.QComboBox(); self.deviceComboBox.addItems(["cuda","cpu","mps"])
        self.modelComboBox  = qt.QComboBox(); self.modelComboBox.addItems([
            "DentalSegmentator","PediatricDentalsegmentator","NasoMaxillaDentSeg","UniversalLabDentalsegmentator"])

        # Resolve-mirroring (spécifique UniversalLab…)
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

        # surface smoothing slider synchronisé avec Show-3D
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

        # Apply / Stop widgets
        self.applyButton = createButton(
            "Apply", callback=self.onApplyClicked,
            toolTip="Run the segmentation.", icon=icon("start_icon.png"))

        self.currentInfoTextEdit = qt.QTextEdit(); self.currentInfoTextEdit.setReadOnly(True)
        self.currentInfoTextEdit.setLineWrapMode(qt.QTextEdit.NoWrap)

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

        # initialise affichage
        self.onInputChangedForLoadedVolume(None)
        self.updateSegmentEditorWidget()

        # observe fermeture de scène
        self.sceneCloseObserver = slicer.mrmlScene.AddObserver(
            slicer.mrmlScene.EndCloseEvent, self.onSceneChanged)
        self.onSceneChanged(doStopInference=False)

        # connecter logique NNUNet
        self._connectSegmentationLogic()



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
        seg   = segmentationNode.GetSegmentation()

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

        # ─── 1. Create an empty label map (correct geometry) ────────────────
        geomLM = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        logic.ExportAllSegmentsToLabelmapNode(
            segmentationNode, geomLM, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)

        ijkToRAS = vtk.vtkMatrix4x4(); geomLM.GetIJKToRASMatrix(ijkToRAS)
        spacing, origin = geomLM.GetSpacing(), geomLM.GetOrigin()
        array_shape     = slicer.util.arrayFromVolume(geomLM).shape
        labelArray      = np.zeros(array_shape, dtype=np.uint16)
        slicer.mrmlScene.RemoveNode(geomLM)  # Remove temporary node

        # ─── 2. Rebuild label map with correct values ───────────────────────
        for segId in seg.GetSegmentIDs():
            segment = seg.GetSegment(segId)

            # a. Get official scalar label value
            tag_val = vtk.mutable("")
            if segment.GetTag("LabelValue", tag_val) and tag_val.get():
                val = int(tag_val.get())
            else:
                val = full_label_map.get(segment.GetName())
                if val is None:
                    print(f"[WARN] Unknown LabelValue for «{segment.GetName()}» — ignored.")
                    continue

            # b. Export THIS segment to a temporary label map
            tmpLM = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            ids   = vtk.vtkStringArray(); ids.InsertNextValue(segId)
            logic.ExportSegmentsToLabelmapNode(
                segmentationNode, ids, tmpLM, volumeNode,
                slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)

            arr = slicer.util.arrayFromVolume(tmpLM)
            labelArray[arr > 0] = val
            slicer.mrmlScene.RemoveNode(tmpLM)

        # ─── 3. Protected mask & mirror table ───────────────────────────────
        protected_vals = {53, 54, 55}
        protected_mask = np.isin(labelArray, list(protected_vals))

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
        def centroid(val):
            pts = np.argwhere(labelArray == val)
            if pts.size == 0:
                return None
            ras_sum = np.zeros(3)
            for z, y, x in pts:
                ras = [0]*4; ijkToRAS.MultiplyPoint([x, y, z, 1.0], ras)
                ras_sum += ras[:3]
            return ras_sum / len(pts)

        incisive_vals = (8, 9, 24, 25)
        inc_centroids = [centroid(v) for v in incisive_vals]
        if any(c is None for c in inc_centroids):
            slicer.util.warningDisplay("Missing central incisors, unable to calculate mirror plane.")
            return
        mirror_x_ras = np.mean([c[0] for c in inc_centroids])

        # ─── 5. Perform mirror correction ────────────────────────────────────
        changed = []
        unique_vals = np.unique(labelArray)
        for i, val in enumerate(unique_vals):
            self.mirroringProgressBar.setValue(int(100 * (i + 1) / len(unique_vals)))
            slicer.app.processEvents()

            if val == 0 or val in protected_vals or val not in mirror_label_map:
                continue

            name        = reverse_full_map.get(val, f"label_{val}")
            mirror_val  = mirror_label_map[val]
            is_left     = "left" in name.lower()

            coords = np.argwhere(labelArray == val)
            fixed  = 0
            for z, y, x in coords:
                if protected_mask[z, y, x]:
                    continue
                ras = [0]*4; ijkToRAS.MultiplyPoint([x, y, z, 1.0], ras)
                if (is_left and ras[0] > mirror_x_ras) or (not is_left and ras[0] < mirror_x_ras):
                    labelArray[z, y, x] = mirror_val; fixed += 1
            if fixed:
                changed.append(f"{name} → {reverse_full_map.get(mirror_val, mirror_val)} ({fixed} vox)")

        self.mirroringProgressBar.setValue(100)

        # ─── 6. Rebuild corrected segmentation ──────────────────────────────
        correctedLM = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(correctedLM, labelArray)
        correctedLM.SetSpacing(spacing)
        correctedLM.SetOrigin(origin)
        correctedLM.SetIJKToRASMatrix(ijkToRAS)

        # ► New name: original segmentation name + suffix
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
        unique_vals      = sorted(np.unique(labelArray[labelArray > 0]))      # [1,2,…,55]
        segIds_sorted    = list(correctedSeg.GetSegmentation().GetSegmentIDs())

        if len(unique_vals) != len(segIds_sorted):
            print("[WARN] Number of values ​​≠ number of segments — check import.")

        for val, segId in zip(unique_vals, segIds_sorted):
            segment = correctedSeg.GetSegmentation().GetSegment(segId)
            segment.SetName(reverse_full_map.get(val, f"label_{val}"))
            segment.SetTag("LabelValue", str(val))

        # DEBUG: Show unique labels after correction
        corr_arr = slicer.util.arrayFromVolume(correctedLM)
        print("Unique labels AFTER correction:", sorted(np.unique(corr_arr[corr_arr > 0])))

        # Cleanup
        slicer.mrmlScene.RemoveNode(correctedLM)
        self.mirroringProgressBar.setVisible(False)

        msg = ("✅ Corrected voxels:\n" + "\n".join(changed)) if changed else "✅ No mirrored voxels detected."
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

    # ──────────────────────────────────────────────────────────────────────────────
    # 3)  _saveSegmentationAsNifti  –  suppression propre du label-map temporaire
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
            self.onProgressInfo(f"✅ Segmentation saved in {output_path}")
        else:
            self.onProgressInfo(f"❌ Failed to save segmentation in {output_path}")

        # ► Nettoyage du label-map
        # labelmapVolumeNode.RemoveAllDisplayNodes()
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)


    def __del__(self):
        slicer.mrmlScene.RemoveObserver(self.sceneCloseObserver)
        super().__del__()

    def selectFolder(self):
        folderPath = qt.QFileDialog.getExistingDirectory(self, "Select Folder Containing Volumes")
        if folderPath:
            self.folderPath = folderPath
            self.folderPathLineEdit.text = folderPath
            folder = Path(folderPath)
            # Filter here according to your formats, e.g., all NIfTI files
            self.folderFiles = list(folder.glob("*.nii*")) + list(folder.glob("*.gipl")) + list(folder.glob("*.gipl.gz"))

            self.currentFileIndex = 0
            self.onProgressInfo(f"Found {len(self.folderFiles)} file(s) in the folder.")

    # ─── Scene change handling ────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────────
    # 2)  onSceneChanged  –  un seul SegmentEditorNode ré-utilisé
    # ──────────────────────────────────────────────────────────────────────────────
    def onSceneChanged(self, *_, doStopInference=True):
        if doStopInference:
            self.onStopClicked()

        # ⇢ Conserver UN unique SegmentEditorNode
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
        self.logic.stopSegmentation()
        self.logic.waitForSegmentationFinished()
        slicer.app.processEvents()
        self.isStopping = False
        self._setApplyVisible(True)

    # ─── Apply segmentation ─────────────────────────────────────────────────────
    

# ──────────────────────────────────────────────────────────────
# 1.  Fonction utilitaire pour exécuter pip de façon asynchrone
# ──────────────────────────────────────────────────────────────

    def onApplyClicked(self, *_):
        # --- validations rapides ---
        if not self.folderPath:
            slicer.util.errorDisplay("Please select a folder containing volumes.")
            return
        if not self.folderFiles:
            slicer.util.errorDisplay("No valid volume file found in the folder.")
            return

        self.currentInfoTextEdit.clear()
        self._setApplyVisible(False)

        # ---------- Étape 1 : installation deps async ----------
        packages = ["numpy<2.0", "numexpr<2.10"]   # complétez votre liste

        _log = logging.getLogger("BATCHDENTALSEG")   # nom de votre extension

        def _onLine(line: str):
            self.currentInfoTextEdit.append(line)    # QTextEdit
            _log.info(line)   

        def _onFinished(ok: bool):
            if not ok:
                qt.QMessageBox.critical(
                    self, "Installation error",
                    "Certaines bibliothèques Python n'ont pas pu être installées.\n"
                    "Veuillez vérifier votre connexion Internet ou relancer Slicer."
                )
                self._setApplyVisible(True)
                return

            # ---------- Étape 2 : dépendances internes ----------
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

            # ---------- Étape 3 : traiter les scans ----------
            self.processNextFile()

        # Lancement *non bloque* (aucun thread Python ⇒ plus de warnings Qt)
        self._pipRunner = PipRunner(packages, _onLine, _onFinished, parent=self)


    def processNextFile(self):
        if self.currentFileIndex >= len(self.folderFiles):
            self.onProgressInfo("All files in the folder have been processed.")
            self._setApplyVisible(True)
            return

        filePath = self.folderFiles[self.currentFileIndex]
        self.onProgressInfo(f"File processing: {filePath}")

        loadedVolume = slicer.util.loadVolume(str(filePath))
        if not loadedVolume:
            self.onProgressInfo(f"Error loading {filePath}. Moving on to the next one.")
            self.currentFileIndex += 1
            self.processNextFile()
            return

        self.currentVolumeNode = loadedVolume
        self.onInputChangedForLoadedVolume(loadedVolume)
        self.onApplyClickedForVolume(loadedVolume)

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
        selectedModel = self.modelComboBox.currentText
        if selectedModel == "PediatricDentalsegmentator":
            print("Selected Model : ",selectedModel)
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
            print("Selected Model : ",selectedModel)
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
            print("Selected Model : ",selectedModel)
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
            print("Selected Model : ",selectedModel)
            parameter = Parameter(folds="0", modelPath=self.nnUnetFolder(), device=self.deviceComboBox.currentText)
                
        if not parameter.isSelectedDeviceAvailable():
            deviceName = parameter.device.upper()
            ret = qt.QMessageBox.question(
                self,
                f"{deviceName} device not available",
                f"Selected device ({deviceName}) is not available and will default to CPU.\n"
                "Running the segmentation may take up to 1 hour.\n"
                "Would you like to proceed?"
            )
            if ret == qt.QMessageBox.No:
                self._setApplyVisible(True)
                return
        slicer.app.processEvents()
        self.logic.setParameter(parameter)
        self.logic.startSegmentation(volumeNode)

    # ─── Inference finished callback ──────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────────
    # 1)  onInferenceFinished  –  appel explicite à _cleanupAfterCase
    # ──────────────────────────────────────────────────────────────────────────────
    def onInferenceFinished(self, *_):
        if self.isStopping:
            self._setApplyVisible(True)
            return

        segNode = None
        volNode = None
        try:
            self.onProgressInfo("Loading inference results...")
            # Chargement des résultats
            self._loadSegmentationResults()

            segNode = self.getCurrentSegmentationNode()
            volNode = self.getCurrentVolumeNode()
            if not segNode:
                raise RuntimeError("No segmentation node found after inference.")

            # ---------- export selon cases cochées ----------
            selected = self.getSelectedExportFormats()
            if selected == ExportFormat(0):
                self.onProgressInfo("⚠️  No export format selected — nothing saved.")
            else:
                if not self.outputFolderPath:
                    raise RuntimeError("Output folder not selected.")
                if selected & ExportFormat.NIFTI:
                    self._saveSegmentationAsNifti(segNode, volNode)
                other = selected & ~ExportFormat.NIFTI
                if other != ExportFormat(0):
                    self.exportSegmentation(segNode, self.outputFolderPath, other)

            self.onProgressInfo("Inference ended successfully.")

        except Exception as e:
            slicer.util.errorDisplay(e)
            self.onProgressInfo(f"Error loading results :\n{e}")

        finally:
            # ►► libération complète des ressources pour le cas courant
            # self._cleanupAfterCase(volNode, segNode)

            # ---------- enchaînement dossier ----------
            if self.folderPath:
                self.currentFileIndex += 1
                if self.currentFileIndex < len(self.folderFiles):
                    self.processNextFile()
                else:
                    self.onProgressInfo("All files in the folder have been processed.")
                    self._setApplyVisible(True)
            else:
                self._setApplyVisible(True)


    # ─── Load segmentation results ────────────────────────────────────────────

    def _loadSegmentationResults(self):
        currentSegmentation = self.getCurrentSegmentationNode()
        segmentationNode = self.logic.loadSegmentation()
        segmentationNode.SetName(self.currentVolumeNode.GetName() + "_Segmentation")
        if currentSegmentation is not None:
            self._copySegmentationResultsToExistingNode(currentSegmentation, segmentationNode)
        else:
            self.segmentationNodeSelector.setCurrentNode(segmentationNode)
        slicer.app.processEvents()
        self._updateSegmentationDisplay()
        # self._postProcessSegments()
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
        # self._keepLargestIsland("Segment_1")
        # self._removeSmallIsland("Segment_3")
        # self._removeSmallIsland("Segment_4")
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
        if self.isStopping:
            return
        self._setApplyVisible(True)
        slicer.util.errorDisplay("Encountered error during inference :\n" + str(errorMsg))


    def onProgressInfo(self, infoMsg):
        infoMsg = self.removeImageIOError(infoMsg)
        self.currentInfoTextEdit.insertPlainText(infoMsg + "\n")
        self.moveTextEditToEnd(self.currentInfoTextEdit)
        self.insertDatedInfoLogs(infoMsg)
        slicer.app.processEvents()

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

        """
        Updates the segmentation editor widget based on the currently selected node.
        """
        if self._prevSegmentationNode:
            self._prevSegmentationNode.SetDisplayVisibility(False)

        segmentationNode = self.getCurrentSegmentationNode()
        self._prevSegmentationNode = segmentationNode
        self._initializeSegmentationNodeDisplay(segmentationNode)
        self.segmentEditorWidget.setSegmentationNode(segmentationNode)
        self.segmentEditorWidget.setSourceVolumeNode(self.getCurrentVolumeNode())
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
        segmentationNode = self.getCurrentSegmentationNode()
        if not segmentationNode:
            slicer.util.warningDisplay("Please select a valid segmentation before exporting.")
            return

        selectedFormats = self.getSelectedExportFormats()
        if selectedFormats == ExportFormat(0):
            slicer.util.warningDisplay("Please select at least one export format before exporting.")
            return

        folderPath = qt.QFileDialog.getExistingDirectory(self, "Please select the export folder")
        if not folderPath:
            return

        with slicer.util.tryWithErrorDisplay(f"Export to {folderPath} failed.", waitCursor=True):
            self.exportSegmentation(segmentationNode, folderPath, selectedFormats)
            slicer.util.infoDisplay(f"Export successful to {folderPath}.")

    def exportSegmentation(self, segNode, folderPath, selectedFormats):
        """
        - STL / OBJ : inchangés
        - VTK_MERGED : pipeline historique (un seul fichier)
        - VTK        : un fichier .vtk par segment
        """
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
        """ancien pipeline ‘merged’, déplacé ici sans changement fonctionnel."""
        import vtk, os, numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        refVol = self.getCurrentVolumeNode()
        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segNode, labelmap)
        img = labelmap.GetImageData()

        mc = vtk.vtkDiscreteMarchingCubes(); mc.SetInputData(img)
        for l in np.unique(vtk_to_numpy(img.GetPointData().GetScalars())):
            if l: mc.SetValue(int(l), int(l))
        mc.Update()

        clean = vtk.vtkCleanPolyData(); clean.SetInputConnection(mc.GetOutputPort()); clean.Update()
        ws = vtk.vtkWindowedSincPolyDataFilter(); ws.SetInputConnection(clean.GetOutputPort())
        ws.SetNumberOfIterations(60); ws.SetPassBand(0.05)
        ws.BoundarySmoothingOn(); ws.FeatureEdgeSmoothingOn()
        ws.NonManifoldSmoothingOn(); ws.NormalizeCoordinatesOn(); ws.Update()

        flatN = vtk.vtkPolyDataNormals(); flatN.SetInputConnection(ws.GetOutputPort())
        flatN.ComputePointNormalsOff(); flatN.ComputeCellNormalsOn()
        flatN.SplittingOff(); flatN.AutoOrientNormalsOn()
        flatN.ConsistencyOn(); flatN.SetFeatureAngle(180); flatN.Update()

        # transforms (parent + RAS→LPS)
        ijk2ras = vtk.vtkMatrix4x4(); labelmap.GetIJKToRASMatrix(ijk2ras)
        parentMat = vtk.vtkMatrix4x4(); parentMat.Identity()
        if refVol and refVol.GetParentTransformNode():
            refVol.GetParentTransformNode().GetMatrixTransformToWorld(parentMat)
        rasMat = vtk.vtkMatrix4x4(); vtk.vtkMatrix4x4.Multiply4x4(parentMat, ijk2ras, rasMat)

        rasT = vtk.vtkTransform(); rasT.SetMatrix(rasMat)
        rasF = vtk.vtkTransformPolyDataFilter(); rasF.SetTransform(rasT); rasF.SetInputConnection(flatN.GetOutputPort()); rasF.Update()
        lpsT = vtk.vtkTransform(); lpsT.Scale(-1, -1, 1)
        lpsF = vtk.vtkTransformPolyDataFilter(); lpsF.SetTransform(lpsT); lpsF.SetInputConnection(rasF.GetOutputPort()); lpsF.Update()

        outPath = os.path.join(folderPath, f"{segNode.GetName()}_merged.vtk")
        w = vtk.vtkPolyDataWriter(); w.SetFileName(outPath); w.SetInputData(lpsF.GetOutput()); w.SetFileTypeToBinary(); w.Write()
        slicer.mrmlScene.RemoveNode(labelmap)

    def _exportVTKPerLabel(self, segNode, folderPath):
        """boucle sur chaque segment, écrit <SegName>_<LabelName>.vtk"""
        import vtk, os, re

        segNode.CreateClosedSurfaceRepresentation()
        seg        = segNode.GetSegmentation()
        segNameRaw = segNode.GetName()
        segSafe    = re.sub(r"[^0-9A-Za-z_-]+", "_", segNameRaw)   # nom de la segmentation “sécurisé”

        # matrice éventuelle de parent transform
        parentMat = vtk.vtkMatrix4x4(); parentMat.Identity()
        tr = segNode.GetParentTransformNode()
        if tr:
            tr.GetMatrixTransformToWorld(parentMat)

        for segId in seg.GetSegmentIDs():
            s = seg.GetSegment(segId)
            poly = s.GetRepresentation("Closed surface")
            if not poly or poly.GetNumberOfPoints() == 0:
                continue

            # --- pipeline nettoyage + lissage (inchangé) ----------------------
            clean = vtk.vtkCleanPolyData(); clean.SetInputData(poly); clean.Update()
            ws = vtk.vtkWindowedSincPolyDataFilter(); ws.SetInputConnection(clean.GetOutputPort())
            ws.SetNumberOfIterations(60); ws.SetPassBand(0.05)
            ws.BoundarySmoothingOn(); ws.FeatureEdgeSmoothingOn()
            ws.NonManifoldSmoothingOn(); ws.NormalizeCoordinatesOn(); ws.Update()

            flatN = vtk.vtkPolyDataNormals(); flatN.SetInputConnection(ws.GetOutputPort())
            flatN.ComputePointNormalsOff(); flatN.ComputeCellNormalsOn()
            flatN.SplittingOff(); flatN.AutoOrientNormalsOn()
            flatN.ConsistencyOn(); flatN.SetFeatureAngle(180); flatN.Update()

            rasT = vtk.vtkTransform(); rasT.SetMatrix(parentMat)
            rasF = vtk.vtkTransformPolyDataFilter(); rasF.SetTransform(rasT); rasF.SetInputConnection(flatN.GetOutputPort()); rasF.Update()
            lpsT = vtk.vtkTransform(); lpsT.Scale(-1, -1, 1)
            lpsF = vtk.vtkTransformPolyDataFilter(); lpsF.SetTransform(lpsT); lpsF.SetInputConnection(rasF.GetOutputPort()); lpsF.Update()

            # --- nom du fichier ------------------------------------------------
            labelSafe = re.sub(r"[^0-9A-Za-z_-]+", "_", s.GetName())
            outPath   = os.path.join(folderPath, f"{segSafe}_{labelSafe}.vtk")

            # --- écriture ------------------------------------------------------
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(outPath)
            writer.SetInputData(lpsF.GetOutput())
            writer.SetFileTypeToBinary()
            writer.Write()


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
            return
        self.logic.progressInfo.connect(self.onProgressInfo)
        self.logic.errorOccurred.connect(self.onInferenceError)
        self.logic.inferenceFinished.connect(self.onInferenceFinished)

    @classmethod
    def nnUnetFolder(cls) -> Path:
        fileDir = Path(__file__).parent
        return fileDir.joinpath("..", "Resources", "ML").resolve()
