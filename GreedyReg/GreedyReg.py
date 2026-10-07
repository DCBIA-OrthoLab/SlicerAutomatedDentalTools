import os
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *

import sys
# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

# Logic reaches ADTLib (env.deps), so the bootstrap has to be above it.
from GreedyReg_Method.Logic import GreedyRegLogic

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
import tempfile
import time

logger = get_logger("GreedyReg")


class GreedyReg(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Greedy Registration"
    self.parent.categories = ["Automated Dental Tools"]
    self.parent.dependencies = []
    self.parent.contributors = ["Your Lab"]
    self.parent.helpText = "ITK-SNAP style registration using Greedy"
    self.parent.acknowledgementText = ""


class GreedyRegWidget(ScriptedLoadableModuleWidget):
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    self.logic = GreedyRegLogic()

    # Greedy binary check
    if not self.logic.isGreedyAvailable():
      self.greedyWarningBox = ctk.ctkCollapsibleButton()
      self.greedyWarningBox.text = "Greedy not found"
      self.greedyWarningBox.collapsed = False
      self.layout.addWidget(self.greedyWarningBox)
      warning_layout = qt.QFormLayout(self.greedyWarningBox)
      warning_label = qt.QLabel("Greedy registration engine not found.\nClick below to download it automatically (from ITK-SNAP, ~200MB).\nSlicer stays usable while it downloads.")
      warning_label.setStyleSheet("color: red;")
      warning_layout.addRow(warning_label)
      self.downloadButton = qt.QPushButton("Download Greedy")
      self.downloadButton.setStyleSheet(
        "QPushButton { background-color: #F44336; color: white; "
        "font-weight: bold; padding: 8px; border-radius: 4px; }")
      self.downloadButton.clicked.connect(self.onDownloadGreedy)
      warning_layout.addRow(self.downloadButton)
      self.downloadProgressBar = qt.QProgressBar()
      self.downloadProgressBar.setRange(0, 100)
      self.downloadProgressBar.setVisible(False)
      warning_layout.addRow(self.downloadProgressBar)
      self.downloadStatusLabel = qt.QLabel("")
      self.downloadStatusLabel.setWordWrap(True)
      warning_layout.addRow(self.downloadStatusLabel)
    self._downloadProcess = None

    # ALI_CBCT Python library check for Distant Registration
    self._aliLibsReady = self.logic.aliLibrariesReady()

    # Create transform node for manual alignment
    self.transformNode = slicer.mrmlScene.AddNewNodeByClass(
      "vtkMRMLLinearTransformNode", "GreedyManualTransform")

    #-- Volume selectors ------------------------------------------
    volumes_box = ctk.ctkCollapsibleButton()
    volumes_box.text = "Input Volumes"
    self.layout.addWidget(volumes_box)
    volumes_layout = qt.QFormLayout(volumes_box)

    self.fixedSelector = slicer.qMRMLNodeComboBox()
    self.fixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.fixedSelector.setMRMLScene(slicer.mrmlScene)
    self.fixedSelector.setToolTip("Fixed image (T1)")
    volumes_layout.addRow("Fixed (T1):", self.fixedSelector)

    self.movingSelector = slicer.qMRMLNodeComboBox()
    self.movingSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.movingSelector.setMRMLScene(slicer.mrmlScene)
    self.movingSelector.setToolTip("Moving image (T2)")
    volumes_layout.addRow("Moving (T2):", self.movingSelector)

    self.maskSelector = slicer.qMRMLNodeComboBox()
    self.maskSelector.nodeTypes = ["vtkMRMLSegmentationNode", "vtkMRMLLabelMapVolumeNode"]
    self.maskSelector.setMRMLScene(slicer.mrmlScene)
    self.maskSelector.setToolTip("Segmentation mask (T1 space), used for registration")
    self.maskSelector.addEnabled = False
    self.maskSelector.noneEnabled = True
    volumes_layout.addRow("Mask (T1):", self.maskSelector)

    self.segmentationSelector = slicer.qMRMLNodeComboBox()
    self.segmentationSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
    self.segmentationSelector.setToolTip("CBCT/MRI volume to segment on, in the Create Mask panel below")
    self.segmentationSelector.addEnabled = False
    self.segmentationSelector.noneEnabled = True
    volumes_layout.addRow("Source Volume (Segmentation):", self.segmentationSelector)
    self.segmentationSelector.connect(
      "currentNodeChanged(vtkMRMLNode*)", self.onSegmentationSelected)

    self.segmentationNodeSelector = slicer.qMRMLNodeComboBox()
    self.segmentationNodeSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.segmentationNodeSelector.setMRMLScene(slicer.mrmlScene)
    self.segmentationNodeSelector.setToolTip("Segmentation to edit in the Create Mask panel below")
    self.segmentationNodeSelector.addEnabled = True
    self.segmentationNodeSelector.noneEnabled = True
    volumes_layout.addRow("Segmentation:", self.segmentationNodeSelector)
    self.segmentationNodeSelector.connect(
      "currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeSelected)

    self.modelSelector = slicer.qMRMLNodeComboBox()
    self.modelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.modelSelector.setMRMLScene(slicer.mrmlScene)
    self.modelSelector.setToolTip("3D model to transform (optional)")
    self.modelSelector.addEnabled = False
    self.modelSelector.noneEnabled = True
    volumes_layout.addRow("3D Model (optional):", self.modelSelector)

    # Connect selectors
    self.movingSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMovingVolumeChanged)
    self.fixedSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onFixedVolumeChanged)
    self.modelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onModelChanged)

    #-- Create Mask -----------------------------------------------
    paint_box = ctk.ctkCollapsibleButton()
    paint_box.text = "Create Mask"
    paint_box.collapsed = True
    self.layout.addWidget(paint_box)
    paint_layout = qt.QVBoxLayout(paint_box)

    # Embed Slicer's standard Segment Editor instead of the custom
    # ROI/lasso/scissors/paint tools, operating on a vtkMRMLSegmentationNode.
    self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
    self.segmentEditorWidget.connect(
      "segmentationNodeChanged(vtkMRMLSegmentationNode*)", self.maskSelector.setCurrentNode)
    self.segmentEditorWidget.connect(
      "segmentationNodeChanged(vtkMRMLSegmentationNode*)", self.segmentationNodeSelector.setCurrentNode)
    self.segmentEditorWidget.sourceVolumeNodeSelectorVisible = False
    self.segmentEditorWidget.segmentationNodeSelectorVisible = False
    self.segmentEditorWidget.setEffectNameOrder(["Paint", "Surface cut"])
    self.segmentEditorWidget.unorderedEffectsVisible = False
    paint_layout.addWidget(self.segmentEditorWidget)
    self._emphasizeEffectButtons(["Paint", "Surface cut"], scale=2.0)

    # The selectors above may have auto-selected a node (and fired their
    # currentNodeChanged signal) before segmentEditorWidget existed, so
    # force a sync now that it does.
    self.onSegmentationSelected(self.segmentationSelector.currentNode())
    self.onSegmentationNodeSelected(self.segmentationNodeSelector.currentNode())

    self.paintStatusLabel = qt.QLabel("")
    self.paintStatusLabel.setAlignment(qt.Qt.AlignCenter)
    paint_layout.addWidget(self.paintStatusLabel)

    #-- Manual alignment ------------------------------------------
    manual_tools_box = ctk.ctkCollapsibleButton()
    manual_tools_box.text = "Manual Alignment"
    self.layout.addWidget(manual_tools_box)
    manual_tools_layout = qt.QFormLayout(manual_tools_box)

    self.centerButton = qt.QPushButton("Center T2 on T1")
    self.centerButton.setStyleSheet(
      "QPushButton { background-color: #607D8B; color: white; "
      "font-weight: bold; padding: 6px; border-radius: 4px; }")
    self.centerButton.clicked.connect(self.onCenterVolumes)
    manual_tools_layout.addRow(self.centerButton)

    self.interactiveButton = qt.QPushButton("Enable Interactive Tool")
    self.interactiveButton.setCheckable(True)
    self.interactiveButton.setStyleSheet(
      "QPushButton { background-color: #2196F3; color: white; "
      "font-weight: bold; padding: 6px; border-radius: 4px; }"
      "QPushButton:checked { background-color: #F44336; }")
    self.interactiveButton.clicked.connect(self.onInteractiveToolToggled)
    manual_tools_layout.addRow(self.interactiveButton)

    self.interactiveHint = qt.QLabel("Drag the arrows to translate, the rings to rotate (Slicer's built-in transform handles)")
    self.interactiveHint.setStyleSheet("color: gray; font-size: 10px;")
    self.interactiveHint.setAlignment(qt.Qt.AlignCenter)
    self.interactiveHint.setVisible(False)
    manual_tools_layout.addRow(self.interactiveHint)

    # Embed Slicer's standard Transforms module Display panel (interaction
    # handle checkboxes, axis enables, glyph/grid options, etc.) instead of
    # only toggling the handles invisibly from code.
    self.transformDisplayWidget = slicer.qMRMLTransformDisplayNodeWidget()
    self.transformDisplayWidget.setMRMLTransformNode(self.transformNode)
    self.transformDisplayWidget.setVisible(False)
    manual_tools_layout.addRow(self.transformDisplayWidget)

    self.hardenButton = qt.QPushButton("Harden Transform & Keep in Scene")
    self.hardenButton.setStyleSheet(
      "QPushButton { background-color: #9C27B0; color: white; "
      "font-weight: bold; padding: 6px; border-radius: 4px; }")
    self.hardenButton.clicked.connect(self.onHardenTransform)
    manual_tools_layout.addRow(self.hardenButton)

    self.resetButton = qt.QPushButton("Reset Transform")
    self.resetButton.clicked.connect(self.onResetTransform)
    manual_tools_layout.addRow(self.resetButton)

    #-- Automatic registration ------------------------------------
    reg_box = ctk.ctkCollapsibleButton()
    reg_box.text = "Automatic Registration"
    self.layout.addWidget(reg_box)
    reg_layout = qt.QFormLayout(reg_box)

    self.metricSelector = qt.QComboBox()
    self.metricSelector.addItems(["NMI - Mutual Information",
                                  "NCC - Cross Correlation",
                                  "SSD - Intensity Difference"])
    reg_layout.addRow("Metric:", self.metricSelector)

    self.transformSelector = qt.QComboBox()
    self.transformSelector.addItems(["Rigid", "Affine"])
    reg_layout.addRow("Transform:", self.transformSelector)

    self.useMaskCheck = qt.QCheckBox("Use segmentation as mask")
    self.useMaskCheck.checked = True
    reg_layout.addRow(self.useMaskCheck)

    self.runButton = qt.QPushButton("Run Registration")
    self.runButton.setStyleSheet(
      "QPushButton { background-color: #4CAF50; color: white; "
      "font-weight: bold; padding: 6px; border-radius: 4px; }")
    self.runButton.clicked.connect(self.onRunRegistration)
    reg_layout.addRow(self.runButton)

    self.statusLabel = qt.QLabel("")
    self.statusLabel.setAlignment(qt.Qt.AlignCenter)
    reg_layout.addRow(self.statusLabel)

    self.saveTransformButton = qt.QPushButton("Save Transform Matrix")
    self.saveTransformButton.clicked.connect(self.onSaveTransform)
    reg_layout.addRow(self.saveTransformButton)

    self.saveVolumeButton = qt.QPushButton("Save Registered Volume")
    self.saveVolumeButton.clicked.connect(self.onSaveVolume)
    reg_layout.addRow(self.saveVolumeButton)

    # Batch automatic registration
    def makeFolderRow(placeholder, browse_slot):
      row = qt.QHBoxLayout()
      edit = qt.QLineEdit()
      edit.setPlaceholderText(placeholder)
      row.addWidget(edit)
      btn = qt.QPushButton("Browse")
      btn.clicked.connect(browse_slot)
      row.addWidget(btn)
      w = qt.QWidget()
      w.setLayout(row)
      return edit, w

    self._batchAutoT1Edit, batch_auto_t1_widget = makeFolderRow(
      "T1 folder (e.g. A01_t1.nii.gz)...",
      lambda: self._browseBatchFolder(self._batchAutoT1Edit, None))
    reg_layout.addRow("T1 Folder:", batch_auto_t1_widget)

    self._batchAutoT2Edit, batch_auto_t2_widget = makeFolderRow(
      "T2 folder (e.g. A01_t2.nii.gz)...",
      lambda: self._browseBatchFolder(self._batchAutoT2Edit, None))
    reg_layout.addRow("T2 Folder:", batch_auto_t2_widget)

    self._batchAutoMaskEdit, batch_auto_mask_widget = makeFolderRow(
      "Mask folder (e.g. A01_MASK.nii.gz) - optional...",
      lambda: self._browseBatchFolder(self._batchAutoMaskEdit, None))
    reg_layout.addRow("Mask Folder:", batch_auto_mask_widget)

    self._batchAutoPairsLabel = qt.QLabel("Select T1 and T2 folders to detect pairs")
    self._batchAutoPairsLabel.setStyleSheet("color: gray; font-size: 10px;")
    self._batchAutoPairsLabel.setWordWrap(True)
    reg_layout.addRow(self._batchAutoPairsLabel)

    self._runBatchAutoButton = qt.QPushButton("Run Batch Registration")
    self._runBatchAutoButton.setStyleSheet(
      "QPushButton { background-color: #FF5722; color: white; "
      "font-weight: bold; padding: 6px; border-radius: 4px; }")
    self._runBatchAutoButton.clicked.connect(self.onRunBatchAuto)
    reg_layout.addRow(self._runBatchAutoButton)

    self._batchAutoStatusLabel = qt.QLabel("")
    self._batchAutoStatusLabel.setAlignment(qt.Qt.AlignCenter)
    self._batchAutoStatusLabel.setWordWrap(True)
    reg_layout.addRow(self._batchAutoStatusLabel)

    #-- Distant Registration --------------------------------------
    distant_box = ctk.ctkCollapsibleButton()
    distant_box.text = "Distant Registration (Large Misalignment)"
    distant_box.collapsed = True
    self.layout.addWidget(distant_box)
    distant_layout = qt.QFormLayout(distant_box)

    self._distantLibsWarning = qt.QLabel(
      "Some Python libraries required for ALI-based Distant Registration "
      "(itk, dicom2nifti, pydicom, monai) are missing.")
    self._distantLibsWarning.setStyleSheet("color: red;")
    self._distantLibsWarning.setWordWrap(True)
    self._distantLibsWarning.setVisible(not self._aliLibsReady)
    distant_layout.addRow(self._distantLibsWarning)

    self._installAliLibsButton = qt.QPushButton("Install ALI Libraries")
    self._installAliLibsButton.setStyleSheet(
      "QPushButton { background-color: #F44336; color: white; "
      "font-weight: bold; padding: 8px; border-radius: 4px; }")
    self._installAliLibsButton.clicked.connect(self.onInstallAliLibraries)
    distant_layout.addRow(self._installAliLibsButton)

    model_folder_row = qt.QHBoxLayout()
    self._aliModelEdit = qt.QLineEdit()
    self._aliModelEdit.setPlaceholderText("Path to ALI models folder...")
    model_folder_row.addWidget(self._aliModelEdit)
    self._aliModelBrowse = qt.QPushButton("Browse")
    self._aliModelBrowse.clicked.connect(self.onBrowseAliModel)
    model_folder_row.addWidget(self._aliModelBrowse)
    model_folder_widget = qt.QWidget()
    model_folder_widget.setLayout(model_folder_row)
    distant_layout.addRow("ALI Models:", model_folder_widget)

    self._distantStructureGroup = qt.QHBoxLayout()
    self._distantMandCheck = qt.QCheckBox("Mandible")
    self._distantMandCheck.checked = True
    self._distantMaxCheck = qt.QCheckBox("Maxilla")
    self._distantCbCheck = qt.QCheckBox("Cranial Base")
    self._distantStructureGroup.addWidget(self._distantMandCheck)
    self._distantStructureGroup.addWidget(self._distantMaxCheck)
    self._distantStructureGroup.addWidget(self._distantCbCheck)
    struct_widget = qt.QWidget()
    struct_widget.setLayout(self._distantStructureGroup)
    distant_layout.addRow("Structures:", struct_widget)

    self._runDistantButton = qt.QPushButton("Run Distant Registration")
    self._runDistantButton.setStyleSheet(
      "QPushButton { background-color: #9C27B0; color: white; "
      "font-weight: bold; padding: 6px; border-radius: 4px; }")
    self._runDistantButton.clicked.connect(self.onRunDistantRegistration)
    distant_layout.addRow(self._runDistantButton)

    self._distantStatusLabel = qt.QLabel("")
    self._distantStatusLabel.setAlignment(qt.Qt.AlignCenter)
    distant_layout.addRow(self._distantStatusLabel)

    # Batch distant registration
    self._batchDistT1Edit, batch_dist_t1_widget = makeFolderRow(
      "T1 folder (e.g. A01_t1.nii.gz)...",
      lambda: self._browseBatchFolder(self._batchDistT1Edit, None))
    distant_layout.addRow("T1 Folder:", batch_dist_t1_widget)

    self._batchDistT2Edit, batch_dist_t2_widget = makeFolderRow(
      "T2 folder (e.g. A01_t2.nii.gz)...",
      lambda: self._browseBatchFolder(self._batchDistT2Edit, None))
    distant_layout.addRow("T2 Folder:", batch_dist_t2_widget)

    self._batchDistPairsLabel = qt.QLabel("Select T1 and T2 folders to detect pairs")
    self._batchDistPairsLabel.setStyleSheet("color: gray; font-size: 10px;")
    self._batchDistPairsLabel.setWordWrap(True)
    distant_layout.addRow(self._batchDistPairsLabel)

    self._runBatchDistButton = qt.QPushButton("Run Batch Distant Registration")
    self._runBatchDistButton.setStyleSheet(
      "QPushButton { background-color: #FF5722; color: white; "
      "font-weight: bold; padding: 6px; border-radius: 4px; }")
    self._runBatchDistButton.clicked.connect(self.onRunBatchDist)
    distant_layout.addRow(self._runBatchDistButton)

    self._batchDistStatusLabel = qt.QLabel("")
    self._batchDistStatusLabel.setAlignment(qt.Qt.AlignCenter)
    self._batchDistStatusLabel.setWordWrap(True)
    distant_layout.addRow(self._batchDistStatusLabel)

    self.layout.addStretch()

    self.onFixedVolumeChanged(self.fixedSelector.currentNode())
    self.onMovingVolumeChanged(self.movingSelector.currentNode())

    slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onSceneCleared)
    slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndImportEvent, self.onSceneCleared)

  def _emphasizeEffectButtons(self, effect_names, scale=2.0):
    """Enlarge specific Segment Editor effect buttons (by their internal
    effect name) so they stand out, e.g. Paint and Surface cut."""
    for effect_name in effect_names:
      effect_button = self.segmentEditorWidget.findChild(qt.QToolButton, effect_name)
      if not effect_button:
        continue
      icon_size = effect_button.iconSize
      effect_button.setIconSize(qt.QSize(
        int(icon_size.width() * scale), int(icon_size.height() * scale)))
      size_hint = effect_button.sizeHint
      effect_button.setMinimumSize(
        int(size_hint.width() * scale), int(size_hint.height() * scale))

  #-- Volume/Model selection methods ----------------------------

  def onMovingVolumeChanged(self, node):
    if node:
      node.SetAndObserveTransformNodeID(self.transformNode.GetID())

  def onFixedVolumeChanged(self, node):
    if node:
      node.SetAndObserveTransformNodeID(None)

  def onModelChanged(self, node):
    if node:
      node.SetAndObserveTransformNodeID(self.transformNode.GetID())

  def onSegmentationSelected(self, node):
    if hasattr(self, 'segmentEditorWidget'):
      self.segmentEditorWidget.setSourceVolumeNode(node)

  def onSegmentationNodeSelected(self, node):
    if hasattr(self, 'segmentEditorWidget'):
      self.segmentEditorWidget.setSegmentationNode(node)

  #-- Manual alignment methods ----------------------------------

  def onResetTransform(self):
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    self.transformNode.SetMatrixTransformToParent(matrix)

  def onInteractiveToolToggled(self, checked):
    if checked:
      self.interactiveButton.setText("Disable Interactive Tool")
      self.interactiveHint.setVisible(True)
      self.startInteractiveTool()
    else:
      self.interactiveButton.setText("Enable Interactive Tool")
      self.interactiveHint.setVisible(False)
      self.stopInteractiveTool()

  def onHardenTransform(self):
    moving = self.movingSelector.currentNode()
    if moving:
        slicer.vtkSlicerTransformLogic().hardenTransform(moving)
    model = self.modelSelector.currentNode()
    if model:
        slicer.vtkSlicerTransformLogic().hardenTransform(model)
    if not moving and not model:
        self.statusLabel.setText("No moving volume or model selected!")
        return
    # Reset transform to identity after hardening
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    self.transformNode.SetMatrixTransformToParent(matrix)
    self.statusLabel.setText("Transform hardened!")

  def startInteractiveTool(self):
    """Show Slicer's built-in transform interaction handles (translate + rotate)
    in both slice and 3D views, plus the standard Transforms module Display
    panel, instead of a custom mouse-driven implementation."""
    moving = self.movingSelector.currentNode()
    if moving:
      moving.SetAndObserveTransformNodeID(self.transformNode.GetID())
    model = self.modelSelector.currentNode()
    if model:
      model.SetAndObserveTransformNodeID(self.transformNode.GetID())

    display_node = self.transformNode.GetDisplayNode()
    if not display_node:
      self.transformNode.CreateDefaultDisplayNodes()
      display_node = self.transformNode.GetDisplayNode()

    display_node.SetVisibility(True)
    display_node.SetEditorVisibility(True)
    display_node.SetEditorVisibility3D(True)
    display_node.SetEditorSliceIntersectionVisibility(True)

    # Rigid alignment only: translation + rotation, no scaling handles
    display_node.SetEditorTranslationEnabled(True)
    display_node.SetEditorRotationEnabled(True)
    display_node.SetEditorScalingEnabled(False)
    display_node.SetEditorTranslationSliceEnabled(True)
    display_node.SetEditorRotationSliceEnabled(True)
    display_node.SetEditorScalingSliceEnabled(False)

    self.transformDisplayWidget.setMRMLTransformNode(self.transformNode)
    self.transformDisplayWidget.setVisible(True)

  def stopInteractiveTool(self):
    display_node = self.transformNode.GetDisplayNode()
    if display_node:
      display_node.SetEditorVisibility(False)
    self.transformDisplayWidget.setVisible(False)

  def onSceneCleared(self, caller, event):
    # Recreate transform node
    self.transformNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLinearTransformNode", "GreedyManualTransform")
    self.transformDisplayWidget.setMRMLTransformNode(self.transformNode)
    self.transformDisplayWidget.setVisible(False)

    # Recreate segment editor node (also scene-owned)
    self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)

    # Reset registration result
    self._regResult = {}

    # Reset status labels
    self.statusLabel.setText("")
    self.paintStatusLabel.setText("")

    # Reset buttons
    self.runButton.setEnabled(True)
    self.interactiveButton.setChecked(False)
    self.interactiveButton.setText("Enable Interactive Tool")
    self.interactiveHint.setVisible(False)

    # Stop any running tools
    self.stopInteractiveTool()

    # Reattach volumes if already selected
    self.onFixedVolumeChanged(self.fixedSelector.currentNode())
    self.onMovingVolumeChanged(self.movingSelector.currentNode())

  #-- Registration methods (delegates the actual work to GreedyReg_CLI) --

  def onRunRegistration(self):
    fixed = self.fixedSelector.currentNode()
    moving = self.movingSelector.currentNode()
    if not fixed or not moving:
      self.statusLabel.setText("Please select fixed and moving volumes!")
      return
    if not self.logic.isGreedyAvailable():
      self.statusLabel.setText("Greedy binary not found - download it above first!")
      return
    if not self.logic.ensureNibabelInstalled():
      self.statusLabel.setText("GreedyReg requires the 'nibabel' package to run registration.")
      return

    self.runButton.setEnabled(False)
    self.statusLabel.setText("Exporting volumes...")
    slicer.app.processEvents()

    patient_id = "CASE0001"
    tmp_dir = tempfile.mkdtemp()
    t1_dir = os.path.join(tmp_dir, "T1"); os.makedirs(t1_dir)
    t2_dir = os.path.join(tmp_dir, "T2"); os.makedirs(t2_dir)
    init_dir = os.path.join(tmp_dir, "INIT"); os.makedirs(init_dir)
    out_dir = os.path.join(tmp_dir, "OUTPUT"); os.makedirs(out_dir)

    fixed_path = os.path.join(t1_dir, f"{patient_id}_t1.nii.gz")
    moving_path = os.path.join(t2_dir, f"{patient_id}_t2.nii.gz")

    # Apply transform to moving volume before export
    moving.SetAndObserveTransformNodeID(self.transformNode.GetID())
    slicer.app.processEvents()
    slicer.util.exportNode(fixed, fixed_path)
    slicer.util.exportNode(moving, moving_path)

    # Save current manual transform as Greedy's initialization. Slicer's
    # ToParent matrix maps moving->fixed; Greedy's init maps fixed->moving
    # (both RAS), so it is the inverse.
    init_path = os.path.join(init_dir, f"{patient_id}_init.mat")
    matrix = vtk.vtkMatrix4x4()
    self.transformNode.GetMatrixTransformFromParent(matrix)
    self.logic.writeInitTransform(init_path, matrix)

    mask_dir = None
    if self.useMaskCheck.checked:
      mask = self.maskSelector.currentNode()
      if mask:
        mask_dir = os.path.join(tmp_dir, "MASK"); os.makedirs(mask_dir)
        mask_path = os.path.join(mask_dir, f"{patient_id}_MASK.nii.gz")
        self.logic.exportMask(mask, mask_path)

    parameters = self.logic.buildGreedyCliParameters(
      t1_dir, t2_dir, out_dir,
      self.metricSelector.currentIndex, self.transformSelector.currentIndex,
      maskFolder=mask_dir, initFolder=init_dir)

    self._regResult = {
      "outputPath": os.path.join(out_dir, f"{patient_id}_registered.nii.gz"),
      "tfmPath": os.path.join(out_dir, f"{patient_id}_transform.tfm"),
      "movingName": moving.GetName(),
    }
    self.statusLabel.setText("Running registration...")
    self._regCliNode = self.logic.runGreedyCli(parameters)
    self._regStartTime = None
    self._regPollTimer = qt.QTimer()
    self._regPollTimer.setInterval(1000)
    self._regPollTimer.connect("timeout()", self.checkRegistrationDone)
    self._regPollTimer.start()

  def checkRegistrationDone(self):
    cli_node = self._regCliNode
    if cli_node.IsBusy():
      if not self._regStartTime:
        self._regStartTime = time.time()
      elapsed = int(time.time() - self._regStartTime)
      self.statusLabel.setText(f"Running registration... {elapsed}s")
      return
    self._regStartTime = None
    self._regPollTimer.stop()
    if cli_node.GetStatus() & cli_node.Cancelled:
      self.statusLabel.setText("Registration cancelled.")
    elif cli_node.GetStatus() & cli_node.ErrorsMask:
      self.statusLabel.setText(f"Registration failed: {cli_node.GetErrorText()}")
      logger.error(cli_node.GetErrorText())
    else:
      slicer.util.loadVolume(self._regResult["outputPath"])
      # The T2-to-T1 transform, as a regular Slicer transform node: apply it to
      # the original moving volume (or anything else in T2 space) in the
      # Transforms module.
      transform_node = slicer.util.loadTransform(self._regResult["tfmPath"])
      transform_node.SetName(slicer.mrmlScene.GenerateUniqueName(
        f"{self._regResult['movingName']}_GreedyTransform"))
      self.statusLabel.setText(f"Registration complete! Transform: {transform_node.GetName()}")
    self.runButton.setEnabled(True)

  def onSaveTransform(self):
    if not hasattr(self, '_regResult') or not self._regResult.get('tfmPath'):
      self.statusLabel.setText("No registration result to save!")
      return
    tfm_path = self._regResult['tfmPath']
    if not os.path.exists(tfm_path):
      self.statusLabel.setText("No transform file found!")
      return
    save_path = qt.QFileDialog.getSaveFileName(
      None, "Save Transform Matrix", f"{self._regResult['movingName']}_transform.tfm",
      "ITK transform files (*.tfm)")
    if save_path:
      if not save_path.lower().endswith(".tfm"):
        save_path += ".tfm"
      import shutil
      shutil.copy(tfm_path, save_path)
      self.statusLabel.setText("Transform saved!")

  def onSaveVolume(self):
    if not hasattr(self, '_regResult') or not self._regResult.get('outputPath'):
      self.statusLabel.setText("No registration result to save!")
      return
    save_path = qt.QFileDialog.getSaveFileName(
      None, "Save Volume", "", "NIfTI files (*.nii.gz);;All files (*)")
    if save_path:
      import shutil
      shutil.copy(self._regResult['outputPath'], save_path)
      self.statusLabel.setText("Volume saved!")

  #-- Greedy Download -------------------------------------------

  def onDownloadGreedy(self):
    """Runs in a separate process (see GreedyRegLogic.startGreedyDownload);
    this only starts it and updates the UI from its callbacks."""
    if self._downloadProcess is not None:
      return
    self.downloadButton.setText("Downloading in background...")
    self.downloadButton.setEnabled(False)
    self.downloadProgressBar.setValue(0)
    self.downloadProgressBar.setVisible(True)
    self.downloadStatusLabel.setStyleSheet("")
    self.downloadStatusLabel.setText("Starting download...")
    try:
      self._downloadProcess = self.logic.startGreedyDownload(
        self.downloadStatusLabel.setText,
        self.downloadProgressBar.setValue,
        self._onGreedyDownloadFinished)
    except Exception as e:
      self._onGreedyDownloadFinished(False, str(e))

  def _onGreedyDownloadFinished(self, success, message):
    self._downloadProcess = None
    self.downloadProgressBar.setVisible(False)
    if success:
      self.greedyWarningBox.setVisible(False)
      self.statusLabel.setText("Greedy downloaded successfully!")
      return
    self.downloadButton.setText("Retry Download")
    self.downloadButton.setEnabled(True)
    self.downloadStatusLabel.setStyleSheet("color: red;")
    self.downloadStatusLabel.setText(f"Download failed: {message}")
    logger.error(f"Greedy download failed: {message}")

  def cleanup(self):
    if self._downloadProcess is not None and self._downloadProcess.poll() is None:
      self._downloadProcess.kill()
    ScriptedLoadableModuleWidget.cleanup(self)

  #-- onCenterVolumes -------------------------------------------

  def onCenterVolumes(self):
    fixed = self.fixedSelector.currentNode()
    moving = self.movingSelector.currentNode()
    if not fixed or not moving:
      self.statusLabel.setText("Please select fixed and moving volumes!")
      return
    tx, ty, tz = self.logic.computeCenteringTranslation(fixed, moving)
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    matrix.SetElement(0, 3, tx)
    matrix.SetElement(1, 3, ty)
    matrix.SetElement(2, 3, tz)
    self.transformNode.SetMatrixTransformToParent(matrix)
    moving.SetAndObserveTransformNodeID(self.transformNode.GetID())
    self.statusLabel.setText(f"Centered! {tx:.1f}, {ty:.1f}, {tz:.1f} mm")

  #-- Distant Registration methods (ALI landmarks via slicer.modules.ali_cbct) --

  def onInstallAliLibraries(self):
    self._distantStatusLabel.setText("Checking ALI libraries, please wait...")
    slicer.app.processEvents()
    try:
      if not self.logic.ensureAliLibrariesInstalled():
        self._distantStatusLabel.setText("Some ALI libraries are still missing - install cancelled or failed.")
        return
      self._aliLibsReady = True
      self._distantLibsWarning.setVisible(False)

      models_dir = self._ensureAliModelsReady(self._distantStatusLabel)
      if not models_dir:
        return
      self._distantStatusLabel.setText("ALI libraries and models are ready!")
    except Exception as e:
      self._distantStatusLabel.setText("Installation failed - check console")
      logger.error(str(e))

  def onBrowseAliModel(self):
    folder = qt.QFileDialog.getExistingDirectory(None, "Select ALI Models Folder")
    if folder:
      self._aliModelEdit.setText(folder)

  def _ensureAliModelsReady(self, status_label, regions=None):
    """Returns the ALI models folder to use (the one typed/browsed into
    _aliModelEdit, or a default under Documents), downloading any missing
    region models into it first. Returns None (after updating statusLabel)
    if the download fails or is declined."""
    models_dir = self._aliModelEdit.text.strip() or self.logic.defaultAliModelsDir()
    if not self.logic.aliModelsReady(models_dir, regions):
      status_label.setText("Downloading ALI landmark models...")
      slicer.app.processEvents()
      def reportStatus(text):
        status_label.setText(text)
        slicer.app.processEvents()
      try:
        self.logic.downloadAliModels(models_dir, regions, statusCallback=reportStatus)
      except Exception as e:
        status_label.setText(f"Failed to download ALI models: {e}")
        logger.error(str(e))
        return None
    self._aliModelEdit.setText(models_dir)
    return models_dir

  def _selectedDistantRegion(self):
    if self._distantCbCheck.checked:
      return "CBMASK"
    if self._distantMandCheck.checked:
      return "MANDMASK"
    if self._distantMaxCheck.checked:
      return "MAXMASK"
    return None

  def onRunDistantRegistration(self):
    fixed = self.fixedSelector.currentNode()
    moving = self.movingSelector.currentNode()
    if not fixed or not moving:
      self._distantStatusLabel.setText("Please select fixed and moving volumes!")
      return
    if not self.logic.ensureAliLibrariesInstalled():
      self._distantStatusLabel.setText("Distant Registration requires the ALI libraries to be installed.")
      return
    self._aliLibsReady = True
    self._distantLibsWarning.setVisible(False)
    region = self._selectedDistantRegion()
    if not region:
      self._distantStatusLabel.setText("Please select a structure!")
      return
    ali_model_dir = self._ensureAliModelsReady(self._distantStatusLabel, regions=[region])
    if not ali_model_dir:
      return

    self._runDistantButton.setEnabled(False)
    self._distantStatusLabel.setText("Exporting volumes...")
    slicer.app.processEvents()

    tmp_dir = tempfile.mkdtemp()
    fixed_path = os.path.join(tmp_dir, "fixed.nii.gz")
    moving_path = os.path.join(tmp_dir, "moving.nii.gz")
    slicer.util.exportNode(fixed, fixed_path)
    slicer.util.exportNode(moving, moving_path)

    self._currentAliRegion = region
    jobs = self.logic.buildAliJobQueue(
      {"fixed": fixed_path, "moving": moving_path}, ali_model_dir, region, tmp_dir)
    self._distantStatusLabel.setText("Running ALI landmark detection...")
    self._startAliJobQueue(jobs, self._onDistantAliAllDone, self._onDistantAliError)

  def _onDistantAliError(self, message):
    self._distantStatusLabel.setText(f"Distant registration failed: {message}")
    self._runDistantButton.setEnabled(True)

  def _onDistantAliAllDone(self, landmarks_acc):
    import numpy as np
    landmarks = self.logic.REGION_CONFIG[self._currentAliRegion]["landmarks"]
    common = [lm for lm in landmarks if lm in landmarks_acc["fixed"] and lm in landmarks_acc["moving"]]
    if len(common) < 3:
      self._onDistantAliError(
        f"only {len(common)} matched landmarks (need >= 3): {common}")
      return
    fixed_pts = np.array([landmarks_acc["fixed"][lm] for lm in common])
    moving_pts = np.array([landmarks_acc["moving"][lm] for lm in common])
    mat4_ras = self.logic.rigidFromLandmarks(fixed_pts, moving_pts)

    vtk_mat = vtk.vtkMatrix4x4()
    for i in range(4):
      for j in range(4):
        vtk_mat.SetElement(i, j, float(mat4_ras[i, j]))
    self.transformNode.SetMatrixTransformToParent(vtk_mat)
    moving = self.movingSelector.currentNode()
    if moving:
      moving.SetAndObserveTransformNodeID(self.transformNode.GetID())
    self._distantStatusLabel.setText(
      "Distant registration complete! Now run Automatic Registration to refine.")
    self._runDistantButton.setEnabled(True)

  #-- ALI job queue: chains slicer.cli.run calls one after another, since
  #  a region can need ALI_CBCT run on more than one model subdirectory
  #  and on both the fixed and moving scan. ----------------------------

  def _startAliJobQueue(self, jobs, on_all_done, on_error):
    self._aliJobs = jobs
    self._aliJobIndex = 0
    self._aliLandmarksAcc = {"fixed": {}, "moving": {}}
    self._aliOnAllDone = on_all_done
    self._aliOnError = on_error
    self._runNextAliJob()

  def _runNextAliJob(self):
    if self._aliJobIndex >= len(self._aliJobs):
      self._aliOnAllDone(self._aliLandmarksAcc)
      return
    job = self._aliJobs[self._aliJobIndex]
    self._aliCurrentJob = job
    cli_node = self.logic.runAliCli(job["parameters"])
    self._aliObserverTag = cli_node.AddObserver("ModifiedEvent", self._onAliJobModified)

  def _onAliJobModified(self, caller, event):
    if caller.IsBusy():
      return
    caller.RemoveObserver(self._aliObserverTag)
    if caller.GetStatus() & caller.Cancelled:
      self._aliOnError("ALI landmark detection cancelled")
      return
    job = self._aliCurrentJob
    job_tag = f"{job['scanKey']}/{job['subdir']}"
    # ALI_CBCT logs missing-model/weight-loading problems as warnings and
    # keeps going rather than failing the CLI, so a "0 landmarks found"
    # result can look identical to success here. Always print what it
    # logged so that case is diagnosable from the Python console.
    output_text = caller.GetOutputText()
    if output_text:
      logger.info(f"ALI_CBCT ({job_tag}) output:\n{output_text}")
    if caller.GetStatus() & caller.ErrorsMask:
      error_text = caller.GetErrorText()
      logger.error(f"ALI_CBCT ({job_tag}) error:\n{error_text}")
      self._aliOnError(error_text or "ALI landmark detection failed")
      return
    found = self.logic.parseAliLandmarksFromOutput(job["outputDir"], job["landmarks"])
    if not found:
      logger.warning(f"ALI_CBCT ({job_tag}) found no landmarks among {job['landmarks']} "
            f"(looked in {job['outputDir']}) - see the output above for why.")
    self._aliLandmarksAcc[job["scanKey"]].update(found)
    self._aliJobIndex += 1
    self._runNextAliJob()

  #-- Batch Processing methods ----------------------------------

  def _browseBatchFolder(self, lineEdit, pairs_label):
    folder = qt.QFileDialog.getExistingDirectory(None, "Select Folder")
    if folder:
      lineEdit.setText(folder)
      if pairs_label:
        pairs_label.setText(f"Selected: {folder}")

  def onRunBatchAuto(self):
    t1_folder = self._batchAutoT1Edit.text.strip()
    t2_folder = self._batchAutoT2Edit.text.strip()
    maskFolder = self._batchAutoMaskEdit.text.strip() or None
    if not t1_folder or not os.path.exists(t1_folder):
      self._batchAutoStatusLabel.setText("Please select a valid T1 folder!")
      return
    if not t2_folder or not os.path.exists(t2_folder):
      self._batchAutoStatusLabel.setText("Please select a valid T2 folder!")
      return
    if not self.logic.isGreedyAvailable():
      self._batchAutoStatusLabel.setText("Greedy binary not found - download it above first!")
      return
    if not self.logic.ensureNibabelInstalled():
      self._batchAutoStatusLabel.setText("GreedyReg requires the 'nibabel' package to run registration.")
      return
    pairs = self.logic.findBatchPairs(t1_folder, t2_folder, maskFolder)
    if not pairs:
      self._batchAutoStatusLabel.setText("No matching pairs found!")
      return

    has_mask = sum(1 for p in pairs if p[3])
    self._batchAutoPairsLabel.setText(
      f"Found {len(pairs)} pair(s), {has_mask} with masks: {', '.join([p[0] for p in pairs])}")
    self._batchAutoPairsLabel.setStyleSheet("color: green; font-size: 10px;")

    self._runBatchAutoButton.setEnabled(False)
    self._batchAutoTotal = len(pairs)
    self._batchAutoStatusLabel.setText(f"Processing {self._batchAutoTotal} pair(s)...")
    self._batchAutoStatusLabel.setStyleSheet("")
    slicer.app.processEvents()

    # GreedyReg_CLI loops over every matched pair itself; the GUI just
    # launches it once and polls until the whole batch is done. Outputs
    # are written back alongside the moving (T2) volumes, like before.
    parameters = self.logic.buildGreedyCliParameters(
      t1_folder, t2_folder, t2_folder,
      self.metricSelector.currentIndex, self.transformSelector.currentIndex,
      maskFolder=maskFolder)
    self._batchAutoCliNode = self.logic.runGreedyCli(parameters)
    self._batchAutoStartTime = None
    self._batchAutoPollTimer = qt.QTimer()
    self._batchAutoPollTimer.setInterval(1000)
    self._batchAutoPollTimer.connect("timeout()", self.checkBatchAutoDone)
    self._batchAutoPollTimer.start()

  def checkBatchAutoDone(self):
    cli_node = self._batchAutoCliNode
    if cli_node.IsBusy():
      if not self._batchAutoStartTime:
        self._batchAutoStartTime = time.time()
      elapsed = int(time.time() - self._batchAutoStartTime)
      self._batchAutoStatusLabel.setText(f"Processing {self._batchAutoTotal} pair(s)... {elapsed}s")
      return
    self._batchAutoStartTime = None
    self._batchAutoPollTimer.stop()
    if cli_node.GetStatus() & cli_node.Cancelled:
      self._batchAutoStatusLabel.setText("Batch cancelled.")
      self._batchAutoStatusLabel.setStyleSheet("color: red;")
    elif cli_node.GetStatus() & cli_node.ErrorsMask:
      self._batchAutoStatusLabel.setText(f"Batch failed: {cli_node.GetErrorText()}")
      self._batchAutoStatusLabel.setStyleSheet("color: red;")
      logger.error(cli_node.GetErrorText())
    else:
      self._batchAutoStatusLabel.setText(
        f"Batch complete! {self._batchAutoTotal} case(s) registered successfully.")
      self._batchAutoStatusLabel.setStyleSheet("color: green;")
    self._runBatchAutoButton.setEnabled(True)

  #-- Batch Distant Registration ---------------------------------------

  def onRunBatchDist(self):
    t1_folder = self._batchDistT1Edit.text.strip()
    t2_folder = self._batchDistT2Edit.text.strip()
    if not t1_folder or not os.path.exists(t1_folder):
      self._batchDistStatusLabel.setText("Please select a valid T1 folder!")
      return
    if not t2_folder or not os.path.exists(t2_folder):
      self._batchDistStatusLabel.setText("Please select a valid T2 folder!")
      return
    if not self.logic.ensureNibabelInstalled():
      self._batchDistStatusLabel.setText("GreedyReg requires the 'nibabel' package to run batch distant registration.")
      return
    if not self.logic.ensureAliLibrariesInstalled():
      self._batchDistStatusLabel.setText("Batch Distant Registration requires the ALI libraries to be installed.")
      return
    self._aliLibsReady = True
    self._distantLibsWarning.setVisible(False)
    pairs = self.logic.findBatchPairsDistant(t1_folder, t2_folder)
    if not pairs:
      self._batchDistStatusLabel.setText("No matching pairs found!")
      return
    region = self._selectedDistantRegion()
    if not region:
      self._batchDistStatusLabel.setText("Please select a structure!")
      return
    ali_model_dir = self._ensureAliModelsReady(self._batchDistStatusLabel, regions=[region])
    if not ali_model_dir:
      return

    self._runBatchDistButton.setEnabled(False)
    self._batchDistPairs       = pairs
    self._batchDistIndex       = 0
    self._batchDistTotal       = len(pairs)
    self._batchDistAliModelDir = ali_model_dir
    self._batchDistRegion      = region
    self._batchDistPairsLabel.setText(
      f"Found {len(pairs)} pair(s): {', '.join([p[0] for p in pairs])}")
    self._batchDistPairsLabel.setStyleSheet("color: green; font-size: 10px;")
    self._runNextBatchDistCase()

  def _runNextBatchDistCase(self):
    if self._batchDistIndex >= self._batchDistTotal:
      self._batchDistStatusLabel.setText(
        f"Batch complete! {self._batchDistTotal} cases aligned successfully.")
      self._batchDistStatusLabel.setStyleSheet("color: green;")
      self._runBatchDistButton.setEnabled(True)
      return
    patient_id, fixed_path, moving_path = self._batchDistPairs[self._batchDistIndex]
    self._batchDistStatusLabel.setText(
      f"Processing {patient_id} ({self._batchDistIndex + 1} of {self._batchDistTotal})...")
    self._batchDistStatusLabel.setStyleSheet("")
    slicer.app.processEvents()

    self._batchDistCurrentCase = {"patientId": patient_id, "movingPath": moving_path}
    tmp_dir = tempfile.mkdtemp()
    jobs = self.logic.buildAliJobQueue(
      {"fixed": fixed_path, "moving": moving_path},
      self._batchDistAliModelDir, self._batchDistRegion, tmp_dir)
    self._startAliJobQueue(jobs, self._onBatchDistCaseAliDone, self._onBatchDistCaseAliError)

  def _onBatchDistCaseAliError(self, message):
    patient_id = self._batchDistCurrentCase["patientId"]
    self._batchDistStatusLabel.setText(f"Failed on {patient_id}: {message}\nBatch halted.")
    self._batchDistStatusLabel.setStyleSheet("color: red;")
    self._runBatchDistButton.setEnabled(True)
    logger.error(f"Batch Dist FAILED on {patient_id}: {message}")

  def _onBatchDistCaseAliDone(self, landmarks_acc):
    import numpy as np
    import nibabel as nib

    region = self._batchDistRegion
    landmarks = self.logic.REGION_CONFIG[region]["landmarks"]
    common = [lm for lm in landmarks if lm in landmarks_acc["fixed"] and lm in landmarks_acc["moving"]]
    patient_id = self._batchDistCurrentCase["patientId"]
    if len(common) < 3:
      self._onBatchDistCaseAliError(f"only {len(common)} matched landmarks (need >= 3)")
      return

    fixed_pts = np.array([landmarks_acc["fixed"][lm] for lm in common])
    moving_pts = np.array([landmarks_acc["moving"][lm] for lm in common])
    mat4_ras = self.logic.rigidFromLandmarks(fixed_pts, moving_pts)

    moving_path = self._batchDistCurrentCase["movingPath"]
    moving_img = nib.load(moving_path)
    R = mat4_ras[:3, :3]
    t = mat4_ras[:3, 3]
    new_affine = moving_img.affine.copy()
    new_affine[:3, :3] = R @ moving_img.affine[:3, :3]
    new_affine[:3, 3] = R @ moving_img.affine[:3, 3] + t
    aligned_img = nib.Nifti1Image(moving_img.get_fdata(), new_affine, moving_img.header)
    t2_folder = os.path.dirname(moving_path)
    output_path = os.path.join(t2_folder, f"{patient_id}_t2_aligned.nii.gz")
    nib.save(aligned_img, output_path)
    logger.info(f"Batch Dist {patient_id} done -> {output_path}")

    self._batchDistIndex += 1
    self._runNextBatchDistCase()
