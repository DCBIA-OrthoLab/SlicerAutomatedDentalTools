import os
import math
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


class StandaloneRotationWheel(qt.QFrame):
  """Standalone angle-based rotation/translation wheel.

  This is still just a Qt overlay on top of a 2D slice view, but unlike the old
  drag pad it behaves like a real wheel: click/drag around the ring and the
  change in polar angle is emitted as degrees. It also emits raw pixel-drag
  deltas so the owner can use the same centered, transparent, zoom-scaling
  control for X/Y/Z translation. Opacity remains fully controlled by QSS.
  """

  BASE_SIZE = 220
  MIN_SIZE = 130
  MAX_SIZE = 420
  RING_THICKNESS = 6
  # The visible ring is thin, but the rotation hit band needs to be a little
  # wider so it is usable. Everything not in this band behaves like the
  # background drag translator, even if it is inside the wheel circle.
  ACTIVE_RING_WIDTH = 22
  INNER_DEAD_ZONE_FRACTION = 0.18

  def __init__(self, parent=None):
    qt.QFrame.__init__(self, parent)
    self._size = self.BASE_SIZE
    self.setFixedSize(self._size, self._size)
    self.setAttribute(qt.Qt.WA_StyledBackground, True)
    self.setCursor(qt.Qt.OpenHandCursor)
    self.onRotate = None      # callback(deltaAngleDegrees)
    self.onTranslate = None   # callback(dxPixels, dyPixels)
    self._dragging = False
    self._interactionMode = None  # "rotate" for ring drag, "translate" elsewhere
    self._lastAngle = None
    self._lastPos = None
    self._opacity = 1.0

    self.hub = qt.QFrame(self)
    self.hub.setFixedSize(26, 26)
    self.hub.move(self._size // 2 - 13, self._size // 2 - 13)
    self.hub.setAttribute(qt.Qt.WA_StyledBackground, True)
    self.hub.setStyleSheet("background-color: rgba(120, 144, 156, 230); border-radius: 13px; border: 1px solid rgba(255,255,255,180);")
    self.hub.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)

    self.indicator = qt.QFrame(self)
    self.indicator.setFixedSize(8, 8)
    self.indicator.setAttribute(qt.Qt.WA_StyledBackground, True)
    self.indicator.setStyleSheet("background-color: #FF7043; border-radius: 4px; border: 1px solid white;")
    self.indicator.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)
    self._placeIndicatorAtAngle(0.0)

    self.label = qt.QLabel("ROTATE", self)
    self.label.setAlignment(qt.Qt.AlignCenter)
    self.label.setFixedSize(70, 18)
    self.label.move(self._size // 2 - 35, self._size // 2 + 18)
    self.label.setStyleSheet("color: white; font-weight: bold; font-size: 9px; background: transparent;")
    self.label.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)

    self._applyStyle()

  def _applyStyle(self):
    fill_alpha = int(45 * self._opacity)
    border_alpha = int(255 * self._opacity)
    inner_alpha = int(130 * self._opacity)
    self.setStyleSheet(
      "QFrame { background-color: rgba(38, 50, 56, %d); "
      "border: %dpx solid rgba(0, 188, 212, %d); "
      "border-radius: %dpx; }" % (fill_alpha, self.RING_THICKNESS, border_alpha, self._size // 2))
    hub_radius = max(1, self.hub.width // 2)
    self.hub.setStyleSheet(
      "background-color: rgba(120, 144, 156, %d); border-radius: %dpx; "
      "border: 1px solid rgba(255,255,255,%d);" % (inner_alpha, hub_radius, border_alpha))

  def setOpacityFraction(self, fraction):
    self._opacity = max(0.05, min(1.0, fraction))
    self._applyStyle()

  def setLabelText(self, text):
    self.label.setText(text)

  def _center(self):
    return self._size / 2.0, self._size / 2.0

  def _angleFromPos(self, pos):
    cx, cy = self._center()
    return math.atan2(pos.y() - cy, pos.x() - cx)

  def _distanceFromCenter(self, pos):
    cx, cy = self._center()
    return math.sqrt((pos.x() - cx) ** 2 + (pos.y() - cy) ** 2)

  def _isRotationBand(self, pos):
    """Return True only for the outer ring band.

    The whole widget is a square Qt child, so relying on normal event routing
    would make the entire circle block slice translation. Instead, only this
    explicit band is rotation-active; all other clicks inside the widget emit
    translation deltas just like the full-slice overlay.
    """
    r = self._distanceFromCenter(pos)
    outer_radius = self._size / 2.0
    inner_radius = outer_radius - max(self.ACTIVE_RING_WIDTH, self.RING_THICKNESS)
    return inner_radius <= r <= outer_radius

  def setWheelSize(self, size):
    size = int(max(self.MIN_SIZE, min(self.MAX_SIZE, size)))
    if size == self._size:
      return
    self._size = size
    self.setFixedSize(self._size, self._size)

    hub_size = max(18, int(self._size * 0.12))
    if hub_size % 2:
      hub_size += 1
    self.hub.setFixedSize(hub_size, hub_size)
    self.hub.move(self._size // 2 - hub_size // 2, self._size // 2 - hub_size // 2)

    # Keep the orange grab/position marker subtle. It should show the wheel
    # position without covering anatomy inside the slice view.
    indicator_size = max(6, int(self._size * 0.035))
    if indicator_size % 2:
      indicator_size += 1
    self.indicator.setFixedSize(indicator_size, indicator_size)
    self.indicator.setStyleSheet(
      "background-color: #FF7043; border-radius: %dpx; border: 1px solid white;"
      % max(1, indicator_size // 2))

    label_w = max(52, int(self._size * 0.32))
    self.label.setFixedSize(label_w, 18)
    self.label.move(self._size // 2 - label_w // 2, self._size // 2 + int(self._size * 0.08))

    self._applyStyle()
    # Reposition the marker after every resize, even before the first drag.
    self._placeIndicatorAtAngle(self._lastAngle if self._lastAngle is not None else 0.0)

  def _normalizedDeltaDegrees(self, old_angle, newAngle):
    delta = math.degrees(newAngle - old_angle)
    while delta > 180.0:
      delta -= 360.0
    while delta < -180.0:
      delta += 360.0
    return delta

  def _placeIndicatorAtAngle(self, angle_radians):
    cx, cy = self._center()
    radius = self._size / 2.0 - self.RING_THICKNESS - max(10, self.indicator.width / 2.0)
    half = self.indicator.width / 2.0
    x = cx + radius * math.cos(angle_radians) - half
    y = cy + radius * math.sin(angle_radians) - half
    self.indicator.move(int(x), int(y))

  def mousePressEvent(self, event):
    self._dragging = True
    self._lastPos = event.pos()

    if self._isRotationBand(event.pos()):
      self._interactionMode = "rotate"
      self._lastAngle = self._angleFromPos(event.pos())
      self._placeIndicatorAtAngle(self._lastAngle)
      self.setCursor(qt.Qt.ClosedHandCursor)
    else:
      # Any click that is not on the actual ring band should translate, even
      # if it is inside the wheel circle/hub. This makes the wheel coexist
      # with click-anywhere slice dragging instead of blocking it.
      self._interactionMode = "translate"
      self._lastAngle = None
      self.setCursor(qt.Qt.SizeAllCursor)

  def mouseMoveEvent(self, event):
    if not self._dragging or self._lastPos is None:
      return

    dx_pixels = event.pos().x() - self._lastPos.x()
    dy_pixels = event.pos().y() - self._lastPos.y()
    self._lastPos = event.pos()

    if self._interactionMode == "rotate":
      new_angle = self._angleFromPos(event.pos())
      delta_degrees = self._normalizedDeltaDegrees(self._lastAngle, new_angle)
      self._lastAngle = new_angle
      self._placeIndicatorAtAngle(new_angle)
      if self.onRotate and abs(delta_degrees) > 1e-6:
        self.onRotate(delta_degrees)
    else:
      if self.onTranslate and (dx_pixels or dy_pixels):
        self.onTranslate(dx_pixels, dy_pixels)

  def mouseReleaseEvent(self, event):
    self._dragging = False
    self._interactionMode = None
    self._lastAngle = None
    self._lastPos = None
    self.setCursor(qt.Qt.OpenHandCursor)


class SliceTranslationOverlay(qt.QFrame):
  """Transparent full-slice mouse layer for click-anywhere translation.

  The centered rotation wheel is a child of this overlay. Mouse events on the
  wheel still rotate; mouse drags anywhere else on the slice view translate the
  moving volume. The overlay is intentionally used only in 2D slice views and is
  hidden whenever the demo panel is collapsed.
  """

  def __init__(self, parent=None):
    qt.QFrame.__init__(self, parent)
    self.setAttribute(qt.Qt.WA_StyledBackground, True)
    self.setStyleSheet("QFrame { background-color: rgba(0,0,0,0); border: none; }")
    self.setMouseTracking(True)
    self.setCursor(qt.Qt.SizeAllCursor)
    self.onTranslate = None  # callback(dxPixels, dyPixels)
    self._dragging = False
    self._lastPos = None

  def resizeToParent(self):
    parent = self.parent()
    if parent is None:
      return
    try:
      w, h = parent.width, parent.height
    except Exception:
      try:
        w, h = parent.width(), parent.height()
      except Exception:
        return
    self.setGeometry(0, 0, int(w), int(h))

  def mousePressEvent(self, event):
    self._dragging = True
    self._lastPos = event.pos()
    self.setCursor(qt.Qt.ClosedHandCursor)

  def mouseMoveEvent(self, event):
    if not self._dragging or self._lastPos is None:
      return
    dx = event.pos().x() - self._lastPos.x()
    dy = event.pos().y() - self._lastPos.y()
    self._lastPos = event.pos()
    if self.onTranslate and (dx or dy):
      self.onTranslate(dx, dy)

  def mouseReleaseEvent(self, event):
    self._dragging = False
    self._lastPos = None
    self.setCursor(qt.Qt.SizeAllCursor)


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
      warning_label = qt.QLabel("Greedy registration engine not found.\nClick below to download it automatically (~60MB).")
      warning_label.setStyleSheet("color: red;")
      warning_layout.addRow(warning_label)
      self.downloadButton = qt.QPushButton("Download Greedy")
      self.downloadButton.setStyleSheet(
        "QPushButton { background-color: #F44336; color: white; "
        "font-weight: bold; padding: 8px; border-radius: 4px; }")
      self.downloadButton.clicked.connect(self.onDownloadGreedy)
      warning_layout.addRow(self.downloadButton)

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
    manual_box = ctk.ctkCollapsibleButton()
    manual_box.text = "Manual Alignment"
    self.layout.addWidget(manual_box)
    manual_layout = qt.QFormLayout(manual_box)

    self.rotX = ctk.ctkSliderWidget()
    self.rotX.minimum, self.rotX.maximum = -180, 180
    self.rotX.value = 0
    manual_layout.addRow("Rotation X:", self.rotX)

    self.rotY = ctk.ctkSliderWidget()
    self.rotY.minimum, self.rotY.maximum = -180, 180
    self.rotY.value = 0
    manual_layout.addRow("Rotation Y:", self.rotY)

    self.rotZ = ctk.ctkSliderWidget()
    self.rotZ.minimum, self.rotZ.maximum = -180, 180
    self.rotZ.value = 0
    manual_layout.addRow("Rotation Z:", self.rotZ)

    self.tranX = ctk.ctkSliderWidget()
    self.tranX.minimum, self.tranX.maximum = -200, 200
    self.tranX.value = 0
    manual_layout.addRow("Translation X (mm):", self.tranX)

    self.tranY = ctk.ctkSliderWidget()
    self.tranY.minimum, self.tranY.maximum = -200, 200
    self.tranY.value = 0
    manual_layout.addRow("Translation Y (mm):", self.tranY)

    self.tranZ = ctk.ctkSliderWidget()
    self.tranZ.minimum, self.tranZ.maximum = -200, 200
    self.tranZ.value = 0
    manual_layout.addRow("Translation Z (mm):", self.tranZ)

    self.rotX.valueChanged.connect(self.onManualTransformChanged)
    self.rotY.valueChanged.connect(self.onManualTransformChanged)
    self.rotZ.valueChanged.connect(self.onManualTransformChanged)
    self.tranX.valueChanged.connect(self.onManualTransformChanged)
    self.tranY.valueChanged.connect(self.onManualTransformChanged)
    self.tranZ.valueChanged.connect(self.onManualTransformChanged)

    self.resetButton = qt.QPushButton("Reset Transform")
    self.resetButton.clicked.connect(self.onResetTransform)
    manual_layout.addRow(self.resetButton)

    manual_tools_box = ctk.ctkCollapsibleButton()
    manual_tools_box.text = "Manual Approximation Tools"
    manual_tools_box.collapsed = True
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

    #-- Sensitivity Demo (standalone, not tied to Slicer Transforms) -
    # Prototype for evaluating whether variable-sensitivity dragging is
    # worth building into Slicer's own transform interaction widget:
    # Slicer's built-in handles recompute their own delta from raw mouse
    # movement on every event, so damping the matrix after the widget
    # reports it has no visible effect (confirmed empirically). This pad
    # sidesteps that entirely by owning the full mouse-to-millimeter
    # mapping itself. It's a plain QSS-styled QFrame (no custom QPainter
    # code - that never rendered reliably in this environment) reparented
    # directly onto a 2D slice view as a normal child widget - shown while
    # this section is expanded.
    demo_box = ctk.ctkCollapsibleButton()
    demo_box.text = "Sensitivity Demo (Standalone Prototype)"
    demo_box.collapsed = True
    self.layout.addWidget(demo_box)
    demo_layout = qt.QFormLayout(demo_box)

    demo_explanation = qt.QLabel(
      "Prototype: expanding this section overlays centered standalone rotation wheels on all 2D slices.\n"
      "Rotation: drag around the thin centered wheel. Translation: click/drag anywhere else in the slice view.\n"
      "Both controls auto-scale with slice zoom and keep transparency/sensitivity controls.")
    demo_explanation.setStyleSheet("color: gray; font-size: 10px;")
    demo_explanation.setWordWrap(True)
    demo_layout.addRow(demo_explanation)

    self.demoSensitivitySlider = ctk.ctkSliderWidget()
    # Much finer control than the first demo. Values are still shown as percent,
    # but the range is intentionally tiny: 5% means a 10° mouse sweep applies
    # only 0.5° to the transform; 0.1% is very fine adjustment.
    self.demoSensitivitySlider.minimum, self.demoSensitivitySlider.maximum = 0.1, 300
    self.demoSensitivitySlider.value = 5
    self.demoSensitivitySlider.singleStep = 0.1
    self.demoSensitivitySlider.pageStep = 1.0
    self.demoSensitivitySlider.setToolTip("Gain for both controls. Wheel rotation uses the slice normal. Background dragging translates within that slice plane.")
    demo_layout.addRow("Control Sensitivity (%):", self.demoSensitivitySlider)

    self.demoAllSlicesLabel = qt.QLabel("Overlay appears on Red, Yellow, and Green slice views at the same time. 3D view is not used.")
    self.demoAllSlicesLabel.setStyleSheet("color: gray; font-size: 10px;")
    self.demoAllSlicesLabel.setWordWrap(True)
    demo_layout.addRow(self.demoAllSlicesLabel)

    self.demoOpacitySlider = ctk.ctkSliderWidget()
    self.demoOpacitySlider.minimum, self.demoOpacitySlider.maximum = 10, 100
    self.demoOpacitySlider.value = 100
    self.demoOpacitySlider.setToolTip("Transparency of all wheels overlaid on the 2D slice views")
    self.demoOpacitySlider.valueChanged.connect(self.onSensitivityDemoOpacityChanged)
    demo_layout.addRow("Wheel Opacity (%):", self.demoOpacitySlider)

    self.demoStatusLabel = qt.QLabel("Wheel transform: rotation X=0.0°, Y=0.0°, Z=0.0° | translation X=0.0mm, Y=0.0mm, Z=0.0mm")
    self.demoStatusLabel.setAlignment(qt.Qt.AlignCenter)
    demo_layout.addRow(self.demoStatusLabel)

    self.demoResetButton = qt.QPushButton("Reset Demo Transform")
    self.demoResetButton.clicked.connect(self.onResetSensitivityDemo)
    demo_layout.addRow(self.demoResetButton)

    # The wheels themselves are NOT added to this panel - they are reparented
    # onto the Red/Yellow/Green slice views so they visually appear over all
    # 2D slices at once, while leaving the 3D view completely untouched.
    self.demoDragPads = {}
    self.demoTranslationOverlays = {}
    self.demoDragPad = None  # kept only so old scene-reset code does not break
    self._setupSensitivityDemoOverlay()
    self._demoZoomTimer = qt.QTimer()
    self._demoZoomTimer.setInterval(200)
    self._demoZoomTimer.connect("timeout()", self._updateSensitivityDemoWheelScales)
    demo_box.toggled.connect(self.onSensitivityDemoBoxToggled)

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

    self._distantTransformSelector = qt.QComboBox()
    self._distantTransformSelector.addItems(["Rigid", "Affine"])
    distant_layout.addRow("Transform:", self._distantTransformSelector)

    self._runDistantButton = qt.QPushButton("Run Distant Registration")
    self._runDistantButton.setStyleSheet(
      "QPushButton { background-color: #9C27B0; color: white; "
      "font-weight: bold; padding: 6px; border-radius: 4px; }")
    self._runDistantButton.clicked.connect(self.onRunDistantRegistration)
    distant_layout.addRow(self._runDistantButton)

    self._distantStatusLabel = qt.QLabel("")
    self._distantStatusLabel.setAlignment(qt.Qt.AlignCenter)
    distant_layout.addRow(self._distantStatusLabel)

    self._saveDistantVolumeButton = qt.QPushButton("Save Registered Volume")
    self._saveDistantVolumeButton.clicked.connect(self.onSaveDistantVolume)
    distant_layout.addRow(self._saveDistantVolumeButton)

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
    self.rotX.value = 0
    self.rotY.value = 0
    self.rotZ.value = 0
    self.tranX.value = 0
    self.tranY.value = 0
    self.tranZ.value = 0

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
    # Reset sliders
    for slider in [self.rotX, self.rotY, self.rotZ,
                   self.tranX, self.tranY, self.tranZ]:
        slider.blockSignals(True)
        slider.value = 0
        slider.blockSignals(False)
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

  #-- Sensitivity demo (standalone wheel, not tied to Slicer Transforms) --

  def _setupSensitivityDemoOverlay(self):
    """Create one standalone Qt wheel on each 2D slice view.

    This keeps the wheel implementation independent from Slicer's native
    transform handles, but places the same control over Red, Yellow, and Green
    simultaneously. The 3D view is intentionally not used.
    """
    self._demoOverlayExpanded = False
    self._demoSliceViews = {}
    self._demoBaseFov = {}
    self._attachSensitivityDemoToAllSlices()

  def _sliceNamesForSensitivityDemo(self):
    return ["Red", "Yellow", "Green"]

  def _attachSensitivityDemoToAllSlices(self):
    if not hasattr(self, 'demoDragPads'):
      self.demoDragPads = {}
    if not hasattr(self, 'demoTranslationOverlays'):
      self.demoTranslationOverlays = {}

    try:
      lm = slicer.app.layoutManager()
    except Exception:
      lm = None

    for slice_name in self._sliceNamesForSensitivityDemo():
      slice_view = None
      if lm is not None:
        try:
          slice_widget = lm.sliceWidget(slice_name)
          slice_view = slice_widget.sliceView() if slice_widget else None
        except Exception:
          slice_view = None
      self._demoSliceViews[slice_name] = slice_view

      if slice_name not in self.demoTranslationOverlays:
        overlay = SliceTranslationOverlay()
        overlay.onTranslate = (lambda dx, dy, sn=slice_name: self.onSensitivityDemoTranslate(dx, dy, sn))
        self.demoTranslationOverlays[slice_name] = overlay
      else:
        overlay = self.demoTranslationOverlays[slice_name]

      if slice_name not in self.demoDragPads:
        pad = StandaloneRotationWheel()
        pad.onRotate = (lambda degrees, sn=slice_name: self.onSensitivityDemoRotate(degrees, sn))
        pad.onTranslate = (lambda dx, dy, sn=slice_name: self.onSensitivityDemoTranslate(dx, dy, sn))
        pad.setOpacityFraction(self.demoOpacitySlider.value / 100.0 if hasattr(self, 'demoOpacitySlider') else 1.0)
        pad.setLabelText("ROTATE")
        self.demoDragPads[slice_name] = pad
      else:
        pad = self.demoDragPads[slice_name]
        pad.onRotate = (lambda degrees, sn=slice_name: self.onSensitivityDemoRotate(degrees, sn))
        pad.onTranslate = (lambda dx, dy, sn=slice_name: self.onSensitivityDemoTranslate(dx, dy, sn))

      overlay.hide()
      pad.hide()
      if slice_view is not None:
        overlay.setParent(slice_view)
        overlay.resizeToParent()
        pad.setParent(overlay)
        self.logic._centerSensitivityDemoWheel(pad)
        if getattr(self, '_demoOverlayExpanded', False):
          overlay.raise_()
          overlay.show()
          pad.raise_()
          pad.show()

    self._updateSensitivityDemoWheelScales()

    # Keep a representative reference for compatibility with old code paths.
    self.demoDragPad = self.demoDragPads.get("Red")


  def onSensitivityDemoBoxToggled(self, expanded):
    self._demoOverlayExpanded = expanded
    self._attachSensitivityDemoToAllSlices()
    if hasattr(self, 'demoTranslationOverlays'):
      for overlay in self.demoTranslationOverlays.values():
        if expanded and overlay.parent() is not None:
          overlay.resizeToParent()
          overlay.raise_()
          overlay.show()
        else:
          overlay.hide()
    for pad in self.demoDragPads.values():
      if expanded and pad.parent() is not None:
        pad.raise_()
        pad.show()
      else:
        pad.hide()
    if hasattr(self, '_demoZoomTimer'):
      if expanded:
        self._updateSensitivityDemoWheelScales()
        self._demoZoomTimer.start()
      else:
        self._demoZoomTimer.stop()

  def _updateSensitivityDemoWheelScales(self):
    if not hasattr(self, 'demoDragPads'):
      return
    for slice_name, pad in self.demoDragPads.items():
      fov_mean = self.logic._currentSliceFovMean(slice_name)
      if fov_mean is None:
        continue
      if slice_name not in self._demoBaseFov or self._demoBaseFov[slice_name] <= 0:
        self._demoBaseFov[slice_name] = fov_mean
      base_fov = self._demoBaseFov[slice_name]
      # Smaller FOV means the clinician zoomed in, so the wheel grows; larger
      # FOV means zoomed out, so the wheel shrinks. Clamp it so it never becomes
      # ridiculous or unusably tiny.
      overlay = self.demoTranslationOverlays.get(slice_name) if hasattr(self, 'demoTranslationOverlays') else None
      if overlay is not None:
        overlay.resizeToParent()
      zoom_factor = max(0.55, min(1.90, base_fov / fov_mean))
      pad.setWheelSize(StandaloneRotationWheel.BASE_SIZE * zoom_factor)
      self.logic._centerSensitivityDemoWheel(pad)

  def onSensitivityDemoOpacityChanged(self, value):
    if hasattr(self, 'demoDragPads'):
      for pad in self.demoDragPads.values():
        pad.setOpacityFraction(value / 100.0)

  def onSensitivityDemoRotate(self, delta_angle_degrees, sliceName="Red"):
    """Apply an angle-based wheel rotation around the current slice normal."""
    moving = self.movingSelector.currentNode()
    if not moving:
      self.demoStatusLabel.setText("Select a moving volume first!")
      return
    moving.SetAndObserveTransformNodeID(self.transformNode.GetID())

    scale = self.demoSensitivitySlider.value / 100.0
    applied_degrees = delta_angle_degrees * scale
    axis_name = self.logic._rotationAxisForSlice(sliceName)

    current_matrix = vtk.vtkMatrix4x4()
    self.transformNode.GetMatrixTransformToParent(current_matrix)

    incremental_transform = vtk.vtkTransform()
    incremental_transform.PostMultiply()
    incremental_transform.SetMatrix(current_matrix)

    if axis_name == "X":
      incremental_transform.RotateX(applied_degrees)
    elif axis_name == "Y":
      incremental_transform.RotateY(applied_degrees)
    else:
      incremental_transform.RotateZ(applied_degrees)

    new_matrix = vtk.vtkMatrix4x4()
    incremental_transform.GetMatrix(new_matrix)
    self.transformNode.SetMatrixTransformToParent(new_matrix)

    if not hasattr(self, '_demoAccumulatedDegrees'):
      self._demoAccumulatedDegrees = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    self._demoAccumulatedDegrees[axis_name] += applied_degrees
    self._updateSensitivityDemoStatus()

  def onSensitivityDemoTranslate(self, dxPixels, dyPixels, sliceName="Red"):
    """Translate by dragging anywhere outside the wheel in a 2D slice view.

    Translation is independent from the wheel and always happens in the plane
    of the slice that was dragged. This gives X/Y/Z translation without any
    extra axis buttons: Red=X/Y, Yellow=Y/Z, Green=X/Z.
    """
    moving = self.movingSelector.currentNode()
    if not moving:
      self.demoStatusLabel.setText("Select a moving volume first!")
      return
    moving.SetAndObserveTransformNodeID(self.transformNode.GetID())

    scale = self.demoSensitivitySlider.value / 100.0
    mm_per_pixel_at_full_sensitivity = 0.5
    mm_per_pixel = mm_per_pixel_at_full_sensitivity * scale
    deltas = self.logic._translationDeltasForSlice(sliceName, dxPixels, dyPixels, mm_per_pixel)
    if all(abs(v) < 1e-9 for v in deltas.values()):
      return

    matrix = vtk.vtkMatrix4x4()
    self.transformNode.GetMatrixTransformToParent(matrix)
    for idx, axis_name in enumerate(["X", "Y", "Z"]):
      matrix.SetElement(idx, 3, matrix.GetElement(idx, 3) + deltas[axis_name])
    self.transformNode.SetMatrixTransformToParent(matrix)

    if not hasattr(self, '_demoAccumulatedMm'):
      self._demoAccumulatedMm = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    for axis_name, delta_mm in deltas.items():
      self._demoAccumulatedMm[axis_name] += delta_mm
    self._updateSensitivityDemoStatus()

  def _updateSensitivityDemoStatus(self):
    if not hasattr(self, '_demoAccumulatedDegrees'):
      self._demoAccumulatedDegrees = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    if not hasattr(self, '_demoAccumulatedMm'):
      self._demoAccumulatedMm = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    self.demoStatusLabel.setText(
      f"Wheel transform: rotation X={self._demoAccumulatedDegrees['X']:.1f}°, "
      f"Y={self._demoAccumulatedDegrees['Y']:.1f}°, "
      f"Z={self._demoAccumulatedDegrees['Z']:.1f}° | "
      f"translation X={self._demoAccumulatedMm['X']:.2f}mm, "
      f"Y={self._demoAccumulatedMm['Y']:.2f}mm, "
      f"Z={self._demoAccumulatedMm['Z']:.2f}mm")

  def onResetSensitivityDemo(self):
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    self.transformNode.SetMatrixTransformToParent(matrix)
    self._demoAccumulatedDegrees = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    self._demoAccumulatedMm = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    self._updateSensitivityDemoStatus()

  def onSceneCleared(self, caller, event):
    # Recreate transform node
    self.transformNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLinearTransformNode", "GreedyManualTransform")
    self.transformDisplayWidget.setMRMLTransformNode(self.transformNode)
    self.transformDisplayWidget.setVisible(False)

    # Recreate segment editor node (also scene-owned)
    self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)

    # Reset sliders without triggering callbacks
    for slider in [self.rotX, self.rotY, self.rotZ,
                   self.tranX, self.tranY, self.tranZ]:
        slider.blockSignals(True)
        slider.value = 0
        slider.blockSignals(False)

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
    if hasattr(self, '_demoBaseFov'):
      self._demoBaseFov = {}
    self.onResetSensitivityDemo()
    if hasattr(self, 'demoTranslationOverlays'):
      for overlay in self.demoTranslationOverlays.values():
        overlay.hide()

    # Stop any running tools
    self.stopInteractiveTool()

    # Reattach volumes if already selected
    self.onFixedVolumeChanged(self.fixedSelector.currentNode())
    self.onMovingVolumeChanged(self.movingSelector.currentNode())

  def onManualTransformChanged(self):
    moving = self.movingSelector.currentNode()
    if moving:
      moving.SetAndObserveTransformNodeID(self.transformNode.GetID())
    model = self.modelSelector.currentNode()
    if model:
      model.SetAndObserveTransformNodeID(self.transformNode.GetID())
    transform = vtk.vtkTransform()
    transform.Identity()
    transform.Translate(self.tranX.value, self.tranY.value, self.tranZ.value)
    transform.RotateX(self.rotX.value)
    transform.RotateY(self.rotY.value)
    transform.RotateZ(self.rotZ.value)
    matrix = vtk.vtkMatrix4x4()
    transform.GetMatrix(matrix)
    self.transformNode.SetMatrixTransformToParent(matrix)

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

    # Save current manual transform as Greedy's initialization
    init_path = os.path.join(init_dir, f"{patient_id}_init.mat")
    matrix = vtk.vtkMatrix4x4()
    self.transformNode.GetMatrixTransformToParent(matrix)
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
      "warpPath": os.path.join(out_dir, f"{patient_id}_warp.mat"),
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
    if not (cli_node.GetStatus() & cli_node.Completed):
      if not self._regStartTime:
        self._regStartTime = time.time()
      elapsed = int(time.time() - self._regStartTime)
      self.statusLabel.setText(f"Running registration... {elapsed}s")
      return
    self._regStartTime = None
    self._regPollTimer.stop()
    if cli_node.GetStatus() & cli_node.ErrorsMask:
      self.statusLabel.setText(f"Registration failed: {cli_node.GetErrorText()}")
      logger.error(cli_node.GetErrorText())
    else:
      slicer.util.loadVolume(self._regResult["outputPath"])
      self.statusLabel.setText("Registration complete!")
    self.runButton.setEnabled(True)

  def onSaveTransform(self):
    if not hasattr(self, '_regResult') or not self._regResult.get('warpPath'):
      self.statusLabel.setText("No registration result to save!")
      return
    warp_path = self._regResult['warpPath']
    if not os.path.exists(warp_path):
      self.statusLabel.setText("No transform file found!")
      return
    save_path = qt.QFileDialog.getSaveFileName(
      None, "Save Transform Matrix", "", "Matrix files (*.mat);;All files (*)")
    if save_path:
      import shutil
      shutil.copy(warp_path, save_path)
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
    self.downloadButton.setText("Downloading... please wait")
    self.downloadButton.setEnabled(False)
    slicer.app.processEvents()
    try:
      def reportStatus(text):
        self.statusLabel.setText(text)
        slicer.app.processEvents()
      self.logic.downloadGreedyBinary(statusCallback=reportStatus)
      self.greedyWarningBox.setVisible(False)
      self.statusLabel.setText("Greedy downloaded successfully!")
    except Exception as e:
      self.downloadButton.setText("Download Greedy")
      self.downloadButton.setEnabled(True)
      self.statusLabel.setText(f"Download failed: {str(e)}")
      logger.error(str(e))

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

  def onSaveDistantVolume(self):
    if not hasattr(self, '_distantResult') or not self._distantResult.get('outputPath'):
      self._distantStatusLabel.setText("No result to save!")
      return
    save_path = qt.QFileDialog.getSaveFileName(
      None, "Save Registered Volume", "", "NIfTI files (*.nii.gz);;All files (*)")
    if save_path:
      import shutil
      shutil.copy(self._distantResult['outputPath'], save_path)
      self._distantStatusLabel.setText("Volume saved!")

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
    if not (caller.GetStatus() & caller.Completed):
      return
    caller.RemoveObserver(self._aliObserverTag)
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
    if not (cli_node.GetStatus() & cli_node.Completed):
      if not self._batchAutoStartTime:
        self._batchAutoStartTime = time.time()
      elapsed = int(time.time() - self._batchAutoStartTime)
      self._batchAutoStatusLabel.setText(f"Processing {self._batchAutoTotal} pair(s)... {elapsed}s")
      return
    self._batchAutoStartTime = None
    self._batchAutoPollTimer.stop()
    if cli_node.GetStatus() & cli_node.ErrorsMask:
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
