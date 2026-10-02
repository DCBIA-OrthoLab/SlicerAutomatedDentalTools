"""
ManualApprox_MRI2CBCT.py

Injects the full GreedyReg manual alignment tools into MRI2CBCT as an
alternative approximation step.

Features (ported from GreedyReg.py):
  - 6 sliders: Rotation X/Y/Z and Translation X/Y/Z
  - Reset Transform button
  - Interactive tool: Slicer's built-in transform interaction handles
    (drag arrows to translate, rings to rotate) shown in slice and 3D views
  - Center MRI on CBCT button
  - Confirm & Save Alignment button

Usage - add to MRI2CBCT.py setup():
    from MRI2CBCT_utils.ManualApprox_MRI2CBCT import ManualApproximation_MRI2CBCT
    self.manual_approx = ManualApproximation_MRI2CBCT(self)
    self.manual_approx.injectUI(self.ui.approxCollapsibleButton)
"""

import os
import glob
import vtk
import qt
import ctk
import slicer
import numpy as np
import SimpleITK as sitk

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("MRI2CBCT_ManualApprox")


class ManualApproximation_MRI2CBCT:

    def __init__(self, widget):
        self.widget = widget          # MRI2CBCTWidget instance
        self.cbctNode = None
        self.mriNode  = None

        # Transform state (mirrors GreedyReg)
        self.transformNode       = None

        # UI references
        self.rotX = self.rotY = self.rotZ = None
        self.tranX = self.tranY = self.tranZ = None
        self.interactiveButton = None
        self.statusLabel = None
        self._cbct_path = None
        self._mri_path  = None

    # ------------------------------------------------------------------ #
    #  UI injection
    # ------------------------------------------------------------------ #

    def injectUI(self, collapsible_button):
        """Add manual alignment UI to the approxCollapsibleButton."""
        # Get the layout - handle both direct and container layouts
        layout = collapsible_button.layout()
        if layout is None:
            for child in collapsible_button.children():
                if hasattr(child, 'layout') and callable(child.layout) and child.layout() is not None:
                    layout = child.layout()
                    break
        if layout is None:
            layout = qt.QFormLayout()
            collapsible_button.setLayout(layout)
        if not hasattr(layout, 'addRow'):
            container = qt.QWidget()
            form = qt.QFormLayout(container)
            layout.addWidget(container)
            layout = form

        # Separator
        sep = qt.QLabel("Manual Approximation")
        sep.setStyleSheet(
            "color: #90CAF9; font-weight: bold; font-size: 12px; "
            "margin-top: 10px; border-top: 1px solid #555; padding-top: 6px;")
        layout.addRow(sep)

        # Volume selectors
        self.cbctSelector = slicer.qMRMLNodeComboBox()
        self.cbctSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.cbctSelector.setMRMLScene(slicer.mrmlScene)
        self.cbctSelector.noneEnabled = True
        self.cbctSelector.addEnabled  = False
        self.cbctSelector.removeEnabled = False
        self.cbctSelector.setToolTip("Fixed image (CBCT)")
        layout.addRow("CBCT (fixed):", self.cbctSelector)

        self.mriSelector = slicer.qMRMLNodeComboBox()
        self.mriSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.mriSelector.setMRMLScene(slicer.mrmlScene)
        self.mriSelector.noneEnabled = True
        self.mriSelector.addEnabled  = False
        self.mriSelector.removeEnabled = False
        self.mriSelector.setToolTip("Moving image (MRI) - this will be transformed")
        layout.addRow("MRI (moving):", self.mriSelector)

        # Connect selectors
        self.cbctSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onCBCTSelected)
        self.mriSelector.connect("currentNodeChanged(vtkMRMLNode*)",  self._onMRISelected)

        # Load from folder button (alternative to selecting from scene)
        self.loadButton = qt.QPushButton("Load from Folders Above")
        self.loadButton.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; "
            "font-weight: bold; padding: 6px; border-radius: 4px; }")
        self.loadButton.setToolTip(
            "Load CBCT and MRI from the folders specified above into the scene")
        self.loadButton.clicked.connect(self._onLoadVolumes)
        layout.addRow(self.loadButton)

        # Rotation sliders
        self.rotX = ctk.ctkSliderWidget()
        self.rotX.minimum, self.rotX.maximum = -180, 180
        self.rotX.value = 0
        layout.addRow("Rotation X:", self.rotX)

        self.rotY = ctk.ctkSliderWidget()
        self.rotY.minimum, self.rotY.maximum = -180, 180
        self.rotY.value = 0
        layout.addRow("Rotation Y:", self.rotY)

        self.rotZ = ctk.ctkSliderWidget()
        self.rotZ.minimum, self.rotZ.maximum = -180, 180
        self.rotZ.value = 0
        layout.addRow("Rotation Z:", self.rotZ)

        # Translation sliders
        self.tranX = ctk.ctkSliderWidget()
        self.tranX.minimum, self.tranX.maximum = -200, 200
        self.tranX.value = 0
        layout.addRow("Translation X (mm):", self.tranX)

        self.tranY = ctk.ctkSliderWidget()
        self.tranY.minimum, self.tranY.maximum = -200, 200
        self.tranY.value = 0
        layout.addRow("Translation Y (mm):", self.tranY)

        self.tranZ = ctk.ctkSliderWidget()
        self.tranZ.minimum, self.tranZ.maximum = -200, 200
        self.tranZ.value = 0
        layout.addRow("Translation Z (mm):", self.tranZ)

        # Connect sliders
        for s in [self.rotX, self.rotY, self.rotZ,
                  self.tranX, self.tranY, self.tranZ]:
            s.valueChanged.connect(self.onManualTransformChanged)

        # Reset button
        reset_btn = qt.QPushButton("Reset Transform")
        reset_btn.setStyleSheet(
            "QPushButton { background-color: #546E7A; color: white; "
            "padding: 6px; border-radius: 4px; }")
        reset_btn.clicked.connect(self.onResetTransform)
        layout.addRow(reset_btn)

        # Interactive tools sub-section
        tools_box = ctk.ctkCollapsibleButton()
        tools_box.text = "Interactive Tools"
        tools_box.collapsed = True
        # We add it as a row spanning both columns
        layout.addRow(tools_box)
        tools_layout = qt.QFormLayout(tools_box)

        self.centerButton = qt.QPushButton("Center MRI on CBCT")
        self.centerButton.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; "
            "font-weight: bold; padding: 6px; border-radius: 4px; }")
        self.centerButton.clicked.connect(self.onCenterVolumes)
        tools_layout.addRow(self.centerButton)

        self.interactiveButton = qt.QPushButton("Enable Interactive Tool")
        self.interactiveButton.setCheckable(True)
        self.interactiveButton.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:checked { background-color: #F44336; }")
        self.interactiveButton.clicked.connect(self.onInteractiveToolToggled)
        tools_layout.addRow(self.interactiveButton)

        self.interactiveHint = qt.QLabel(
            "Drag the arrows to translate, the rings to rotate (Slicer's built-in transform handles)")
        self.interactiveHint.setStyleSheet("color: gray; font-size: 10px;")
        self.interactiveHint.setVisible(False)
        tools_layout.addRow(self.interactiveHint)

        # Embed the standard Transforms module Display panel (interaction
        # handle checkboxes, axis enables, glyph/grid options, etc.) instead
        # of only toggling the handles invisibly from code - same as GreedyReg.
        self.transformDisplayWidget = slicer.qMRMLTransformDisplayNodeWidget()
        self.transformDisplayWidget.setVisible(False)
        tools_layout.addRow(self.transformDisplayWidget)

        # Confirm button
        self.confirmButton = qt.QPushButton("Confirm & Save Alignment")
        self.confirmButton.setStyleSheet(
            "QPushButton { background-color: #2E7D32; color: white; "
            "font-weight: bold; padding: 8px; border-radius: 4px; }")
        self.confirmButton.setEnabled(False)
        self.confirmButton.clicked.connect(self.onConfirm)
        layout.addRow(self.confirmButton)

        self.statusLabel = qt.QLabel(
            "Set CBCT and MRI folders above, then click Load.")
        self.statusLabel.setStyleSheet("color: #aaa; font-size: 11px;")
        self.statusLabel.setWordWrap(True)
        layout.addRow(self.statusLabel)

    # ------------------------------------------------------------------ #
    #  Load volumes
    # ------------------------------------------------------------------ #

    def _findFirstNifti(self, folder):
        for ext in ["*.nii.gz", "*.nii"]:
            files = glob.glob(os.path.join(folder, ext))
            if files:
                return files[0]
        return None

    def _onCBCTSelected(self, node):
        """Called when CBCT selector changes."""
        if node:
            self.cbctNode = node
            self._cbct_path = None  # path unknown when selected from scene
            self._updateSliceViews()
            self.confirmButton.setEnabled(
                self.mriNode is not None and self.transformNode is not None)

    def _onMRISelected(self, node):
        """Called when MRI selector changes."""
        if node:
            self.mriNode = node
            self._mri_path = None  # path unknown when selected from scene
            # Create transform node if needed
            if not self.transformNode:
                self.transformNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLinearTransformNode", "ManualApproxTransform")
                self.transformDisplayWidget.setMRMLTransformNode(self.transformNode)
            self.mriNode.SetAndObserveTransformNodeID(self.transformNode.GetID())
            self._updateSliceViews()
            self.confirmButton.setEnabled(True)

    def _updateSliceViews(self):
        if self.cbctNode and self.mriNode:
            slicer.util.setSliceViewerLayers(
                background=self.cbctNode,
                foreground=self.mriNode,
                foregroundOpacity=0.5)
            if self.statusLabel:
                self.statusLabel.setText(
                    "Volumes selected. Use sliders or Interactive Tool to align.")
                self.statusLabel.setStyleSheet("color: #4CAF50;")

    def _onLoadVolumes(self):
        cbct_folder = self.widget.ui.lineEditApproxCBCT.text.strip()
        mri_folder  = self.widget.ui.lineEditApproxMRI.text.strip()

        if not cbct_folder or not os.path.isdir(cbct_folder):
            self.statusLabel.setText("Please set the CBCT folder first.")
            self.statusLabel.setStyleSheet("color: red;")
            return
        if not mri_folder or not os.path.isdir(mri_folder):
            self.statusLabel.setText("Please set the MRI folder first.")
            self.statusLabel.setStyleSheet("color: red;")
            return

        self._cbct_path = self._findFirstNifti(cbct_folder)
        self._mri_path  = self._findFirstNifti(mri_folder)

        if not self._cbct_path:
            self.statusLabel.setText("No NIfTI files found in CBCT folder.")
            self.statusLabel.setStyleSheet("color: red;")
            return
        if not self._mri_path:
            self.statusLabel.setText("No NIfTI files found in MRI folder.")
            self.statusLabel.setStyleSheet("color: red;")
            return

        self.cbctNode = slicer.util.loadVolume(self._cbct_path)
        self.mriNode  = slicer.util.loadVolume(self._mri_path)

        if not self.cbctNode or not self.mriNode:
            self.statusLabel.setText("Failed to load volumes.")
            self.statusLabel.setStyleSheet("color: red;")
            return

        self.cbctNode.SetName("CBCT_manual_approx")
        self.mriNode.SetName("MRI_manual_approx")

        # Update selectors to show loaded volumes
        self.cbctSelector.setCurrentNode(self.cbctNode)
        self.mriSelector.setCurrentNode(self.mriNode)

        # Create/recreate transform node
        if self.transformNode:
            slicer.mrmlScene.RemoveNode(self.transformNode)
        self.transformNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", "ManualApproxTransform")
        self.transformDisplayWidget.setMRMLTransformNode(self.transformNode)
        self.mriNode.SetAndObserveTransformNodeID(self.transformNode.GetID())

        slicer.util.setSliceViewerLayers(
            background=self.cbctNode,
            foreground=self.mriNode,
            foregroundOpacity=0.5)
        slicer.util.resetSliceViews()

        self.confirmButton.setEnabled(True)
        self.statusLabel.setText(
            "Volumes loaded. Use sliders or Enable Interactive Tool to align.\n"
            "Then click Confirm & Save Alignment.")
        self.statusLabel.setStyleSheet("color: #4CAF50;")

    # ------------------------------------------------------------------ #
    #  Slider-based transform (exact port from GreedyReg)
    # ------------------------------------------------------------------ #

    def onManualTransformChanged(self):
        if not self.transformNode or not self.mriNode:
            return
        self.mriNode.SetAndObserveTransformNodeID(self.transformNode.GetID())
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.Translate(self.tranX.value, self.tranY.value, self.tranZ.value)
        transform.RotateX(self.rotX.value)
        transform.RotateY(self.rotY.value)
        transform.RotateZ(self.rotZ.value)
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrix(matrix)
        self.transformNode.SetMatrixTransformToParent(matrix)

    def onResetTransform(self):
        if not self.transformNode:
            return
        identity = vtk.vtkMatrix4x4()
        self.transformNode.SetMatrixTransformToParent(identity)
        for s in [self.rotX, self.rotY, self.rotZ,
                  self.tranX, self.tranY, self.tranZ]:
            s.blockSignals(True)
            s.value = 0
            s.blockSignals(False)
        self.statusLabel.setText("Transform reset.")

    # ------------------------------------------------------------------ #
    #  Interactive tool - Slicer's built-in transform interaction handles
    # ------------------------------------------------------------------ #

    def onInteractiveToolToggled(self, checked):
        if checked:
            self.interactiveButton.setText("Disable Interactive Tool")
            self.interactiveHint.setVisible(True)
            self.startInteractiveTool()
        else:
            self.interactiveButton.setText("Enable Interactive Tool")
            self.interactiveHint.setVisible(False)
            self.stopInteractiveTool()

    def onCenterVolumes(self):
        if not self.cbctNode or not self.mriNode or not self.transformNode:
            self.statusLabel.setText("Load volumes first.")
            return
        cbct_bounds = [0]*6
        self.cbctNode.GetRASBounds(cbct_bounds)
        mri_bounds  = [0]*6
        self.mriNode.GetRASBounds(mri_bounds)
        dx = ((cbct_bounds[0]+cbct_bounds[1]) - (mri_bounds[0]+mri_bounds[1])) / 2
        dy = ((cbct_bounds[2]+cbct_bounds[3]) - (mri_bounds[2]+mri_bounds[3])) / 2
        dz = ((cbct_bounds[4]+cbct_bounds[5]) - (mri_bounds[4]+mri_bounds[5])) / 2
        matrix = vtk.vtkMatrix4x4()
        self.transformNode.GetMatrixTransformToParent(matrix)
        matrix.SetElement(0, 3, matrix.GetElement(0, 3) + dx)
        matrix.SetElement(1, 3, matrix.GetElement(1, 3) + dy)
        matrix.SetElement(2, 3, matrix.GetElement(2, 3) + dz)
        self.transformNode.SetMatrixTransformToParent(matrix)
        self.statusLabel.setText("MRI centered on CBCT.")

    def startInteractiveTool(self):
        """Show Slicer's built-in transform interaction handles (translate + rotate)
        in both slice and 3D views, instead of a custom mouse-driven implementation."""
        if not self.transformNode or not self.mriNode:
            self.statusLabel.setText("Load CBCT and MRI volumes first.")
            self.statusLabel.setStyleSheet("color: red;")
            self.interactiveButton.setChecked(False)
            self.interactiveButton.setText("Enable Interactive Tool")
            self.interactiveHint.setVisible(False)
            return
        self.mriNode.SetAndObserveTransformNodeID(self.transformNode.GetID())

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
        if not self.transformNode:
            return
        display_node = self.transformNode.GetDisplayNode()
        if display_node:
            display_node.SetEditorVisibility(False)
        self.transformDisplayWidget.setVisible(False)

    # ------------------------------------------------------------------ #
    #  Confirm & Save
    # ------------------------------------------------------------------ #

    def onConfirm(self):
        if not self.mriNode or not self.transformNode:
            self.statusLabel.setText("Load volumes first.")
            return
        output_folder = self.widget.ui.lineEditOutputApprox.text.strip()
        if not output_folder:
            self.statusLabel.setText("Please set the output folder first.")
            self.statusLabel.setStyleSheet("color: red;")
            return

        approx_folder = os.path.join(output_folder, "first_approximation")
        os.makedirs(approx_folder, exist_ok=True)

        try:
            # Stop interactive tool if running
            if self.interactiveButton and self.interactiveButton.isChecked():
                self.interactiveButton.setChecked(False)
                self.stopInteractiveTool()

            # Harden transform into a clone and export
            clone = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
            clone.Copy(self.mriNode)
            clone.SetAndObserveTransformNodeID(self.transformNode.GetID())
            slicer.modules.transforms.logic().hardenTransform(clone)

            mri_basename = os.path.splitext(
                os.path.splitext(os.path.basename(self._mri_path))[0])[0]
            out_path = os.path.join(approx_folder,
                                    f"{mri_basename}_approximate.nii.gz")
            slicer.util.saveNode(clone, out_path)
            slicer.mrmlScene.RemoveNode(clone)

            # Also save the transform
            m = vtk.vtkMatrix4x4()
            self.transformNode.GetMatrixTransformToParent(m)
            ras = np.array([[m.GetElement(i,j) for j in range(4)] for i in range(4)])
            flip = np.diag([-1.,-1.,1.,1.])
            lps  = flip @ ras @ flip
            sitk_t = sitk.AffineTransform(3)
            sitk_t.SetMatrix(lps[:3,:3].flatten().tolist())
            sitk_t.SetTranslation(lps[:3,3].tolist())
            tfm_path = os.path.join(approx_folder, f"{mri_basename}_approximate.tfm")
            sitk.WriteTransform(sitk_t, tfm_path)

            self.statusLabel.setText(
                f"Saved to:\n{out_path}\nYou can now run Registration.")
            self.statusLabel.setStyleSheet("color: #4CAF50;")
            logger.info(f"Manual approximation saved: {out_path}")

        except Exception as e:
            self.statusLabel.setText(f"Error: {e}")
            self.statusLabel.setStyleSheet("color: red;")
            logger.error(f"Error saving manual approximation: {e}", exc_info=True)
