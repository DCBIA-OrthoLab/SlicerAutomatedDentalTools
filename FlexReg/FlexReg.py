"""
FlexReg: build a registration patch on an intraoral arch, and register on it.

Two arches are aligned on a REGION the clinician chooses rather than on the whole
mesh, because teeth move between timepoints and the palate does not: registering
on everything drags the result toward whatever moved most.

Thin GUI over the remote `FlexReg` tool. The patch and the registration run on
the server, so nothing is installed into Slicer's interpreter -- the former
module shipped 191 lines of `install_pytorch.py` for exactly that reason, its
patch propagation calling `.cuda()` with no availability test and no device
argument.

What did NOT move to the server is the preview. Dragging a pad recomputes the
patch outline here, on this machine, in about 18 ms on a 294k-point arch: a round
trip per gesture is not a preview. It needs nothing but vtk, numpy and
matplotlib, all of which ship with Slicer. The server is asked once, when Apply
is pressed, for the real patch -- geodesic propagation on the mesh rather than a
polygon fill, and what the registration actually runs on.

The five pads come from the schema. `run()` declares each corner as a pair of
floats and `layout.py` gives the axes their ranges and their end labels, so
`formgen` builds the joysticks with no code here. This module only wires them to
the preview.

Authors:
- Nathan Hutin (UoM)
- Luc Anchling (UoM)
"""

import qt
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib.base_widget import ServerToolWidgetBase

# The corner pads, in the order the preview names them.
CORNERS = ("anterior_right", "anterior_left", "posterior_right", "posterior_left")

# The tooth number driving each corner, by argument name.
TEETH = {corner: "tooth_" + corner for corner in CORNERS}

# What the preview draws into the scene. Kept out of the saved scene: it is a
# working overlay, and a study reopened a year later should not carry one.
CONTOUR_NODE = "FlexReg patch preview"


class FlexReg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("FlexReg")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = []
        self.parent.contributors = ["Nathan Hutin (UoM)", "Luc Anchling (UoM)"]
        self.parent.helpText = _(
            "Build a registration patch on an intraoral arch and register two "
            "timepoints on it. The patch and the registration run on the remote "
            "server; the outline you drag is previewed here."
        )


class FlexRegWidget(ServerToolWidgetBase):
    """The panel. Everything except the preview comes from the schema."""

    TOOL_NAME = "FlexReg"
    AUTO_UI = True

    # A registered arch is a MODEL, not a segmentation: FlexReg moves a surface,
    # it does not label one. `*.tfm` is deliberately absent -- the transform
    # carries a measurement back onto the original acquisition, and loading it
    # into the scene applies nothing by itself.
    # `reference` is the surface a patch is registered ONTO, but the word is
    # ambiguous on its own -- AREG's `cbct_reference` is a volume -- so the
    # name rule leaves it alone rather than guessing. Said here instead.
    SCENE_INPUTS = {"reference": ("model",)}

    _LOADABLE = (
        ("*.vtk", "model"),
        ("*.vtp", "model"),
        ("*.stl", "model"),
    )
    MAX_RESULTS_TO_LOAD = 12
    RESULT_KIND = "save_as"

    def __init__(self, parent=None):
        super().__init__(parent)
        # Imported lazily: it pulls numpy, vtk and matplotlib. Slicer ships all
        # three, but a module that will not IMPORT takes its whole panel with
        # it, and the preview is the one thing here that can be done without.
        from FlexRegLib.butterfly_preview import ButterflyPreview

        self._preview = ButterflyPreview()
        self._contourNode = None
        self._previewSurface = None
        self._previewCheckBox = None
        self._previewStatus = None

    # -- the preview --------------------------------------------------

    def addExtraWidgets(self, layout) -> None:
        """A switch and a status line, under the form.

        The switch defaults ON: watching the outline follow the pads is the
        whole point of having pads, and a refresh costs 18 ms.
        """
        self._previewCheckBox = qt.QCheckBox(_("Preview the patch while I drag"))
        self._previewCheckBox.checked = True
        self._previewCheckBox.toggled.connect(self._onPreviewToggled)
        layout.addWidget(self._previewCheckBox)

        self._previewStatus = qt.QLabel("")
        self._previewStatus.setWordWrap(True)
        self._previewStatus.setVisible(False)
        layout.addWidget(self._previewStatus)

        # Wired here because _argWidgets is filled by _buildForm, which runs
        # before addExtraWidgets.
        self._wirePreview()

    def _wirePreview(self) -> None:
        """Recompute on every value that moves the patch.

        The pads move the outline; the four tooth numbers move the centroids it
        is built from, which is the expensive path -- the cache has to go.
        """
        for name in CORNERS + ("shift",):
            widget = self._argWidgets.get(name)
            if widget is None:
                continue
            for box in (getattr(widget, "xBox", None), getattr(widget, "yBox", None)):
                if box is not None:
                    box.valueChanged.connect(self._onPatchChanged)

        for argument in TEETH.values():
            widget = self._argWidgets.get(argument)
            if widget is not None and hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._onTeethChanged)

    def _onPreviewToggled(self, checked) -> None:
        if checked:
            self._onTeethChanged()
        else:
            self._clearContour()
            self._setPreviewStatus("")

    def _onTeethChanged(self, *_args) -> None:
        """The teeth changed, so the cached centroids are stale."""
        self._preview.clear()
        self._onPatchChanged()

    def _onPatchChanged(self, *_args) -> None:
        if self._previewCheckBox is None or not self._previewCheckBox.checked:
            return

        surface = self._selectedSurface()
        if surface is None:
            self._clearContour()
            self._setPreviewStatus(_("Pick a labelled arch to preview the patch."))
            return

        teeth = self._selectedTeeth()
        if (not self._preview.ready
                or self._previewSurface is not surface
                or not self._preview.matches(teeth)):
            self._previewSurface = surface
            # Rebuilt only here: it walks every vertex and projects it, which is
            # the expensive half. Dragging a pad reuses it.
            self._preview.prepare(surface, teeth)

        if not self._preview.ready:
            self._clearContour()
            self._setPreviewStatus(self._preview.error or _("Cannot preview this arch."))
            return

        values = self._padValues()
        try:
            # with_fill=False: filling labels every vertex inside the contour,
            # which is what the SERVER computes properly. The outline is what a
            # hand needs to follow, and it is the cheap half.
            contour, _labels, _corners = self._preview.compute(
                values["ratios"], values["adjusts"], values["shift"], with_fill=False
            )
        except Exception as error:  # a preview must never take the panel down
            self._clearContour()
            self._setPreviewStatus(str(error))
            return

        self._showContour(contour)
        self._setPreviewStatus("")

    def _padValues(self) -> dict:
        """The five pads, named as the preview names them."""
        ratios, adjusts = {}, {}
        for corner in CORNERS:
            widget = self._argWidgets.get(corner)
            pair = widget.value() if widget is not None else [0.5, 0.0]
            ratios[corner], adjusts[corner] = float(pair[0]), float(pair[1])

        widget = self._argWidgets.get("shift")
        shift = widget.value() if widget is not None else [0.0, 0.0]
        return {
            "ratios": ratios,
            "adjusts": adjusts,
            "shift": (float(shift[0]), float(shift[1])),
        }

    def _selectedTeeth(self) -> dict:
        teeth = {}
        for corner, argument in TEETH.items():
            widget = self._argWidgets.get(argument)
            value = getattr(widget, "value", 0)
            teeth[corner] = int(value() if callable(value) else value)
        return teeth

    def _selectedSurface(self):
        """The polydata behind the `surfaces` row, when it points at a node.

        None when the row holds a path instead, which is the batch case: a
        folder of forty arches has no single outline to draw.
        """
        widget = self._argWidgets.get("surfaces")
        node = getattr(widget, "currentNode", None)
        node = node() if callable(node) else node
        if node is None or not hasattr(node, "GetPolyData"):
            return None
        return node.GetPolyData()

    def _showContour(self, contour) -> None:
        if self._contourNode is None:
            self._contourNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", CONTOUR_NODE
            )
            self._contourNode.CreateDefaultDisplayNodes()
            display = self._contourNode.GetDisplayNode()
            display.SetColor(1.0, 0.85, 0.1)
            display.SetLineWidth(3)
            # A working overlay, not a result: a study reopened a year later
            # should not carry one.
            self._contourNode.SetSaveWithScene(False)
        self._contourNode.SetAndObservePolyData(contour)
        self._contourNode.Modified()

    def _clearContour(self) -> None:
        if self._contourNode is not None:
            slicer.mrmlScene.RemoveNode(self._contourNode)
            self._contourNode = None

    def _setPreviewStatus(self, message) -> None:
        if self._previewStatus is not None:
            self._previewStatus.setText(message)
            self._previewStatus.setVisible(bool(message))

    def cleanup(self) -> None:
        self._clearContour()
        super().cleanup()
