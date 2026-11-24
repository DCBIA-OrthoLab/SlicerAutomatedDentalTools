"""
Impacted canine segmentation on CBCT (CLIC).

Thin GUI over the remote `CLIC` tool. The Mask R-CNN inference runs on the
server, so torchvision is never installed into Slicer's interpreter and no
checkpoint is downloaded to this machine.

It replaces a local module that ran that inference here; `runner/` and the
hand-written `Resources/UI/CLIC.ui` went with it. The panel is generated from
the server's `GET /tools` entry.
"""

from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from SADTCoreLib.base_widget import ServerToolWidgetBase
from SADTCoreLib.mode_switch import legacy_icon


class CLICVisor(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("CLIC (VISOR local)")
        self.parent.categories = ["Automated Dental Tools.VISOR local"]
        self.parent.icon = legacy_icon(__file__, "CLIC.png")
        self.parent.dependencies = ["SADTCore"]
        self.parent.contributors = [
            "Enzo Tulissi (UoM)",
            "Lucia Cevidanes (UoM)",
            "Juan Carlos Prieto (UoNC)",
        ]
        self.parent.helpText = _("""
        Segments impacted canines in CBCT scans — one scan or a folder of them — computed
        remotely by the Automated Dental Tools server.
        Detections scoring below the threshold are not painted, so raising it trades recall for
        certainty; the value used is recorded with every result.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("This module was supported by NIDCR R01 024450.")


class CLICVisorWidget(ServerToolWidgetBase):
    """Thin GUI: everything else (HTTP, async, form generation, styling, lifecycle)
    lives in SADTCoreLib. See ARCHITECTURE.md.

    The old panel had a hand-written .ui with its own input, model and output
    rows plus a progress bar; every one of those is a schema argument or part of
    the shared panel now, so none of it is restated here.
    """

    TOOL_NAME = "CLIC"
    LOAD_RESULTS_LABEL = _("Load the segmentations into the scene when done")

    MAX_RESULTS_TO_LOAD = 12

    # A painted canine is a LABELMAP: it holds the label the network assigned,
    # not intensities, and loading it as a volume would render it as grey.

    # This module works on CBCTs, so any scan it is given or produces is shown
    # in 3D with this preset -- an input the clinician just picked as much as a
    # result. "" for a module whose data is not a CT-like volume; nothing then
    # happens, and nothing happens anyway for one whose files load as meshes.
    VOLUME_RENDERING = "CT-AAA"

    _LOADABLE = (
        ("*.nii.gz", "labelmap"),
        ("*.nii", "labelmap"),
        ("*.nrrd", "labelmap"),
        ("*.nrrd.gz", "labelmap"),
    )

    def handleResult(self, result) -> None:
        """Unpack the archive (base class), then optionally load what it held."""
        super().handleResult(result)

        self._maybeLoadResults()
