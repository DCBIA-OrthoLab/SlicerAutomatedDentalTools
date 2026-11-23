"""
Teeth and jaw structures on dental CT and CBCT (Batch_Dental_Seg).

Thin GUI over the remote `Batch_Dental_Seg` tool. The nnU-Net inference runs on
the server, so torch is never installed into Slicer's interpreter and no weight
bundle is downloaded to this machine.

What the remote tool deliberately does not carry over is most of what the local
module was: a queue table, a RAM watchdog, killing nnUNet processes a crashed
scan left behind, a cool-down between scans, restoring the queue from disk. All
of that existed because inference ran inside Slicer on a clinician's laptop and
had to survive being out of memory. Here the queue is a folder argument and the
memory is the server's, and `BATCHDENTALSEGLib/` has gone with the local
inference it existed to drive.
"""

from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib.base_widget import ServerToolWidgetBase
from ServerToolsCoreLib.mode_switch import legacy_icon


class BATCHDENTALSEGVisor(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("BatchDentalSegmentator (VISOR local)")
        self.parent.categories = ["Automated Dental Tools.VISOR local"]
        self.parent.icon = legacy_icon(__file__, "DentalSegmentator.png")
        self.parent.dependencies = ["ServerToolsCore"]
        self.parent.contributors = [
            "Enzo Tulissi (UoM)",
            "Gauthier DOT (AP-HP)",
            "Laurent GAJNY (ENSAM)",
            "Roman FENIOUX (KITWARE SAS)",
            "Thibault PELLETIER (KITWARE SAS)",
        ]
        self.parent.helpText = _("""
        Segments teeth and jaw structures in dental CT and CBCT scans — one scan or a whole
        cohort — computed remotely by the Automated Dental Tools server.
        The hosted model bundle chooses the weights AND the label table together: the four
        trained models do not name the same anatomy, so which bundle ran is recorded with every
        result.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("This module was supported by NIDCR R01 024450.")


class BATCHDENTALSEGVisorWidget(ServerToolWidgetBase):
    """Thin GUI: everything else (HTTP, async, form generation, styling, lifecycle)
    lives in ServerToolsCoreLib. See ARCHITECTURE.md.

    Nothing about the anatomy or the models lives here. Which bundles exist is
    `GET /tools/Batch_Dental_Seg/data`, and the label table each one implies is
    the server's to publish with the run — so a newly trained model appears in
    this panel with no client release.
    """

    TOOL_NAME = "Batch_Dental_Seg"
    LOAD_RESULTS_LABEL = _("Load the segmentations into the scene when done")

    # A cohort legitimately returns one segmentation per scan, and
    # `separate_segments` multiplies that by the label count. Twelve is what the
    # other file-producing panels use.
    MAX_RESULTS_TO_LOAD = 12

    # A segmentation is a LABELMAP, not a greyscale volume: loading it as a
    # volume would render anatomy the network separated as one continuous grey
    # ramp, which is exactly the mistake the colour table exists to prevent.

    # This module works on CBCTs, so any scan it is given or produces is shown
    # in 3D with this preset -- an input the clinician just picked as much as a
    # result. "" for a module whose data is not a CT-like volume; nothing then
    # happens, and nothing happens anyway for one whose files load as meshes.
    VOLUME_RENDERING = "CT-AAA"

    # A segmentation comes back as a label VOLUME or as surfaces, depending on
    # what the Export format box asked for, and the two load as different kinds
    # of node: a labelmap carries the model's colour table, a mesh does not and
    # would be rendered as one grey ramp if it were loaded as a volume.
    _LOADABLE = (
        ("*.nii.gz", "labelmap"),
        ("*.nii", "labelmap"),
        ("*.nrrd", "labelmap"),
        ("*.nrrd.gz", "labelmap"),
        ("*.stl", "model"),
        ("*.obj", "model"),
        ("*.vtk", "model"),
    )

    def handleResult(self, result) -> None:
        """Unpack the archive (base class), then optionally load what it held."""
        super().handleResult(result)

        self._maybeLoadResults()
