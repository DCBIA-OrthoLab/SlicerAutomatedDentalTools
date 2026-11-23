"""
Crop CBCT volumes to a region of interest drawn in Slicer (AutoCrop3D).

Thin GUI over the remote `AutoCrop3D` tool. The crop itself is arithmetic on an
image grid -- no weights, no GPU, no network -- but it runs on the server like
every other tool, so a cohort of 512x512x365 volumes is never resampled in
Slicer's own interpreter.

It replaces a local module that computed here; `Crop_Volumes_CLI/` and the
hand-written `Crop_Volumes_UI/` went with it. The panel is generated from the
server's `GET /tools` entry.
"""

from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib.base_widget import ServerToolWidgetBase


class AutoCrop3D(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("AutoCrop3D")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["ServerToolsCore"]
        self.parent.contributors = [
            "Enzo Tulissi (UoM)",
            "Lucia Cevidanes (UoM)",
            "Juan Carlos Prieto (UoNC)",
        ]
        self.parent.helpText = _("""
        Crops CBCT scans — one or a whole cohort — to a region of interest, computed remotely by
        the Automated Dental Tools server.
        Draw the box with the Markups module's ROI tool and pick it below; the scans and the box
        are matched per patient by name, so a cohort needs one ROI per patient.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("This module was supported by NIDCR R01 024450.")


class AutoCrop3DWidget(ServerToolWidgetBase):
    """Thin GUI: everything else (HTTP, async, form generation, styling, lifecycle)
    lives in ServerToolsCoreLib. See ARCHITECTURE.md.

    Of the 1 457 lines upstream, 1 136 were the Qt panel — its own scan row, ROI
    row, suffix field, check boxes and progress bar. Every one of those is a
    schema argument or part of the shared panel now, so none of it is restated
    here.
    """

    TOOL_NAME = "AutoCrop3D"
    LOAD_RESULTS_LABEL = _("Load the cropped scans into the scene when done")

    # The box is DRAWN in Slicer -- that is the whole reason a clinician has it
    # open beside the panel -- so it is already a node when this row is filled.
    # Declared because the schema cannot say it: `describe.py` publishes no
    # extensions for a packaged tool, and the name `roi` means nothing to the
    # generic rule. `scans` needs no entry: "scan" is in that vocabulary.
    #
    # A ROI is `vtkMRMLMarkupsROINode`, which is NOT a markups fiducial, hence
    # its own kind in formgen.SCENE_NODE_KINDS.
    SCENE_INPUTS = {"roi": ("roi",)}

    # A crop of a CBCT is a CBCT: it keeps the intensities, only the extent
    # changes. So it is a volume, never a labelmap -- unlike a segmentation
    # tool's output, which carries labels.
    #
    # `*_vtk.vtk` is the surface a cropped label map can be turned into, and it
    # loads as a model. One pattern per extension: two that can match one file
    # would open that file twice.
    _LOADABLE = (
        ("*.nii.gz", "volume"),
        ("*.nii", "volume"),
        ("*.nrrd", "volume"),
        ("*.nrrd.gz", "volume"),
        ("*.gipl", "volume"),
        ("*.gipl.gz", "volume"),
        ("*.vtk", "model"),
    )

    # This module works on CBCTs, so any scan it is given or produces is shown
    # in 3D with this preset -- an input the clinician just picked as much as a
    # result.
    VOLUME_RENDERING = "CT-AAA"

    # A cohort crops one file per scan, and `keep_original_size` can double that
    # with a surface each. The shared cap is what stops forty patients landing
    # in the scene at once.
    MAX_RESULTS_TO_LOAD = 12

    RUN_REPORT = "AutoCrop3D_report.json"

    def handleResult(self, result) -> None:
        """Unpack the archive (base class), then optionally load what it held."""
        super().handleResult(result)

        self._maybeLoadResults()
        self._sayWhatWasCropped()

    def _sayWhatWasCropped(self) -> None:
        """One line from the run report, because a crop looks like nothing.

        The panel otherwise says only "done": a cropped volume opens at the same
        place as the scan it came from, and a patient the run SKIPPED leaves no
        trace on screen at all. That was upstream's defect in its purest form --
        a cohort whose identifiers held an underscore matched no ROI, was
        skipped in full, and exited 0 with an empty folder.

        Best effort, like every other use of the report: the results are on disk
        and in the scene whatever this says.
        """
        report = self._readRunReport(self._producedRoot)
        if not report:
            return
        summary = report.get("summary") or {}
        cropped, found = summary.get("cropped"), summary.get("scans_found")
        if cropped is None or found is None:
            return

        message = _("{cropped} of {found} scan(s) cropped.").format(
            cropped=cropped, found=found)
        # Named, not counted. Which patient found no box is the one thing a
        # clinician can act on, and it is the difference between "the run did
        # less than you think" and a number nobody reads.
        absent = [
            str(entry.get("patient") or entry.get("scan"))
            for entry in (report.get("without_a_roi") or [])
            if isinstance(entry, dict)
        ]
        if absent:
            message += " " + _("No ROI for: {names}.").format(
                names=", ".join(absent[:5]))
        self._showPhase(message)
