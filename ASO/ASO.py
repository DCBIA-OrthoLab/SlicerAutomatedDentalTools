"""
Automated Standardized Orientation (ASO) of cone-beam computed tomography scans
(CBCT) and intraoral surface scans (IOS).

Thin GUI over the remote `ASO` tool. Orientation runs on the server, so nothing
is installed into Slicer's interpreter, no conda environment is created and no
reference bundle is downloaded to this machine.

Replaces the former local module, which drove the PRE/SEMI_ASO_CBCT and
PRE/SEMI_ASO_IOS CLIs from inside Slicer through a conda environment and a
pytorch3d wheel. `ASO_Method/` went with it and is no longer in the tree.

Authors:
- Nathan Hutin (UoM)
- Luc Anchling (UoM)
"""

import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib import slicer_io
from ServerToolsCoreLib.base_widget import ServerToolWidgetBase


class ASO(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ASO")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["ServerToolsCore"]
        self.parent.contributors = [
            "Nathan Hutin (UoM)",
            "Luc Anchling (UoM)",
            "Lucia Cevidanes (UoM)",
            "Juan Carlos Prieto (UoNC)",
        ]
        self.parent.helpText = _("""
        Orients CBCT scans and intraoral surface scans into a standardized coordinate frame,
        computed remotely by the Automated Dental Tools server. Give it one scan or a whole
        folder, pick the reference frame, and each case comes back oriented, with its landmarks
        and the transform back to the original acquisition.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was developed by Nathan Hutin (UoM) and Luc Anchling (UoM) and was supported by
        NIDCR R01 024450, AA0F Dewel Memorial Biomedical Research award and by Research
        Enhancement Award Activity 141 from the University of the Pacific,
        Arthur A. Dugoni School of Dentistry.
        """)


class ASOWidget(ServerToolWidgetBase):
    """Thin GUI: everything else (HTTP, async, form generation, styling, lifecycle)
    lives in ServerToolsCoreLib. See ARCHITECTURE.md.

    Nothing about the anatomy lives here: the landmark catalog, the tooth chart,
    which options start checked, and whether a run is CBCT or IOS, semi- or
    fully-automated, are all arguments in the server's schema rendered by
    formgen - so a landmark added server-side appears in this panel with no
    client release.

    Nor does the *layout*, which is the part that used to force a hand-written
    panel. The four collapsible boxes, the landmark tabs, the tooth chart and
    the fact that a CBCT run is never asked about teeth all come from the
    schema's presentation hints (`section`, `visible_when`, `ui`, `groups` - 
    see ARCHITECTURE.md). The old local module spent a four-page
    QStackedWidget and ~700 lines of checkbox plumbing on the same result, with
    the tooth numbering written out twice inside the widget.

    In particular the mode is *asked for*, never guessed: a `.zip` can hold CBCT
    volumes or intraoral meshes, so no extension could tell this module which
    pipeline the user wanted.
    """

    TOOL_NAME = "ASO"
    # No FILE_INPUTS: the schema types `input` as ["volume_or_zip_file",
    # "surface_file", "folder"] and `reference` as ["zip_file", "folder"], so
    # "auto" already gives each one a path field taking either a file or a whole
    # folder, the folder being zipped before upload. `reference` is also flagged
    # server_selectable, which puts the reference bundles the server hosts above
    # that picker. Nothing here names a type, an extension, or a mode.
    #
    # No RESULT_KIND either: output_kind "files" is one oriented scan, its
    # landmarks and its transform per case, plus ASO_report.json, bundled into
    # one .zip and unpacked into the output folder the user picks.
    AUTO_UI = True

    # Loading is a courtesy for the single-case run: a cohort legitimately
    # returns hundreds of files, and loading them all would be worse than
    # useless.
    MAX_RESULTS_TO_LOAD = 12

    # The oriented scan IS the result here, so the 3D view starting empty reads
    # as a run that did nothing. CT-AAA because the output is a dental CBCT;
    # see slicer_io.show_volume_rendering for the shift applied over it.
    VOLUME_RENDERING = "CT-AAA"
    # Pattern -> how to load it. The oriented CBCT is a VOLUME, not a
    # segmentation: ASO moves a scan, it does not label one. `*.tfm` is
    # deliberately absent - the transform is there to carry a measurement back
    # onto the acquisition, and loading it into the scene applies nothing.
    _LOADABLE = (
        ("*.nii.gz", "volume"),
        ("*.nii", "volume"),
        ("*.nrrd.gz", "volume"),
        ("*.nrrd", "volume"),
        ("*.gipl.gz", "volume"),
        ("*.gipl", "volume"),
        ("*.vtk", "model"),
        ("*.vtp", "model"),
        ("*.mrk.json", "markups"),
    )

    def handleResult(self, result) -> None:
        """Unpack the archive (base class), then optionally load what it held."""
        super().handleResult(result)

        self._maybeLoadResults()
