"""
Automatic multi-anatomical skull structure segmentation (AMASSS) of cone-beam
computed tomography scans (CBCT).

Thin GUI over the remote `AMASSS` tool. Segmentation runs on the server, so
nothing is installed into Slicer's interpreter and no model is ever downloaded
to this machine.

Authors:
- Maxime Gillot (UoM)
- Baptiste Baquero (UoM)
"""


import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib import slicer_io
from ServerToolsCoreLib.base_widget import ServerToolWidgetBase


class AMASSS(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("AMASSS")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["ServerToolsCore"]
        self.parent.contributors = [
            "Maxime Gillot (CPE Lyon & UoM)",
            "Baptiste Baquero (CPE Lyon & UoM)",
            "Lucia Cevidanes (UoM)",
            "Juan Carlos Prieto (UoNC)",
        ]
        self.parent.helpText = _("""
        Automatically segments skull structures (mandible, maxilla, cranial base, cervical
        vertebrae, upper airway, skin and the AREG masks) in CBCT scans, computed remotely
        by the Automated Dental Tools server.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was developed by Maxime Gillot (CPE Lyon & UoM), Baptiste Baquero (CPE Lyon & UoM)
        and was supported by NIDCR R01 024450, AA0F Dewel Memorial Biomedical Research award and by
        Research Enhancement Award Activity 141 from the University of the Pacific,
        Arthur A. Dugoni School of Dentistry.
        """)


class AMASSSWidget(ServerToolWidgetBase):
    """Thin GUI: everything else (HTTP, async, form generation, styling, lifecycle)
    lives in ServerToolsCoreLib. See ARCHITECTURE.md.

    Nothing about the anatomy lives here: the structure list, its display names
    and which ones start checked are a `multichoice` argument in the server's
    schema, rendered by formgen - so a new model appears in this panel with no
    client release.
    """

    TOOL_NAME = "AMASSS"
    # "auto": the schema says `scans` is a "path", which a packaged tool uses
    # for a file OR a folder, so the panel takes either - one path field with a
    # File and a Folder browse button - a single scan or a whole cohort, the
    # folder being zipped before upload. Nothing here names a type, an
    # extension, or a mode.
    #
    # `scans`, not `input`: packaging renamed it. The argument name is part of
    # what the client sends, so it has to match the tool's run() signature.
    FILE_INPUTS = {"scans": "auto"}
    # output_kind "files": one <scan>_<ID>_SegOut/ folder per scan plus
    # AMASSS_report.json, bundled into one .zip and unpacked into the output
    # folder the user picks. The model is picked server-side, hence no
    # "Download latest models" button.
    RESULT_KIND = "save_as"
    AUTO_UI = True

    # Loading is a courtesy for the single-scan case: a cohort run legitimately
    # returns hundreds of files, and loading them all would be worse than useless.
    MAX_RESULTS_TO_LOAD = 12
    # Extension -> the slicer_io kind that loads it. Surfaces are only there
    # when the run asked for them (the schema's generate_surface).

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
        ("*.vtk", "model"),
    )

    def handleResult(self, result) -> None:
        """Unpack the archive (base class), then optionally load what it held."""
        super().handleResult(result)

        self._maybeLoadResults()
