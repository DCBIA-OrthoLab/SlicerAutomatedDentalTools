"""
Crop of both temporomandibular joints of a CBCT.

Thin facade over SADT-VISOR's `TMJ_Crop` tool, run on this machine through the
local client (ServerToolsCoreLib.local_client): the panel is generated from
the tool's own run() schema, and the tool orients the scan with ASO (which
places its landmarks with ALI_CBCT), segments the mandible with AMASSS and
cuts a box around each condyle, each tool in its own environment. Nothing is
installed into Slicer's interpreter.

Authors:
- Jules Grivot Pélisson (UNC)
"""

from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib.base_widget import ServerToolWidgetBase


class TMJ_Crop(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("TMJ_Crop")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["ServerToolsCore"]
        self.parent.contributors = ["Jules Grivot Pélisson (UNC)"]
        self.parent.helpText = _("""
        Crops both temporomandibular joints of a CBCT into fixed-size volumes: the scan is
        oriented to the Frankfurt horizontal frame, the mandible is segmented, and a box is
        cut around each condyle.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools#tmj_crop">module documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """)


class TMJ_CropWidget(ServerToolWidgetBase):
    """Thin GUI: the form, the run, the progress and the results all come from
    ServerToolsCoreLib, driven by the tool's schema."""

    TOOL_NAME = "TMJ_Crop"
    # `scans` is a "path": one CBCT or a folder of them.
    FILE_INPUTS = {"scans": "auto"}
    RESULT_KIND = "save_as"
    AUTO_UI = True
    VOLUME_RENDERING = "CT-AAA"

    # A run writes, per scan, the two crops, the report and a quick view, plus
    # the oriented scan and the mandible when asked: loading the crops of one
    # scan is a courtesy, a cohort's are left on disk.
    MAX_RESULTS_TO_LOAD = 6
    _LOADABLE = (
        ("*_R.nii.gz", "volume"),
        ("*_L.nii.gz", "volume"),
    )

    def handleResult(self, result) -> None:
        super().handleResult(result)
        self._maybeLoadResults()
