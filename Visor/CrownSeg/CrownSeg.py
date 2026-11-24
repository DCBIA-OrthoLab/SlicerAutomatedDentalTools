"""
Segmentation and labelling of the teeth crowns of an intraoral scan.

Thin facade over SADT-VISOR's `Crown_Seg` tool, run on this machine through the
local client (SADTCoreLib.local_client): the panel is generated from the
tool's own run() schema, and the tool runs in its own environment. Nothing is
installed into Slicer's interpreter.
"""

from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from SADTCoreLib.base_widget import ServerToolWidgetBase


class CrownSeg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Crown_Seg")
        self.parent.categories = ["Automated Dental Tools.VISOR local"]
        self.parent.dependencies = ["SADTCore"]
        self.parent.contributors = ["Juan Carlos Prieto (UNC)", "Lucia Cevidanes (UoM)"]
        self.parent.helpText = _("""
        Segments and labels each tooth crown of an intraoral surface scan.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """)


class CrownSegWidget(ServerToolWidgetBase):
    """Thin GUI: the form, the run, the progress and the results all come from
    SADTCoreLib, driven by the tool's schema."""

    TOOL_NAME = "Crown_Seg"
    RESULT_KIND = "save_as"
    AUTO_UI = True
    VOLUME_RENDERING = ""
    MAX_RESULTS_TO_LOAD = 12
    _LOADABLE = (
        ("*.vtk", "model"),
    )

    def handleResult(self, result) -> None:
        super().handleResult(result)
        self._maybeLoadResults()
