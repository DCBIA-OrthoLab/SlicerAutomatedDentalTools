from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib.base_widget import ServerToolWidgetBase


class SurgMovPred(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SurgMovPred")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["ServerToolsCore"]
        self.parent.contributors = ["Paul Dumont, University of North Carolina, Chapel Hill"]
        self.parent.helpText = _("""
        Predicts surgical movement outcomes from cephalometric measurements, computed by a
        model served remotely by the Automated Dental Tools server.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """)


class SurgMovPredWidget(ServerToolWidgetBase):
    """Thin GUI: everything else (HTTP, async, form generation, styling, lifecycle)
    lives in ServerToolsCoreLib. See ARCHITECTURE.md."""

    TOOL_NAME = "Surg_Mov_Pred"
    # No FILE_INPUTS. There used to be one, `{"input": "folder_zip"}`, and it
    # named an argument this tool stopped having when it was packaged: the
    # table is `measurements` now. An override naming an argument the schema
    # does not declare used to be ADDED to the form, so the panel grew a row
    # labelled "Input", styled optional because there was no spec to say
    # otherwise, uploading to an argument the server would have rejected.
    # `measurements` is a packaged tool's `path`, which the generic picker
    # already takes as either one table or a folder, zipping the folder on the
    # way out -- which is all the override ever wanted.
    #
    # output_kind is "file", which says a file comes back but not what to do
    # with it: save it, rather than load it into the scene.
    RESULT_KIND = "save_as"
    AUTO_UI = True
