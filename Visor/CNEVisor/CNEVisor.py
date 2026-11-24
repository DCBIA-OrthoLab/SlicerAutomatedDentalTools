"""CNE - clinical note extraction, computed on the tool server.

Replaces the former local module, which pip-installed llama-cpp-python, PyMuPDF
and python-docx into Slicer's interpreter, downloaded a 4.4-4.9 GB GGUF model
from Hugging Face into the user's Documents folder, and drove the CNE_CLIRemote
command-line module. Nothing is installed or downloaded here any more: the
panel is generated from the server's `GET /tools` entry, the notes go up, the
extractions come back. CNE_CLIRemote/, the hand-written Resources/UI/CNE.ui and
the bundled Resources/testfiles/ went with it.

What the old panel offered, and where it went:

* **Notes folder** -> `notes`, which takes one note or a folder of them and
  filters the dialog on the extensions the tool's readers read.
* **TMJ / Ortho radio buttons** -> `notes_type`. It still picks the model: the
  `model` field defaults to "(automatic)", which the tool resolves to the
  bundle fine-tuned for that note type. Naming a model is possible, and a model
  that does not match the note type is reported in the run's warnings.
* **Summary folder** -> the shared output-folder row.
* **Download Test Files** -> the test notes the server hosts for this tool,
  offered in the `notes` row's own dropdown like every other module's.
* **The model download dialog and the pip-install prompt** -> dropped: the
  model lives on the server and so does the inference engine.
* **The CLI progress bar and Cancel button** -> the shared run controls.
"""

import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from SADTCoreLib.base_widget import ServerToolWidgetBase
from SADTCoreLib.mode_switch import legacy_icon


class CNEVisor(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("CNE (VISOR local)")
        self.parent.categories = ["Automated Dental Tools.VISOR local"]
        self.parent.icon = legacy_icon(__file__, "CNE.png")
        self.parent.dependencies = ["SADTCore"]
        self.parent.contributors = ["Paul Dumont, University of North Carolina, Chapell Hill"]
        self.parent.helpText = _("""
        This tool helps to create summaries of clinical notes.
        The language model runs on the Automated Dental Tools server: nothing is installed
        into Slicer and no model is downloaded to this machine.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """)


class CNEVisorWidget(ServerToolWidgetBase):
    """Thin GUI: HTTP, async, form generation, styling and lifecycle all live
    in SADTCoreLib. See ARCHITECTURE.md."""

    TOOL_NAME = "CNE"
    RUN_REPORT = "CNE_report.json"

    # No FILE_INPUTS: `notes` is a packaged tool's `path`, which the client
    # already gives a picker taking a file or a folder, filtered on the
    # extensions the schema publishes for it.
    #
    # No RESULT_KIND: output_kind "files" is one `Extraction_<note>.json` and
    # `.txt` per note plus the run report, bundled into one .zip and unpacked
    # into the output folder the user picks.
    #
    # No _LOADABLE: an extraction is text, and there is nothing in it to put in
    # a scene, so the panel offers no "load the results" box.

    def handleResult(self, result) -> None:
        """Unpack the archive (base class), then say how many notes made it.

        A note the model could not answer for -- an image-only PDF, an answer
        cut off at the token limit -- is skipped rather than failing the run,
        so the count and the warnings are what a user has to read.
        """
        super().handleResult(result)

        outputDir = self._outputFolderWidget.currentPath if self._outputFolderWidget else None
        if not outputDir:
            return

        report = self._readRunReport(outputDir)
        if report:
            slicer.util.showStatusMessage(self._summarize(report), 8000)

    @classmethod
    def _summarize(cls, report: dict) -> str:
        summary = _("CNE: {summary} with the {notes_type} model.").format(
            summary=report.get("summary") or "?",
            notes_type=report.get("notes_type") or "?",
        )
        warnings = report.get("warnings") or []
        if warnings:
            summary += " " + _("{count} warning(s), see {report}.").format(
                count=len(warnings), report=cls.RUN_REPORT,
            )
        return summary
