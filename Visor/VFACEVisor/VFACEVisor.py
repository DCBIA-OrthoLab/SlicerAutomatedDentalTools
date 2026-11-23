"""
V FACE -- facial asymmetry and longitudinal change, measured and classified.

Thin GUI over the remote `VFACE` tool. Everything runs on the Automated Dental
Tools server: the resampling, both orientations, the AMASSS segmentation, the
registrations, the landmark prediction, the measurements and the classification.
Nothing is installed into Slicer's interpreter, no conda environment is created,
and none of the six model bundles is ever downloaded to this machine.

Replaces the former local module, which was 1594 lines: `slicer.util.pip_install`
for joblib and lightgbm at the click of a button, ten archives downloaded into
`~/Documents/SlicerDownloads`, and a chain of a dozen processes rebuilt by hand in
`VFACE_utils/createlistprocess.py` and stepped through with a QTimer. That
sequencing is the server's now -- the tool calls its seven neighbours in-process
-- so this module makes ONE request.

`VFACE_utils/` is left in the tree but is no longer wired to this one, the same
way `AREG_Method` and `ASO_Method` were left.

What did NOT survive the port, and deliberately:

* the "keep intermediate files" and "pause for visualization" check boxes. Both
  exist server-side as `keep_intermediate` and `stop_after`, and both are hidden
  by the tool's own layout: stopping a cohort halfway for review is a workflow
  this deployment has not decided on. A module that offered them would be a
  second place to keep that decision;
* `VFACELogic.process()` and `VFACETest`, which thresholded a scalar volume --
  Slicer's module template, never VFACE, and tested only itself;
* `registerSampleData`, which registered two generic Slicer test volumes under
  the VFACE name. The hosted test cohorts the server publishes take its place,
  and they are real VFACE inputs.

**There is no `.ui` file here, and that is the design rather than an omission.**
The panel is built from the server's schema by `ServerToolsCoreLib.formgen`, so a
field added to the tool server-side appears here with no client release. The
former module's `Resources/UI/VFACE.ui` was deleted rather than left behind,
because a resource CMakeLists no longer ships and no code reads still looks like
the panel to whoever opens Qt Designer -- and editing it is a no-op that reports
no error. One such file cost a day of looking for a missing UI change on the
wrong side of the client/server line.

Author:
- Alexandre Buisson (University of North Carolina at Chapel Hill)
"""

import qt
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib.base_widget import ServerToolWidgetBase


class VFACE(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        # "V FACE", with the space: that is the name in the module selector
        # today, and somebody looking for it types what they saw.
        self.parent.title = _("V FACE")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["ServerToolsCore"]
        self.parent.contributors = [
            "Alexandre Buisson (University of North Carolina at Chapel Hill)",
        ]
        self.parent.helpText = _("""
        Measures facial asymmetry, or change between two timepoints, remotely on the Automated
        Dental Tools server.
        Give it a folder of CBCT scans. An asymmetry assessment compares each patient against
        their own mirror image; a longitudinal study compares a baseline against a follow-up.
        Either way the scans are put on one voxel grid, oriented into the cranial base and the
        maxillary frames, segmented, registered on the regions you pick, and measured on the
        landmarks those measurements name.
        The results come back as measurement tables and a classification, Excel workbooks to
        open outside Slicer, plus the heat maps if you asked for them -- those are surfaces, and
        this panel can load them into the scene for you.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was developed by Alexandre Buisson (University of North Carolina at Chapel
        Hill) and was supported by NIDCR R01 024450, AA0F Dewel Memorial Biomedical Research
        award and by Research Enhancement Award Activity 141 from the University of the Pacific,
        Arthur A. Dugoni School of Dentistry.
        """)


class VFACEWidget(ServerToolWidgetBase):
    """Thin GUI: everything else (HTTP, async, form generation, styling, lifecycle)
    lives in ServerToolsCoreLib. See ARCHITECTURE.md.

    Nothing about the pipeline lives here. Which studies exist, which regions can
    be measured, what a run returns, and the fact that a longitudinal study asks
    for a second timepoint where an asymmetry assessment does not, are all in the
    server's schema and rendered by formgen -- so a region added server-side
    appears in this panel with no client release. The old module carried the same
    switches as four `currentTextChanged` handlers showing and hiding labels.

    Eleven of the tool's eighteen arguments are hidden by ITS layout, and none of
    them is hidden here: the classifier, the mirror transform, the two
    orientation references, the measurement lists and the feature template are
    resolved by the deployment, and the two quality-control knobs are not offered
    yet. Naming them again in this module would be a second place to keep one
    decision.

    The three inputs are typed `path`, which means a folder is as acceptable as
    one file, so `formgen.auto_file_mode` gives each a folder picker that zips
    before upload -- and `t1`/`t2` are `server_selectable`, which puts the hosted
    test cohorts above it. That is why there is no `FILE_INPUTS` here. It also
    matters more than it looks: `t1` wants a FOLDER of scans, and sending one
    file is a mistake the picker should not make easy.
    """

    TOOL_NAME = "VFACE"
    # No FILE_INPUTS and no RESULT_KIND: see the class docstring for the first,
    # and for the second, output_kind "files" is the measurement workbooks, the
    # classification, VFACE_report.json and the heat maps, bundled into one .zip
    # and unpacked into the output folder the user picks.
    AUTO_UI = True

    # A cohort of forty patients measured on three regions returns a hundred and
    # twenty surfaces, and loading them all would be worse than useless. The same
    # courtesy AREG's panel offers for the single-pair run.
    MAX_RESULTS_TO_LOAD = 12

    # Pattern -> how to load it, matched by `base_widget._loadResults` against
    # the BASE NAME of every member the archive actually held. Surfaces only, and
    # that is not an oversight: what a run returns besides them is Excel
    # workbooks, which Slicer has no loader for, and a JSON report. The tables
    # ARE the answer -- they are left in the output folder for a spreadsheet to
    # open, and the help text says so.
    #
    # Patterns rather than a "Heat maps" folder name, because the base class
    # never looks at the folder: it reads the archive's member list, precisely so
    # that a second run into the same output folder does not report the first
    # run's files as its own.
    _LOADABLE = (
        ("*.vtk", "model"),
        ("*.vtp", "model"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loadResultsCheckBox = None

    def addExtraWidgets(self, layout) -> None:
        # "if there are any", because there are none unless the run was asked for
        # them: the heat maps are one of the three values of `outputs`, and a
        # measurements-only run returns tables and nothing to draw. A box
        # promising to load what the request did not ask for would be a bug
        # report waiting to happen.
        self._loadResultsCheckBox = qt.QCheckBox(
            _("Load the heat maps into the scene when done, if there are any")
        )
        self._loadResultsCheckBox.setChecked(True)
        layout.addWidget(self._loadResultsCheckBox)

    def handleResult(self, result) -> None:
        """Unpack the archive (base class), then optionally load what it held."""
        super().handleResult(result)

        if not (self._loadResultsCheckBox and self._loadResultsCheckBox.isChecked()):
            return
        self._loadResults()

    # No `_findResults` here, unlike AREG's panel, and its absence is the point:
    # `base_widget._loadResults` does not call one. It matches `_LOADABLE`
    # against the archive's own member list, which is what makes "what did THIS
    # run make" a different question from "what is in this folder". AREG's copy
    # globs the output directory and nothing calls it -- see that module.
