"""AutoMatrix - apply a registration matrix, computed on the tool server.

Replaces the former local module (a Slicer widget driving Automatrix_CLI, plus
a Mirror check box that downloaded a matrix from a GitHub release and typed its
path into the matrix field). Nothing is computed in Slicer any more: the panel
is generated from the server's `GET /tools` entry, the scans and the matrices
go up, the moved files come back. AutoMatrix_Method/ went with it; the unused
Resources/UI/ is all that is left of the hand-written panel.

Three things about AutoMatrix's schema are worth knowing when reading this file:

* **There is no mode, so nothing is hidden.** Every argument is read on every
  run, which is why no `visible_when` appears anywhere in the schema and why
  there is no conditional code here. The sections separate the two paths a user
  must fill in from the things they only sometimes change.
* **The Mirror button is gone and the mirroring is not.** The server's tool
  still applies a matrix whose name contains "mirror" in the scan's own space,
  as upstream does; what has gone is the client downloading `Mirror.zip` to get
  hold of one. A mirroring matrix is now named like any other file.
* **A landmark file is a result too.** AutoMatrix moves `.mrk.json` control
  points as readily as it resamples a volume, so what comes back is a mixture
  and both kinds are offered to the scene.
"""

import json
import logging
import os

import qt
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib import formgen, slicer_io
from ServerToolsCoreLib.base_widget import ServerToolWidgetBase

logger = logging.getLogger("AutoMatrix")


class AutoMatrix(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("AutoMatrix")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["ServerToolsCore"]
        self.parent.contributors = [
            "Gaelle Leroux (UoM)",
            "Lucia Cevidanes (UoM)",
            "Juan Carlos Prieto (UoNC)",
        ]
        self.parent.helpText = _("""
        Applies a registration matrix that AREG, ASO or a mirroring transform already
        produced, on the Automated Dental Tools server. Give it a folder of scans and a
        folder of matrices; each scan is paired with its own matrix by patient key, and
        the moved volumes and landmark files come back. Segmentations are resampled with
        nearest-neighbour so no label is invented.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This module was supported by NIDCR R01 024450.
        """)


class AutoMatrixWidget(ServerToolWidgetBase):
    """Thin GUI: HTTP, async, form generation, styling and lifecycle all live
    in ServerToolsCoreLib. See ARCHITECTURE.md."""

    TOOL_NAME = "AutoMatrix"
    LOAD_RESULTS_LABEL = _("Load the moved scans and landmarks into the scene when done")
    RUN_REPORT = "AutoMatrix_report.json"

    # No FILE_INPUTS. `scans`, `matrices` and `reference` are a packaged tool's
    # `path`, and the client already gives that a picker taking a file OR a
    # folder (client.accepts_folder: "a packaged tool's 'path' always does").
    # All three genuinely take both -- a cohort is a folder, one case is a file,
    # and a single named matrix is deliberately applied to every patient, which
    # is how one mirroring transform serves a whole batch. Naming a mode here
    # would restate the schema at best and contradict it at worst.
    #
    # No RESULT_KIND: output_kind "files" is the input tree rebuilt, one moved
    # file per scan and matrix plus AutoMatrix_report.json, bundled into one
    # .zip and unpacked into the output folder the user picks.

    # A cohort moved through four region matrices legitimately returns dozens
    # of files. Twelve is what AREG and GreedyReg use for the same kind of
    # output.
    MAX_RESULTS_TO_LOAD = 12

    # This module works on CBCTs, so any scan it is given or produces is shown
    # in 3D with this preset -- an input the clinician just picked as much as a
    # result. "" for a module whose data is not a CT-like volume; nothing then
    # happens, and nothing happens anyway for one whose files load as meshes.
    VOLUME_RENDERING = "CT-AAA"

    # Pattern -> how to load it. A moved scan is a VOLUME whether or not it was
    # a segmentation: AutoMatrix resamples a label map, it does not create one,
    # and loading a mask as a segmentation node here would relabel a file the
    # user already has labelled. `AutoMatrix_report.json` is deliberately
    # absent: the report is a `.json` and not a `.mrk.json`, so the markups
    # pattern below never picks it up.
    _LOADABLE = (
        ("*.nii.gz", "volume"),
        ("*.nii", "volume"),
        ("*.nrrd", "volume"),
        ("*.mrk.json", "markups"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mirrorCheckBox = None

    # ------------------------------------------------------------------
    # Panel
    # ------------------------------------------------------------------

    # The mirror matrix, as the tool hosts it. The legacy module fetched
    # Mirror.zip from GitHub into the user's Documents and filled the matrix
    # field with it; here that file is one of AutoMatrix's own hosted test files,
    # so there is nothing to download from anywhere else -- only an entry in the
    # Transforms dropdown that nobody would think to look for.
    _MIRROR_TEST_FILE = "Mirror"

    def addExtraWidgets(self, layout) -> None:
        # A check box rather than a button: it is a state of the run -- "the
        # transform is the mirror matrix" -- not a one-off action, and it reads
        # as one more option beside the others instead of a link floating above
        # Apply.
        self._mirrorCheckBox = qt.QCheckBox(_("Use the mirror matrix"))
        self._mirrorCheckBox.connect("toggled(bool)", self._onMirrorToggled)
        layout.addWidget(self._mirrorCheckBox)


    def configureFields(self) -> None:
        """Grey the mirror box off what the server actually hosts, and keep it
        honest when the user picks a transform themselves.

        Runs on every build, unlike addExtraWidgets: the panel is rebuilt when a
        server that was down comes back, and that is exactly when the hosted list
        goes from empty to populated. The widgets are new each time too, so the
        connection below is made once per widget, not once per panel.
        """
        super().configureFields()
        box = getattr(self, "_mirrorCheckBox", None)
        if box is None:
            return
        widget = self._inputWidgets.get("transforms")
        entries = widget.hosted_entries() if widget is not None else []
        available = any(e.get("name") == self._MIRROR_TEST_FILE for e in entries)
        box.setEnabled(available)
        box.setToolTip(_(
            "Apply the mirror matrix to every patient — reflecting a patient "
            "across the midsagittal plane is how left is compared with right.")
            if available else _(
            "The server does not host the mirror matrix. Fetch it with "
            "setup-testfiles.sh --tool AutoMatrix."))
        if widget is not None:
            # Choosing a transform by hand contradicts the box. Untick it rather
            # than leave a tick claiming the mirror matrix is in a field that now
            # holds something else.
            formgen.connect_changed(widget, self._onTransformsChanged)

    def _onTransformsChanged(self, *_args) -> None:
        """Untick the box when Transforms stops holding the mirror matrix.

        Compared against the field's contents rather than tracked with a "I am
        writing this" flag, because the hosted file is fetched on a BackgroundJob
        (base_widget._onHostedTestFile): the flag would already have been cleared
        by the time the download lands and writes the path, and this handler
        would then untick the box the download was fulfilling.
        """
        box = getattr(self, "_mirrorCheckBox", None)
        if box is None or not box.checked:
            return
        widget = self._inputWidgets.get("transforms")
        path = (widget.currentPath if widget is not None else "") or ""
        if os.path.basename(path.rstrip(os.sep)) != self._MIRROR_TEST_FILE:
            box.setChecked(False)

    def _onMirrorToggled(self, checked: bool) -> None:
        """Both halves of the mirror workflow, on one tick.

        Filling Transforms goes through the same path the dropdown uses --
        download, cache, preview. Ticking "same transform for every patient" is
        the other half and cannot be skipped: a mirror matrix belongs to no
        patient, so no name can pair it, and without that a cohort is refused
        rather than guessed at.
        """
        widget = self._inputWidgets.get("transforms")
        if checked:
            self._onHostedTestFile("transforms", self._MIRROR_TEST_FILE)
        elif widget is not None:
            # Unticking must not leave the mirror matrix sitting in the field,
            # where the next Apply would silently still use it.
            widget.clear()
        share = self._argWidgets.get("same_transform_for_every_patient")
        if share is not None:
            share.setChecked(bool(checked))

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def handleResult(self, result) -> None:
        """Unpack the archive (base class), then report and optionally load.

        The report is the half that is not visible on disk. A scan whose matrix
        never matched produces no file and no error, and the count of skipped
        pairings is the only thing that says so -- which is exactly the case
        where a user would otherwise conclude the tool had run on everything.
        """
        super().handleResult(result)

        outputDir = self._outputFolderWidget.currentPath if self._outputFolderWidget else None
        if not outputDir:
            return

        report = self._readRunReport(outputDir)
        if report:
            slicer.util.showStatusMessage(self._summarize(report), 8000)

        self._maybeLoadResults()

    @staticmethod
    def _summarize(report: dict) -> str:
        skipped = report.get("skipped") or 0
        if skipped:
            return _("AutoMatrix: {written} file(s) written, {skipped} pairing(s) "
                     "skipped - see AutoMatrix_report.json.").format(
                         written=report.get("written") or 0, skipped=skipped)
        return _("AutoMatrix: {written} file(s) written.").format(
            written=report.get("written") or 0)
