"""ALI - Automatic Landmark Identification, computed on the tool server.

Replaces the former local module (a Slicer widget driving the ALI_CBCT and
ALI_IOS CLIs, a conda `shapeaxi` environment and a WSL detour on Windows).
Nothing is computed in Slicer any more: the panel is generated from the
server's `GET /tools` entry, the data goes up, the landmarks come back. The
old CLI modules are left in the tree but are no longer wired to this one.

Three things about ALI's schema are worth knowing when reading this file:

* **There is no `mode` argument.** The server inspects what the input actually
  holds - volumes, DICOM series or surface meshes - and picks the CBCT or IOS
  engine itself. This module must not guess: a `.zip` can carry either kind,
  so an extension tells you nothing.
* **Both engines' selections are always shown, and one of them is always
  inert.** The engine isn't known until the server has looked at the data, so
  there is no field to hide the other one behind - `visible_when` needs a
  `choice` argument to test, and ALI declares none on purpose. What the schema
  does instead is put each engine's selection in its own collapsible box
  (`section`) and repeat "CBCT only" / "IOS only" in each description, which
  formgen renders as a visible hint. Hiding one by guessing would be wrong more
  often than the clutter is annoying.
* **CBCT landmarks can be chosen by region or one by one.** `cbct_regions` is
  the granularity a human wants; `landmarks` (119 options, rendered as tabs
  from the server's own grouping) is for asking for named points, and naming
  any of them *replaces* the region selection rather than narrowing it. It
  exists because ASO's fully-automated mode registers on seven landmarks that
  straddle two regions, and going through regions would run 58 deep-RL agents
  to use seven.
* **A partial run is normal and is not an error.** A landmark whose weights
  are missing from the bundle, or whose agent never converged, is reported in
  `run_report.json` while every other landmark is still written out. Surfacing
  that report is this module's one real job - see handleResult.
"""

import logging
import os

import qt
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ServerToolsCoreLib import slicer_io
from ServerToolsCoreLib.base_widget import ServerToolWidgetBase
from ServerToolsCoreLib.errors import plain_text

logger = logging.getLogger("ALI")

# Loading a markups node per predicted file is instant for a handful of scans
# and minutes of frozen UI for a cohort. Past this count the user is asked
# rather than made to wait for something they may not have wanted.
_CONFIRM_LOAD_ABOVE = 20


class ALI(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ALI")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["ServerToolsCore"]
        self.parent.contributors = [
            "Maxime Gillot (UoM)",
            "Baptiste Baquero (UoM)",
            "Lucia Cevidanes (UoM)",
            "Juan Carlos Prieto (UoNC)",
        ]
        self.parent.helpText = _("""
        Places anatomical landmarks automatically, on CBCT scans or on intraoral surface scans,
        using models hosted on the Automated Dental Tools server. Give it one scan or a whole
        folder (DICOM series included) and pick a model bundle; the server decides which engine
        applies and returns one Slicer markups file per scan, plus a report of what it could and
        could not find.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was developed by Maxime Gillot (CPE Lyon & UoM) and Baptiste Baquero (CPE Lyon & UoM)
        and was supported by NIDCR R01 024450, AA0F Dewel Memorial Biomedical Research award and by
        Research Enhancement Award Activity 141 from the University of the Pacific, Arthur A. Dugoni
        School of Dentistry.
        """)


class ALIWidget(ServerToolWidgetBase):
    """Thin GUI: HTTP, async, form generation, styling and lifecycle all live
    in ServerToolsCoreLib. See ARCHITECTURE.md."""

    TOOL_NAME = "ALI"
    RUN_REPORT = "run_report.json"

    # The one thing ALI's schema cannot state. `input` is typed
    # ("volume_or_zip_file", "surface_or_zip_file") - both file types, no
    # "folder" - so the schema-driven rule would give it a file picker only.
    # But a cohort, and *any* DICOM series, is a directory: the user picks a
    # folder, base_widget zips it, and the server extracts it. Which of the two
    # was given is read off the path at upload time, never asked.
    FILE_INPUTS = {"input": "file_or_folder"}

    # The CBCT engine takes a scan in, so one picked as input is shown in 3D
    # with this preset. Its own results are markups, loaded by this module's
    # own path below, and an IOS surface loads as a mesh -- neither is a volume,
    # so neither reaches this.
    VOLUME_RENDERING = "CT-AAA"

    # RESULT_KIND is deliberately absent: output_kind is "files", which can
    # only mean "save the archive", and formgen.result_kind_for derives it.

    def __init__(self, parent=None):
        ServerToolWidgetBase.__init__(self, parent)
        self.loadResultsCheckBox = None

    # ------------------------------------------------------------------
    # Panel
    # ------------------------------------------------------------------

    def configureFields(self) -> None:
        """Show the server's own default for `prediction_ID` as a placeholder.

        A placeholder rather than a pre-filled value, on purpose: an empty
        optional field is dropped from the request (see
        ServerToolWidgetBase.collectArgs), so the default stays written down
        once, server-side, instead of being duplicated here where the two
        copies would drift.
        """
        field = self._argWidgets.get("prediction_ID")
        if field is not None:
            field.setPlaceholderText("Pred")

    def addExtraWidgets(self, layout) -> None:
        self.loadResultsCheckBox = qt.QCheckBox(_("Load the predicted landmarks into the scene"))
        self.loadResultsCheckBox.setChecked(True)
        layout.addWidget(self.loadResultsCheckBox)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def handleResult(self, result) -> None:
        """Unpack the archive, load what was predicted, and report what wasn't.

        Overriding rather than extending the base "save_as" handling: its info
        dialog would pop before the report has even been read, and the report
        is the whole point - a landmark missing from the scene means one of two
        very different things, and only `run_report.json` says which.
        """
        resultDir = os.path.dirname(result.path)
        # Normally unpacked already, off the main thread, before this is called
        # (base_widget._unpackedOnWorker); then the archive is gone.
        if getattr(result, "extracted", None) is None and slicer_io.is_extractable_archive(result.path):
            slicer_io.unzip_folder(result.path, resultDir)
            os.remove(result.path)

        report = self._readRunReport(resultDir)
        markupFiles = self._findMarkupFiles(resultDir)
        loaded = self._loadMarkups(markupFiles)

        slicer.util.infoDisplay(plain_text(self._summarize(resultDir, report, len(markupFiles), loaded)))

    @staticmethod
    def _findMarkupFiles(resultDir: str) -> list:
        """Every predicted markups file, in the per-scan tree the server built."""
        found = []
        for root, _dirs, names in os.walk(resultDir):
            found.extend(
                os.path.join(root, name) for name in sorted(names) if name.endswith(".mrk.json")
            )
        return sorted(found)

    def _loadMarkups(self, paths: list) -> int:
        """Load the markups files into the scene; return how many made it.

        A file that fails to load is logged and skipped rather than aborting
        the rest: one unreadable result must not cost the user the other 199.
        """
        if not paths or not (self.loadResultsCheckBox and self.loadResultsCheckBox.isChecked()):
            return 0

        if len(paths) > _CONFIRM_LOAD_ABOVE and not slicer.util.confirmYesNoDisplay(
            _("This run produced {count} landmark files. Load them all into the scene?").format(
                count=len(paths)
            )
        ):
            return 0

        loaded = 0
        for path in paths:
            try:
                slicer.util.loadMarkups(path)
                loaded += 1
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)
        return loaded

    @staticmethod
    def _summarize(resultDir: str, report, produced: int, loaded: int) -> str:
        """The end-of-run message, built around what did NOT work.

        The successes are visible in the scene and on disk; the failures exist
        only in the report, and a landmark silently absent reads as a bug in
        the tool. Both failure kinds are named, and they are kept apart because
        the fix differs: a missing model means "use another bundle", a failed
        search means "this scan is hard".
        """
        lines = []

        if report:
            mode = report.get("mode") or "?"
            selection = report.get("regions") or report.get("networks") or []
            line = _("ALI finished in {mode} mode.").format(mode=mode)
            if selection:
                line += " " + _("Selection: {selection}.").format(selection=", ".join(selection))
            lines.append(line)

            # `cases` is what the tools write; `scans` was the older name.
            scans = report.get("cases") or report.get("scans") or {}
            found_total = sum(len(scan.get("landmarks_found") or []) for scan in scans.values())
            lines.append(
                _("{scans} scan(s), {found} landmark(s) written.").format(
                    scans=len(scans), found=found_total
                )
            )

            failed = {
                key: sorted((scan.get("landmarks_failed") or {}).keys())
                for key, scan in scans.items()
                if scan.get("landmarks_failed")
            }
            if failed:
                lines.append("")
                lines.append(_("Did not converge (the scan, not the model):"))
                lines.extend(f"  {key}: {', '.join(names)}" for key, names in sorted(failed.items()))

            without_model = report.get("landmarks_without_model") or []
            if without_model:
                lines.append("")
                lines.append(
                    _("Not in the selected model bundle: {landmarks}").format(
                        landmarks=", ".join(without_model)
                    )
                )
        else:
            lines.append(
                _("ALI finished. {produced} landmark file(s) produced.").format(produced=produced)
            )
            lines.append(_("No run report was found in the results."))

        lines.append("")
        if loaded:
            lines.append(_("{loaded} landmark file(s) loaded into the scene.").format(loaded=loaded))
        lines.append(_("Results saved to {path}").format(path=resultDir))
        return "\n".join(lines)
