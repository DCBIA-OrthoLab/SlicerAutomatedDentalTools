"""
Automated REGistration (AREG) of two timepoints of the same patient, for
cone-beam computed tomography (CBCT) and intraoral surface scans (IOS).

Thin GUI over the remote `AREG` tool. The registration runs on the server, so
nothing is installed into Slicer's interpreter, no conda environment is created,
and neither the segmentation models nor the registration checkpoint are ever
downloaded to this machine.

Replaces the former local module, which drove the AREG_CBCT / AREG_IOS /
AREG_IOSCBCT CLIs from inside Slicer and chained them with AMASSS, ASO, ALI and
CrownSeg by hand. `AREG_Method/` is left in the tree but is no longer wired to
this one.

**There is no `.ui` file here, and that is the design rather than an omission.**
The panel is built from the server's schema by `SADTCoreLib.formgen`, so a
field added to the tool server-side appears here with no client release. The
former module's `Resources/UI/AREG.ui` was kept for a while and then deleted,
because keeping it invited edits that changed nothing: CMakeLists did not ship
it and no code read it, so a UI change made there was a silent no-op -- and one
cost a day of looking for it in the wrong place.

Authors:
- Nathan Hutin (UoM)
- Luc Anchling (UoM)
"""

import glob
import json
import logging
import os
import re

import qt
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from SADTCoreLib.base_widget import ServerToolWidgetBase
from SADTCoreLib.mode_switch import legacy_icon

logger = logging.getLogger("AREG")


# How the intraoral-to-CBCT engine pairs its two sides, mirrored so the panel
# sends only the files of the patient a single file names. The server's
# `sadt_areg_ioscbct.pipeline.patient_key` and `timepoint_of` are the reference:
# the digits of the name once a timepoint word is dropped, and the timepoint
# (T0, T1, T2) the name -- or the folder above it -- gives.
_VOLUME_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")
_SURFACE_EXTENSIONS = (".vtk", ".stl")
_TIMEPOINTS = ("t1", "t2", "t0")
_JAWS = ("u", "up", "upper", "maxilla", "max", "mx",
         "l", "low", "lower", "mandible", "mandibule", "mand", "md")
_SEPARATORS = re.compile(r"[_\-.\s]+")


def _strip_extension(name: str) -> str:
    lowered = name.lower()
    for extension in _VOLUME_EXTENSIONS + _SURFACE_EXTENSIONS:
        if lowered.endswith(extension):
            return name[: -len(extension)]
    return name


def _tokens(stem: str) -> list:
    """Lowercase words, with `UpperT2` split into `upper` and `t2` as the
    server splits it -- otherwise the 2 would count as a patient digit."""
    words = []
    for part in _SEPARATORS.split(stem.lower()):
        if not part:
            continue
        for point in _TIMEPOINTS:
            if part.endswith(point) and part[: -len(point)] in _JAWS:
                words.extend((part[: -len(point)], point))
                break
        else:
            words.append(part)
    return words


def subject_of(relative_path: str) -> tuple:
    """`(patient, timepoint)` of a file, as the server pairs it.

    The timepoint is read off the name first, then off the folders above it;
    None when nothing names one, or when two different ones are named.
    """
    parts = relative_path.replace("\\", "/").split("/")
    stem = _strip_extension(parts[-1])
    words = _tokens(stem)
    digits = "".join(c for word in words if word not in _TIMEPOINTS
                     for c in word if c.isdigit())
    patient = digits.lstrip("0") or digits or stem
    timepoint = None
    for part in [stem] + list(reversed(parts[:-1])):
        named = {word for word in _tokens(part) if word in _TIMEPOINTS}
        if named:
            timepoint = named.pop() if len(named) == 1 else None
            break
    return patient, timepoint


def matching_entries(folder: str, anchor: str, extensions) -> list:
    """The files under `folder` that belong with the single file `anchor`.

    Relative paths, sorted. Same patient, and the same timepoint when both
    name one; files naming no timepoint are taken only when none names the
    anchor's. Raises ValueError, said in the panel's terms, when nothing
    matches or when the anchor cannot tell two timepoints apart.
    """
    patient, timepoint = subject_of(os.path.basename(anchor))
    same = []
    for root, directories, names in os.walk(folder):
        directories.sort()
        for name in sorted(names):
            if name.startswith(".") or not name.lower().endswith(tuple(extensions)):
                continue
            relative = os.path.relpath(os.path.join(root, name), folder).replace(os.sep, "/")
            found_patient, found_point = subject_of(relative)
            if found_patient == patient:
                same.append((relative, found_point))
    if timepoint:
        chosen = [path for path, point in same if point == timepoint]
        chosen = chosen or [path for path, point in same if point is None]
    else:
        points = {point for _path, point in same if point}
        if len(points) > 1:
            raise ValueError(_(
                "'{anchor}' names no timepoint, and the folder holds {points} for "
                "this patient. Put T1 or T2 in its name so the right files are sent."
            ).format(anchor=os.path.basename(anchor),
                     points=" and ".join(sorted(p.upper() for p in points))))
        chosen = [path for path, _point in same]
    if not chosen:
        raise ValueError(_(
            "Nothing in '{folder}' matches '{anchor}'. Files are matched on the "
            "digits of their names and on T1/T2 when the names give one."
        ).format(folder=os.path.basename(os.path.normpath(folder)),
                 anchor=os.path.basename(anchor)))
    return sorted(chosen)


class AREGVisor(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("AREG (VISOR local)")
        self.parent.categories = ["Automated Dental Tools.VISOR local"]
        self.parent.icon = legacy_icon(__file__, "AREG.png")
        self.parent.dependencies = ["SADTCore"]
        self.parent.contributors = [
            "Nathan Hutin (UoM)",
            "Luc Anchling (UoM)",
            "Lucia Cevidanes (UoM)",
            "Juan Carlos Prieto (UoNC)",
        ]
        self.parent.helpText = _("""
        Registers a follow-up scan onto a baseline scan of the same patient, remotely, on the
        Automated Dental Tools server. Give it a T1 folder and a T2 folder; each pair comes back
        with the T2 moved into the T1's frame and the transform that moved it, so growth or
        treatment change can be measured.
        CBCT registration is confined to the anatomy you pick (cranial base, mandible, maxilla);
        intraoral registration uses the palate, which orthodontic treatment does not move.
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was developed by Nathan Hutin (UoM) and Luc Anchling (UoM) and was supported by
        NIDCR R01 024450, AA0F Dewel Memorial Biomedical Research award and by Research
        Enhancement Award Activity 141 from the University of the Pacific,
        Arthur A. Dugoni School of Dentistry.
        """)


class AREGVisorWidget(ServerToolWidgetBase):
    """Thin GUI: everything else (HTTP, async, form generation, styling, lifecycle)
    lives in SADTCoreLib. See ARCHITECTURE.md.

    Nothing about the pipeline lives here. Which anatomical regions can be
    registered on, which modes exist, what each mode additionally needs, and the
    fact that a CBCT run is never asked for an intraoral checkpoint, are all in
    the server's schema and rendered by formgen — so a region added server-side
    appears in this panel with no client release.

    The old local module spent 2574 lines on the same panel, plus three
    `AREG_Method` classes whose job was to build the argument dictionaries for
    five different CLIs and run them in sequence. That sequencing is the
    server's now: AREG calls AMASSS, ASO and CrownSeg in-process, so this module
    makes ONE request where the old one launched up to six processes and polled
    a log file for progress.

    In particular the mode is *asked for*, never guessed: a `.zip` can hold CBCT
    volumes or intraoral meshes, so no extension could tell this module which
    pipeline the user wanted. The same goes for `ios_patch`, which picks the
    region the two timepoints are aligned on and, with it, the arch: the palate
    (predicted, upper) or the mucogingival line (built from landmarks, lower).
    The panel shows the checkpoint for one and the landmark folder for the
    other, and that switch is `visible_when` in the schema, not code here.
    """

    TOOL_NAME = "AREG"
    # What the server runs for each `mode`, as the console names it: "AREG"
    # alone does not say which of the three registrations is talking.
    CONSOLE_NAMES = {"CBCT": "AREG CBCT", "IOS": "AREG IOS", "IOS to CBCT": "AREG IOS-CBCT"}
    # FILE_INPUTS names `patients` only (below): the schema types `t1`, `t2`
    # and `t1_masks` as ["folder", "zip_file"], so "auto" already gives each
    # one a picker that takes a folder and zips it before upload. `t1`/`t2`
    # are also flagged server_selectable, which puts the hosted test cohorts
    # above that picker. Nothing else here names a type, an extension, or a
    # mode.
    #
    # No RESULT_KIND either: output_kind "files" is one registered scan and its
    # transform per case and per region, plus AREG_report.json, bundled into one
    # .zip and unpacked into the output folder the user picks.
    AUTO_UI = True
    # The one input the schema cannot describe precisely: `patients` is a
    # FOLDER of patient folders, never one file, and a packaged tool's path
    # takes either. Forced to the folder picker so the panel does not offer
    # a choice that can only fail.
    FILE_INPUTS = {"patients": "folder_zip"}

    # Loading is a courtesy for the single-pair run: a cohort registered on
    # three regions legitimately returns hundreds of files, and loading them all
    # would be worse than useless.
    MAX_RESULTS_TO_LOAD = 12
    # The CBCT every IOS was registered onto is the scene's frame of reference;
    # without a rendering the 3D view showed the arches floating on nothing.
    # The base class renders a scan only when the run returned exactly one, so
    # a CBCT run handing back both timepoints keeps an empty 3D view.
    VOLUME_RENDERING = "CT-AAA"
    # One colour per landmark set and one per mesh, in the order the files sort.
    # Every markups file loaded in Slicer's default colour, so the orientation
    # landmarks, the CBCT's and each arch's moved copy were one indistinct cloud,
    # and both arches the same grey. Kept apart so a landmark never takes the
    # colour of a mesh it sits on.
    _MARKUPS_COLORS = (
        (0.90, 0.20, 0.20),  # red
        (0.20, 0.75, 0.25),  # green
        (0.20, 0.45, 1.00),  # blue
        (1.00, 0.80, 0.10),  # yellow
        (0.80, 0.30, 0.90),  # purple
        (0.10, 0.80, 0.85),  # cyan
    )
    # The colours a landmark file carries when nobody chose one: the green ALI
    # and ASO write into every `.mrk.json`, and Slicer's own default for a
    # file that names none (a plain `.json`). Only these are replaced; any
    # other colour was set in the file on purpose and is kept.
    _DEFAULT_MARKUPS_COLORS = (
        (0.2667, 0.6745, 0.3922),  # ALI / ASO
        (1.0, 0.5, 0.5),  # Slicer
    )
    _MODEL_COLORS = (
        (0.95, 0.85, 0.65),  # ivory
        (0.65, 0.80, 1.00),  # light blue
        (1.00, 0.70, 0.75),  # pink
        (0.70, 0.95, 0.70),  # light green
    )
    # Pattern -> how to load it. A registered CBCT is a VOLUME, not a
    # segmentation: AREG moves a scan, it does not label one. `*.tfm` is
    # deliberately absent — the transform is there to carry a measurement back
    # onto the original acquisition, and loading it into the scene applies
    # nothing by itself.
    _LOADABLE = (
        ("*.nii.gz", "volume"),
        ("*.nii", "volume"),
        ("*.nrrd.gz", "volume"),
        ("*.nrrd", "volume"),
        ("*.gipl.gz", "volume"),
        ("*.gipl", "volume"),
        ("*.vtk", "model"),
        ("*.vtp", "model"),
        # The landmarks a run hands back when "Registration landmarks" is
        # ticked (each mesh's, moved with it, and the CBCT's it was registered
        # on) or "Orientation landmarks" (the ones ASO oriented the CBCT on).
        # Without these the scene showed the meshes on the CBCT and no point
        # to judge the fit by. `*_lm.json` is the moved copy of a plain-JSON
        # input; AREG_report.json matches neither pattern.
        ("*.mrk.json", "markups"),
        ("*_lm.json", "markups"),
    )

    def consoleName(self, run) -> str:
        return self.CONSOLE_NAMES.get((run.args or {}).get("mode"), self.TOOL_NAME)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loadResultsCheckBox = None

    # The two sides of an intraoral-to-CBCT run, each with the file types of
    # the OTHER side's folder. Either may be one file while the other is a
    # folder: the folder is then narrowed to that file's patient and timepoint.
    _PAIRED_SIDES = {
        "ios": ("cbct", _SURFACE_EXTENSIONS),
        "cbct": ("ios", _VOLUME_EXTENSIONS),
    }

    def _prepareOneInputFile(self, workspace, arg_name: str, mode: str, entries=None):
        """Send only what pairs with a single file on the other side.

        A folder of a whole cohort sent beside one CBCT uploaded every mesh of
        every patient for one registration. The single file may also be a
        node picked from the scene, read by its node name -- the name the base
        class uploads it under.
        """
        sides = self._PAIRED_SIDES.get(arg_name)
        if sides is None or arg_name in self._hiddenArgs:
            return super()._prepareOneInputFile(workspace, arg_name, mode, entries)
        widget = self._inputWidgets.get(arg_name)
        other, extensions = sides
        anchor = self._singleInputName(other)
        folder = getattr(widget, "currentPath", "") if widget else ""
        if not anchor or not folder or not os.path.isdir(folder):
            return super()._prepareOneInputFile(workspace, arg_name, mode, entries)
        wanted = matching_entries(folder, anchor, extensions)
        if entries is not None:
            batch = set(entries)
            wanted = [path for path in wanted if path.split("/", 1)[0] in batch]
        logger.info("Sending %d file(s) of '%s' that match the %s", len(wanted), arg_name, other)
        return self._zipFolder(workspace, arg_name, folder, wanted)

    def _sceneNodeOf(self, widget):
        reader = getattr(widget, "volume_name", None)
        name = reader() if reader else ""
        return self._sceneVolumes.get(name) if name else None

    def _singleInputName(self, arg_name: str) -> str:
        """The name of the ONE file `arg_name` holds, or "" when it holds a
        folder, nothing, or is hidden."""
        if arg_name in self._hiddenArgs:
            return ""
        widget = self._inputWidgets.get(arg_name)
        node = self._sceneNodeOf(widget)
        if node is not None:
            return node.GetName()
        path = getattr(widget, "currentPath", "") if widget else ""
        return os.path.basename(path) if path and os.path.isfile(path) else ""

    def addExtraWidgets(self, layout) -> None:
        self._loadResultsCheckBox = qt.QCheckBox(_("Load the results into the scene when done"))
        self._loadResultsCheckBox.setChecked(True)
        layout.addWidget(self._loadResultsCheckBox)

    def handleResult(self, result) -> None:
        """Unpack the archive (base class), then optionally load what it held."""
        super().handleResult(result)

        if not (self._loadResultsCheckBox and self._loadResultsCheckBox.isChecked()):
            return
        self._loadResults()

    def _showLoadedResults(self, opened: list) -> None:
        """Show the results (base class), each landmark set and mesh in its own
        colour, and a CBCT pair as the pair it is."""
        super()._showLoadedResults(opened)
        self._colorResults(opened)
        self._showRegisteredPair(opened)

    def _showRegisteredPair(self, opened: list) -> None:
        """The T1 under the registered T2, in the slices and in 3D.

        The base class puts the FIRST scan in the slices and renders a scan in
        3D only when there is one, so a CBCT run -- the T1 and the registered
        T2 -- ended on one grey scan and an empty 3D view with the landmarks
        floating in it. Which is which is what the run's `scene.json` says
        (`reference`, `registered`); without one, nothing is guessed.
        Best effort, like every view after a finished run.
        """
        roles = self._sceneRoles()
        volumes = [node for kind, node in opened if kind == "volume" and node is not None]
        registered = [node for node in volumes if roles.get(self._nodePath(node)) == "registered"]
        if not registered:
            return
        reference = next((node for node in volumes
                          if roles.get(self._nodePath(node)) == "reference"), None)
        if reference is None:
            # Oriented first, the T1 travels under `T1_Oriented/` whether or
            # not it was ticked, and the scene only names it when it was.
            reference = next((node for node in volumes if node not in registered
                              and "T1_Oriented" in self._nodePath(node).split(os.sep)), None)
        try:
            slicer.util.setSliceViewerLayers(
                background=reference or registered[0],
                foreground=registered[0] if reference else None,
                foregroundOpacity=0.5, fit=True)
        except Exception as exc:  # noqa: BLE001 - a view is never worth a failure
            logger.warning("Could not show the registered pair: %s", exc)
        for node in ([reference] if reference else []) + registered[:1]:
            self._renderScan(node)

    def _sceneRoles(self) -> dict:
        """{resolved path: role} from every `scene.json` this run produced."""
        roles = {}
        for path in getattr(self, "_producedFiles", None) or ():
            if not path.endswith("scene.json"):
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    listed = json.load(handle).get("files", [])
            except (OSError, ValueError) as exc:
                logger.warning("Could not read %s: %s", os.path.basename(path), exc)
                continue
            folder = os.path.dirname(path)
            for entry in listed:
                if isinstance(entry, dict) and entry.get("path") and entry.get("role"):
                    roles[os.path.realpath(os.path.join(folder, entry["path"]))] = entry["role"]
        return roles

    @staticmethod
    def _nodePath(node) -> str:
        """The resolved file a loaded node was read from, or ""."""
        try:
            storage = node.GetStorageNode()
            return os.path.realpath(storage.GetFileName()) if storage else ""
        except Exception:  # noqa: BLE001
            return ""

    @classmethod
    def _colorResults(cls, opened: list) -> None:
        """Give every markups node and every model a colour of its own.

        Ordered by node name, which is the file's, so a lower arch and its
        landmarks get the same colours on every run. A landmark file that
        carries a colour of its own keeps it: only the default one is replaced
        (see _DEFAULT_MARKUPS_COLORS), so a colour set in the `.mrk.json` reads
        the same here as in any other module. Best effort: a colour is never
        worth failing a run that finished.
        """
        for kind, palette in (("markups", cls._MARKUPS_COLORS), ("model", cls._MODEL_COLORS)):
            nodes = sorted((node for k, node in opened if k == kind and node is not None),
                           key=lambda node: node.GetName())
            for index, node in enumerate(nodes):
                color = palette[index % len(palette)]
                try:
                    display = node.GetDisplayNode()
                    if display is None:
                        continue
                    if kind == "markups" and not cls._hasDefaultColor(display):
                        continue
                    display.SetColor(*color)
                    if kind == "markups":
                        # A markups point is drawn in its SELECTED colour, which
                        # every point is until clicked.
                        display.SetSelectedColor(*color)
                except Exception as exc:  # noqa: BLE001 - never worth failing a run
                    logger.warning("Could not colour %s: %s", node.GetName(), exc)

    @classmethod
    def _hasDefaultColor(cls, display) -> bool:
        """Whether a markups display still shows a colour nobody chose."""
        reader = getattr(display, "GetSelectedColor", None)
        current = tuple(reader()) if reader else ()
        if len(current) != 3:
            return True
        return any(all(abs(a - b) < 0.01 for a, b in zip(current, default))
                   for default in cls._DEFAULT_MARKUPS_COLORS)

    @classmethod
    def _findResults(cls, outputDir: str) -> list:
        """[(path, kind)] for every result with a loader, in the per-region,
        per-case tree the server built."""
        return sorted(
            (path, kind)
            for pattern, kind in cls._LOADABLE
            for path in glob.glob(os.path.join(outputDir, "**", pattern), recursive=True)
        )
