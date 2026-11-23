"""What a folder of results holds, grouped so a reader can step through it.

The panel's job is to put a scan on screen with the landmarks, masks and
surfaces that belong to it. Nothing in this extension could answer which file
belongs to which, so this module does: it walks the folders it is given,
groups their files by patient, and says which anchor each overlay may be drawn
against.

**Why the grouping cannot be `os.listdir` and a sort.** Measured against the
tools as they are written today, four naming conventions coexist:

* `ALI`, `Crown_Seg` and `FlexReg` APPEND to the input stem, so
  `P1_scan.nii.gz` becomes `P1_scan_lm_Pred.mrk.json`;
* `ASO` CBCT TRUNCATES it -- the same input becomes `P1_Or.nii.gz` and
  `P1_lm_Or.mrk.json`, sharing no stem at all with ALI's output;
* `ASO` IOS writes its surface, its landmarks and its transform under THREE
  different stems, sharing only a patient-and-jaw key;
* `Crown_Seg` files half a batch under an extra `crownseg_input_<suffix>/`
  level and the other half beside it, and `AMASSS` invents a
  `<stem>_<id>_SegOut/` folder per scan.

Truncating at a known tool suffix, on token boundaries, is what reconciles the
first two -- and it is what `ASO` itself already does to read `ALI`'s output
back. `_normalise_directory` handles the last one.

**This is a third copy of an algorithm, deliberately.** The reference lives in
`SADT-VISOR` at `tools/AREG/common/src/sadt_areg_common/pairing.py`, and `ASO`
carries a second, older copy whose suffix table is missing `_Seg` -- so the two
already disagree on `arch_Seg.vtk`. A client cannot import either: they are
inside tool virtualenvs in another repository. The copy ends the day a run
publishes a manifest naming its own outputs, which is the only source that can
be right by construction rather than by guessing at names.
"""

import json
import os
import re

# Longest first: `.nii.gz` must never resolve as `.gz`, and `.nrrd.gz` never as
# `.gz` either. The list is the union of what the tools read and write.
SCAN_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")
SURFACE_EXTENSIONS = (".vtk", ".vtp", ".stl", ".obj", ".off")
MARKUPS_EXTENSIONS = (".mrk.json",)
# ASO IOS's landmarks are a BARE `.json` on the hosted test data
# (`Upper_new_9_Upper_O_Pred.json`), so extension alone under-counts them and
# over-counts every other json in the folder. The head of the file settles it.
_MARKUPS_AMBIGUOUS_EXTENSION = ".json"
_MARKUPS_MARKER = '"markups"'
_MARKUPS_SNIFF_BYTES = 512
TRANSFORM_EXTENSIONS = (".tfm", ".h5")

# What a file IS, in the vocabulary `slicer_io` already loads by.
VOLUME = "volume"
LABELMAP = "labelmap"
MODEL = "model"
MARKUPS = "markups"
TRANSFORM = "transform"

# An anchor is something an overlay can be drawn ON. A markups file has no
# geometry of its own and a transform is not viewable at all.
ANCHOR_KINDS = (VOLUME, LABELMAP, MODEL)
# What a case can be BUILT on: something with a shape of its own. A labelmap
# is not here, and that is the difference -- it anchors a view when nothing
# else does, but it is derived from a scan and belongs to one.
GEOMETRY_KINDS = (VOLUME, MODEL)
OVERLAY_KINDS = (MARKUPS, LABELMAP, MODEL)

# Truncated at the FIRST of these that ends on a token boundary. The table is
# the superset from AREG's `pairing.py`; `_Seg`/`_seg` are the two entries ASO's
# own copy lacks, and leaving them out is what makes `arch_Seg.vtk` and
# `arch_Seg_lm_Pred.mrk.json` look like two different patients.
PATIENT_SUFFIXES = (
    "_lm_Pred", "_Scanreg", "_MERGED", "_OutReg", "_SegOr",
    "_scan", "_Scan", "_Seg", "_seg", "_Or", "_OR", "_lm",
)

# Dropped from a stem only when the caller asks. The viewer does NOT: two
# timepoints of one patient are two scans to step through, and collapsing them
# would hide one behind the other.
TIMEPOINT_TOKENS = ("t1", "t2", "t0")

# A scan whose stem carries one of these is shown as labelled voxels rather
# than greyscale. Whole tokens, never substrings: `"max" in name` makes a
# patient called MAX_01 a maxillary mask.
#
# This disagrees with AutoCrop3D on purpose. That tool declares its output a
# volume even when it cropped a segmentation, because a cropped segmentation is
# still whatever the caller sent; here the question is only how to draw it, and
# `P1_seg_cropped.nii.gz` is more useful drawn as labels.
MASK_TOKENS = ("mask", "seg", "segmentation", "pred", "label", "labels", "merged")

# Skipped whole. `.amasss_work/` and `.batchdentalseg_work/` are scratch
# directories created INSIDE the output folder and removed in a `finally` -- so
# they are only ever present after a run was killed, holding nnUNet
# intermediates named `p_000_0000.nii.gz` that are not anybody's patient.
_HIDDEN_PREFIX = "."

# Every tool writes one, none of them is viewable, and `run_report.json` would
# otherwise index as a patient called `run`.
_REPORT_NAMES = ("run_report.json",)
_REPORT_SUFFIX = "_report.json"

# A DICOM series is a DIRECTORY of single-slice files -- 577 of them for one
# scan on the hosted ASO fixture -- and Slicer reads the whole series from the
# folder. Indexed as one volume named after the folder, which is also the only
# place that scan's patient name appears: the slices are `IMG0375.dcm`.
_DICOM_EXTENSION = ".dcm"

_SEPARATORS = re.compile(r"[_\-.\s]+")
_SEPARATOR = re.compile(r"[_\-.\s]")

# `crownseg_input_<suffix>/` for the meshes Crown_Seg segmented itself, nothing
# for the ones it passed through -- so one batch lands at two depths. It names
# no patient: the files inside keep their own stems.
_EXTRA_LEVELS = (re.compile(r"^crownseg_input(_.+)?$"),)

# AMASSS is the other way round: `<scan stem>_<prediction_ID>_SegOut/` is the
# ONLY place the scan's name survives whole. Inside it every file is
# `<stem>_<prediction_ID>_<CODE>`, and `prediction_ID` is free text a caller
# passes -- so no table of suffixes can strip it off a file name, while the
# folder gives it up to a greedy match.
_DECLARING_LEVEL = re.compile(r"^(?P<base>.+)_.+_SegOut$")

# What a stopped run hands back: `intermediate/<NN>_<Tool>/`, one level per
# step, nested when a step was itself a chain (`01_ASO/intermediate/01_ALI_CBCT/`).
# Each step writes what it made and nothing else -- Crown_Seg the meshes,
# ALI_IOS the landmarks on them -- so a patient is spread across the steps,
# and filed by step it was two cases that each lacked half of it. Both levels
# name a step, never a patient, and go. A numbered level only counts right
# under `intermediate/`, which keeps a patient folder called `01_P1` a patient.
_STEPS_LEVEL = "intermediate"
_STEP_LEVEL = re.compile(r"^\d+_[A-Za-z0-9][A-Za-z0-9_]*$")

# A folder that says what KIND of file is in it rather than whose it is:
# `<root>/CBCT/` beside `<root>/Landmarks/` is one cohort filed by role, not
# two cohorts. Stripped from the case key so a patient's scan and its points
# are one case -- and remembered on the artifact, because whether two files
# sit in the same real directory is what decides whether the points may be
# drawn on that scan without a caveat.
ROLE_TOKENS = frozenset((
    "cbct", "ios", "scan", "scans", "volume", "volumes", "surface", "surfaces",
    "mesh", "meshes", "landmark", "landmarks", "markup", "markups",
    "mask", "masks", "segmentation", "segmentations", "seg", "segs",
    "transform", "transforms", "output", "outputs", "result", "results",
    # A bare `T1/` or `T2/` level too. It looks like a timepoint and it is
    # one, but the timepoint is already in the stems under it -- the hosted
    # AREG fixture files `P_0001_T2.nii.gz` under `T2/` -- so the level adds
    # nothing and splits the case away from its landmarks. Where two
    # timepoints DID share a stem they become one case holding both, which a
    # viewer shows as two views of one subject rather than as two subjects.
    "t0", "t1", "t2",
))


def is_role_level(part: str) -> bool:
    """Is this directory named after a KIND of file rather than after a subject?

    Every token has to be one, which is what keeps a patient out. The hosted
    AREG fixture is why it is not a list of exact names: it files one case
    under `CBCT Landmarks/`, `IOS Landmarks/` and `T2/`, and matching
    `Landmarks` alone left the same patient indexed three times.

    A level is only a role when EVERY token is one, so `patient_T1/` stays a
    patient while `T1/` does not.
    """
    parts = tokens(part)
    return bool(parts) and all(token.lower() in ROLE_TOKENS for token in parts)


def split_extension(filename: str) -> tuple:
    """`("P1_scan", ".nii.gz")`, compound extensions kept whole."""
    lowered = filename.lower()
    for extension in (
        MARKUPS_EXTENSIONS + SCAN_EXTENSIONS + SURFACE_EXTENSIONS + TRANSFORM_EXTENSIONS
    ):
        if lowered.endswith(extension) and len(filename) > len(extension):
            return filename[: -len(extension)], filename[-len(extension):]
    stem, extension = os.path.splitext(filename)
    return stem, extension


def tokens(stem: str) -> tuple:
    """The stem's words, separators dropped: `P1_scan_lm` -> `("P1","scan","lm")`."""
    return tuple(part for part in _SEPARATORS.split(stem) if part)


def _token_aligned_index(stem: str, suffix: str) -> int:
    """Where `suffix` starts in `stem`, or -1 if it never sits on a boundary.

    A plain `find` is what collapsed `P_Seg1_T1` and `P_Seg2_T1` onto one
    patient upstream: `_Seg` matches inside `_Seg1`, two subjects become one,
    and one of them is silently lost. The match has to END on a separator or at
    the end of the stem; every entry of `PATIENT_SUFFIXES` already begins with
    one, which is what settles the other side.
    """
    start = 0
    while True:
        index = stem.find(suffix, start)
        if index < 0:
            return -1
        end = index + len(suffix)
        if end == len(stem) or _SEPARATOR.match(stem[end]):
            return index
        start = index + 1


def _drop_tokens(stem: str, unwanted) -> str:
    kept = [token for token in tokens(stem) if token.lower() not in unwanted]
    return "_".join(kept) if kept else stem


def patient_stem(filename: str, drop_timepoint: bool = False) -> str:
    """The patient a file is about, whichever tool named it.

    `P1_scan.nii.gz`, `P1_scan_lm_Pred.mrk.json`, `P1_Or.nii.gz` and
    `P1_Pred_MERGED.nii.gz` all answer `P1`.
    """
    stem, _ = split_extension(filename)
    cut = len(stem)
    for suffix in PATIENT_SUFFIXES:
        index = _token_aligned_index(stem, suffix)
        if 0 <= index < cut:
            cut = index
    # Never truncate to nothing: a file literally named `_Or.nii.gz` is a
    # patient whose name we cannot read, not a patient with no name.
    trimmed = stem[:cut] or stem
    return _drop_tokens(trimmed, TIMEPOINT_TOKENS) if drop_timepoint else trimmed


def kind_of(filename: str, path: str = ""):
    """Which Slicer node the file opens as, or None for one that opens as none.

    `path` is only read for a bare `.json`, where the extension does not say.
    """
    lowered = filename.lower()
    if lowered.endswith(MARKUPS_EXTENSIONS):
        return MARKUPS
    if lowered.endswith(_MARKUPS_AMBIGUOUS_EXTENSION):
        return MARKUPS if path and looks_like_markups(path) else None
    if lowered.endswith(SURFACE_EXTENSIONS):
        return MODEL
    if lowered.endswith(TRANSFORM_EXTENSIONS):
        return TRANSFORM
    if lowered.endswith(SCAN_EXTENSIONS):
        stem, _ = split_extension(filename)
        lowered_tokens = {token.lower() for token in tokens(stem)}
        return LABELMAP if lowered_tokens & set(MASK_TOKENS) else VOLUME
    return None


def looks_like_markups(path: str) -> bool:
    """Does this `.json` open as landmarks? Read the head rather than guess.

    Cheap -- a few hundred bytes -- and it is the difference between showing
    ASO IOS's points and showing none of them, its writer having dropped the
    `.mrk` half of the extension on the data we have.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(_MARKUPS_SNIFF_BYTES)
    except OSError:
        return False
    return _MARKUPS_MARKER.encode() in head


def _is_report(filename: str) -> bool:
    return filename in _REPORT_NAMES or filename.endswith(_REPORT_SUFFIX)


def _normalise_directory(relative: str, root_name: str = "") -> tuple:
    """`(directory, declared patient or "")` for a walked path.

    Four kinds of level are removed. The steps of a stopped run
    (`intermediate/01_ASO/`) say which step wrote a file, not whose it is. A ROLE level (`CBCT/`, `Landmarks/`,
    `IOS Landmarks/`) is how a reader files a cohort, not who is in it. The other two a tool
    invented, and they are removed for opposite reasons. Crown_Seg's carries nothing -- dropping it is what stops
    a mesh it segmented from indexing two directories away from one it passed
    through. AMASSS's carries EVERYTHING: the scan's stem is in the folder name
    and nowhere else recoverable, its files being `<stem>_<prediction_ID>_<CODE>`
    with a free-text id in the middle.
    """
    parts = [part for part in relative.split(os.sep) if part not in ("", ".")]
    kept, declared = [], ""
    for position, part in enumerate(parts):
        if part == _STEPS_LEVEL:
            continue
        # The level above the first is the folder that was opened, so opening
        # `intermediate/` itself still reads its children as steps.
        above = parts[position - 1] if position else root_name
        if above == _STEPS_LEVEL and _STEP_LEVEL.match(part):
            continue
        if is_role_level(part):
            continue
        if any(pattern.match(part) for pattern in _EXTRA_LEVELS):
            continue
        match = _DECLARING_LEVEL.match(part)
        if match:
            declared = match.group("base")
            continue
        kept.append(part)
    return os.sep.join(kept), declared


def is_token_prefix(prefix: str, stem: str) -> bool:
    """Is `prefix` the start of `stem`, ending on a token boundary?

    What pairs `Upper_new_9_Upper_O_Pred.json` with `Upper_new_9.vtk` -- ASO
    IOS being the one producer whose landmark stem is not its surface stem.
    The boundary is what keeps patient `P1` off patient `P10`, which is the
    substring match that paired them upstream.
    """
    if not prefix or len(prefix) >= len(stem):
        return False
    return stem.startswith(prefix) and bool(_SEPARATOR.match(stem[len(prefix)]))


class Artifact:
    """One file on disk, and what is known about it without opening it."""

    __slots__ = ("path", "kind", "source", "directory", "origin", "name", "patient")

    def __init__(self, path: str, kind: str, source: str, directory: str,
                 patient: str, origin: str = ""):
        self.path = path
        self.kind = kind
        self.source = source
        # Where the case is filed: role levels stripped, so a scan and its
        # landmarks meet.
        self.directory = directory
        # Where the file really is. Two artifacts share it only when they were
        # written side by side, which is the evidence `View` needs and the one
        # thing `directory` can no longer answer.
        self.origin = origin if origin else directory
        self.name = os.path.basename(path)
        self.patient = patient

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<Artifact {self.name} {self.kind} from {self.source}>"


class View:
    """One anchor and the overlays that are in ITS frame.

    A case can hold several: `ASO` writes an oriented scan beside its
    landmarks, and the ORIGINAL scan is still in the input folder. Both are
    worth looking at and they are not the same picture -- so they are two
    views, never one anchor with everybody's points on it.
    """

    # `basis` is reassigned when a view gains overlays it did not start with,
    # so the two travel together and a caller cannot read one without the other.
    __slots__ = ("anchor", "overlays", "basis")

    def __init__(self, anchor, overlays, basis: str):
        self.anchor = anchor
        self.overlays = list(overlays)
        self.basis = basis

    @property
    def label(self) -> str:
        return self.anchor.name if self.anchor is not None else "(no scan)"


# What `View.basis` says, and it is shown to the reader rather than kept here.
#
# The distinction is the whole reason this class exists. ASO's landmarks carry
# the recentring AND the ICP rotation, so they match the scan ASO wrote and not
# the one it read; ALI writes no scan at all, so its landmarks can only be
# drawn on the input -- where they are correct. Draw either on the other and
# the result renders without an error and is wrong by a rotation, which is
# exactly the mistake a reviewer is looking at the screen to catch.
BASIS_COLOCATED = "written beside this scan"
BASIS_ACQUISITION = "assumed: the acquisition"
BASIS_NONE = "no scan found"
BASIS_SCENE = "declared by the tool: one space"
BASIS_NAMED = "named after this scan"
BASIS_COLOCATED_AND_NAMED = "written beside or named after this scan"

# The label of a source a caller adds only so landmarks with no scan of their
# own have one to land on -- a run's inputs, beside what one of its steps
# wrote. Such a source never makes a case and never shows a scan nothing is
# drawn on: what the reader opened is the result, not the inputs.
SUPPORT = "inputs"

# The file a tool writes to say which of its outputs make one scene, in one
# space (SADT-VISOR, `sadt_areg_common/scene.py`). `scene.json` when a folder
# holds one scene, `<prefix>_scene.json` when several patients share it.
SCENE_SUFFIX = "scene.json"
# What a scene's `kind` may be, in this module's vocabulary. A transform is
# claimed by its scene like the rest, but is not something to draw.
_SCENE_KINDS = {"volume": VOLUME, "labelmap": LABELMAP, "model": MODEL,
                "markups": MARKUPS, "transform": TRANSFORM}


class Case:
    """Everything one patient has, across every folder that was indexed."""

    __slots__ = ("key", "artifacts", "scene")

    def __init__(self, key: str, scene: bool = False):
        self.key = key
        self.artifacts = []
        # True when a tool's `scene.json` named these files: they share one
        # space by the tool's own word, so the case is ONE view and nothing
        # is guessed from names or folders.
        self.scene = scene

    @property
    def patient(self) -> str:
        return os.path.basename(self.key)

    @property
    def label(self) -> str:
        """How the case reads to a person.

        `key` is an identity and is built to be unique -- it carries the
        subfolder, because two patients of the same name in two folders are
        two patients. That makes it a path, and a path read as a name is
        noise: what a reader wants first is WHO, with where it was found as
        context behind it.
        """
        folder = os.path.dirname(self.key)
        return f"{self.patient}  ({folder})" if folder else self.patient

    def of_kind(self, *kinds) -> list:
        return [artifact for artifact in self.artifacts if artifact.kind in kinds]

    def views(self, acquisition: str = "") -> list:
        """The pictures this case can honestly produce.

        One per anchor, holding the overlays written beside it. Overlays with
        no anchor in their own folder fall back to the acquisition's scan and
        say so -- that is `ALI`, whose landmark file is alone in its directory
        and belongs to the scan the caller sent.
        """
        if self.scene:
            return [self._scene_view()]
        every_anchor = self.of_kind(*ANCHOR_KINDS)
        # A support scan anchors only what is NAMED after it, below.
        anchors = [artifact for artifact in every_anchor if artifact.source != SUPPORT]
        overlays = self.of_kind(*OVERLAY_KINDS)
        # A labelmap and a surface are both: they anchor a view of their own and
        # they are drawn on the greyscale scan beside them. Only a scan is never
        # somebody else's overlay.
        volumes = [artifact for artifact in anchors if artifact.kind == VOLUME]

        views = []
        placed = set()
        if not volumes:
            # Masks and no scan beside them: AMASSS at a stop, which writes the
            # labels and nothing else. They go on the scan they are named
            # after -- `P1_T1_Or_seg_CBMASK` on the `P1_T1_Or` an earlier step
            # oriented -- or a reader judged a segmentation on black.
            masks = [artifact for artifact in anchors if artifact.kind == LABELMAP]
            scans = [artifact for artifact in every_anchor
                     if artifact.source == SUPPORT and artifact.kind == VOLUME]
            if masks and scans:
                unplaced = self._place_by_name(masks, scans, views)
                placed.update(id(mask) for mask in masks if mask not in unplaced)
                anchors = [artifact for artifact in anchors if id(artifact) not in placed]
        for anchor in volumes or anchors:
            beside = [
                artifact for artifact in overlays
                if artifact is not anchor
                and (artifact.source, artifact.origin) == (anchor.source, anchor.origin)
            ]
            placed.update(id(artifact) for artifact in beside)
            views.append(View(anchor, beside, BASIS_COLOCATED))

        anchored = {id(view.anchor) for view in views}
        homeless = [
            artifact for artifact in overlays
            if id(artifact) not in placed and id(artifact) not in anchored
            and artifact.source != SUPPORT
        ]
        homeless = self._place_by_name(homeless, every_anchor, views)
        if not homeless:
            return views

        fallback = next(
            (view for view in views if view.anchor.source == acquisition), None
        ) or (views[0] if views else None)
        if fallback is None:
            return views + [View(None, homeless, BASIS_NONE)]
        if not fallback.overlays:
            # The same picture, so the same view. Appending a second one on
            # the same anchor put the points behind a drop-down the reader had
            # no reason to open -- and the empty view came first.
            fallback.overlays.extend(homeless)
            fallback.basis = BASIS_ACQUISITION
        else:
            # That anchor already carries overlays written beside it, and
            # these were not. Two claims about the frame cannot share a line,
            # so they do not share a view.
            views.append(View(fallback.anchor, homeless, BASIS_ACQUISITION))
        return views


    @staticmethod
    def _place_by_name(homeless, anchors, views) -> list:
        """Put each overlay on the scan its file is named after; return the rest.

        A step that writes only landmarks names them after the file it read:
        ALI_IOS turns `P1_U_Seg.vtk`, which Crown_Seg wrote one step earlier,
        into `P1_U_Seg_lm_Pred.mrk.json`. The longest stem that prefixes the
        overlay's on a token boundary is that file, and it settles what no
        folder can when a stopped run spreads a patient across its steps:
        `P1_U.vtk`, the mesh the caller sent, is a prefix too and is the wrong
        frame for nothing here, but it is not what the points were placed on.

        Added to the anchor's own view when it has one: a name is as good a
        ground as a shared folder, and the view's basis says which held.
        """
        stems = [(split_extension(anchor.name)[0], anchor) for anchor in anchors]
        named = {}
        rest = []
        for overlay in homeless:
            stem = split_extension(overlay.name)[0]
            # Ties go to a step's output over the reader's own file: a cohort
            # sent already named `P1_T1_Or` meets the `P1_T1_Or` ASO wrote, and
            # what a later step read is the one it wrote.
            match = max(((anchor_stem, anchor) for anchor_stem, anchor in stems
                         if anchor is not overlay and is_token_prefix(anchor_stem, stem)
                         and not _ran_inside(overlay, anchor)),
                        key=lambda pair: (len(pair[0]), _is_step_output(pair[1])),
                        default=None)
            if match is None:
                rest.append(overlay)
            else:
                named.setdefault(id(match[1]), (match[1], []))[1].append(overlay)
        for anchor, placed in named.values():
            same = next((view for view in views if view.anchor is anchor), None)
            if same is None:
                views.append(View(anchor, placed, BASIS_NAMED))
                continue
            # One scan, one view. A name says the points were placed on this
            # very file, which is as strong as having been written beside it,
            # so they are drawn together: at an AREG IOS-to-CBCT stop the
            # oriented CBCT carries ASO's points beside it and ALI_CBCT's
            # named after it, and split in two the landmarks the registration
            # uses sat in a third view nobody opened.
            same.basis = BASIS_NAMED if not same.overlays else BASIS_COLOCATED_AND_NAMED
            same.overlays.extend(placed)
        return rest

    def _scene_view(self):
        """Everything on one anchor: the scan when there is one, else the
        first surface. The tool said they share a space, so they share a view."""
        volumes = self.of_kind(VOLUME)
        anchors = volumes or self.of_kind(*ANCHOR_KINDS)
        anchor = anchors[0] if anchors else None
        overlays = [artifact for artifact in self.artifacts if artifact is not anchor]
        return View(anchor, overlays, BASIS_SCENE if anchor else BASIS_NONE)


def _ran_inside(overlay, anchor) -> bool:
    """Whether `overlay` came from a step nested inside the one that wrote `anchor`.

    Such a file was written BEFORE the anchor and cannot be in its frame: at a
    stop after ASO, ALI_CBCT's points (`01_ASO/intermediate/01_ALI_CBCT/`) were
    placed on the scan the reader sent, never on the oriented one ASO wrote
    afterwards (`01_ASO/`) -- even when both scans carry one name.
    """
    outer = (anchor.origin or "").replace("\\", "/").rstrip("/")
    inner = (overlay.origin or "").replace("\\", "/")
    return bool(outer) and _is_step_output(anchor) and inner.startswith(outer + "/")


def _is_step_output(artifact) -> bool:
    """Whether a stopped run's step wrote this, rather than the reader."""
    return _STEPS_LEVEL in (artifact.origin or "").replace("\\", "/").split("/")


def is_scene(filename: str) -> bool:
    return filename == SCENE_SUFFIX or filename.endswith("_" + SCENE_SUFFIX)


def read_scene(path: str):
    """`[(relative path, kind)]` for the files a scene lists, or
    None when the file is not a scene this module can read -- the folder is
    then indexed as if it held none.

    A path may climb (`../`): a CBCT run lists the oriented T1 every region
    shares, one folder up. Whether it stays inside the folder that was opened
    is `_scene_cases`' check, the one place that knows that folder."""
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        listed = payload["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    found = []
    for entry in listed if isinstance(listed, list) else ():
        if not isinstance(entry, dict):
            continue
        relative, kind = entry.get("path"), _SCENE_KINDS.get(entry.get("kind"))
        if isinstance(relative, str) and kind and not os.path.isabs(relative):
            found.append((relative.replace("\\", "/"), kind))
    return found


def _scene_cases(directory: str, filenames, label: str, relative: str, raw: str,
                 root: str) -> tuple:
    """The cases this directory's scene files declare, and the file names
    they claimed. A listed file that is missing is skipped, not invented, and
    so is one outside `root`: a scene may point at a sibling folder of the
    same result, never at the rest of the disk."""
    cases, claimed = [], set()
    inside = os.path.join(os.path.realpath(root), "")
    for filename in sorted(name for name in filenames if is_scene(name)):
        listed = read_scene(os.path.join(directory, filename))
        if listed is None:
            continue
        claimed.add(filename)
        prefix = filename[: -len(SCENE_SUFFIX)].rstrip("_")
        patient = prefix or (os.path.basename(relative) if relative else os.path.basename(root))
        parent = relative if prefix else os.path.dirname(relative)
        key = os.path.join(parent, patient) if parent else patient
        case = Case(key, scene=True)
        for path, kind in listed:
            full = os.path.normpath(os.path.join(directory, *path.split("/")))
            if not os.path.realpath(full).startswith(inside):
                continue
            if "/" not in path:
                claimed.add(path)
            if kind == TRANSFORM or not os.path.isfile(full):
                continue
            case.artifacts.append(Artifact(full, kind, label, relative, patient, origin=raw))
        if case.artifacts:
            cases.append(case)
    return cases, claimed


# What a case with no directory of its own is called in the picker. A blank
# chip is unclickable and reads as a bug.
AT_THE_TOP = "(at the top)"


def folder_of(case) -> str:
    """The first level under the folder that was opened, or `AT_THE_TOP`.

    What a reader picks between when one folder holds several cohorts: the
    hosted ASO fixture is eight subjects across `CBCT_SemiAuto/`,
    `IOS_FullyAuto/` and four others, and wanting to look at the CBCT ones is
    not a reason to open a different folder.

    The FIRST level and not the whole path: deeper levels are how a tool
    mirrors an input tree, and offering one chip per patient would be a
    second case list.
    """
    directory = os.path.dirname(case.key)
    return directory.split(os.sep)[0] if directory else AT_THE_TOP


def folders_in(cases) -> list:
    """The levels those cases came from, in the order they are listed."""
    found = []
    for case in cases:
        name = folder_of(case)
        if name not in found:
            found.append(name)
    return found


def build(sources, drop_timepoint: bool = False, supports=(), focus=None) -> list:
    """Index `sources` -- `[(label, root), ...]` -- into cases, ordered by key.

    `supports` are paths -- folders or single files -- indexed under the
    SUPPORT label: a case is never made of them alone, and their scans only
    carry overlays named after them (see `Case._place_by_name`).

    `focus` is a directory inside one of the sources: what a stopped run's
    LAST step wrote. Everything outside it becomes support too, so a stop
    shows what that step changed -- the meshes Crown_Seg made, the points
    ALI_CBCT placed -- on whatever earlier file it needs to be seen on, and
    not every file every earlier step left behind.

    `label` is how the reader named the folder ("Scans", "Results"); it travels
    with every artifact so a view can say which folder its anchor came from,
    and so the acquisition can be preferred when an overlay has no anchor of
    its own.
    """
    cases = {}
    every = list(sources) + [(SUPPORT, path) for path in supports or ()]
    for label, root in every:
        if root and os.path.isfile(root):
            _index_file(cases, label, root, drop_timepoint)
            continue
        if not root or not os.path.isdir(root):
            continue
        for directory, subdirectories, filenames in os.walk(root):
            subdirectories[:] = sorted(
                name for name in subdirectories if not name.startswith(_HIDDEN_PREFIX)
            )
            raw = os.path.relpath(directory, root)
            raw = "" if raw == "." else raw
            relative, declared = _normalise_directory(
                raw, os.path.basename(os.path.normpath(root)))

            if any(name.lower().endswith(_DICOM_EXTENSION) for name in filenames):
                # One volume, named after the folder, filed with its PARENT --
                # which is where ASO's fixture keeps the landmarks that go with
                # it. Nothing below a series is indexed: its slices are not
                # patients.
                subdirectories[:] = []
                parent, series = os.path.split(relative)
                patient = patient_stem(series or os.path.basename(root),
                                       drop_timepoint=drop_timepoint)
                key = os.path.join(parent, patient) if parent else patient
                case = cases.setdefault(key, Case(key))
                case.artifacts.append(
                    Artifact(directory, VOLUME, label, parent, patient,
                             origin=os.path.dirname(raw))
                )
                continue

            declared_cases, claimed = _scene_cases(
                directory, filenames, label, relative, raw, root)
            for case in declared_cases:
                cases[case.key] = case

            for filename in sorted(filenames):
                if filename.startswith(_HIDDEN_PREFIX) or _is_report(filename):
                    continue
                if filename in claimed:
                    continue
                full = os.path.join(directory, filename)
                kind = kind_of(filename, full)
                if kind is None:
                    continue
                # A declaring folder outranks the file name: it is the only
                # place AMASSS's scan stem survives the prediction id.
                patient = patient_stem(
                    declared or filename, drop_timepoint=drop_timepoint
                )
                key = os.path.join(relative, patient) if relative else patient
                case = cases.setdefault(key, Case(key))
                case.artifacts.append(
                    Artifact(full, kind, label, relative, patient, origin=raw)
                )

    if focus:
        inside = os.path.join(os.path.realpath(focus), "")
        for case in cases.values():
            for artifact in case.artifacts:
                if not os.path.realpath(artifact.path).startswith(inside):
                    artifact.source = SUPPORT
    cases = {key: case for key, case in cases.items()
             if any(artifact.source != SUPPORT for artifact in case.artifacts)}
    return [cases[key] for key in sorted(_absorb_orphans(cases))]


def _index_file(cases: dict, label: str, path: str, drop_timepoint: bool) -> None:
    """One file given as a source of its own: a scan a run was sent alone.
    Its folder is not walked, because nothing else in it was chosen."""
    filename = os.path.basename(path)
    kind = kind_of(filename, path)
    if kind is None:
        return
    patient = patient_stem(filename, drop_timepoint=drop_timepoint)
    case = cases.setdefault(patient, Case(patient))
    case.artifacts.append(Artifact(path, kind, label, "", patient, origin=""))


def _absorb_orphans(cases: dict) -> dict:
    """Fold a case that is only overlays into the anchored case it belongs to.

    A suffix table cannot reach every producer: ASO IOS names its landmarks
    after the landmark FILE it was handed, so `Upper_new_9_Upper_O_Pred.json`
    keys nowhere near `Upper_new_9.vtk`. Prefix on a token boundary does reach
    it, and only within the same directory -- across directories the same
    prefix is two different patients.

    **What counts as needing a host is holding no GEOMETRY**, not holding no
    anchor. A mask anchors a view of its own and is still somebody's mask:
    AMASSS names one `<stem>_<prediction_ID>_MAND.nii.gz`, with a free-text id
    in the middle that no suffix table can strip, so loose in a folder it
    indexed as a patient called `p1_Pred_MAND` instead of joining `p1`.
    """
    anchored = sorted(key for key, case in cases.items()
                      if case.of_kind(*GEOMETRY_KINDS) and not case.scene)
    for key in sorted(cases):
        case = cases[key]
        if case.of_kind(*GEOMETRY_KINDS) or case.scene:
            continue
        directory = os.path.dirname(key)
        # Longest first: with `P1` and `P1_T1` both anchored, an overlay of
        # `P1_T1_lm` belongs to the more specific one.
        host = max(
            (other for other in anchored
             if os.path.dirname(other) == directory
             and is_token_prefix(os.path.basename(other), os.path.basename(key))),
            key=len, default=None,
        )
        if host is None:
            continue
        cases[host].artifacts.extend(case.artifacts)
        del cases[key]
    return cases
