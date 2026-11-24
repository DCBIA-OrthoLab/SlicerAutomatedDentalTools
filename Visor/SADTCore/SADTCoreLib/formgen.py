"""Builds Qt widgets from a server tool's argument schema, and reads them back.

Imports neither `requests` nor anything HTTP — see ARCHITECTURE.md dependency
rule. The server is the single source of truth for the schema: adding a field
to a tool server-side makes it appear here without touching any module code.

File-type arguments (any type accepted by `is_file_type` — "file", "zip_file",
"nifti_file", ...) are skipped by build()/collect(): they are not generic
scalar fields, they get their own row in base_widget's "Inputs" section. The
*widget* for such an input is still built here (`file_widget`), so that every
"schema shape -> Qt widget" decision lives in one file.

So is the translation from the server's vocabulary to the one base_widget acts
on — `file_input_modes`, `auto_file_mode`, `result_kind_for` — for the same
reason: it is all "what does the schema say this panel should be". A module
then declares only what the schema *cannot* say (see file_input_modes).

Some schema types render as several widgets rather than one, so they get a
small Python holder class each (`MultiChoiceGroup`, `FileOrFolderInput`,
`JoystickInput`) instead of a QWidget subclass: PythonQt makes subclassing
awkward, and everything the rest of this module needs fits in a plain object
exposing `container` for layout. The one genuine QWidget subclass, the
joystick pad itself, lives in joystick.py.

Escape hatch: a hand-written .ui can still be used by giving its widgets a Qt
dynamic property named "serverArgName" matching the schema argument name —
collect() will pick them up as if they had been generated. Not used by
SurgMovPred; documented for future modules that need custom layout.
"""

import logging
import os

import ctk
import qt

from . import accepts_folder, argument_types, design, file_extensions_for, is_file_type
from .errors import plain_text, preformatted_text
from .joystick import JoystickPad

logger = logging.getLogger("SADTCore.formgen")

ARG_NAME_PROPERTY = "serverArgName"

# The collapsible box an argument declaring no `section` goes into — i.e. the
# single box every tool's panel is today.
DEFAULT_SECTION = "Inputs"

# Options per row inside a "tabs" tab. Fixed rather than computed from the
# panel width: the module panel is resizable and a reflow on every drag would
# move check boxes under the user's cursor mid-click.
# How many options a dense grid puts on a line. Derived from the LONGEST label
# rather than fixed: ALI's cranial base is `Ba`, `S`, `N` and fits six across,
# while its lower region runs to `UR3OIP` and fits three. A single number had to
# be chosen for the worst case, which wasted half the width on every short
# catalogue.
_MIN_COLUMNS = 3
_MAX_COLUMNS = 8
# Roughly the character width a chip's padding and border add, in characters.
_CHIP_OVERHEAD = 4
# The width a tab has to spend, in characters. Calibrated against the panel at
# its usual width -- which is what a Slicer module panel is, not resizable in
# practice: four chips of five characters measured 225 px of the 570 the box
# offers, so the first guess spent under half of it.
_GRID_BUDGET = 64


def _columns_for(options) -> int:
    """A column count the longest option still fits in."""
    longest = max((len(str(option)) for option in options), default=1)
    return max(_MIN_COLUMNS, min(_MAX_COLUMNS, _GRID_BUDGET // (longest + _CHIP_OVERHEAD)))

# Where the leftovers go when `groups` doesn't mention every option. The server
# rejects a group naming an option that doesn't exist, but not the reverse —
# and silently dropping an option would mean the user cannot select something
# the tool offers.
_UNGROUPED_LABEL = "Other"

# Per TAB. The original extension had both a per-tab `Switch group selection`
# and a global `Select All` / `Clear All` pair (ALI.py's LandmarkTabWidget); a
# clinician who wants the cranial base wants ten boxes ticked, not ten clicks.
#
# TWO buttons rather than one toggle: a single button has to say which of the
# two a click will do, so its label moves under the pointer as the group fills,
# and a control whose name changes is a control you have to read before every
# click. Two named actions are always true.
#
# The SAME two words as the bar above a flat group, from the one pair in
# design.py. They were written twice and had drifted to two wordings and two
# casings, so the same action read as two different controls depending on which
# layout the tool happened to ask for.
SELECT_GROUP_LABEL = design.SELECT_ALL_TEXT
CLEAR_GROUP_LABEL = design.SELECT_NONE_TEXT

# Leads the dropdown of an OPTIONAL scalar `server_selectable` argument, and
# reads back as "" so collectArgs drops the argument entirely and the server
# applies its own rule.
#
# It exists because a QComboBox cannot be empty: `addItems` selects index 0 the
# moment the list arrives, so an optional argument whose schema says "leave
# empty and the server decides" (ALI's `model`, ASO's `landmark_models`) had no
# way to be left empty — the first hosted name was submitted by a user who
# never chose it. For ASO that list is DATA/ASO/models/, which holds reference
# bundles next to weight bundles, so the silent default was routinely a
# reference and the run died on "No CBCT landmark weights found in ...".
#
# Only for OPTIONAL arguments: a required one has no server-side fallback to
# defer to, so offering the entry would only produce a 422.
# This extension runs its tools on this machine: no wording here names a server.
AUTOMATIC_OPTION = "(automatic — the tool chooses)"

# The two browse buttons of an argument accepting a file or a folder. Which of
# the two the user ends up giving is read back from the path, not from these.
# What the ONE browse button says, and what the dialog it opens is titled.
# The button no longer names the kind it picks -- the segmented control above
# the row does that, and a button reading `File...` under a pressed `File`
# segment was the same word twice, the larger of the two saying the less.
SELECT_LABEL = "Select"
BROWSE_FILE_LABEL = "Select a file"
BROWSE_FOLDER_LABEL = "Select a folder"
PATH_PLACEHOLDER = "Select a file or a folder"

# What the caption says when the argument holds nothing. The path field's
# placeholder used to carry this, and the caption is where it lands now
# that there is no field.
NOTHING_CHOSEN = "Nothing selected"

# Room for the scroll bar and the item margins, so the widest entry is not
# elided by a pixel.
_POPUP_PADDING = 40
# A popup may be wider than the box it drops from -- that is the whole point --
# but not wider than the screen it drops onto. The longest entry any tool
# publishes today measures 265 px, so this is a guard rail, not a budget.
_POPUP_MAX_WIDTH = 720


# 1024-based, like every other size this extension prints (transfer._Meter,
# client._download_message). A test file is a download the user is about to
# pay for, and "648 MB" is the only form of 679477248 that says so.
_SIZE_UNITS = ("B", "KB", "MB", "GB", "TB")


def human_size(size) -> str:
    """"2.9 KB", "7.4 MB", "648 MB" - and "" when the size is unknown.

    A `size` the server could not compute arrives as `null`, and rendering
    that as "0 B" would be a claim rather than a gap: the picker shows the
    name alone instead. A genuinely empty entry reads the same way, which is
    the harmless half of the same rule.
    """
    if not isinstance(size, (int, float)) or isinstance(size, bool) or size <= 0:
        return ""
    value = float(size)
    unit = _SIZE_UNITS[0]
    for unit in _SIZE_UNITS:
        if value < 1024 or unit == _SIZE_UNITS[-1]:
            break
        value /= 1024
    if unit == "B" or value >= 10:
        return f"{round(value)} {unit}"
    # One decimal below 10, where dropping it would round 2.9 KB to "3 KB".
    return f"{value:.1f} {unit}"


def hosted_entry_label(entry: dict) -> str:
    """How one server-hosted test file reads in the input dropdown:
    "CBCT_FullyAuto  (folder, 339 MB)".

    The name alone is what the dropdown used to show, and it hides both things
    a user needs before clicking: whether this is one scan or a whole cohort,
    and how many bytes are about to cross the link. Either may be unknown (an
    older server, or a backend that cannot size a tree cheaply) and is simply
    left out - an entry that says nothing extra still reads as its own name.
    """
    name = entry.get("name", "")
    details = [detail for detail in (entry.get("kind"), human_size(entry.get("size"))) if detail]
    return f"{name}  ({', '.join(details)})" if details else name

# What a file IS, in the words a clinician uses, keyed by extension. The panel
# shows a path; a path says where a file sits, not what it holds, and ".vtk" or
# ".nrrd" at the tail of an elided temp directory says neither. Longest suffix
# first, so `.nii.gz` never matches as `.gz`.
FILE_KINDS = (
    (".nii.gz", "NIfTI volume"), (".nii", "NIfTI volume"),
    (".nrrd", "NRRD volume"), (".nhdr", "NRRD volume"),
    (".mha", "MetaImage volume"), (".mhd", "MetaImage volume"),
    (".gipl.gz", "GIPL volume"), (".gipl", "GIPL volume"),
    (".dcm", "DICOM slice"),
    (".vtk", "VTK surface"), (".vtp", "VTK surface"),
    (".stl", "STL surface"), (".obj", "OBJ surface"), (".ply", "PLY surface"),
    (".mrk.json", "Slicer markups"), (".fcsv", "Slicer markups"),
    (".tfm", "transform"), (".h5", "transform"), (".mat", "transform"),
    (".csv", "CSV table"), (".tsv", "TSV table"),
    (".xlsx", "Excel table"), (".xls", "Excel table"), (".ods", "table"),
    (".json", "JSON file"), (".txt", "text file"), (".md", "text file"),
    (".zip", "ZIP archive"),
)


def file_kind(path: str) -> str:
    """"NIfTI volume", "VTK surface", "folder" -- or "" for a name that says
    nothing recognisable, which is better than guessing at one."""
    if not path:
        return ""
    if os.path.isdir(path):
        return "folder"
    lowered = os.path.basename(path).lower()
    for suffix, kind in FILE_KINDS:
        if lowered.endswith(suffix):
            return kind
    return ""


def _size_on_disk(path: str) -> int:
    """Bytes, walking a directory when it is one. Metadata only -- nothing is
    read -- so a 339 MB cohort costs a stat per file and no I/O."""
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for directory, _subdirs, names in os.walk(path):
            for name in names:
                try:
                    total += os.path.getsize(os.path.join(directory, name))
                except OSError:
                    continue
        return total
    except OSError:
        return 0


def _folder_contents(path: str):
    """(total bytes, {kind: count}) for a directory, walked ONCE.

    Recursively, because that is how every tool reads a cohort: a folder whose
    scans sit one level down under per-patient directories is the normal shape,
    and counting only the top level would report zero for it.

    Metadata only -- nothing is read -- so a 339 MB cohort costs a stat per
    file and no I/O, which is what it already cost to report a size.
    """
    total, counts = 0, {}
    try:
        for directory, _subdirs, names in os.walk(path):
            for name in names:
                if name.startswith("."):
                    continue
                try:
                    total += os.path.getsize(os.path.join(directory, name))
                except OSError:
                    continue
                kind = file_kind(name) or "file"
                counts[kind] = counts.get(kind, 0) + 1
    except OSError:
        return 0, {}
    return total, counts


def _plural(count: int, noun: str) -> str:
    """"14 VTK surfaces", "1 NIfTI volume", "3 Slicer markups".

    A kind already ending in `s` is left alone: `FILE_KINDS` holds "Slicer
    markups", and a blind `+ "s"` produced "Slicer markupss" on screen.
    """
    plural = noun if count == 1 or noun.endswith("s") else noun + "s"
    return "{} {}".format(count, plural)


def describe_folder(path: str, name_only: bool = True) -> str:
    """What a folder HOLDS, which is the only thing that confirms it is the
    right one.

    "2_TAD_VTKFiles_L_T2 - folder, 73 MB" says nothing a wrong folder would
    not also say. The count and the kind do: pointing one level too high shows
    `0 files`, and pointing at the T1 cohort shows a different count.

    One kind is named on its own ("14 VTK surfaces"); a mixed folder gives the
    total and its two largest groups, because naming all of them turns the
    caption into a paragraph.
    """
    total, counts = _folder_contents(path)
    # The full path when the caption names its source ("Folder: /data/..."):
    # confirming you picked the right one means seeing WHERE it is, and the
    # caption wraps rather than eliding, so it can afford to.
    name = (os.path.basename(path.rstrip(os.sep)) or path) if name_only else path
    if not counts:
        return "{} - folder, empty".format(name)

    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    files = sum(counts.values())
    if len(ordered) == 1:
        held = _plural(files, ordered[0][0])
    else:
        named = ", ".join(_plural(count, kind) for kind, count in ordered[:2])
        more = " and more" if len(ordered) > 2 else ""
        held = "{} ({}{})".format(_plural(files, "file"), named, more)
    # The size is dropped when there is none to report: `human_size(0)` is
    # empty, and a trailing comma with nothing after it reads as a value that
    # failed to load rather than as a folder of empty files.
    parts = [part for part in ("folder", held, human_size(total)) if part]
    return "{} - {}".format(name, ", ".join(parts))


def describe_file(path: str, name_only: bool = True) -> str:
    """One line naming what is in an input: "sample_scan.nii.gz - NIfTI volume, 94 MB".

    The row itself cannot say this. A downloaded test file lands on a path like
    `/tmp/Slicer-user/ADTRemoteTestFiles2026-09-09_09+26+53.117/sample_scan.nii.gz`,
    which a path field renders as `:TestFiles2026-09-09_09+26+53.117/sample_sc...`
    -- the name is the first thing to be cut, and the kind is never visible at
    all. This is written under the row, wraps rather than elides, and so cannot
    lose the two things a user needs: which file, and what kind of file.
    """
    if not path:
        return ""
    if os.path.isdir(path):
        return describe_folder(path, name_only)
    name = (os.path.basename(path.rstrip(os.sep)) or path) if name_only else path
    details = [detail for detail in (file_kind(path), human_size(_size_on_disk(path))) if detail]
    return "{} - {}".format(name, ", ".join(details)) if details else name


# Extensions Slicer holds as a scalar volume in the scene. A file argument
# accepting one of these can equally be satisfied by a volume the user already
# has open, exported at upload time (base_widget._prepareOneInputFile).
_VOLUME_EXTENSIONS = {".nii", ".nii.gz", ".nrrd", ".gipl", ".gipl.gz", ".mha", ".mhd"}

# Surfaces, for the same question asked of a mesh argument. Separate from
# _VOLUME_EXTENSIONS because the two answer different node classes, and a
# caller almost always wants one or the other rather than both.
_SURFACE_EXTENSIONS = {".vtk", ".vtp", ".stl", ".obj", ".ply"}

# What the scene can be asked for, per kind: the MRML class to offer and the
# format to write it back out in. Written here rather than derived, because
# "what can satisfy this argument" is a widget decision and this file is where
# every other one lives.
SCENE_NODE_KINDS = {
    # The two shapes a SCAN takes in this extension: a CBCT is a scalar volume,
    # an intraoral scan is a surface. Both are "a scan" to a clinician, and the
    # dropdown says so in those words.
    "volume": ("vtkMRMLScalarVolumeNode", ".nii.gz"),
    "model": ("vtkMRMLModelNode", ".vtk"),
    # **Landmarks are deliberately absent**, and were offered here until
    # 2026-09-24. The scene list is the imported SCAN a run is about, and a set
    # of points sitting in the same list read as another scan -- the two are
    # picked from one dropdown, one row apart, and choosing the wrong one is a
    # run that fails on a file the tool cannot open. What it costs: landmarks
    # placed by hand have to be saved and browsed to, rather than picked.
    # A crop box, and a kind of its own because Slicer's is NOT a markups
    # fiducial: `vtkMRMLMarkupsROINode.IsA("vtkMRMLMarkupsFiducialNode")` is
    # false, so offering it under "markups" offers nothing at all. Same file
    # format, different node -- and worth the entry, because a ROI is DRAWN in
    # Slicer. It is already in the scene when the panel is opened; asking for it
    # as a file would mean saving it first for no reason.
    "roi": ("vtkMRMLMarkupsROINode", ".mrk.json"),
}

# What a pick of each kind is called on the row's second line. A row accepting
# more than one says "Scene", because naming one of them would be wrong for
# the others -- ALI takes all three.
# What a pick of each kind is called on the row's second line, and what its
# dropdown offers to fill the row with. A CBCT volume and an intraoral surface
# are both "a scan", so a row taking either says the one word that is true of
# both rather than falling back on "Scene".
SCENE_LABELS = {"volume": "Scan", "model": "Scan", "roi": "ROI"}

# The first entry of the scene dropdown, per label -- the one place that says
# what the list holds. Keyed by the LABEL and not by the kind, so a row taking
# a volume or a surface gets one prompt rather than two that disagree.
SCENE_PROMPTS = {
    "Scan": "Imported scan...",
    # Drawn rather than imported: a ROI is made in Slicer, which is the whole
    # reason it is offered here instead of being asked for as a file.
    "ROI": "Drawn ROI...",
}
_SCENE_PROMPT_FALLBACK = "From the scene..."


# What an argument's NAME says about the scene node that could satisfy it.
#
# Here, and not on the server, for the reason PRETTY_NAMES is here: this is a
# dental vocabulary, and the server is built not to hold one. It is needed
# because `describe.py` publishes no extensions for any packaged tool, so the
# schema cannot narrow a row on its own -- every file argument in the extension
# would otherwise be offered nothing at all.
#
# Matched on WORDS in the name, longest first, and an unknown name gets
# NOTHING: a spreadsheet or a transform has no counterpart in a scene, and
# guessing one wrong is worse than offering none.
SCENE_NAME_KINDS = (
    # FIRST, and answering NOTHING. `cbct_landmarks` holds `cbct`, so a
    # landmark argument falling through to the rest of this table would be
    # offered the scene's volumes -- the one answer that is certainly wrong.
    ("landmark", ()),
    # Before the rest, because it is the narrowest: a box, not points.
    ("roi", ("roi",)),
    # A mask is labelled voxels, which is a volume node like any other.
    ("mask", ("volume",)),
    ("mesh", ("model",)),
    ("surface", ("model",)),
    # Intraoral, which in this extension always means a surface.
    ("ios", ("model",)),
    ("scan", ("volume",)),
    ("cbct", ("volume",)),
    ("volume", ("volume",)),
    # Deliberately BOTH. `t1`/`t2` are CBCT volumes in AREG_CBCT and GreedyReg
    # and intraoral SURFACES in AREG_IOS; `input` and `files` are whatever the
    # tool was pointed at. Offering both is the honest answer -- the scene is
    # listed, and what the user picks is what they meant.
    ("t1", ("volume", "model")),
    ("t2", ("volume", "model")),
    ("input", ("volume", "model")),
    ("files", ("volume", "model")),
)


def scene_kinds_for(spec: dict, name: str = "") -> tuple:
    """Which kinds of scene node can satisfy this file argument.

    Read off the schema like every other widget decision: an argument is
    narrowed by the formats it declares, and a `.csv` input must never offer a
    scan.

    An argument declaring NO format gets nothing, and that is deliberate even
    though it is not always right. `describe.py` cannot express extensions at
    all today, so every packaged tool publishes none -- reading that as "takes
    anything" would put a list of scans under every file row in the extension,
    including the ones that want a spreadsheet. A module that knows better says
    so itself: see `ServerToolWidgetBase.SCENE_INPUTS`.
    """
    extensions = {e.lower() for e in file_extensions_for(spec)}
    kinds = []
    if not extensions and name:
        # The schema said nothing, which is every packaged tool. The name is
        # the only thing left, and it is a better guess than none: a row called
        # `scans` beside an empty dropdown is a feature nobody can find.
        lowered = name.lower()
        for word, answer in SCENE_NAME_KINDS:
            if word in lowered:
                return answer
        return ()
    if extensions & _VOLUME_EXTENSIONS or any(
            "volume" in name or "nifti" in name for name in argument_types(spec)):
        kinds.append("volume")
    if extensions & _SURFACE_EXTENSIONS or any(
            "surface" in name or "mesh" in name for name in argument_types(spec)):
        kinds.append("model")
    # No markups branch, and no `_MARKUP_EXTENSIONS` table to go with it: an
    # argument declaring `.mrk.json` gets nothing from the scene. See
    # SCENE_NODE_KINDS for why the list is scans only.
    return tuple(kinds)


def scene_label_for(kinds) -> str:
    """The word a scene pick goes under.

    Read off the kinds' own labels rather than off their number: a volume and a
    surface are both a Scan, so a row taking either says `Scan` where counting
    would have said `Scene` -- a word that names the container instead of the
    thing, on the one row where a true word exists.
    """
    words = {SCENE_LABELS[kind] for kind in kinds if kind in SCENE_LABELS}
    if len(words) == 1:
        return words.pop()
    return "Scene"


def scene_prompt_for(label: str) -> str:
    """The scene dropdown's first entry, for a row labelled `label`."""
    return SCENE_PROMPTS.get(label, _SCENE_PROMPT_FALLBACK)


def accepts_volume(spec: dict, name: str = "") -> bool:
    """Whether the scene can satisfy this file argument at all.

    Read off the schema, like every other widget decision. An argument that
    names its formats is narrowed by them -- a `.csv` input must never offer a
    scan. One that names NONE takes whatever the scene holds: ALI's `input` is
    exactly that, a single argument accepting a CBCT or an intraoral surface,
    and the tool decides from the data rather than from an extension list.
    Refusing it was what left ALI with no scene entries at all.
    """
    return bool(scene_kinds_for(spec, name))

# `ArgSpec.ui` values on the scalar types (the multichoice ones are LAYOUTS
# below). "slider" turns a bounded int/float into a ctkSliderWidget; "joystick"
# gives a vec2 the 2D pad. Like every presentation hint, an unknown one falls
# back to the plain rendering with a warning: a newer server must never be
# able to break an older client's panel.
SLIDER_UI = "slider"
JOYSTICK_UI = "joystick"
# A `"choice"` drawn as a row of segments rather than a drop-down: for a
# two- or three-way decision that changes what the rest of the section asks
# for, where every option should be visible at once. Such a field also heads
# its section (see `build`), since what follows it depends on it.
SEGMENTED_UI = "segmented"


class MultiChoiceGroup:
    """The checkboxes rendered for a `"multichoice"` argument.

    Holds one QCheckBox per option, in the schema's declaration order (the
    order `choices` arrives in — never sorted), and reads back the *complete*
    {option: checked} state. Sending the full state is required, not a
    convenience: see ToolServerClient._stringify for why a missing option is
    not the same as an unchecked one.

    **The argument's `description` is not drawn here any more.** It was, as a
    wrapped hint above the boxes, on the reasoning that a tooltip nobody
    hovers is not where a field's meaning belongs. What that produced on a
    real panel is several small grey paragraphs stacked between the fields —
    ALI's `landmarks` note alone is 304 characters — at a size and a contrast
    that made them text a reader scrolls past rather than reads. The
    description is the row LABEL's tooltip now (see `build`), and the label is
    marked with a dotted rule so it is visibly a thing that has more to say.

    **`layout` and `groups` change only where the boxes are put.** `self.boxes`
    is keyed and ordered by `choices` whatever the layout, so `value()`,
    `collect()`, `connect_changed()` and `all_required_filled()` cannot tell
    the four apart — which is what makes the layouts safe to add: a wrong one
    is ugly, never wrong on the wire. See `LAYOUTS` for what each does and the
    server's `ArgSpec.ui` for why they exist at all.
    """

    def __init__(self, choices: dict, layout=None, groups=None,
                 option_help=None, select_all=False, note=None):
        self.container = qt.QWidget()
        column = qt.QVBoxLayout(self.container)
        # Air UNDER the block, and only under it. A multichoice is several rows
        # tall where every other field is one, so its last option sat as close
        # to the next argument's label as its own options sit to each other --
        # and a reader has no way to tell where the group ends. The other three
        # margins stay zero: the row's own label has to line up with the first
        # option, not with a gap.
        column.setContentsMargins(0, 0, 0, design.SPACING_LG)
        column.setSpacing(design.SPACING_XS)

        self._column = column
        self._layout = layout
        self._groups = groups
        # Two buttons above the options. Only a hint: whatever they do, what
        # `value()` reads back is the same complete {option: checked} dict.
        self._select_all = select_all
        self.selectAllButton = None
        self.selectNoneButton = None
        # {option: one line saying what it is}. The tool's own words, published
        # per option because a catalogue of CODES cannot be read off its labels.
        self._option_help = option_help
        # One line the tool wants read before anything is ticked, drawn above
        # the options on every redraw. Not the description, which stays the
        # label's tooltip; see `ArgSpec.note` on the server.
        self._note = note
        self._draw(choices, groups)

    def _draw(self, choices: dict, groups) -> None:
        """Lay the options out. Split from __init__ so `rebuild` can redraw a
        group whose option set changed with the mode."""
        layout = self._layout
        column = self._column
        if self._note:
            # A plain-text label: schema text is never rendered as HTML.
            column.addWidget(design.hint_label(self._note))
        self._add_select_all(column, choices)
        builder = _LAYOUT_BUILDERS.get(layout)
        if builder is None:
            if layout is not None:
                logger.warning(
                    "Unknown multichoice layout '%s', falling back to a single column", layout
                )
            builder = _build_flat_boxes
        made = builder(column, choices, groups, self._option_help)

        # Declaration order, whatever order the layout visited the options in.
        self.boxes = {option: made[option] for option in choices}

        # After the boxes exist, because that is what they act on.
        if self.selectAllButton is not None:
            self.selectAllButton.connect("clicked()", self._checkEverything)
            self.selectNoneButton.connect("clicked()", self._checkNothing)

    def _add_select_all(self, column, choices: dict) -> None:
        """A "Select all" / "Deselect all" pair, for a catalogue nobody would
        tick one box at a time.

        Skipped below two options, where it would be two buttons commanding one
        check box -- which reads as more of a decision than the check box is.

        `ghost_button` rather than the underlined links this started as: two
        underlined captions in a row read as one broken sentence, and an
        underline is what this extension uses for something that opens
        elsewhere. See that factory for why it is not a filled button either.
        """
        self.selectAllButton = None
        self.selectNoneButton = None
        if not self._select_all or len(choices) < 2:
            return
        self.selectAllButton = design.ghost_button(design.SELECT_ALL_TEXT)
        self.selectNoneButton = design.ghost_button(design.SELECT_NONE_TEXT)
        # A WIDGET holding the row, not a bare sub-layout: `rebuild` empties the
        # column with takeAt()/setParent(None), which reaches a widget and not a
        # layout -- a sub-layout would be dropped while its buttons stayed
        # parented to the container, and a redraw would leave two of each.
        holder = qt.QWidget()
        row = qt.QHBoxLayout(holder)
        # Air UNDER the pair, so it reads as a heading over the options rather
        # than as the first line of the list. The column's own spacing is the
        # gap between two OPTIONS, and at that distance a control and the
        # things it controls look like the same kind of thing.
        row.setContentsMargins(0, 0, 0, design.SPACING_SM)
        row.setSpacing(design.SPACING_XS)
        row.addWidget(self.selectAllButton)
        row.addWidget(self.selectNoneButton)
        row.addStretch(1)
        column.addWidget(holder)

    def _checkEverything(self) -> None:
        self.setEverything(True)

    def _checkNothing(self) -> None:
        self.setEverything(False)

    def setEverything(self, checked: bool) -> None:
        """Tick or clear every option. Each box emits its own signal, so
        whatever was connected to this group reacts exactly as it does to a
        click -- Apply re-evaluates, a dependent field re-renders."""
        for box in self.boxes.values():
            box.setChecked(checked)


    def rebuild(self, choices: dict, groups=None) -> None:
        """Draw this group again for a different set of options.

        A facade publishes the UNION of its engines' options and says per mode
        which apply; a combo box is narrowed by refilling it, and a check-box
        group has to be redrawn the same way. What survives is the SELECTION:
        an option still offered keeps its state, so switching mode and back does
        not silently clear what the user ticked.
        """
        kept = {option: box.isChecked() for option, box in self.boxes.items()}
        wanted = {option: kept.get(option, default) for option, default in choices.items()}
        if list(self.boxes) == list(wanted) and self._groups == groups:
            return

        while self._column.count():
            item = self._column.takeAt(0)
            widget = item.widget() if hasattr(item, "widget") else None
            if widget is not None:
                widget.setParent(None)

        self._groups = groups
        self._draw(wanted, groups)

    def value(self) -> dict:
        return {option: box.isChecked() for option, box in self.boxes.items()}

    # -- the slice of the QWidget API build()/base_widget use on a field ----

    def setProperty(self, name, value) -> None:
        self.container.setProperty(name, value)

    def setToolTip(self, _text) -> None:
        """Deliberately nothing. The description belongs to the row's LABEL.

        Qt hands a container's tooltip to every child that has none, so
        accepting it here put ALI's 304-character note on `landmarks` under
        each of its 236 chips -- the hovered copy being the one nobody asked
        for. `build` puts it on the label beside the field instead, which has
        no children to hand it down to.

        A chip's own tooltip is a different thing: it says what THAT landmark is
        and where it goes, which the schema cannot express yet.
        """


def _make_box(option: str, checked, help_text: str = "") -> qt.QCheckBox:
    box = qt.QCheckBox(option)
    box.setChecked(bool(checked))
    _explain(box, help_text)
    return box


def _make_chip(option: str, checked, help_text: str = ""):
    """The dense layouts' option: the label itself, checkable (see
    design.option_chip). Reads back exactly as a check box does."""
    chip = design.option_chip(option)
    chip.setChecked(bool(checked))
    _explain(chip, help_text)
    return chip


def _help_for(help_texts, option: str) -> str:
    """This option's line, or nothing. A table the tool did not send, or one
    that arrived as something other than a mapping, must leave every widget
    exactly as it was rather than take the panel down."""
    if not isinstance(help_texts, dict):
        return ""
    text = help_texts.get(option)
    return text if isinstance(text, str) else ""


def _explain(widget, help_text: str) -> None:
    """Say what THIS option is, on the option itself.

    A catalogue of codes needs it and a catalogue of words does not: `Ba` and
    `UR1MB` tell a clinician nothing, while `Mandible` already reads. So it is
    set only where the tool named the option -- an argument declaring no
    `option_help` leaves every widget without one, which is what it had before
    this existed.

    Never a fallback to the argument's own description: that paragraph is
    already rendered above the options, and Qt hands a container's tooltip to
    every child without one, so ALI's 304-character note used to pop up under
    each of its 236 chips.
    """
    if not help_text:
        return
    setter = getattr(widget, "setToolTip", None)
    if setter:
        setter(plain_text(help_text))


def _grouped(choices: dict, groups) -> list:
    """[(group name, [option, ...])] — the declared groups, then whatever they
    left out. Options keep `choices` order within each group, so a group
    listing them in a different order does not reorder the display."""
    if not groups:
        return [("", list(choices))]

    claimed = {option for options in groups.values() for option in options}
    grouped = [
        (name, [option for option in choices if option in set(options)])
        for name, options in groups.items()
    ]
    leftovers = [option for option in choices if option not in claimed]
    if leftovers:
        grouped.append((_UNGROUPED_LABEL, leftovers))
    return [(name, options) for name, options in grouped if options]


def _build_flat_boxes(column, choices: dict, _groups=None, help_texts=None) -> dict:
    """One box per line. The default, and what every tool declaring no `ui`
    gets — unchanged from before layouts existed."""
    boxes = {}
    for option, checked in choices.items():
        boxes[option] = _make_box(option, checked, _help_for(help_texts, option))
        column.addWidget(boxes[option])
    return boxes


def _build_inline_boxes(column, choices: dict, _groups=None, help_texts=None) -> dict:
    """One horizontal row. For a handful of short options (ASO's two jaws, its
    eight landmark types) that waste a line each stacked vertically."""
    row_container = qt.QWidget()
    row = qt.QHBoxLayout(row_container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(design.SPACING_MD)

    boxes = {}
    for option, checked in choices.items():
        boxes[option] = _make_box(option, checked, _help_for(help_texts, option))
        row.addWidget(boxes[option])
    row.addStretch(1)

    column.addWidget(row_container)
    return boxes


def _build_chips_boxes(column, choices: dict, groups=None, help_texts=None) -> dict:
    """Options as chips, wrapped over as many lines as they need.

    The tabbed layout's grid without the tabs, and it exists because the two
    answer different problems. `tabs` is for a CATALOGUE -- ALI's 119 landmarks,
    which nobody reads in one piece and which needs a per-tab button to be
    usable at all. This is for a handful: AMASSS's nine structures fit on two
    lines, and putting them behind a single tab would be a tab bar with nowhere
    to go.

    So it carries no group button either. Nine chips are nine clicks, and a
    control that takes all of them earns its place at a hundred, not at nine.

    Groups, when a tool declares them, become a heading and a grid of their own
    rather than a tab -- which keeps a two-group argument readable without
    hiding half of it behind a click.

    `group_heading`, not `section_title`: the headings used to sit at the
    column's own 4px option spacing, so AMASSS's `Soft tissue` was as close to
    the last chip of `Bones` as two chips of one group are to each other, and
    its three groups read as one run of nine. The air and the rule under each
    heading are what separate them.
    """
    boxes = {}
    for group_name, options in _grouped(choices, groups):
        if group_name and (groups or {}):
            column.addWidget(design.group_heading(group_name))

        page = qt.QWidget()
        grid = qt.QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(design.SPACING_XS)
        # Wider than tall, for the same reason as the tabbed grid: chips carry
        # their own padding, so touching columns read as one long word while
        # touching rows read as a list.
        grid.setHorizontalSpacing(design.SPACING_MD)

        columns = _columns_for(options)
        for index, option in enumerate(options):
            boxes[option] = _make_chip(option, choices[option],
                                       _help_for(help_texts, option))
            grid.addWidget(boxes[option], index // columns, index % columns)
        # No scroll area here, so the height follows the chips; only the spare
        # WIDTH needs somewhere to go, or the columns stretch apart.
        _pack_to_top_left(grid, rows=-(-len(options) // columns), columns=columns)
        column.addWidget(page)

    return boxes


def _build_grid_boxes(column, choices: dict, groups=None, help_texts=None) -> dict:
    """One row per group, options as columns — the chart layout.

    For options whose *position* carries meaning: ASO asks for teeth "spread
    across the arch", and a column of 32 check boxes is the one layout that
    cannot show whether a selection is spread or clustered.

    Sixteen teeth do not fit in a Slicer module panel, so the grid scrolls
    horizontally rather than being squeezed or wrapped — wrapping an arch onto
    two lines would destroy the very adjacency the layout exists to show. The
    old module did the same (`ASO.ui`'s scrollArea around LayoutSemiIOS_tooth).

    It is drawn on a `table_frame`: the same filled, strongly-bordered surface
    a tab pane gives the other dense layouts, which this one has no pane to
    inherit. Without it a chart of thirty-two chips and two row headings sat
    directly on the panel with no edge anywhere, and where the table stopped
    and the next argument started was left to the reader.
    """
    grid_container = qt.QWidget()
    grid = qt.QGridLayout(grid_container)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(design.SPACING_XS)

    boxes = {}
    for row_index, (group_name, options) in enumerate(_grouped(choices, groups)):
        if group_name:
            # A row header, not a hint: 8pt muted put the name of the arch in
            # the smallest type on the panel, beside the chips it names.
            grid.addWidget(design.section_title(group_name), row_index, 0)
        for offset, option in enumerate(options):
            boxes[option] = _make_chip(option, choices[option],
                                       _help_for(help_texts, option))
            grid.addWidget(boxes[option], row_index, offset + 1)

    # Rows only: the COLUMNS are the arch, and letting them take the slack would
    # spread a tooth chart across whatever width the panel happens to have --
    # destroying the adjacency this layout exists to show.
    grid.setRowStretch(grid.rowCount(), 1)

    frame = design.table_frame()
    inside = qt.QVBoxLayout(frame)
    inside.setContentsMargins(design.SPACING_SM, design.SPACING_SM,
                              design.SPACING_SM, design.SPACING_SM)
    inside.addWidget(_horizontal_scroll(grid_container))
    column.addWidget(frame)
    return boxes


def _build_tabs_boxes(column, choices: dict, groups=None, help_texts=None) -> dict:
    """One tab per group, options in a scrollable multi-column grid.

    For a catalog too long to scroll through in one piece: ASO publishes 130
    CBCT landmarks, and the grouping (cranial base / upper / lower) is how the
    people who use them already think about them — the server sends it, so the
    tabs are the server's own grouping rather than a client-side guess.
    """
    tabs = qt.QTabWidget()
    boxes = {}
    grouped = list(_grouped(choices, groups))
    # Per TAB, from that tab's own longest label. `Ba`, `S`, `N` fit six across
    # where `UR3OIP` fits four, and one count for the whole argument had to be
    # the worst case -- half the width wasted on every short region. It changes
    # only the arrangement INSIDE the box, which already resizes with the tab.
    for group_name, options in grouped:
        columns = _columns_for(options)
        page = qt.QWidget()
        grid = qt.QGridLayout(page)
        grid.setContentsMargins(design.SPACING_SM, design.SPACING_SM, design.SPACING_SM, design.SPACING_SM)
        grid.setVerticalSpacing(design.SPACING_XS)
        # Wider than tall: chips carry their own padding, so touching columns
        # read as one long word while touching rows read as a list.
        grid.setHorizontalSpacing(design.SPACING_MD)
        for index, option in enumerate(options):
            boxes[option] = _make_chip(option, choices[option],
                                       _help_for(help_texts, option))
            grid.addWidget(boxes[option], index // columns, index % columns)
        page_boxes = [boxes[option] for option in options]
        # The page is stretched to the scroll area's height, and a QGridLayout
        # hands that slack to its ROWS: measured on ALI's cranial base, eleven
        # 20 px check boxes sat 94 px apart -- three sparse lines floating in a
        # tall empty box. A trailing row and column take the slack instead, so
        # the options pack at the top left and read as a list.
        _pack_to_top_left(grid, rows=-(-len(options) // columns), columns=columns)
        tabs.addTab(_group_page(page, page_boxes), group_name or _UNGROUPED_LABEL)

    # Fixed both ways, and PER TAB. A minimum alone let the panel's spare
    # vertical space stretch the box -- 380 px for ten cranial landmarks. Sizing
    # every tab to the TALLEST one instead was the first fix, and it still left
    # ALI's four-landmark tab in a box built for fifty-seven: the box has to
    # follow the number of boxes, which is what a reader is looking at.
    #
    # The heights live in this closure, never on the widget: PythonQt refuses a
    # new attribute on a C++ object ("creating new attributes on C++ objects is
    # not allowed") and takes the panel down with it.
    heights = [design.tabs_height_for(-(-len(options) // _columns_for(options)))
               for _name, options in grouped]

    def fit(index=None):
        chosen = index if isinstance(index, int) else tabs.currentIndex
        height = heights[chosen] if 0 <= chosen < len(heights) else max(heights, default=0)
        tabs.setMinimumHeight(height)
        tabs.setMaximumHeight(height)

    tabs.currentChanged.connect(fit)
    fit()

    column.addWidget(tabs)
    return boxes


def _group_page(grid_page, page_boxes):
    """One tab: its options, and one click that takes the whole group.

    The original extension put a `Switch group selection` button in every
    landmark tab, and dropping it made a ten-landmark region cost ten clicks.
    The button sits OUTSIDE the scroll area so it does not scroll away from the
    options it acts on, and its label says which of the two a click will do --
    so it cannot lie about itself once the group is already ticked.
    """
    container = qt.QWidget()
    column = qt.QVBoxLayout(container)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(design.SPACING_XS)
    column.addWidget(_vertical_scroll(grid_page))

    if not page_boxes:
        return container

    # Real buttons, splitting the full width of their tab, rather than the small
    # links they started as. They are the ONLY bulk control on a tabbed
    # multichoice -- the global All / None / Default bar is not drawn beside
    # them -- and a control that acts on everything above it should look like it
    # does.
    #
    # Blue adds, red takes away, which is the extension's own vocabulary
    # everywhere else: Apply is blue, Cancel is red. Grey was tried first and
    # read as disabled -- two large, evenly weighted slabs of DISABLED_BG next to
    # each other look like a control that is off, not like two you may press.
    row = qt.QWidget()
    bar = qt.QHBoxLayout(row)
    bar.setContentsMargins(0, 0, 0, 0)
    bar.setSpacing(design.SPACING_XS)

    def setter(state):
        def apply_state():
            for box in page_boxes:
                box.setChecked(state)
        return apply_state

    for label, state, make in ((SELECT_GROUP_LABEL, True, design.primary_button),
                               (CLEAR_GROUP_LABEL, False, design.danger_button)):
        button = make(label)
        button.clicked.connect(setter(state))
        # Equal stretch: the pair spans exactly what it acts on, and neither
        # half looks like the more important one.
        bar.addWidget(button, 1)

    column.addWidget(row)
    return container


def _pack_to_top_left(grid, rows: int, columns: int) -> None:
    """Send a grid's spare space to one trailing row and column.

    A QGridLayout inside a resizable QScrollArea is stretched to the area's
    height and shares that height between its rows -- so eleven check boxes in a
    220 px box end up 94 px apart. Qt has no "pack" flag; an empty stretched row
    and column at the far edge is the idiom, and the same reason every
    hand-written `.ui` in this repo ends with a vertical spacer.
    """
    grid.setRowStretch(max(rows, 0), 1)
    grid.setColumnStretch(max(columns, 0), 1)


def _horizontal_scroll(widget):
    area = qt.QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
    # Without this the area collapses to a couple of pixels: a QScrollArea's
    # size hint ignores its child, so it has to be told how tall one row of
    # check boxes is.
    area.setMinimumHeight(design.CHART_MIN_HEIGHT)
    return area


def _vertical_scroll(widget):
    area = qt.QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
    area.setMinimumHeight(design.TABS_MIN_HEIGHT)
    return area


# The server's ArgSpec.ui values. A layout it does not know about falls back to
# the flat column with a warning rather than failing the panel: a presentation
# hint from a newer server must never be able to break an older client.
_LAYOUT_BUILDERS = {
    None: _build_flat_boxes,
    "inline": _build_inline_boxes,
    "chips": _build_chips_boxes,
    "grid": _build_grid_boxes,
    "tabs": _build_tabs_boxes,
}

LAYOUTS = tuple(name for name in _LAYOUT_BUILDERS if name)


class JoystickInput:
    """The widgets rendered for a `"vec2"` argument: two numbers set together.

    The two spin boxes ARE the value: `value()` reads them and nothing else.
    The pad (built only when the schema says `ui: "joystick"`) is a second way
    of writing into them (a drag sets both at once) while the boxes remain
    for typing an exact number, the same pairing FlexReg keeps between its
    pads and line edits. Change notification hangs off the boxes alone, so
    every input path (drag, wheel, keys, typing) is one code path.

    A `spring_back` pad is relative: the knob deals out displacements from its
    rest position and the boxes accumulate them (clamped by their own ranges),
    the running total becoming the new base when the gesture ends. Without it
    the pad is absolute and simply mirrors the boxes.
    """

    def __init__(self, x_range=(0.0, 1.0), y_range=(0.0, 1.0), initial=None, step=None,
                 x_axis="X", y_axis="Y", x_labels=None, y_labels=None,
                 spring_back=False, with_pad=True):
        self._syncing = False

        self.container = qt.QWidget()
        column = qt.QVBoxLayout(self.container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(design.SPACING_XS)

        row_container = qt.QWidget()
        row = qt.QHBoxLayout(row_container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(design.SPACING_MD)

        x0, y0 = _vec2_initial(initial, x_range, y_range)
        self._base = (x0, y0)

        self.xBox = _axis_spinbox(x_range, step)
        self.yBox = _axis_spinbox(y_range, step)
        self.xBox.setValue(x0)
        self.yBox.setValue(y0)

        self.pad = None
        if with_pad:
            self.pad = JoystickPad(
                x_range=x_range, y_range=y_range, x_step=step, y_step=step,
                x_labels=x_labels, y_labels=y_labels, spring_back=spring_back,
            )
            self.pad.setDefaults(x0, y0)
            self.pad.setValues(x0, y0)
            self.pad.onChanged = self._onPadMoved
            self.pad.onReleased = self._onPadReleased
            row.addWidget(self.pad)

        boxes_container = qt.QWidget()
        boxes = qt.QFormLayout(boxes_container)
        boxes.addRow(design.section_title(x_axis), self.xBox)
        boxes.addRow(design.section_title(y_axis), self.yBox)
        row.addWidget(boxes_container, 1)
        column.addWidget(row_container)

        self.xBox.valueChanged.connect(self._onBoxEdited)
        self.yBox.valueChanged.connect(self._onBoxEdited)

    def value(self) -> list:
        return [self.xBox.value, self.yBox.value]

    def _onPadMoved(self, pad) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            if pad.spring_back:
                # The knob's offset from its rest position is a displacement
                # dealt onto the committed base, not a value of its own.
                self.xBox.setValue(self._base[0] + (pad.value_x - pad.default_x))
                self.yBox.setValue(self._base[1] + (pad.value_y - pad.default_y))
            else:
                self.xBox.setValue(pad.value_x)
                self.yBox.setValue(pad.value_y)
        finally:
            self._syncing = False

    def _onPadReleased(self, _pad) -> None:
        # The gesture's running total becomes the new base; the pad has
        # already sprung home silently.
        self._base = (self.xBox.value, self.yBox.value)

    def _onBoxEdited(self, *_args) -> None:
        if self._syncing:
            return
        self._base = (self.xBox.value, self.yBox.value)
        if self.pad is not None and not self.pad.spring_back:
            self._syncing = True
            try:
                self.pad.setValues(self.xBox.value, self.yBox.value)
            finally:
                self._syncing = False

    # -- the slice of the QWidget API build()/base_widget use on a field ----

    def setProperty(self, name, value) -> None:
        self.container.setProperty(name, value)

    def setToolTip(self, text) -> None:
        self.container.setToolTip(text)


def _axis_spinbox(bounds, step) -> qt.QDoubleSpinBox:
    """The number a pad is showing, read-only.

    The pad IS the input: it sets both axes with one gesture, and the knob sits
    where the point sits on the arch. A box that also accepts typing gives the
    same value two owners and reads as a form to fill in, which is not what the
    original was -- there the numbers report what the pad is doing.
    """
    box = qt.QDoubleSpinBox()
    box.setReadOnly(True)
    box.setButtonSymbols(qt.QAbstractSpinBox.NoButtons)
    box.setFocusPolicy(qt.Qt.NoFocus)
    low, high = sorted((float(bounds[0]), float(bounds[1])))
    box.setRange(low, high)
    box.setDecimals(_decimals_for_step(step))
    if step:
        box.setSingleStep(float(step))
    return box


def _vec2_initial(initial, x_range, y_range):
    """The pair the panel opens at: the declared `initial`, or the centre of
    both axes (a joystick's rest position, and where a spring_back pad deals
    its displacements from)."""
    if isinstance(initial, (list, tuple)) and len(initial) == 2:
        return float(initial[0]), float(initial[1])
    return ((float(x_range[0]) + float(x_range[1])) / 2.0,
            (float(y_range[0]) + float(y_range[1])) / 2.0)


def _decimals_for_value(initial, step, maximum=6) -> int:
    """Decimal places for a float spin box the schema gave none for: enough
    for its initial value and its step, and never fewer than 2. A flat 6 drew
    a temperature of 0 as `0.000000`, which reads as a measured quantity."""
    places = _decimals_for_step(step, maximum)
    try:
        places = max(places, _decimals_for_step(initial, maximum))
    except (TypeError, ValueError):
        pass
    return max(2, places)


def _decimals_for_step(step, maximum=6) -> int:
    """Decimal places that make `step` representable (0.05 needs 2), with 2
    (the ctk default) when no step is declared."""
    if not step:
        return 2
    step = abs(float(step))
    decimals = 0
    while decimals < maximum and abs(round(step) - step) > 1e-9:
        step *= 10.0
        decimals += 1
    return decimals


class FileOrFolderInput:
    """The local half of an input row: browse buttons and nothing to type in.

    **There is no path field.** There was one, and it said the same thing as
    the caption under the row -- badly: truncated to the width left over after
    two dropdowns and two buttons, it showed a fragment of
    `/tmp/...TestFiles.../sample_scan.nii.gz` where the caption already reads
    `sample_scan.nii.gz - 94 MB - test data, fetched to a temporary folder`.
    Five controls competing on one line, two of them answering the same
    question. The full path stays one hover away, on the container's tooltip.

    Which buttons appear is the argument's own answer: an argument accepting a
    folder gets `Folder...`, one accepting a file gets `File...`, one accepting
    both gets both. The user still never DECLARES which they are providing --
    `is_folder()` reads it off the filesystem, because a folder pasted into a
    field set to "File" used to be uploaded as one and fail at `open()`.

    It replaces `ctkPathLineEdit` on every input row, which is a simplification
    and not only a cosmetic one: that widget emits `currentPathChanged` only
    for input its name filters accept, so a `*.csv` restriction silently
    swallowed the change signal for every FOLDER (measured against Slicer
    5.13). Driving the dialogs here keeps them filtered by the declared
    extensions *and* every selection observable.
    """

    def __init__(self, extensions=(), modes=("file", "folder"), destination=False):
        self._extensions = tuple(extensions)
        # Whether this row is where results GO rather than where inputs come
        # from. Two things follow from it and both are the same reason -- a
        # destination is about a PLACE, not about a file:
        #
        #   * it shows the whole path. A patient's scan is identified by its
        #     name and the directory above it is noise; a folder results are
        #     about to be written into is identified by WHERE it is, and the
        #     name alone (`out`, `Documents`) says nothing.
        #   * it never takes the accent. See design.set_input_filled.
        self._destination = bool(destination)
        # Which KINDS this argument accepts. Published rather than consumed and
        # forgotten: the sources wrapper turns them into segments, and it can
        # no longer read them off a pair of buttons that no longer exists.
        self.modes = tuple(modes)
        self._path = ""
        # Plain Python callbacks, not a Qt signal: this class is an ordinary
        # object, and the field that used to carry the signal is gone.
        self._listeners = []

        # ONE line: what the row holds on the left, the button that changes it
        # on the right. It was two -- a full-width `File...` slab, and a
        # sentence under it saying what that button had produced -- which is
        # three lines per input once the source bar is counted, on a panel
        # where ASO has four of them. The slab was also the loudest thing on
        # the row while being the least informative: it said `File` under a
        # pressed `File` segment.
        #
        # A CARD, not a bare container: the box is what says at a glance
        # whether this input has been given anything (see design.input_card).
        # When a `ServerFileInput` wraps this picker it is that wrapper's card
        # the panel shows, and this one is never added to a layout -- the value
        # field is shared between them, so both are painted all the same.
        self.container = design.input_card()
        row_layout = qt.QHBoxLayout(self.container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(design.SPACING_SM)

        # `caption` rather than `valueField`, because it is what every caller
        # and every test already calls the thing that says what the row holds.
        # What changed is its SHAPE, not its job.
        self.caption = design.value_field(NOTHING_CHOSEN)
        row_layout.addWidget(self.caption, 1)

        # ONE button, whatever the row accepts. Which dialog it opens follows
        # the source the segmented control has chosen (`setBrowseMode`), so a
        # row taking both a file and a folder no longer needs two of them --
        # and the button can say what it DOES rather than which of two kinds it
        # happens to be at this moment.
        self._mode = "folder" if "folder" in modes and "file" not in modes else "file"
        self.selectButton = design.compact_button(SELECT_LABEL)
        self.selectButton.clicked.connect(self._onSelect)
        row_layout.addWidget(self.selectButton, 0)
        self._row = row_layout

    @property
    def currentPath(self) -> str:
        """Same name as ctkPathLineEdit's, so base_widget's readiness check
        treats this field like any other path input."""
        return self._path

    def setCurrentPath(self, value: str) -> None:
        """Set the chosen path and tell whoever is listening.

        Called by a browse dialog and by `set_local_path` when a downloaded
        test file lands -- and it must notify either way, because the caption
        and the Apply button are both driven off that notification.
        """
        value = (value or "").strip()
        if value == self._path:
            return
        self._path = value
        self.describe()
        for listener in list(self._listeners):
            listener()

    def setBrowseMode(self, mode: str) -> None:
        """Which dialog the one button opens: "file" or "folder".

        Set by the wrapper from the segmented control, so the button never has
        to name the kind it picks -- the pressed segment already does, and it
        does it in a place a reader is looking before they reach for the
        button.
        """
        if mode in ("file", "folder"):
            self._mode = mode

    def detachRow(self):
        """Hand this row's two widgets over, and empty the layout holding them.

        The wrapper lays out FOUR sources on one line and shows one, so it
        needs the value field and the button as widgets rather than as the row
        this class packs them into. Emptied explicitly rather than left to Qt's
        re-parenting, so the two layouts never both believe they hold them.
        """
        while self._row.count():
            self._row.takeAt(0)
        return self.caption, self.selectButton

    def describe(self, text: str = None) -> None:
        """Say what this row holds, on its second line.

        `text` is how the sources wrapper speaks for it: a hosted test file or
        a node picked from the scene is not a local path, and only that wrapper
        knows which was chosen. Left out, the row describes its own path.
        """
        if text is None:
            path = self._path
            whole = self._destination
            if not path:
                text = NOTHING_CHOSEN
            elif os.path.isdir(path):
                # A destination needs no `Folder:` in front of it: it only ever
                # holds one, and the word costs a fifth of the box.
                text = describe_folder(path, name_only=not whole)
                text = text if whole else "Folder: {}".format(text)
            else:
                text = describe_file(path, name_only=not whole)
                text = text if whole else "File: {}".format(text)
            if whole:
                # A destination is shown whole, so the home directory is the
                # part to fold: `~/Documents/...` fits the box where
                # `/home/<user>/Documents/...` was cut on its left.
                home = os.path.expanduser("~")
                if home and home != os.sep and text.startswith(home + os.sep):
                    text = "~" + text[len(home):]
            # An input shows the NAME. The box is half a row wide, and a full
            # path elided into it is the one part of itself that means nothing
            # -- `/tmp/tmpgt7qr_f5/pati...` where a reader is looking for
            # `patient1.nii.gz`. The whole path is on the row's tooltip.
            self.container.setToolTip(path)
        self.caption.setText(text)
        # setText leaves the cursor at the end, which scrolls the box to the
        # tail; `value_field` shows the start, and so must every later text.
        self.caption.setCursorPosition(0)
        # The card and the line inside it are one statement, so they are
        # painted together and can never disagree about whether the row is
        # satisfied. A wrapper's own words are never NOTHING_CHOSEN unless
        # nothing was chosen, which is what makes that comparison the answer.
        design.set_input_filled(self.container, self.caption,
                                text != NOTHING_CHOSEN, accent=not self._destination)

    def onPathChanged(self, callback) -> None:
        self._listeners.append(callback)

    def is_folder(self) -> bool:
        """Whether what the user picked is a folder — read off the filesystem,
        never off a mode the user had to set correctly beforehand."""
        path = self.currentPath
        return bool(path) and os.path.isdir(path)

    def _onSelect(self) -> None:
        """Open the dialog the current source asks for."""
        if self._mode == "folder":
            chosen = qt.QFileDialog.getExistingDirectory(
                self.container, BROWSE_FOLDER_LABEL, self.currentPath)
        else:
            chosen = qt.QFileDialog.getOpenFileName(
                self.container, BROWSE_FILE_LABEL, self.currentPath,
                ";;".join(name_filters(self._extensions)))
        if chosen:
            self.setCurrentPath(chosen)

    # -- the slice of the QWidget API build()/base_widget use on a field ----

    def setProperty(self, name, value) -> None:
        self.container.setProperty(name, value)

    def setToolTip(self, text) -> None:
        self.container.setToolTip(text)


class ServerFileInput:
    """One input row for a file argument that can be satisfied three ways: a
    local file or folder to upload, one of the TEST FILES the server hosts for
    this tool, or a scalar volume already OPEN in the scene (any argument
    `accepts_volume` says yes to).

    Everything sits on ONE line, [sources dropdown][path field + browse], and
    the dropdown is the single place a tool's test data is reached from. There
    used to be two: this list, which sent the hosted NAME and left the file on
    the server, and a separate "Test data" button fetching a hardcoded GitHub
    release URL four modules declared by hand. They answered the same question
    differently, and only one of them put the scan where a clinician could
    open it beside the panel.

    The dropdown's entries, in order: the prompt, the hosted test files (each
    labelled with its kind and size, see `hosted_entry_label`), then the open
    volumes (fed by base_widget; formgen never touches the MRML scene). Which
    kind is selected is decided by INDEX (`_selection`), never by parsing the
    text back, so a hosted file whose name happens to start with the volume
    prefix cannot be misread.

    **Picking a hosted TEST FILE is an action, not a state.** It hands the name
    to the callback base_widget registers (`setHostedCallback`), which
    downloads the file and writes the local path back through `set_local_path`
    - at which point the row holds an ordinary local path like any other, the
    combo returns to its prompt, and the run uploads it. Nothing travels as a
    bare name: the file the user asked to see is on their disk, and a file on
    disk is uploaded.

    Two entries are still a state, for the same reason - there is no local path
    to write. An OPEN VOLUME is exported at upload time
    (base_widget._prepareOneInputFile). A hosted MODEL (`hosted_downloads`
    False, ASO's `reference`) is not downloadable at all - the server declines
    to stream weights, which are selected by name and used in place - so it is
    read back by `server_name()` and sent as a plain form value.

    The sources are kept mutually exclusive by clearing the other one, not by
    letting one silently win: picking a dropdown entry empties the path field,
    and typing/browsing a path resets the dropdown. A precedence rule the user
    cannot see is how you end up uploading a file you thought you had replaced.

    Rebuilding the dropdown (`setChoices`/`setVolumeChoices`) preserves the
    current selection by text when it is still offered: both lists are
    refreshed on every `enter()`, and a refresh must not silently reset a
    chosen entry to the first one in the list.
    """

    # First entry, and the state that means "nothing picked FROM THIS LIST": a
    # combo box cannot express an empty selection in a way a user reads as
    # deliberate.
    #
    # It used to be the path field's own placeholder, word for word, on the
    # reasoning that two halves of one row should say one thing. Photographed,
    # that reasoning does not survive: the panel shows "Select a file or a
    # folder" twice, side by side, and the dropdown reads as a duplicate of the
    # field rather than as the one place a tool's test data is reached from.
    # Nobody opens a control that appears to repeat its neighbour.
    #
    # So the prompt now NAMES WHAT IS INSIDE, and this constant is only the
    # fallback for a list with nothing in it. `_prompt` picks the words.
    CHOOSE_OPTION = PATH_PLACEHOLDER
    PROMPT_HOSTED = "Test data..."
    PROMPT_MODEL = "Installed model..."

    # The four ways one file argument can be satisfied. Keys, not labels: the
    # words are `SOURCE_LABELS` and change with the row (`Test data` is
    # `Model` where the hosted entry is weights), the identity does not.
    SOURCE_FILE = "file"
    SOURCE_FOLDER = "folder"
    SOURCE_HOSTED = "hosted"
    SOURCE_SCENE = "scene"
    SOURCE_ORDER = (SOURCE_FILE, SOURCE_FOLDER, SOURCE_HOSTED, SOURCE_SCENE)
    SOURCE_LABELS = {
        SOURCE_FILE: "File",
        SOURCE_FOLDER: "Folder",
        SOURCE_HOSTED: "Test data",
        SOURCE_SCENE: "Imported",
    }
    SOURCE_MODEL_LABEL = "Model"
    SOURCE_HINTS = {
        SOURCE_FILE: "One file on this computer",
        SOURCE_FOLDER: "A folder on this computer",
        SOURCE_HOSTED: "Test data installed with this tool",
        SOURCE_SCENE: "Something already open in Slicer",
    }
    SOURCE_MODEL_HINT = "A model installed with this tool, used where it is"

    def __init__(self, local, hosted_downloads=True, on_hosted=None):
        self.local = local
        self._syncing = False
        self._source = None
        self._hosted = []  # [{"name", "kind", "size"}], in server order
        # How a scene pick is named on the second line, and what its dropdown
        # calls itself. Set by base_widget from the kinds this argument
        # accepts: a CBCT and an intraoral surface are both a Scan, a crop box
        # is not.
        self._scene_label = SCENE_LABELS["volume"]
        # Whether this row can EVER be filled from the scene. Distinct from
        # having something to offer right now: "never" is a property of the
        # argument and hides the control, "nothing at the moment" is a property
        # of the scene and only greys it.
        self._scene_supported = False
        self._popup_warned = False  # the widening failure is reported once
        self._volume_names = []
        # Whether picking a hosted entry FETCHES it. True for the tool's test
        # files, which exist to be looked at. False for a hosted MODEL: the
        # server refuses to stream one on purpose (weights are selected by name
        # and used in place), so for those the name is still what travels.
        self.hosted_downloads = bool(hosted_downloads)
        self._on_hosted = on_hosted

        # A column, not a row: the controls sit on one line and the caption
        # under them. The caption is the only place that can say what is loaded
        # without truncating it -- see `describe_file`.
        #
        # The card is HERE rather than on the picker inside it, because this is
        # the widget the panel actually shows for a wrapped argument: the box
        # has to hold every way of filling the row -- both dropdowns included
        # -- or it would outline two of the four and look like a mistake.
        self.container = design.input_card()
        column = qt.QVBoxLayout(self.container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        # A segmented control over the row: which of the four sources this
        # input is being filled from. Only the ones this argument can actually
        # take are drawn, and it is hidden entirely when there is only one --
        # a choice of one is not a choice.
        self.sourceBar = qt.QWidget()
        self._sourceRow = qt.QHBoxLayout(self.sourceBar)
        self._sourceRow.setContentsMargins(0, 0, 0, design.SPACING_SM)
        self._sourceRow.setSpacing(design.SPACING_XS)
        self.sourceButtons = {}
        self.sourceBar.setVisible(False)
        column.addWidget(self.sourceBar)

        controls = qt.QWidget()
        row = qt.QHBoxLayout(controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(design.SPACING_XS)

        self.combo = qt.QComboBox()
        # Without this a long hosted entry (cohort_10_patients.zip  (file,
        # 94 MB)) widens the dropdown until the path field has no room left on
        # the line.
        # The COLLAPSED box stays narrow, so a long hosted name cannot widen
        # the row until the path field has no room left on the line. The POPUP
        # is a different question: a clinician choosing between
        # `CBCT_Or_FullyAuto_DCM (folder, 532 MB)` and
        # `CBCT_FullyAuto_DCM (folder, 183 MB)` has to read the whole entry --
        # truncated, the two are the same word. `_widenPopup` sizes the list to
        # its longest entry on every rebuild.
        self.combo.sizeAdjustPolicy = qt.QComboBox.AdjustToMinimumContentsLengthWithIcon
        self.combo.minimumContentsLength = 14
        # Replaced on every rebuild, once the list knows what it holds.
        self.combo.setToolTip(self.CHOOSE_OPTION)
        self.combo.addItems([self.CHOOSE_OPTION])
        # Empty until `setChoices` says otherwise, and hidden while it is --
        # the same rule the scene list follows, applied from the start rather
        # than only on the first rebuild.
        self.combo.setVisible(False)

        # A SECOND dropdown, not more entries in the first. The two answer
        # different questions -- "fetch the tool's sample data" and "use what is
        # already open in Slicer" -- and one list mixing them read as a single
        # jumbled menu. It also lets a row that cannot take a scene node simply
        # not show it, which one list cannot express.
        self.sceneCombo = qt.QComboBox()
        self.sceneCombo.sizeAdjustPolicy = \
            qt.QComboBox.AdjustToMinimumContentsLengthWithIcon
        self.sceneCombo.minimumContentsLength = 14
        self.sceneCombo.addItems([self._scenePrompt()])
        self.sceneCombo.setToolTip(self._scenePrompt())
        # Hidden until there is something in it: an empty dropdown is a control
        # that can only disappoint, and most rows never get one.
        self.sceneCombo.setVisible(False)

        # The row is the same shape whatever the source: WHAT IT HOLDS on the
        # left, and on the right the one control that changes it -- a `Select`
        # button for a file or a folder, the list itself for the other two.
        #
        # The value field is shared with the picker inside, which owns the
        # words: a row with dropdowns and a row without then say the same thing
        # about the same file, in the same place.
        # What the picker inside accepts. A bare Qt field declares nothing and
        # is treated as a plain file input, which is what it is.
        self._localModes = tuple(getattr(local, "modes", ("file",)))
        detach = getattr(local, "detachRow", None)
        self._selectButton = None
        if detach:
            self.caption, self._selectButton = detach()
        else:
            # A bare Qt field, which only a test builds now. It has no value
            # field of its own, so the wrapper makes one.
            self.caption = design.value_field(NOTHING_CHOSEN)
            self._selectButton = row_widget(local)
        row.addWidget(self.caption, 1)

        # The right-hand controls, in SOURCE_ORDER. Only one is ever visible,
        # so the order changes nothing on screen -- it keeps the code readable
        # in the order the segments above are drawn.
        self._pickers = {
            self.SOURCE_FILE: self._selectButton,
            self.SOURCE_FOLDER: self._selectButton,
            self.SOURCE_HOSTED: self.combo,
            self.SOURCE_SCENE: self.sceneCombo,
        }
        for control in (self._selectButton, self.combo, self.sceneCombo):
            row.addWidget(control, 0)
        column.addWidget(controls)

        self.combo.currentTextChanged.connect(self._onComboChoice)
        self.sceneCombo.currentTextChanged.connect(self._onSceneChoice)
        connect_changed(local, self._onLocalChoice)
        # Its own connection, not a call from the two handlers: a path also
        # arrives through `set_local_path` when a download lands, which is
        # exactly the case the caption exists for.
        connect_changed(local, self._describe)
        self.combo.currentTextChanged.connect(self._describe)
        self.sceneCombo.currentTextChanged.connect(self._describe)

        self._rebuildSources()

    # -- which of the four sources this row is being filled from -----------

    def _availableSources(self) -> list:
        """The sources this argument can actually be filled from, right now.

        Two of them are properties of the ARGUMENT and never change: whether it
        takes a file, whether it takes a folder. Two are properties of the
        SERVER and the SCENE and change under the panel -- the hosted list
        arrives with the schema, the scene list is refreshed on every enter().
        """
        available = [key for key in (self.SOURCE_FILE, self.SOURCE_FOLDER)
                     if key in self._localModes]
        if self._hosted:
            available.append(self.SOURCE_HOSTED)
        if self._scene_supported:
            available.append(self.SOURCE_SCENE)
        return [key for key in self.SOURCE_ORDER if key in available]

    def _sourceLabel(self, key: str) -> str:
        if key == self.SOURCE_HOSTED and not self.hosted_downloads:
            return self.SOURCE_MODEL_LABEL
        return self.SOURCE_LABELS[key]

    def _sourceHint(self, key: str) -> str:
        if key == self.SOURCE_HOSTED and not self.hosted_downloads:
            return self.SOURCE_MODEL_HINT
        if key == self.SOURCE_SCENE:
            return "A {} already open in Slicer".format(self._scene_label.lower())
        return self.SOURCE_HINTS[key]

    def _rebuildSources(self) -> None:
        """Redraw the segmented control, and keep the chosen source when it is
        still offered.

        Rebuilt rather than merely re-labelled because what is on offer moves:
        a row has one source at `setup()` and three once the schema and the
        scene have been read, and a bar built once would show the first state
        for ever.
        """
        available = self._availableSources()
        while self._sourceRow.count():
            item = self._sourceRow.takeAt(0)
            widget = item.widget() if hasattr(item, "widget") else None
            if widget is not None:
                widget.setParent(None)
        self.sourceButtons = {}

        # One source is not a choice: the control speaks for itself, and a
        # single pressed segment over it would be a decoration that looks like
        # a decision.
        if len(available) > 1:
            for key in available:
                button = design.segment_button(self._sourceLabel(key))
                button.setToolTip(self._sourceHint(key))
                button.connect("clicked()", self._sourcePicker(key))
                self.sourceButtons[key] = button
                # Equal stretch: four sources of equal standing, and a bar that
                # spans exactly the row it commands.
                self._sourceRow.addWidget(button, 1)
        self.sourceBar.setVisible(bool(self.sourceButtons))

        if self._source not in available:
            self._source = available[0] if available else None
        self._showActive()

    def _sourcePicker(self, key: str):
        """A click on one segment, as a callable Qt can hold.

        A closure rather than `functools.partial` on a bound method: PythonQt
        keeps no reference to a partial's target, and the slot stops firing as
        soon as it is collected.
        """
        def picked():
            self._chooseSource(key)
        return picked

    def _chooseSource(self, key: str) -> None:
        """Switch the row to `key`, and EMPTY it.

        Emptying is the point rather than a side effect: one source at a time
        is what the segments say, and a row that kept its scan while showing
        the folder button would be saying two things at once. It is also the
        only moment a clinician can lose a pick by accident, which is why the
        caption underneath goes straight back to saying nothing was chosen.
        """
        if key == self._source:
            return
        self._source = key
        self._clearOthers(keep=None)
        self._showActive()
        self._describe()

    def _showActive(self) -> None:
        """One source's control on the right of the row, one pressed segment.

        The value field on the left is never hidden: every source fills the
        same row, and what it holds is the one thing that does not depend on
        where it came from.
        """
        for key, button in self.sourceButtons.items():
            button.setChecked(key == self._source)
        wanted = self._pickers.get(self._source)
        for control in set(self._pickers.values()):
            control.setVisible(control is wanted)
        # The button opens whichever dialog the chosen source asks for, so it
        # can say `Select` rather than naming a kind the segment already names.
        setter = getattr(self.local, "setBrowseMode", None)
        if setter and self._source in (self.SOURCE_FILE, self.SOURCE_FOLDER):
            setter(self._source)
        # Greyed rather than hidden when the scene holds nothing of the right
        # kind: the segment is how a clinician learns the row can be filled
        # that way at all, and a control that vanishes teaches nobody.
        self.sceneCombo.setEnabled(bool(self._volume_names))

    def setHostedCallback(self, callback) -> None:
        """What to do when the user picks a hosted test file: base_widget
        downloads it and writes the resulting local path back here. Registered
        rather than called directly because the download is HTTP, and this
        module does not speak it (see ARCHITECTURE.md dependency rule)."""
        self._on_hosted = callback

    def setChoices(self, entries) -> None:
        """Fill the dropdown with the server-hosted test files. Called once the
        schema is known and again on every enter(); formgen never talks HTTP.

        Takes either the normalised entries (`client.testfile_entries`) or bare
        names, so a caller that only has names - and every test that only cares
        about order - needs no ceremony.
        """
        self._hosted = [_hosted_entry(entry) for entry in entries]
        self._rebuild()

    def setSceneSupported(self, supported: bool) -> None:
        """Whether the scene can answer this argument AT ALL.

        A transform or a spreadsheet has no counterpart in a scene, so the
        control is not shown -- there is nothing to learn from it. Everything
        else keeps it, greyed while the scene holds nothing of the right kind:
        hidden, the feature is invisible until the day it happens to be
        available, and nobody discovers a control that is not there.
        """
        self._scene_supported = bool(supported)
        self._rebuildScene()

    def setSceneLabel(self, label: str) -> None:
        """What a scene pick is called on the caption -- Scan, ROI -- and,
        through `scene_prompt_for`, what its dropdown calls itself.

        Redraws, because the label IS the prompt's source: set after the list
        was last built, it would otherwise name the row correctly on the
        caption and leave the old words at the top of the dropdown.
        """
        self._scene_label = label or SCENE_LABELS["volume"]
        self._rebuildScene()

    def _scenePrompt(self) -> str:
        return scene_prompt_for(self._scene_label)

    def setVolumeChoices(self, names) -> None:
        """What the scene currently offers THIS argument, as display names.

        base_widget owns the name-to-node mapping, the refresh triggers and the
        narrowing by kind; this widget only offers the entries. An empty list
        hides the dropdown rather than leaving an inert one on the row.
        """
        self._volume_names = list(names)
        self._rebuildScene()

    def hosted_entries(self) -> list:
        """The hosted test files currently offered, as the entries they were
        set from - base_widget reads `kind` off this to decide whether the
        download is unpacked and whether it is loaded into the scene."""
        return list(self._hosted)

    def _prompt(self) -> str:
        """The hosted list's first entry, naming what IT holds.

        No longer has to speak for the scene as well: that is a dropdown of its
        own now, with its own prompt. `PROMPT_BOTH` went with the merge -- one
        list describing two unrelated sources is what read as a jumble.

        A model row is its own case: those entries are not fetched, they are
        the value, so "Test data" would be wrong twice over.
        """
        if self._hosted and not self.hosted_downloads:
            return self.PROMPT_MODEL
        if self._hosted:
            return self.PROMPT_HOSTED
        return self.CHOOSE_OPTION

    def _entries(self) -> list:
        return [self._prompt()] + [
            hosted_entry_label(entry) for entry in self._hosted
        ]

    def _sceneEntries(self) -> list:
        """The prompt, then the scene's own node names, unadorned.

        They used to be prefixed `Open volume: `, from when one dropdown held
        the hosted files and the scene together and a reader needed telling
        which was which. The scene has had a list of its own since, whose first
        entry names what it holds -- so the prefix was the same four words
        repeated down every row of it.
        """
        return [self._scenePrompt()] + list(self._volume_names)

    def _rebuildScene(self) -> None:
        """Redraw the scene list, keeping the current pick when still offered.

        A node closed in Slicer takes its entry with it, and a selection that
        is gone must not silently become the prompt's neighbour: `_selection`
        reads by INDEX, so a stale index would name the wrong scan.
        """
        previous = self.sceneCombo.currentText
        self._syncing = True
        try:
            self.sceneCombo.clear()
            entries = self._sceneEntries()
            self.sceneCombo.addItems(entries)
            # Searched from index 1, never from 0: the entries are bare node
            # names now, so a node named exactly like the prompt would
            # otherwise restore to the prompt and read as nothing chosen.
            if previous in entries[1:]:
                self.sceneCombo.setCurrentIndex(entries.index(previous, 1))
            self.sceneCombo.setToolTip(entries[0])
        finally:
            self._syncing = False
        # What is VISIBLE is the segmented control's business, not this
        # method's: a list may be refreshed while another source is showing.
        self._rebuildSources()

    def _rebuild(self) -> None:
        previous = self.combo.currentText
        # Guarded: clear()+addItems reselects index 0, which would otherwise
        # run the mutual-exclusion sync for a choice the user never made -- and
        # for a hosted entry, would start a download nobody asked for.
        self._syncing = True
        try:
            self.combo.clear()
            entries = self._entries()
            self.combo.addItems(entries)
            if previous in entries:
                self.combo.setCurrentIndex(entries.index(previous))
            self._widenPopup()
            self.combo.setToolTip(entries[0])
        finally:
            self._syncing = False
        # Visibility belongs to the segmented control. What this decides is
        # whether the source exists at all -- most arguments host no test files
        # and never get the segment.
        self._rebuildSources()

    def _widenPopup(self) -> None:
        """Let the dropdown LIST show a whole entry, however narrow the box is.

        The collapsed box is deliberately narrow -- 168 px on AREG -- so that a
        long entry cannot push the path field off the line. The open list has no
        such excuse, and AREG offers `CBCT_FullyAuto`, `CBCT_Or_FullyAuto` and
        `CBCT_Or_FullyAuto_DCM`: elided to the box's width all three read
        `CBCT_Or_Full...`, and choosing between them is guesswork.

        **Measured by the VIEW, never by font metrics.** `combo.fontMetrics` is
        a SLOT under PythonQt, not a property. The previous version read it
        without calling it, so every measurement raised `AttributeError` into a
        bare `except` and the list was never widened once -- measured in Slicer,
        `view.minimumWidth` stayed 0 and the open popup was 166 px wide.
        `sizeHintForColumn` asks the view what its own items need, delegate
        included, and touches no font API at all.

        A failure here is still cosmetic and must not take the panel down. It is
        logged as a WARNING rather than at debug, and once per widget: a silent
        debug line is how a dead feature stayed dead through 728 passing tests.
        """
        try:
            view = self.combo.view()
            needed = view.sizeHintForColumn(0)
            if needed and needed > 0:
                view.setMinimumWidth(min(needed + _POPUP_PADDING, _POPUP_MAX_WIDTH))
        except Exception:  # noqa: BLE001 - a dropdown that is merely narrow
            if not self._popup_warned:
                self._popup_warned = True
                logger.warning("could not widen the dropdown list", exc_info=True)

    def _selection(self):
        """("none" | "hosted" | "volume", name) for whichever list holds a pick.

        Decided by INDEX in both, never by parsing the text back: a hosted file
        whose name happens to start with the volume prefix cannot be misread,
        and the two lists cannot both be picked -- choosing in one resets the
        other, the same way either resets the path field.
        """
        index = self.combo.currentIndex
        if 0 < index <= len(self._hosted):
            return "hosted", self._hosted[index - 1]["name"]
        index = self.sceneCombo.currentIndex
        if 0 < index <= len(self._volume_names):
            return "volume", self._volume_names[index - 1]
        return "none", ""

    def hosted_name(self) -> str:
        """The hosted entry currently showing in the dropdown, or "".

        For a downloadable one (a test file) this is only non-empty between the
        pick and the moment the download hands back a path -- which resets the
        dropdown -- so it is progress feedback rather than a value.
        """
        kind, name = self._selection()
        return name if kind == "hosted" else ""

    def server_name(self) -> str:
        """The hosted name that TRAVELS to the server as a plain form value,
        or "".

        Only a MODEL does. A model is never downloadable -- the weights stay
        where they are and the run names them -- so that selection is a value
        the way it always was. A test file is fetched instead and becomes an
        ordinary local path, which is why it answers "" here.
        """
        return "" if self.hosted_downloads else self.hosted_name()

    def volume_name(self) -> str:
        """The chosen open volume's display name, or "" otherwise."""
        kind, name = self._selection()
        return name if kind == "volume" else ""

    @property
    def currentPath(self) -> str:
        """The LOCAL path to upload: empty while an open volume or a hosted
        model is chosen, so nothing is read off disk for an argument that is
        already satisfied another way. A downloaded test file IS an ordinary
        local path and reads back here like one."""
        if self.volume_name() or self.server_name():
            return ""
        return _local_path(self.local)

    def is_folder(self) -> bool:
        checker = getattr(self.local, "is_folder", None)
        return bool(checker()) if checker else False

    def clear(self) -> None:
        """Empty this input: no local path, no hosted pick, no open volume.

        For a caller OUTSIDE the dropdown that decides this argument is no longer
        satisfied -- AutoMatrix's mirror check box fills the field when it is
        ticked, and must not leave the mirror matrix behind when it is cleared.
        Guarded by `_syncing` like every other programmatic write here, so
        emptying the field does not read as the user having picked something.
        """
        self._syncing = True
        try:
            _set_local_path(self.local, "")
            self.combo.setCurrentIndex(0)  # the prompt: `_entries` leads with it
        finally:
            self._syncing = False
        self._describe()

    def _onComboChoice(self, _text=None) -> None:
        if self._syncing or self.combo.currentIndex <= 0:
            return
        index = self.combo.currentIndex
        if index > len(self._hosted):
            return
        name = self._hosted[index - 1]["name"]
        # Whatever else satisfied this argument is not what the user just asked
        # for. Cleared before the download starts, not after it lands, so a run
        # launched mid-download cannot send the previous file.
        self._clearOthers(keep=self.combo)
        if self.hosted_downloads and self._on_hosted is not None:
            self._on_hosted(name)

    def _onSceneChoice(self, _text=None) -> None:
        if self._syncing or self.sceneCombo.currentIndex <= 0:
            return
        self._clearOthers(keep=self.sceneCombo)

    def _clearOthers(self, keep) -> None:
        """One source at a time, cleared rather than ranked.

        A precedence rule the user cannot see is how you end up sending a file
        you thought you had replaced -- which is why picking in either list
        empties the path field and resets the other list, and why typing a path
        resets both.
        """
        self._syncing = True
        try:
            if keep is not self.local:
                _set_local_path(self.local, "")
            if keep is not self.combo:
                self.combo.setCurrentIndex(0)
            if keep is not self.sceneCombo:
                self.sceneCombo.setCurrentIndex(0)
        finally:
            self._syncing = False

    def _describe(self, *_args) -> None:
        """Say what this input holds, under the row, whatever satisfied it.

        Three sources, three sentences, because "what is in this field" has
        three different answers and a path field can only ever show one of
        them -- badly.
        """
        volume = self.volume_name()
        model = self.server_name()
        path = _local_path(self.local)
        if volume:
            # Named for what it IS, from the node's own class: "Volume" over a
            # surface would be wrong on ALI, which takes either.
            text = "{}: {}".format(self._scene_label, volume)
        elif model:
            text = "Model: {}".format(model)
        elif path and self._is_fetched(path):
            # Its hosted NAME, not the path it landed on: a fetched file sits
            # in a session directory that is swept on exit, and a user who
            # mistakes that path for their own copy will look for it later.
            text = "Test File: {}".format(describe_file(path))
        elif path:
            # The picker's own wording, so a row with dropdowns and one without
            # say the same thing about the same file.
            text = None
        else:
            text = NOTHING_CHOSEN
        self._say(text)
        # The full path stays reachable without taking a line of its own.
        set_tooltip(self.local, path)

    def _say(self, text) -> None:
        """Put `text` on the row's second line, wherever that line lives.

        `None` means "describe your own path", which only a picker that owns a
        caption can do -- a bare Qt field gets the neutral words instead.
        """
        # `None` means the picker answers for itself, so what it holds is what
        # decides; anything else is this wrapper's own sentence.
        filled = (bool(_local_path(self.local)) if text is None
                  else text != NOTHING_CHOSEN)
        describe = getattr(self.local, "describe", None)
        if describe is not None:
            # It owns the caption -- detached into this column, but still its
            # widget -- so it paints that, and this paints the box around it.
            describe(text)
            design.set_input_filled(self.container, None, filled)
        else:
            self.caption.setText(text or NOTHING_CHOSEN)
            design.set_input_filled(self.container, self.caption, filled)

    def _is_fetched(self, path: str) -> bool:
        """Whether this path is one of the hosted entries this row offered.

        By NAME rather than by directory: the panel owns where a download
        lands, this widget does not, and asking would couple the two.
        """
        name = os.path.basename(path.rstrip(os.sep))
        return any(entry.get("name") == name for entry in self._hosted)

    def _onLocalChoice(self, *_args) -> None:
        if self._syncing or not _local_path(self.local):
            return
        self._clearOthers(keep=self.local)

    # -- the slice of the QWidget API build()/base_widget use on a field ----

    def setProperty(self, name, value) -> None:
        self.container.setProperty(name, value)

    def setToolTip(self, text) -> None:
        self.container.setToolTip(text)


def _hosted_entry(entry) -> dict:
    """One dropdown entry, from either shape a caller may hold: the normalised
    `{"name", "kind", "size"}` of `client.testfile_entries`, or a bare name."""
    if isinstance(entry, dict):
        return {
            "name": entry.get("name", ""),
            "kind": entry.get("kind"),
            "size": entry.get("size"),
        }
    return {"name": str(entry), "kind": None, "size": None}


def local_input(widget):
    """The half of an input row that holds a LOCAL path.

    A row is sometimes a composite: an argument the server hosts test files for
    is wrapped in a `ServerFileInput`, whose `local` is the real picker. Anyone
    reaching for the picker has to go through here, and forgetting to is not a
    visible mistake -- `getattr(wrapper, "setSceneCallback", None)` simply
    answers None, and whatever was being wired stays silently unwired. That is
    exactly how the Scene button shipped inert: ALI's input is the one that has
    hosted test files, so it is the one that is wrapped.
    """
    return widget.local if isinstance(widget, ServerFileInput) else widget


def set_local_path(widget, value: str) -> None:
    """Write a local path into any input-row kind: what base_widget fills in
    once a hosted test file has been downloaded. Writing the local half of a
    ServerFileInput also resets its dropdown, through its own sync."""
    _set_local_path(local_input(widget), value)


# Which browse buttons each input mode gets. The user still never declares
# which of the two they are giving -- `is_folder` reads it off the filesystem --
# this only says which are OFFERED, from what the argument accepts.
_BROWSE_MODES = {
    "single_file": ("file",),
    "folder_zip": ("folder",),
    "file_or_folder": ("file", "folder"),
}


def _local_path(widget) -> str:
    return (getattr(widget, "currentPath", "") or "").strip()


def _set_local_path(widget, value: str) -> None:
    """Write a path into whichever picker kind `widget` is.

    Every row a panel builds is a FileOrFolderInput now, the output folder
    included. The `currentPath` branch is kept for a bare ctkPathLineEdit,
    which only a test constructs -- and for anything a module puts on its own
    panel through `addExtraWidgets`.
    """
    setter = getattr(widget, "setCurrentPath", None)
    if setter is not None:
        setter(value)
    else:
        widget.currentPath = value


def set_tooltip(widget, text: str) -> None:
    """Put `text` on whichever picker kind `widget` is, and on nothing else.

    The caption says which file; the tooltip says where it sits. A composite
    row has no tooltip of its own, so it goes on the container the pointer is
    actually over -- which is now the whole button group, the path field having
    gone.
    """
    target = getattr(widget, "container", widget)
    setter = getattr(target, "setToolTip", None)
    if setter:
        setter(plain_text(text or ""))


def row_widget(field):
    """The QWidget to put in a form row for `field` — composite fields lay
    several widgets out inside a container."""
    return getattr(field, "container", field)


# Words a clinician reads as one unit, kept in their own case rather than
# sentence-cased into nonsense: "CBCT regions", not "Cbct regions".
#
# **This table lives here and nowhere else, on purpose.** It used to sit in the
# server, which put the names of the served tools inside a server built not to
# know them -- an executable claim of that project, checked on every build by
# scripts/domain_coupling.py, and it failed. A dental vocabulary belongs to the
# side that is dental: this extension. The server derives "Cbct regions" from
# the argument name, knowing nothing; this finishes the word.
ACRONYMS = {
    "cbct": "CBCT", "ios": "IOS", "mri": "MRI", "ct": "CT", "roi": "ROI",
    "id": "ID", "gpu": "GPU", "cpu": "CPU", "vram": "VRAM", "dicom": "DICOM",
    "vtk": "VTK", "stl": "STL", "nifti": "NIfTI", "tmj": "TMJ", "llm": "LLM",
    "3d": "3D", "2d": "2D", "fdi": "FDI", "icp": "ICP", "aso": "ASO",
    "ali": "ALI", "areg": "AREG", "amasss": "AMASSS",
}


def spell_acronyms(text: str) -> str:
    """Put the vocabulary's own case back into a label, word by word.

    Applied to whatever is displayed, declared or derived. A tool that wrote
    "Cbct landmarks" by hand gets the same courtesy, and a tool that wrote
    something no token matches -- "Scan / Landmark Folder" -- comes through
    untouched, which is the case that matters most.
    """
    words = []
    for word in text.split(" "):
        stripped = word.strip()
        replacement = ACRONYMS.get(stripped.lower())
        words.append(word.replace(stripped, replacement) if replacement else word)
    return " ".join(words)


def label_for(name: str, spec: dict) -> str:
    """The text shown next to an argument's widget.

    **The schema's `label` when it declares one**, so the words a user reads
    are the tool's own — "Scan / Landmark Folder", not something this file
    invented. The fallback prettifies the argument name and is exactly that: a
    fallback for a tool that declares none. It has no way to know that ASO's
    `input` is the folder holding both the scans and their landmarks.

    What it CAN do, and the server cannot, is spell the vocabulary: the server
    hands over "Cbct regions" because knowing that CBCT is a word would make it
    know its tools. See ACRONYMS above.

    There is ONE rule and it lives here. There used to be two — `build()` used
    the raw schema name while base_widget prettified it — so a single panel
    showed "Reference" and "cbct_landmarks" one above the other.
    """
    declared = (spec.get("label") or "").strip()
    return spell_acronyms(declared or name.replace("_", " ").capitalize())


def section_of(spec: dict) -> str:
    """The collapsible box this argument belongs in. An argument declaring no
    `section` — every argument of every tool but ASO today — lands in the one
    box a panel has always had, so the grouping is opt-in per tool and no
    existing panel moves."""
    return spec.get("section") or DEFAULT_SECTION


# A section whose arguments are laid out in a grid rather than one per row.
# Declared per ARGUMENT (`section_columns`) because that is the only place the
# schema has to hang a hint, and read back per section: every argument in one
# section must agree, and the first that speaks wins.
#
# FlexReg is why. Its four patch corners are a 2x2 that MIRRORS THE ARCH -- left
# column one side, right column the other, top row anterior -- so a pad's
# position on screen is where that corner is in the mouth. Stacked one per row
# that meaning is gone, and the panel is four identical pads in a column.
def cell_of(name: str, spec: dict) -> str:
    """Which grid cell an argument shares. Its own name when it names none.

    Several arguments describing ONE thing belong together: FlexReg's anterior
    right corner is a tooth number and a position along it, and upstream drew
    them in one box with the pad. One argument per cell puts the four teeth in a
    column and the four pads in another, which is a table of arguments rather
    than a picture of an arch.
    """
    return spec.get("cell") or name


def section_columns(arguments_schema: dict, section: str) -> int:
    """How many columns `section` is laid out in. 1 is one argument per row."""
    for spec in arguments_schema.values():
        if section_of(spec) == section:
            declared = spec.get("section_columns")
            if declared:
                try:
                    return max(1, int(declared))
                except (TypeError, ValueError):
                    return 1
    return 1


def sections_of(arguments_schema: dict, extra=()) -> list:
    """Every distinct section a tool's arguments name, in the order they are
    first mentioned — the schema's declaration order, which is the tool
    author's intended reading order. `extra` names boxes the client adds on its
    own (the output folder), appended unless an argument already claimed them.
    """
    ordered = []
    for spec in arguments_schema.values():
        name = section_of(spec)
        if name not in ordered:
            ordered.append(name)
    for name in extra:
        if name not in ordered:
            ordered.append(name)
    return ordered


def is_visible(spec: dict, values: dict) -> bool:
    """Whether `visible_when` is satisfied by the panel's current values.

    `{"modality": "CBCT", "automation": "Fully-Automated"}` — every entry must
    match, and a tuple/list of values means "any of these". An argument
    declaring nothing is always visible.

    A controlling argument absent from `values` counts as NOT matching. That
    only happens when the schema could not be fetched (so the panel holds an
    error, not a form) or when a server declares a `visible_when` naming an
    argument it doesn't publish — which its own check_schema rejects at boot.
    Hiding is the safe answer either way: a field whose precondition cannot be
    evaluated is a field the user cannot fill in meaningfully.
    """
    # `hidden` is not a condition: it is never rendered, whatever the panel
    # holds. It carries the arguments a clinician has no business being asked
    # -- which CUDA device, nnUNet's tile step size -- set by whoever deploys
    # the server. The tool still declares them and still applies its own
    # defaults; the client simply does not ask.
    if spec.get("hidden"):
        return False

    return conditions_hold(spec.get("visible_when"), values)


def conditions_hold(conditions, values: dict) -> bool:
    """Whether every `{argument: value(s)}` entry matches the panel's values.

    No conditions hold trivially. Shared by `visible_when` and `required_when`,
    which are written the same way.
    """
    if not conditions:
        return True
    for other_name, expected in conditions.items():
        if other_name not in values:
            return False
        wanted = expected if isinstance(expected, (list, tuple)) else (expected,)
        if values[other_name] not in wanted:
            return False
    return True


def is_required(spec: dict, values: dict) -> bool:
    """Whether this argument must be filled, given what the panel holds.

    `required` is one answer for every request. `required_when` is the answer
    for an input needed in one shape of the request and absent from another --
    AREG's T1 and T2 with one folder per type, its folder of patients with one
    per patient -- where the schema has to say `required: false`, since either
    may be the one not sent.
    """
    if spec.get("required"):
        return True
    conditions = spec.get("required_when")
    return bool(conditions) and conditions_hold(conditions, values)


def may_be_required(spec: dict) -> bool:
    """Whether the field is ever required: what its label says. A
    `required_when` input is shown only where its condition holds, so it reads
    as required wherever a reader sees it."""
    return bool(spec.get("required") or spec.get("required_when"))


def allowed_groups(spec: dict, values: dict):
    """How this argument's options are grouped, given what the panel holds.

    `groups_when` is `options_when` one level up, and it exists for the same
    reason: a facade puts two engines behind one argument, and ALI's four
    anatomical regions are not another spelling of its five intraoral families.
    Without it the panel showed whichever engine the server composed first, so
    intraoral landmarks were laid out under `Cranial base`.

    None means "no rule": render `groups` as declared.
    """
    rules = spec.get("groups_when")
    if not rules:
        return None
    for other_name, by_value in rules.items():
        chosen = values.get(other_name)
        if isinstance(chosen, dict):  # a multichoice controlling one is not a case
            return None
        groups = by_value.get(chosen)
        if groups is not None:
            return groups
    return None


def allowed_options(spec: dict, values: dict):
    """The options a choice argument may offer, given what the panel holds.

    None means "no rule, offer them all". `visible_when` can only show or hide
    a whole field; this narrows one that stays. AREG's three automation modes
    are all real, but IOS has no "Oriented + Fully-Automated" — offering it and
    refusing the run at the end is the worst of both.
    """
    rules = spec.get("options_when")
    if not rules:
        return None
    allowed = None
    for other_name, by_value in rules.items():
        chosen = values.get(other_name)
        if chosen is None or chosen not in by_value:
            # Nothing said about this state: a rule that cannot be evaluated
            # must not silently empty the box.
            continue
        permitted = list(by_value[chosen])
        allowed = permitted if allowed is None else [o for o in allowed if o in permitted]
    return allowed


def controlling_arguments(arguments_schema: dict) -> set:
    """Every argument the panel has to re-evaluate on — the ones some other
    argument's visibility, or its set of options, depends on."""
    return {
        other_name
        for spec in arguments_schema.values()
        for key in ("visible_when", "options_when", "required_when")
        for other_name in (spec.get(key) or {})
    }


def build(arguments_schema: dict, layout, sections=None, rows=None) -> dict:
    """Add one row per non-file argument. Returns {arg_name: widget}.

    `layout` is a qt.QFormLayout — the single-box behavior, kept as the default
    so a caller that knows nothing about sections is unaffected. `sections` is
    {section name: QFormLayout}; when given, each argument goes to the layout
    its `section` names and `layout` is only the fallback for a section the
    caller didn't create.

    `rows`, if given, is filled with `{arg_name: (label, field)}` — the two
    widgets a caller has to show or hide together to make a row appear or
    disappear. An out-parameter rather than a second return value so the
    signature stays what every existing caller and test expects; the labels are
    created in here, and a caller cannot recover a QFormLayout's label for a
    field reliably across PythonQt versions.
    """
    widgets = {}
    # {(layout, cell name): the QWidget holding that cell}, so several arguments
    # naming one cell stack inside it instead of taking a cell each.
    grid_cells = {}
    # {id(layout): how many segmented choices already head it}.
    heads = {}
    for name, spec in arguments_schema.items():
        if is_file_type(spec.get("type", "")):
            continue

        widget = _make_widget(name, spec)
        widget.setProperty(ARG_NAME_PROPERTY, name)
        description = spec.get("description")
        if description:
            # Schema text, from the server: never rendered as HTML.
            widget.setToolTip(plain_text(description))

        text = label_for(name, spec)
        # The description hangs off the LABEL as well as off the field, and for
        # the widgets that refuse it (a multichoice, whose container would hand
        # it to each of its chips) the label is the only place it survives at
        # all. It is also where a reader looks for it: the label is what names
        # the thing they do not understand.
        explained = bool(description)
        label = design.fit_row_label(
            design.required_label(text, explained) if spec.get("required")
            else design.section_title(text, explained))
        if description:
            label.setToolTip(plain_text(description))
        target = (sections or {}).get(section_of(spec), layout)
        field = row_widget(widget)
        if isinstance(widget, SegmentedChoice) and hasattr(target, "insertRow"):
            # Above the section's other rows, the file inputs included, which a
            # panel adds before any scalar: a segmented choice decides what the
            # rows under it ask for, so it is read first. In declaration order
            # among themselves.
            position = heads.get(id(target), 0)
            target.insertRow(position, label, field)
            heads[id(target)] = position + 1
        elif isinstance(widget, qt.QCheckBox) and hasattr(target, "addRow"):
            # A yes/no is read as one sentence, box first, across the whole
            # row. In the label column its words set the column's width, and
            # "Standardise the CBCT orientation (Frankfurt)" pushed every
            # field of the section to the side to put a box under them.
            field = _toggle_row(widget, label)
            target.addRow(field)
        elif hasattr(target, "addRow"):
            target.addRow(label, field)
        else:
            # A grid section: the caller handed a QGridLayout instead, and the
            # label goes above its field rather than beside it, so a 2x2 of pads
            # reads as a 2x2 rather than as four labelled rows.
            cell = qt.QWidget()
            stack = qt.QVBoxLayout(cell)
            stack.setContentsMargins(0, 0, 0, 0)
            stack.addWidget(label)
            stack.addWidget(field)
            # Read back from the schema, never stored on the layout: PythonQt
            # forbids creating an attribute on a C++ object, so `grid.columns =
            # 2` fails with "creating new attributes on C++ objects is not
            # allowed" and takes the whole panel down.
            columns = section_columns(arguments_schema, section_of(spec))
            key = (id(target), cell_of(name, spec))
            holder = grid_cells.get(key)
            if holder is None:
                holder = qt.QWidget()
                qt.QVBoxLayout(holder).setContentsMargins(0, 0, 0, 0)
                placed = len(
                    [k for k in grid_cells if k[0] == id(target)])
                target.addWidget(holder, placed // columns, placed % columns)
                grid_cells[key] = holder
            holder.layout().addWidget(cell)
        widgets[name] = widget
        if rows is not None:
            rows[name] = (label, field)
    return widgets


def _toggle_row(box, label):
    """A check box with its label to its right, as one widget for a row that
    spans the form. The label stays a QLabel, so it keeps its style and the
    dotted rule that says it has a tooltip. It is not made clickable: PythonQt
    refuses a Python attribute on a C++ object, so `mousePressEvent` cannot be
    replaced, and an event filter for one label is more than it is worth."""
    row = qt.QWidget()
    line = qt.QHBoxLayout(row)
    line.setContentsMargins(0, 0, 0, 0)
    line.setSpacing(design.SPACING_SM)
    line.addWidget(box)
    line.addWidget(label)
    line.addStretch(1)
    return row


def _make_widget(name: str, spec: dict):
    arg_type = spec.get("type", "str")

    # A scalar argument flagged server_selectable (e.g. SurgMovPred's
    # "model": the *name* of a model hosted on the server) is a choice among
    # server-side files, not free text: render a dropdown. base_widget
    # populates it from GET /tools/{tool}/data once the schema is known —
    # formgen itself never talks HTTP (dependency rule, see ARCHITECTURE.md).
    # Checked before the type so a server-filled dropdown is never overwritten
    # with a schema-declared choice list.
    if spec.get("server_selectable"):
        return qt.QComboBox()

    # `initial` is the scalar counterpart of a choice argument's `choices`: the
    # value the SERVER wants the widget to start at. It matters because collect()
    # always sends every widget, so a field the user never touched still travels
    # — a spin box left at Qt's own 0 sent 0, never letting the tool's own
    # default apply. None means the tool declared none; leave Qt's default then.
    initial = spec.get("initial")

    if arg_type == "str":
        widget = qt.QLineEdit()
        if initial is not None:
            widget.setText(str(initial))
        return widget
    if arg_type == "int":
        return _make_numeric_widget(name, spec, integer=True, initial=initial)
    if arg_type == "float":
        return _make_numeric_widget(name, spec, integer=False, initial=initial)
    if arg_type == "bool":
        widget = qt.QCheckBox()
        if initial is not None:
            widget.setChecked(bool(initial))
        return widget
    if arg_type == "vec2":
        return _make_vec2_widget(name, spec)
    if arg_type == "choice":
        return _make_choice_widget(name, spec)
    if arg_type == "multichoice":
        return MultiChoiceGroup(
            _choices(name, spec),
            layout=spec.get("ui"),
            groups=spec.get("groups"),
            option_help=spec.get("option_help"),
            select_all=bool(spec.get("select_all")),
            note=spec.get("note"),
        )
    if is_file_type(arg_type):
        return file_widget(spec)

    logger.warning("Unknown argument type '%s' for '%s', falling back to QLineEdit", arg_type, name)
    return qt.QLineEdit()


def _make_numeric_widget(name: str, spec: dict, integer: bool, initial):
    """An int/float argument. `ui: "slider"` (with min/max declared) renders
    the combined slider+spinbox; otherwise a spin box whose range and step
    still honour any declared bounds. min/max alone constrain the field, they
    never switch the widget kind, so a bound added server-side for validation
    cannot silently turn a spin box into a slider."""
    ui = spec.get("ui")
    if ui == SLIDER_UI:
        slider = _make_slider_widget(name, spec, integer)
        if slider is not None:
            return slider
    elif ui is not None:
        logger.warning(
            "Unknown %s ui '%s' for '%s', falling back to a spin box",
            "int" if integer else "float", ui, name,
        )

    if integer:
        widget = qt.QSpinBox()
        widget.setRange(
            -2147483648 if spec.get("min") is None else int(spec["min"]),
            2147483647 if spec.get("max") is None else int(spec["max"]),
        )
        if spec.get("step") is not None:
            widget.setSingleStep(int(spec["step"]))
        if initial is not None:
            widget.setValue(int(initial))
        return widget

    widget = qt.QDoubleSpinBox()
    widget.setRange(
        -1e12 if spec.get("min") is None else float(spec["min"]),
        1e12 if spec.get("max") is None else float(spec["max"]),
    )
    declared = spec.get("decimals")
    widget.setDecimals(int(declared) if declared is not None
                       else _decimals_for_value(initial, spec.get("step")))
    if spec.get("step") is not None:
        widget.setSingleStep(float(spec["step"]))
    if initial is not None:
        widget.setValue(float(initial))
    return widget


def _make_slider_widget(name: str, spec: dict, integer: bool):
    """A bounded int/float rendered as a ctkSliderWidget, the slider + spin
    box combination GreedyReg's manual-alignment rows use. Returns None when
    the schema asked for a slider without both bounds: an unbounded slider has
    no geometry, so the argument falls back to a plain spin box rather than
    failing the panel."""
    minimum, maximum = spec.get("min"), spec.get("max")
    if minimum is None or maximum is None:
        logger.warning(
            "Argument '%s' asks for ui \"slider\" but declares no min/max bounds, "
            "falling back to a spin box", name,
        )
        return None

    widget = ctk.ctkSliderWidget()
    widget.minimum = float(minimum)
    widget.maximum = float(maximum)
    step = spec.get("step")
    if integer:
        widget.decimals = 0
        widget.singleStep = float(step) if step is not None else 1.0
    else:
        declared = spec.get("decimals")
        widget.decimals = int(declared) if declared is not None else _decimals_for_step(step)
        if step is not None:
            widget.singleStep = float(step)
    initial = spec.get("initial")
    if initial is not None:
        widget.value = float(initial)
    return widget


def _make_vec2_widget(name: str, spec: dict):
    """A `"vec2"` argument: two numbers set together. `ui: "joystick"` adds
    the 2D pad next to the boxes; any other hint falls back to the boxes
    alone, same rule as the multichoice layouts: a newer server's presentation
    hint must never break an older client."""
    ui = spec.get("ui")
    if ui is not None and ui != JOYSTICK_UI:
        logger.warning("Unknown vec2 ui '%s' for '%s', falling back to two spin boxes", ui, name)
    return JoystickInput(
        x_range=_axis_range(name, spec, "x_range"),
        y_range=_axis_range(name, spec, "y_range"),
        initial=spec.get("initial"),
        step=spec.get("step"),
        x_axis=spec.get("x_label") or "X",
        y_axis=spec.get("y_label") or "Y",
        x_labels=_axis_labels(spec.get("x_labels")),
        y_labels=_axis_labels(spec.get("y_labels")),
        spring_back=bool(spec.get("spring_back")),
        with_pad=ui == JOYSTICK_UI,
    )


def _axis_range(name: str, spec: dict, key: str):
    """One vec2 axis. Index 0 is the left/bottom end, index 1 the right/top,
    so declaring the bounds inverted mirrors the axis (see JoystickPad)."""
    declared = spec.get(key)
    if isinstance(declared, (list, tuple)) and len(declared) == 2 and declared[0] != declared[1]:
        return float(declared[0]), float(declared[1])
    if declared is not None:
        logger.warning("Argument '%s' declares an invalid %s %r, using (0, 1)", name, key, declared)
    return (0.0, 1.0)


def _axis_labels(declared):
    if isinstance(declared, (list, tuple)) and len(declared) == 2:
        return tuple(str(label) for label in declared)
    return None


class SegmentedChoice:
    """A `"choice"` drawn as segments: every option on screen, one pressed.

    The same pills a file input uses to say where its data comes from
    (`design.segment_button`), for the same kind of decision: one of a few,
    exactly one at a time, and the rest of the row depends on it. A drop-down
    hides the alternative until it is opened, which is the wrong shape for a
    choice whose whole point is that there IS one.

    Read like a combo box by the rest of the form: `currentText` is the pressed
    option, and `onChanged` is told the new one.
    """

    def __init__(self, options, selected, help_texts=None):
        self.container = qt.QWidget()
        row = qt.QHBoxLayout(self.container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(design.SPACING_XS)
        self.buttons = {}
        self._callbacks = []
        self._selected = selected
        for option in options:
            button = design.segment_button(option)
            # Each segment says what IT means -- AREG shows the folder tree
            # each layout expects -- rather than the row repeating one text.
            # Preformatted: a tree is its indentation, which a plain tooltip
            # collapses.
            explained = _help_for(help_texts, option)
            if explained:
                button.setToolTip(preformatted_text(explained))
            button.connect("clicked()", self._picker(option))
            self.buttons[option] = button
            # Equal stretch: options of equal standing, across the whole row.
            row.addWidget(button, 1)
        self._show()

    @property
    def currentText(self) -> str:
        return self._selected

    def onChanged(self, callback) -> None:
        self._callbacks.append(callback)

    def _picker(self, option):
        """A closure rather than `functools.partial`: see
        FileOrFolderInput._sourcePicker."""
        def picked():
            self._choose(option)
        return picked

    def _choose(self, option) -> None:
        changed = option != self._selected
        self._selected = option
        # Re-pressed even when unchanged: a click on the pressed segment
        # un-checks a checkable button, and one option is always chosen.
        self._show()
        if changed:
            for callback in list(self._callbacks):
                callback(option)

    def _show(self) -> None:
        for option, button in self.buttons.items():
            button.setChecked(option == self._selected)

    # -- the slice of the QWidget API build()/base_widget use on a field ----

    def setProperty(self, name, value) -> None:
        self.container.setProperty(name, value)

    def setToolTip(self, _text) -> None:
        """Deliberately nothing, as for MultiChoiceGroup: Qt hands a container's
        tooltip to every child without one, and each segment carries its own.
        The description goes on the row's label (see `build`)."""


def _make_choice_widget(name: str, spec: dict):
    """A `"choice"` argument: one option among `choices`, whose single true
    entry is the server's declared default."""
    choices = _choices(name, spec)
    options = list(choices)
    if spec.get("ui") == SEGMENTED_UI and len(options) > 1:
        selected = [option for option, on in choices.items() if on]
        return SegmentedChoice(options, selected[0] if selected else options[0],
                               help_texts=spec.get("option_help"))

    widget = qt.QComboBox()
    widget.addItems(options)
    selected = [option for option, on in choices.items() if on]
    if selected:
        widget.setCurrentIndex(options.index(selected[0]))
    return widget


# What each single-kind file-input mode means for a ctkPathLineEdit. There is
# deliberately no "file_or_folder" entry: an argument accepting both is a
# FileOrFolderInput, for the reasons spelled out in that class.
_PATH_FILTERS = {
    "single_file": ctk.ctkPathLineEdit.Files,
    "folder_zip": ctk.ctkPathLineEdit.Dirs,
}


def path_widget(extensions=(), mode: str = "single_file"):
    """A ctkPathLineEdit for one file-input mode, restricted to `extensions`
    where that applies.

    A ctkPathLineEdit is configured **once, here, at construction**, and never
    touched again: re-assigning `nameFilters` on a live one corrupts it and
    takes Slicer down with it — reproduced against Slicer 5.13, and the reason
    the mode is a constructor argument rather than something the widget
    switches between later. Hence also the `if`: an unrestricted picker is left
    with its default rather than handed an empty list.
    """
    widget = ctk.ctkPathLineEdit()
    widget.filters = _PATH_FILTERS.get(mode, ctk.ctkPathLineEdit.Files)
    if mode != "folder_zip" and extensions:
        widget.nameFilters = name_filters(extensions)
    return widget


def auto_file_mode(spec: dict) -> str:
    """Which kind of picker a file argument gets, from what its `types` accept.

    The general rule, in one place: an argument accepting "folder" may be given
    a whole folder (zipped before upload); one accepting a file type as well
    gets the choice between the two. Returns a base_widget FILE_INPUTS mode,
    because the answer is needed twice — to build the widget, and again at
    upload time to know whether to zip (see base_widget._prepareOneInputFile).
    """
    if not accepts_folder(spec):
        return "single_file"
    if any(is_file_type(type_name) for type_name in argument_types(spec)):
        return "file_or_folder"
    return "folder_zip"


def file_input_modes(arguments_schema: dict, overrides=None) -> dict:
    """`{argument_name: mode}` for every file argument the client provides, in
    schema order.

    **Which arguments those are is the schema's answer, not a module's**: every
    file-typed argument gets an input row. A module's `FILE_INPUTS` is merged
    on top and only has to say what the schema cannot express —

    - `"volume_node"`: filled from a node in the MRML scene rather than from
      disk. The server does not know a scene exists;
    - a forced `"folder_zip"`/`"single_file"`: SurgMovPred's `input` is typed
      `zip_file`, and the module still wants to hand the user a folder picker
      and zip it client-side. "Give me a zip" is the contract; "let them pick a
      folder" is an ergonomics decision that lives here;
    - `"none"`: an optional file argument this module deliberately doesn't
      offer.

    Everything else stays `"auto"` and is resolved by `auto_file_mode`.
    """
    modes = {
        name: "auto"
        for name, spec in arguments_schema.items()
        if is_file_type(spec.get("type", ""))
    }
    # An override may only MODIFY an argument the tool declares. Naming one it
    # does not is always a module left behind by a rename, and taking it at its
    # word grew a phantom row: SurgMovPred's `input` became `measurements` when
    # the tool was packaged, and the panel kept offering an "Input" picker that
    # uploaded to an argument the server would have refused. Skipped and said
    # out loud -- silence is what let that one sit there.
    for name, mode in (overrides or {}).items():
        if arguments_schema and name not in arguments_schema:
            logger.warning(
                "FILE_INPUTS names '%s', which this tool does not declare; ignored. "
                "Its arguments are: %s", name, ", ".join(sorted(arguments_schema))
            )
            continue
        modes[name] = mode

    resolved = {}
    for name, mode in modes.items():
        if mode == "none":
            continue
        resolved[name] = auto_file_mode(arguments_schema.get(name, {})) if mode == "auto" else mode
    return resolved


# How a tool's server-side `output_kind` maps onto the client's RESULT_KIND.
_RESULT_KIND_FOR_OUTPUT = {
    "text": "text",
    "segmentation": "segmentation",
    "file": "save_as",
    "files": "save_as",
}


def result_kind_for(output_kind, declared=None) -> str:
    """The client's RESULT_KIND for a tool's declared `output_kind`.

    Three of the server's four output kinds settle the question on their own:
    `text` is text, `segmentation` is a segmentation, and `files` can only be
    saved (a zip of several files cannot become one MRML node).

    **`file` is the one genuinely ambiguous case**: the server says a single
    file comes back, it cannot say whether that file is meant to be loaded into
    the scene as a volume or as a mesh, or just written to disk — that is MRML
    knowledge, and the server has no business holding it. It defaults to
    `save_as`, and a module wanting the result loaded declares
    `RESULT_KIND = "volume"` / `"model"`. A declared value always wins.
    """
    return declared or _RESULT_KIND_FOR_OUTPUT.get(output_kind, "text")


def file_widget(spec: dict, mode: str = "auto", name: str = ""):
    """The picker for a file argument. `mode` defaults to the schema-driven
    rule above; base_widget passes an explicit one for what the schema cannot
    express (or to force a single selection kind).

    Kept here (rather than in base_widget) so every "schema shape -> Qt
    widget" decision lives in one file; `build()` itself never emits one (see
    the module docstring and FILE_INPUTS).
    """
    if mode == "auto":
        mode = auto_file_mode(spec)

    # One dropdown serves both extra sources: the test files the server hosts
    # for this tool (server_selectable), and a volume already open in the
    # scene (accepts_volume). Only file-typed arguments reach here: a SCALAR
    # server_selectable argument (a model, which must never leave the server)
    # is a plain combo box built by _make_widget, with no local picker at all.
    wrap = bool(spec.get("server_selectable")) or accepts_volume(spec, name)

    extensions = file_extensions_for(spec)
    local = FileOrFolderInput(extensions, _BROWSE_MODES.get(mode, ("file",)))
    if not wrap:
        return local
    # A hosted MODEL is not downloadable and never was: the server declines to
    # stream one, so its name travels and the weights stay put. Only the tool's
    # TEST FILES are fetched on selection.
    return ServerFileInput(local, hosted_downloads=spec.get("server_selectable") != "model")


def _choices(name: str, spec: dict) -> dict:
    """`choices` is a {option: initially_selected} dict, and its key order is
    the declaration order — preserved as-is, never sorted."""
    choices = spec.get("choices")
    if not choices:
        logger.warning("Argument '%s' is a '%s' but declares no choices", name, spec.get("type"))
        return {}
    return choices


def name_filters(extensions) -> list:
    """Qt name filters for a file picker restricted to `extensions` (an empty
    list — no restriction — when it is empty)."""
    if not extensions:
        return []
    patterns = " ".join(f"*{extension}" for extension in extensions)
    return [f"Supported files ({patterns})", "All files (*)"]


def collect(arg_widgets: dict) -> dict:
    return {name: _read_widget(widget) for name, widget in arg_widgets.items()}


def _read_widget(widget):
    if isinstance(widget, MultiChoiceGroup):
        # The complete state of every box, including the unchecked ones — the
        # server reads what it receives as the selection itself. Encoding it
        # for the wire is client.py's job (JSON, never the `a,b` shortcut).
        return widget.value()
    if isinstance(widget, JoystickInput):
        # Both numbers, as a two-element list; client.py sends it as JSON.
        return widget.value()
    if isinstance(widget, ctk.ctkSliderWidget):
        # ctk reports a double whatever `decimals` says; an integer slider
        # (decimals == 0) reads back as the int the server declared.
        value = widget.value
        return int(round(value)) if widget.decimals == 0 else value
    if isinstance(widget, qt.QCheckBox):
        return widget.isChecked()
    if isinstance(widget, SegmentedChoice):
        return widget.currentText
    if isinstance(widget, qt.QComboBox):
        # The selected option's name for a "choice" argument, sent in clear.
        # "" while a server-side list hasn't been loaded (or is empty) — which
        # keeps all_required_filled() False and the Apply button disabled.
        #
        # The "(automatic)" entry reads back as "" so the argument is dropped
        # rather than sent: matched on the text because that is what the entry
        # IS here, and no server-side file name nor `choices` option can
        # collide with it (both come from the server; one is a file name, the
        # other a declared option, and this string is neither).
        text = widget.currentText
        return "" if text == AUTOMATIC_OPTION else text
    if isinstance(widget, (qt.QSpinBox, qt.QDoubleSpinBox)):
        return widget.value
    if isinstance(widget, ctk.ctkPathLineEdit):
        return widget.currentPath
    if isinstance(widget, qt.QLineEdit):
        return widget.text
    raise TypeError(f"Don't know how to read value from widget {widget!r}")


def all_required_filled(arg_widgets: dict, arguments_schema: dict, hidden=()) -> bool:
    """Whether every required scalar argument holds a value.

    A `hidden` argument is skipped: it is not sent (see base_widget.collectArgs),
    so the server applies its default and an empty widget behind a hidden row
    must not be able to disable Apply forever with nothing on screen to explain
    why. No tool declares a required argument under a `visible_when` today, and
    this is what keeps that from becoming a dead-locked panel if one does.
    """
    for name, spec in arguments_schema.items():
        if is_file_type(spec.get("type", "")) or not spec.get("required") or name in hidden:
            continue
        widget = arg_widgets.get(name)
        if widget is None:
            return False
        value = _read_widget(widget)
        if value in ("", None):
            return False
    return all_minimums_met(arg_widgets, arguments_schema, hidden)


def all_minimums_met(arg_widgets: dict, arguments_schema: dict, hidden=()) -> bool:
    """Whether every multichoice declaring `min_selected` has that many ticked.

    Separate from the loop above because it answers a different question. An
    empty multichoice is normally FILLED -- every box unchecked is a meaningful
    selection, and ALI's empty `landmarks` is how a caller says "let the regions
    decide". `min_selected` is a tool saying that for THIS argument it is not:
    AMASSS cannot run with no structure and cannot write with no output form,
    and it raises on both.

    Checked here so Apply greys out instead of a clinician sending a request and
    being told no. The server refuses it too, and the tool refuses it again --
    this is the earliest of the three, not the only one.
    """
    for name, spec in arguments_schema.items():
        minimum = spec.get("min_selected")
        if not minimum or name in hidden:
            continue
        widget = arg_widgets.get(name)
        if widget is None:
            continue
        value = _read_widget(widget)
        if not isinstance(value, dict):
            continue
        if sum(1 for on in value.values() if on) < minimum:
            return False
    return True


def connect_changed(widget, callback) -> None:
    if isinstance(widget, MultiChoiceGroup):
        for box in widget.boxes.values():
            box.toggled.connect(callback)
    elif isinstance(widget, ServerFileInput):
        # EVERY source can satisfy the argument, so every one of them changing
        # must re-evaluate whether Apply can be enabled.
        #
        # The scene list was missing here, and the row it fills leaves no local
        # path behind -- an imported scan is exported at upload time -- so
        # picking one satisfied the argument and told nobody: Apply stayed grey
        # over a row the user had just filled. It went unseen because the stub's
        # combo box emitted on every `setCurrentIndex`, change or not, so the
        # reset of the OTHER list fired a signal that real Qt does not.
        widget.combo.currentTextChanged.connect(callback)
        widget.sceneCombo.currentTextChanged.connect(callback)
        connect_changed(widget.local, callback)
    elif isinstance(widget, FileOrFolderInput):
        # Its own callback list rather than a Qt signal: the field that used to
        # carry one is gone, and this class is a plain Python object.
        widget.onPathChanged(callback)
    elif isinstance(widget, JoystickInput):
        # The pad writes into the spin boxes (see JoystickInput), so the two
        # boxes cover every input path: drag, wheel, keys and typing.
        widget.xBox.valueChanged.connect(callback)
        widget.yBox.valueChanged.connect(callback)
    elif isinstance(widget, ctk.ctkSliderWidget):
        widget.valueChanged.connect(callback)
    elif isinstance(widget, qt.QCheckBox):
        widget.toggled.connect(callback)
    elif isinstance(widget, SegmentedChoice):
        widget.onChanged(callback)
    elif isinstance(widget, qt.QComboBox):
        widget.currentTextChanged.connect(callback)
    elif isinstance(widget, (qt.QSpinBox, qt.QDoubleSpinBox)):
        widget.valueChanged.connect(callback)
    elif isinstance(widget, ctk.ctkPathLineEdit):
        widget.currentPathChanged.connect(callback)
    elif isinstance(widget, qt.QLineEdit):
        widget.textChanged.connect(callback)
    else:
        logger.warning("Don't know how to connect change signal for widget %r", widget)
