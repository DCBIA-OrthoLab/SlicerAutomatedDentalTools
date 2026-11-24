"""The Slicer bridge, isolated: MRML export, zipping, and result loading.

Nothing here speaks HTTP. Every temporary file used by a tool module (an
exported volume, a zipped folder, a downloaded result) should go through
TempWorkspace so cleanup on error is never forgotten.
"""

import logging
import contextlib
import os
import re
import shutil
import tempfile
import zipfile
from typing import Optional
from urllib.parse import urlparse

import qt
import slicer

from . import config, console
from .errors import ServerToolError

logger = logging.getLogger("SADTCore.slicer_io")

# Also installed by the SADTCore module at startup; repeated here for a
# module reloaded on its own, which imports this file and nothing at startup.
console.install()


class TempWorkspace:
    """Context manager for a temp directory, removed on exit including on error."""

    def __init__(self, prefix="ServerTools_"):
        self._prefix = prefix
        self.path = None

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix=self._prefix)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.path and os.path.isdir(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
        self.path = None
        return False

    def file(self, name: str) -> str:
        return os.path.join(self.path, name)


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def upload_name(node_name: str, extension: str = "") -> str:
    """A node's name as a file name: what the scene shows, safe on any disk.

    Runs of anything but letters, digits, `.`, `_` and `-` become one `_`, and
    an extension the name already ends with is dropped so it is not doubled.
    Empty when nothing usable is left.
    """
    name = (node_name or "").strip()
    for suffix in {extension, ".nii.gz", ".nii", ".nrrd", ".vtk", ".vtp", ".stl", ".obj"}:
        if suffix and name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    # Accents are kept as their letter ("élève" -> "eleve") rather than lost:
    # the name is how the user finds their file among the results.
    import unicodedata

    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return _UNSAFE.sub("_", name).strip("._")


def export_volume(volume_node, dest_path: str) -> str:
    return export_node(volume_node, dest_path)


def export_node(node, dest_path: str) -> str:
    """Write any savable scene node out, and return where it went.

    Not volumes only: `saveNode` writes a model just as readily, and a scene
    pick has to answer for whatever the argument accepts -- ALI's `input` takes
    a CBCT or an intraoral surface, and declares no extensions precisely
    because the tool decides from the data.
    """
    ok = slicer.util.saveNode(node, dest_path)
    if not ok:
        raise IOError(f"Failed to export {getattr(node, 'GetName', lambda: '')()} "
                      f"to {dest_path}")
    return dest_path


def is_extractable_archive(path: str) -> bool:
    """Whether `path` should be unpacked as a delivery archive.

    Deliberately extension-based, not `zipfile.is_zipfile()`: OOXML formats
    (.xlsx, .docx, .ods, .pptx, ...) are zip containers structurally, so a
    signature check would "extract" a result .xlsx into its raw XML parts
    instead of keeping it as the file it actually is. Only a genuine `.zip`
    is meant to be unpacked here.
    """
    return path.lower().endswith(".zip")


# Extensions whose bytes are already compressed. DEFLATE gains ~0% on them and
# runs at ~45 MB/s on one core, so a folder of .nii.gz scans used to spend
# seconds per 100 MB shrinking nothing -- measured 2.3s to pack 105 MB of
# gzipped CBCT into an archive of exactly the same 105 MB, before a single byte
# was sent, and the server paid it again inflating them. `.gz` covers the
# compound medical extensions (.nii.gz, .nrrd.gz, .gipl.gz); the OOXML formats
# are zip containers by design. Mirrors the server's own table in
# file_utils.py.
_STORED_EXTENSIONS = (
    ".gz", ".bz2", ".xz", ".zip", ".7z",
    ".xlsx", ".ods", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg",
)

# Level 1 for everything else: it compresses at roughly twice the rate of the
# default 6 and gives up about 3% of size on the one kind of member still worth
# deflating here (binary .vtk, ~2.7:1 at either level).
_COMPRESS_LEVEL = 1


# Hosts whose link is faster than the compressor, so deflating on the way out
# costs more than it saves (see config.ZIP_COMPRESS for the measurements).
# 0.0.0.0 is compared against, never bound.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")  # nosec B104


def _link_is_fast(server_url: str) -> bool:
    """Is the server close enough that compressing is a net loss?

    Loopback and the private ranges only. Anything else is treated as remote,
    which is the safe way round: guessing "fast" on a real link would spend a
    third more bytes on the wire to save CPU that was not the constraint.
    """
    host = urlparse(server_url).hostname or ""
    if host in _LOCAL_HOSTS:
        return True
    return host.startswith(("10.", "192.168.")) or bool(
        re.match(r"172\.(1[6-9]|2[0-9]|3[01])\.", host)
    )


def zip_folder(folder: str, dest_path: str, compress: Optional[bool] = None) -> str:
    """Pack a folder for upload, choosing the compression per member.

    A folder argument is zipped only because HTTP has no notion of a folder --
    the archive is a container, not an attempt to make the data smaller. So
    already-compressed members are STORED as-is and only what genuinely
    deflates is deflated, which is 14x faster to pack for exactly the same
    bytes on the wire.

    `compress` decides whether the rest is deflated at all. None reads
    config.ZIP_COMPRESS, which in turn defaults to "not against a local
    server": deflating runs at 57 MB/s on one core, and a link faster than
    about 27 MB/s carries the raw bytes sooner than the compressor can shrink
    them.
    """
    if not os.path.isdir(folder):
        raise IOError(f"Not a folder: {folder}")
    with _archive(dest_path, compress) as archive:
        _add_tree(archive, folder, folder)
    return dest_path


def zip_subset(folder: str, entries, dest_path: str, compress: Optional[bool] = None) -> str:
    """Pack SOME of a folder's top-level entries, named as if the whole folder.

    This is how a cohort is sent in batches without costing a byte of local
    disk: `split_cohort` decides which entries go together and this writes those
    straight out of the user's own folder. Copying them into per-batch staging
    folders first would duplicate the cohort on a laptop that was chosen to send
    it in pieces precisely because it is large.

    Member names stay relative to `folder`, so the server unpacks each batch
    into the same tree shape the whole folder would have had -- which is what
    lets a tool that mirrors its input tree keep working per batch.
    """
    if not os.path.isdir(folder):
        raise IOError(f"Not a folder: {folder}")
    with _archive(dest_path, compress) as archive:
        for entry in entries:
            _add_tree(archive, os.path.join(folder, entry), folder)
    return dest_path


def _archive(dest_path: str, compress: Optional[bool]):
    if compress is None:
        compress = config.ZIP_COMPRESS
    if compress is None:
        compress = not _link_is_fast(config.SERVER_URL)
    default_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    return zipfile.ZipFile(dest_path, "w", default_type, compresslevel=_COMPRESS_LEVEL)


def _add_tree(archive, path: str, relative_to: str) -> None:
    """Write `path` -- one file, or every file under one directory -- into `archive`."""
    if os.path.isfile(path):
        files = [path]
    else:
        files = [
            os.path.join(root, name)
            for root, _dirs, names in os.walk(path)
            for name in names
        ]
    for full_path in files:
        # compress_type=None defers to the archive's default (DEFLATED at
        # _COMPRESS_LEVEL); already-compressed members opt out.
        stored = os.path.basename(full_path).lower().endswith(_STORED_EXTENSIONS)
        archive.write(
            full_path,
            os.path.relpath(full_path, relative_to),
            compress_type=zipfile.ZIP_STORED if stored else None,
        )


def _entry_size(path: str) -> int:
    """Bytes one cohort entry occupies, a whole subfolder included."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                # A file that vanished between the walk and the stat. Sizing is
                # advisory -- it decides how big a batch is, not what is in it.
                pass
    return total


def split_cohort(folder: str, max_mb: int, max_files: int) -> list:
    """Divide a folder into batches, as lists of its top-level entry names.

    **The unit is a TOP-LEVEL ENTRY, and that is the correctness rule of the
    whole feature.** A file is one; a subfolder is one, whole and never opened.
    A cohort is filed in subfolders -- one per patient -- and a DICOM series is
    a folder of hundreds of slices that means nothing apart. Splitting inside
    either would hand the tool half a patient and call it a batch.

    So `max_files` counts entries, not leaf files: 25 DICOM folders are 25, the
    same as 25 `.nii.gz`. Whichever cap binds first decides, and an entry larger
    than `max_mb` on its own travels alone rather than being dropped or split.
    A cap of 0 does not bind; with neither binding there is one batch.

    Sorted by name, so the same cohort always divides the same way -- a rerun
    after a failure resends the same batches, and two timepoints of one patient
    stay adjacent.
    """
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []

    max_bytes = max_mb * 1024 * 1024 if max_mb > 0 else 0
    batches, batch, batch_bytes = [], [], 0
    for name in names:
        size = _entry_size(os.path.join(folder, name))
        too_many = max_files > 0 and len(batch) >= max_files
        too_big = max_bytes > 0 and batch and batch_bytes + size > max_bytes
        if too_many or too_big:
            batches.append(batch)
            batch, batch_bytes = [], 0
        batch.append(name)
        batch_bytes += size
    if batch:
        batches.append(batch)
    return batches


def unzip_folder(zip_path: str, dest_dir: str) -> list:
    """Extract into `dest_dir`; return the absolute path of every FILE written.

    The names matter, not just the directory. A result archive is unpacked into
    the folder the user picked, which is the same folder their earlier runs
    wrote to -- so "what did this run produce" cannot be answered by looking at
    what is in there afterwards. It is answered here, by the paths extraction
    actually wrote -- never by the archive's raw names, which zipfile rewrites
    (`../x`, `/abs`) and which would otherwise name files nothing produced.

    The archive comes from a server, so it is untrusted, and the whole of it is
    checked before anything is written: a member that is absolute, climbs out
    with `..`, resolves outside `dest_dir` or is a symbolic link, or an archive
    past config.UNZIP_MAX_MEMBERS / UNZIP_MAX_BYTES, refuses the archive.
    """
    os.makedirs(dest_dir, exist_ok=True)
    root = os.path.realpath(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()
        _check_archive(members, root)
        written = []
        for member in members:
            path = archive.extract(member, root)
            if not member.is_dir():
                written.append(os.path.realpath(path))
        return written


def _check_archive(members, root: str) -> None:
    if len(members) > config.UNZIP_MAX_MEMBERS:
        raise ServerToolError(
            f"Refused an archive of {len(members)} entries: the limit is "
            f"{config.UNZIP_MAX_MEMBERS}."
        )
    total = sum(member.file_size for member in members)
    if total > config.UNZIP_MAX_BYTES:
        raise ServerToolError(
            f"Refused an archive that would expand to {total / 1024 ** 3:.1f} GB: "
            f"the limit is {config.UNZIP_MAX_BYTES / 1024 ** 3:.0f} GB."
        )
    for member in members:
        name = member.filename.replace("\\", "/")
        parts = [part for part in name.split("/") if part]
        unsafe = (
            name.startswith("/")
            or re.match(r"^[A-Za-z]:", name) is not None
            or ".." in parts
            or (member.external_attr >> 16) & 0o170000 == 0o120000   # a symlink
        )
        target = os.path.realpath(os.path.join(root, *parts)) if parts else root
        if unsafe or os.path.commonpath([root, target]) != root:
            # The name is not echoed: it came from the server and may carry
            # anything, a patient's file name included.
            raise ServerToolError(
                "Refused an archive with an entry that would be written outside "
                "the destination folder."
            )


# What a downloaded test file can be shown as, by extension. The point of
# downloading a hosted test file rather than naming it is that the clinician
# gets to LOOK at it beside the panel, so a single scan is put in the scene as
# soon as it lands. A COHORT never is: forty patients would flood the scene.
# `sole_scan_in` below is what tells those two apart for a folder.
#
# Distinct from formgen._VOLUME_EXTENSIONS on purpose. That table answers "can
# a volume already in the scene satisfy this argument"; this one answers "what
# node type is this file", and it has to cover meshes, which no scene volume
# could ever stand in for.
_INPUT_LOAD_KINDS = (
    ((".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz", ".mha", ".mhd"), "volume"),
    ((".vtk", ".vtp", ".stl", ".obj", ".ply"), "model"),
)


# Where a run's results land when nobody says otherwise. Named rather than
# nested under the tool, so a clinician running three tools on one patient finds
# all three in the same place.
OUTPUT_ROOT_NAME = "Slicer Output"

# Enough that a user who never cleans up still gets a fresh folder for years,
# small enough that the search is instant. Past it the last one is reused, which
# is worse than a new folder and much better than a panel that cannot open.
_MAX_OUTPUT_FOLDERS = 1000


def documents_dir() -> str:
    """Where THIS operating system puts a user's documents.

    `~/Documents` on Linux and macOS, `C:\\Users\\<name>\\Documents` on Windows --
    and whatever a localised Windows calls it, which is exactly why Qt is asked
    instead of a path being built by hand. The same call every other module in
    this extension already makes (AREG, GreedyReg, VFACE).

    Falls back to the home directory: a Qt with no documents location is not a
    reason to have no default at all.
    """
    try:
        location = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DocumentsLocation)
    except Exception:  # noqa: BLE001 - a default must never break a panel
        location = ""
    return location or os.path.expanduser("~")


def default_output_folder() -> str:
    """`<documents>/Slicer Output/<n>`, the first `n` that is free.

    So Apply works on a panel nobody has configured. The point is speed: a
    clinician trying a tool should not have to decide where the results go
    before finding out whether the tool helps them.

    "Free" means absent OR empty -- an empty folder left by a run that failed
    before writing is reusable, and skipping it would count upward forever. The
    folder is NOT created here: this only proposes a name, and a run that never
    happens must leave nothing behind.
    """
    root = os.path.join(documents_dir(), OUTPUT_ROOT_NAME)
    for number in range(1, _MAX_OUTPUT_FOLDERS + 1):
        candidate = os.path.join(root, str(number))
        try:
            if not os.path.isdir(candidate) or not os.listdir(candidate):
                return candidate
        except OSError:
            # Unreadable is not free, but it is not a reason to stop either.
            continue
    return os.path.join(root, str(_MAX_OUTPUT_FOLDERS))


def load_kind_for(path: str):
    """"volume", "model", or None for a file that is neither (a .zip, a .csv, a
    single DICOM slice). Extension-based, longest match first so `.nii.gz` never
    resolves as `.gz`.

    What a file IS, which is what counting a folder's contents needs. What
    OPENING it produces is `scene_kind_for`, and the two differ on DICOM."""
    lowered = path.lower()
    best, best_kind = "", None
    for extensions, kind in _INPUT_LOAD_KINDS:
        for extension in extensions:
            if lowered.endswith(extension) and len(extension) > len(best):
                best, best_kind = extension, kind
    return best_kind


# A DICOM series is a DIRECTORY of single-slice files, so it is one scan spread
# over hundreds of paths -- the one input shape where counting files answers the
# wrong question entirely. Slicer reads the whole series from any one of them
# (its "archetype"): measured on ASO's own test folder, 365 slices became one
# 512x512x365 volume in 1.0 s.
#
# By extension only. A DICOM file is identified properly by the "DICM" magic at
# offset 128, and plenty of real archives ship slices with no extension at all
# -- but confirming that means opening every file in the folder to decide
# whether to show a preview, which is a lot of I/O to spend on a courtesy. Every
# hosted test file uses `.dcm`; a folder of extensionless slices is simply not
# previewed, exactly as it is not today.
_DICOM_EXTENSION = ".dcm"


def sole_scan_in(folder: str):
    """The one scan a folder holds, as a path to open -- or None if it is not one.

    This is the rule that decides whether a folder appears in the scene, and it
    is deliberately about the SCAN COUNT rather than the file count. "A folder
    is never loaded" was the first version, and it was too blunt by half: every
    CBCT test file except ALI's is a folder, and most of them hold exactly one
    patient, so choosing one showed nothing and read as a broken button. What
    must never happen is a forty-patient cohort landing in the scene at once,
    and that is what is actually checked here.

    Exactly one volume wins. Failing that, exactly one DICOM directory. Failing
    that -- no volume at all -- one lone surface, so a folder holding a single
    mesh behaves like the file it contains. Anything else is a cohort (or
    nothing a scene can hold) and answers None, which shows nothing.

    Note the priority: AutoMatrix's test folder holds one CBCT *and* one surface,
    and the scan is what a clinician opened that folder to see. Counting them
    together would make it two objects and show neither.

    Metadata only -- nothing is read, nothing is opened -- so this costs a walk
    of the tree and no I/O.
    """
    volumes, models, dicom_dirs = [], [], {}
    try:
        for directory, _subdirs, names in os.walk(folder):
            for name in sorted(names):
                if name.startswith("."):
                    continue
                path = os.path.join(directory, name)
                kind = load_kind_for(name)
                if kind == "volume":
                    volumes.append(path)
                elif kind == "model":
                    models.append(path)
                elif name.lower().endswith(_DICOM_EXTENSION):
                    # One entry per DIRECTORY, keeping its first slice: that is
                    # the archetype to open, and the directory count is the scan
                    # count. AREG's DICOM test file is `T1/<case>/` and
                    # `T2/<case>/` -- two timepoints, 884 files, and merging
                    # them into one volume is precisely the accident this
                    # counts its way out of.
                    dicom_dirs.setdefault(directory, path)
    except OSError:
        return None

    if len(volumes) == 1:
        return volumes[0]
    if not volumes and len(dicom_dirs) == 1:
        return next(iter(dicom_dirs.values()))
    if not volumes and not dicom_dirs and len(models) == 1:
        return models[0]
    return None


def scene_nodes(node_class: str) -> list:
    """The nodes of `node_class` a user would recognise as their OWN data.

    `getNodesByClass` answers with the whole scene, and a Slicer scene holds
    more than what was loaded into it. Every slice view keeps a model node for
    the plane it draws in the 3D view -- `Red Volume Slice`, `Yellow Volume
    Slice`, `Green Volume Slice` -- so a surface argument's scene dropdown
    offered three rectangles nobody put there, that no tool could use, and that
    sat above the meshes the user actually loaded.

    Slicer answers that question itself (`vtkMRMLSliceLogic.IsSliceModelNode`),
    and deferring to it is the point: the three names are only the default
    layout's, and a custom one adds `Slice4`, `Compare1` and the rest. The name
    list would have to grow with Slicer; this does not.

    `GetHideFromEditors` is the general case beside it -- the flag Slicer sets
    on a node that is scaffolding rather than data. A node carrying it has
    already been told not to appear in a chooser, and this is a chooser.
    """
    try:
        found = slicer.util.getNodesByClass(node_class)
    except Exception:  # noqa: BLE001 - a class this build lacks is not fatal
        return []
    return [node for node in found if not _is_scaffolding(node)]


def _is_scaffolding(node) -> bool:
    """Whether the scene made this node for itself rather than for the user.

    Every probe is guarded: this runs against whatever MRML the host Slicer
    ships, and a missing method must cost one filter rather than the dropdown.
    An unanswerable question is answered "it is data", which at worst offers one
    node too many -- the failure the other way round hides a user's own mesh.
    """
    with contextlib.suppress(Exception):
        if node.GetHideFromEditors():
            return True
    try:
        return bool(slicer.vtkMRMLSliceLogic.IsSliceModelNode(node))
    except Exception:  # noqa: BLE001
        return False


def scene_kind_for(path: str):
    """What OPENING this path puts in the scene: "volume", "model", or None.

    `load_kind_for` answers what a file IS, which is what counting a folder's
    contents needs. This answers what opening it produces, and the two differ on
    exactly one input: a DICOM slice is not a volume, but opening it loads the
    series around it, which is. Counting with this one would make a 365-slice
    series read as 365 scans; opening with the other one would show nothing.
    """
    kind = load_kind_for(path)
    if kind is None and path.lower().endswith(_DICOM_EXTENSION):
        return "volume"
    return kind


def load_input(path: str, name: str = ""):
    """Show a downloaded input file in the scene, if it is something a scene
    can hold. Returns the loaded node, or None when the file is not one.

    `name` renames what was loaded, for the one case where the file's own name
    is not the scan's: a DICOM series is opened through one of its slices, so it
    would otherwise arrive in the scene called `IMG0001`. The folder is what the
    user picked and what they will look for.

    **Never raises.** This is a courtesy - the input is already filled in and
    the run works whether or not the scene shows anything - so a reader that
    Slicer refuses (a mesh in a dialect its loader does not take, a truncated
    volume) is a log line, not a failed pick.
    """
    kind = scene_kind_for(path)
    if kind is None:
        return None
    try:
        node = load_result(path, kind)
    except Exception:  # noqa: BLE001 - a preview must never break the input
        logger.warning("Could not load %s into the scene as a %s", os.path.basename(path), kind)
        return None
    if node is not None and name:
        try:
            node.SetName(name)
        except Exception as exc:  # noqa: BLE001 - a name is never worth a failure
            logger.warning("Could not rename the loaded scan: %s", exc)
    return node


_LOADERS = {
    "segmentation": lambda path: slicer.util.loadSegmentation(path),
    # Labelled VOXELS, kept as voxels. A segmentation node builds a closed
    # surface representation to show itself in 3D, so a labelmap loaded as one
    # arrives as triangles - an appearance the tool never produced. AMASSS
    # writes a labelled grid and generates no surface unless asked
    # (generate_surface defaults to False), so the mesh was the panel's doing,
    # not the tool's. The caller can still build a surface in Slicer when that
    # is what they want.
    "labelmap": lambda path: slicer.util.loadLabelVolume(path),
    "volume": lambda path: slicer.util.loadVolume(path),
    "model": lambda path: slicer.util.loadModel(path),
    "transform": lambda path: slicer.util.loadTransform(path),
    # A `.mrk.json`. Missing until a run actually produced one: ASO has
    # declared ("*.mrk.json", "markups") since it was converted, and every run
    # with "load the results" ticked ended on "No MRML loader registered for
    # result kind 'markups'" -- the landmarks were on disk and correct, and the
    # panel said the run had failed to deliver them. ALI had the call all
    # along, in a loader of its own.
    "markups": lambda path: slicer.util.loadMarkups(path),
}


# How far the transfer functions are shifted after a preset is applied, in the
# units Slicer's own Shift slider reads in -- because it IS that slider we set.
#
# Set by eye on one scanner: on a CBCT out of this pipeline, with intensities
# spanning -6266 to 19272, 580 is the value that looked right. Absolute rather
# than a fraction of that span, and the measurement is why: a preset's curve is
# written in absolute units (CT-AAA is defined in Hounsfield), so moving it by a
# number of units is the natural operation. 60% of that scan's range would have
# been 15323 units and shown nothing at all.
VOLUME_RENDERING_SHIFT = 580.0

# The two controls driven inside the Volume Rendering module, by the object
# names its own .ui gives them. Named constants because they are the one part
# of this that a Slicer release could rename under us.
_PRESET_COMBO = "PresetComboBox"
_PRESET_OFFSET_SLIDER = "PresetOffsetSlider"
_VISIBILITY_CHECKBOX = "VisibilityCheckBox"


# A view ID no view carries. A hidden display node is pointed at it so that no
# view counts it: see `hide_scene`.
_PARKED_VIEW_ID = "vtkMRMLViewNodeHiddenByServerTools"


def hide_scene() -> None:
    """Hide every node in the scene; remove none.

    Every display node of every displayable node is switched off -- a volume's
    rendering as much as a model -- and the slice views are emptied, for the
    caller to fill with what it is about to show. Each node is still in the
    Data module, one eye-click away. Nodes hidden from editors (the slice-plane
    models) are Slicer's own and left alone.

    Called before a run's results are loaded and before VISU draws a view: a
    pause showed the oriented CBCT it was reviewing on top of the input scan
    the panel had already rendered, two volumes in one place.

    **Hidden is not enough for the 3D view's box.** Slicer sizes that box to
    every node DISPLAYABLE in the view, which is a matter of the display node's
    view list and not of its visibility -- so at a pause after ASO the box
    spanned the hidden original and the oriented CBCT, grew, and the camera
    centred on the gap between them. Each hidden display node is therefore also
    parked on a view that does not exist, and given its own views back the
    moment it is shown again, by the eye in the Data module or by anyone else
    (see `_park`).

    Best effort: one node that refuses does not keep the others shown, and
    nothing here is worth failing the caller for.
    """
    try:
        nodes = slicer.util.getNodesByClass("vtkMRMLDisplayableNode")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list the scene to hide it: %s", exc)
        return
    for node in nodes:
        try:
            if node.GetHideFromEditors():
                continue
            for index in range(node.GetNumberOfDisplayNodes()):
                display = node.GetNthDisplayNode(index)
                if display is None:
                    continue
                display.SetVisibility(False)
                try:
                    _park(display)
                except Exception as exc:  # noqa: BLE001 - hidden is what matters most
                    logger.warning("Could not take %s out of the views: %s", node.GetName(), exc)
        except Exception as exc:  # noqa: BLE001 - one node must not keep the rest shown
            logger.warning("Could not hide %s: %s", node.GetName(), exc)
    try:
        slicer.util.setSliceViewerLayers(background=None, foreground=None, label=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not empty the slice views: %s", exc)


def _park(display) -> None:
    """Take a hidden display node out of every view until it is shown again.

    Its own view list is kept and restored as it was: empty means every view,
    and a node the reader had limited to one view must not come back in all
    of them. Parking twice keeps the first list, which is the reader's.
    """
    views = [display.GetNthViewNodeID(i) for i in range(display.GetNumberOfViewNodeIDs())]
    if _PARKED_VIEW_ID in views:
        return
    display.RemoveAllViewNodeIDs()
    display.AddViewNodeID(_PARKED_VIEW_ID)
    tag = []

    def restore(caller, _event):
        if not caller.GetVisibility():
            return
        caller.RemoveObserver(tag[0])
        caller.RemoveAllViewNodeIDs()
        for view in views:
            caller.AddViewNodeID(view)

    tag.append(display.AddObserver("ModifiedEvent", restore))


def show_volume_rendering(node, preset: str, shift: float = None):
    """Turn a loaded scan into a 3D rendering, with `preset` applied.

    Loading a volume puts it in the slice views and nothing more; the 3D view
    stays empty until a volume-rendering display node exists and is switched
    on. For a tool whose whole output IS a scan -- an orientation, a
    registration -- that empty 3D view is what a clinician reads as "it did not
    work".

    **Driven through the module's own widgets, not by writing to the nodes.**
    Writing the curve directly does produce the right image, and that is what
    this did first -- but the preset list and the Shift slider are the module's
    own state, with no counterpart in MRML (`offsetPreset` is a widget slot;
    the logic library holds no offset at all). So the picture was right while
    the panel said "no preset, shift 0", and the next person to touch the
    slider would have started from a curve that was already moved without
    anything saying by how much.

    Setting the two controls instead means Slicer applies both through its own
    code path: the module reads CT-AAA and 580, and the shift cannot drift from
    what that slider would have done.

    Which preset suits a result is the MODULE's to say (see
    `ServerToolWidgetBase.VOLUME_RENDERING`): a dental CBCT and an MRI want
    different curves, and this library has no way to tell them apart.

    Returns the display node, or None when this build has no volume-rendering
    module -- which is not an error, only a scene without a 3D rendering in it.
    """
    module = getattr(slicer.modules, "volumerendering", None)
    if module is None or node is None:
        return None
    logic = module.logic()

    display = logic.CreateDefaultVolumeRenderingNodes(node)
    if display is None:
        return None

    # The presets live in a scene of their own, loaded ON DEMAND, and
    # `GetPresetByName` looks in it without loading it: a module nobody has
    # opened yet answers nothing, and the default curve silently stays.
    try:
        logic.GetPresetsScene()
    except Exception as exc:  # noqa: BLE001 - the lookup below says what happened
        logger.warning("Could not load the volume rendering presets: %s", exc)

    found = logic.GetPresetByName(preset)
    if found is None:
        # Named by a module, installed by this build: one that is not there
        # leaves the default curve rather than no rendering at all.
        logger.warning("No volume rendering preset named '%s'; keeping the default", preset)
    else:
        _drive_module(module, node, found,
                      VOLUME_RENDERING_SHIFT if shift is None else shift)

    # LAST, and on the node itself. Reaching the module instantiates its widget,
    # which reacts to the volume being selected and settles the rendering's own
    # state -- a visibility set before that is simply undone, which is what left
    # a rendering loaded, correct and switched off, waiting for someone to find
    # the check box. Setting it here too costs nothing and does not depend on a
    # Qt object name holding still.
    display.SetVisibility(True)
    return display


def _drive_module(module, node, preset, shift: float) -> None:
    """Set the module's volume, preset, Shift and Visibility, in that order.

    All four, because all four are the module's own state: a rendering can be
    loaded, carry the right curve and still be switched off, which is a 3D view
    that looks broken with nothing to say why.

    The order is not cosmetic and each step was paid for:

    * the VOLUME first -- the module applies a preset to whatever IS selected
      in it, and the Shift slider's range is computed from that volume, so a
      value set earlier is clamped against the wrong scan;
    * the SHIFT after the preset -- applying a preset resets the offset, so a
      shift set first is thrown away without a word;
    * VISIBILITY last -- reaching the module instantiates its widget, which
      reacts to the volume being selected and settles the rendering's state.

    Best effort, and loudly so. If a Slicer release renames one of these
    controls the rendering is still there, with the default curve, and the log
    says which name went missing rather than leaving a scene nobody can explain.
    """
    try:
        widget = module.widgetRepresentation()
        widget.setMRMLVolumeNode(node)
        combo = slicer.util.findChild(widget, _PRESET_COMBO)
        slider = slicer.util.findChild(widget, _PRESET_OFFSET_SLIDER)
    except Exception as exc:  # noqa: BLE001 - never worth failing a finished run
        logger.warning("Could not reach the volume rendering controls: %s", exc)
        return

    try:
        combo.setCurrentNode(preset)
        slider.value = shift
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not apply preset '%s': %s", preset.GetName(), exc)
        return

    # Separately, because a build that renamed only this one should still get
    # its preset -- and because the node's own flag below covers it anyway.
    try:
        slicer.util.findChild(widget, _VISIBILITY_CHECKBOX).setChecked(True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not tick the rendering's visibility: %s", exc)


def load_result(path: str, kind: str):
    loader = _LOADERS.get(kind)
    if loader is None:
        raise ValueError(f"No MRML loader registered for result kind '{kind}'.")
    return loader(path)
