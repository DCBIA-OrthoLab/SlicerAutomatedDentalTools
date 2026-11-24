"""VISU -- look at what a tool produced, one case at a time.

Every other module in this extension asks the server to compute something.
This one computes nothing: it opens folders that are already on disk and puts
them on screen in the order a reader wants them, with the arrow keys to step
from one patient to the next.

**Why it is not "drag the folder into Slicer".** Dragging works and tells you
nothing about which file goes with which. A cohort of forty comes back as a
scan here, a mask two directories down under a name the tool invented, and a
landmark file whose stem matches nothing -- and the one arrangement that is
wrong renders exactly as well as the right one. `SADTVisuLib.index` decides the
pairing; this file is the panel over it.

**The one thing it must never do is lie about the frame.** ASO's landmarks
carry its recentring and its ICP rotation, so they belong to the scan ASO
WROTE; ALI writes no scan at all, so its landmarks belong to the one the
caller sent. Draw either on the other and the picture renders without an error
and is wrong by a rotation -- which is precisely the mistake a reviewer opened
this panel to catch. So a view names the scan its overlays are drawn on, and
says whether that was read off the folder or assumed.
"""

import logging
import os
import shutil
import tempfile
import threading

import ctk
import qt
import slicer
import vtk
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget

from SADTCoreLib import design, formgen, get_client, slicer_io, testfile_entries
from SADTCoreLib.errors import plain_text
from SADTCoreLib.worker import BackgroundJob
from SADTVisuLib import edits, index, review

logger = logging.getLogger("VISU")

_SETTINGS_GROUP = "SADTVisu"
_KEY_FOLDER = f"{_SETTINGS_GROUP}/Folder"

# One folder in, and the index carries one source label for everything under
# it. Two fields were a worse question: a reader has one folder in front of
# them, and whether it holds the acquisition, a run's output or both is
# something the folder answers rather than something to be declared. An
# overlay with no anchor in its own directory falls back to a scan from this
# source, which is what puts ALI's landmarks on the scan beside them.
SOURCE = "folder"

# The hosted datasets worth opening a VIEWER on, named rather than discovered.
#
# Asking every tool for its test files answered 61 entries -- 28 once the
# facades were folded together, since `deployment.toml` points several tool
# names at one bundle. That list is not too long by accident: it is every
# tool's REGRESSION FIXTURE, and most of it is a second timepoint, a transform
# or a spreadsheet that a viewer has nothing to do with. Size does not separate
# them either: 13 entries sit under 250 MB and 9 under 100.
#
# The rule a sample has to pass: AT LEAST THREE subjects, each one carrying
# both a scan and the landmarks placed on it. Three because the arrows are the
# point and two of anything demonstrates nothing; matched because the panel
# exists to show what belongs to one scan together, and a subject that comes
# up bare teaches the reader that the panel lost something.
#
# Counted with this module's own index against the bundles on disk. **Exactly
# one hosted folder passes.** Four carry landmarks at all and the other three
# hold one or two subjects -- the server's test data is each tool's regression
# fixture, assembled to prove a tool still runs rather than to be looked at.
# A folder of one's own is where a richer set lives; `SADTVisuLib.index` walks any
# directory, so pointing the field at it needs nothing from here.
# Each entry is (tool, the server's own name, what to call it here). The third
# is not decoration: the server's names are each tool's fixture names --
# `IOSCBCT_RegTestFiles`, `CBCT_SemiAuto_DCM` -- and they say which REGRESSION
# TEST the folder belongs to, not what is inside it. Renaming them on the
# server would break the tools that name them; naming them here costs nothing
# and is the only place that knows a viewer is asking.
SAMPLE_DATA = (
    ("AREG", "IOSCBCT_RegTestFiles", "3 subjects - CBCT and IOS, with landmarks"),
    # Staged by hand under `DATA/`, which is gitignored: three CBCT and three
    # IOS with the landmarks ALI placed on them, filed `CBCT/` beside
    # `Landmarks/`. A deployment that has not staged them offers them not at
    # all, which is what makes naming them here safe.
    ("ASO", "VISU_CBCT_3", "3 subjects - CBCT, with landmarks"),
    ("ASO", "VISU_IOS_3", "3 subjects - IOS, with landmarks"),
)

# What a CBCT is rendered with in 3D, and it is the tools' own choice: AMASSS
# and ASO both name CT-AAA for the scans they return. `slicer_io` applies the
# shift it measured on a scan out of this pipeline over the top.
VOLUME_RENDERING = "CT-AAA"

# How a mask is opened when it cannot be the slice label layer. Not one of
# `index`'s kinds: what the file IS stays a labelmap, this is only how it is
# shown.
SEGMENTATION = "segmentation"

# VISU draws a landmark exactly as its file asks. A `.mrk.json` carries three
# fields for it -- `glyphScale` (percent of the view), `glyphSize`
# (millimetres) and `useGlyphScale`, which picks between them -- and they are
# the tool's to set. Overriding them here was tried and taken back out: the
# panel would then show something no other reader of the same file sees, and
# a size that looks wrong is a thing to fix where it is WRITTEN, in
# `sadt_ali_common.markups`, so every consumer gets the fix at once.

# What an adjustment is called on disk, beside the scan it moves. Never the
# scan's own name: this file is the reader's, and the tool's output has to
# stay recognisable as the tool's.
ADJUSTMENT_SUFFIX = "_VISU_adjust.tfm"

# What the check boxes offer, in the order they are drawn, and the kind each
# one governs. Words a reader uses, not the loader's vocabulary: nobody calls
# a mask a labelmap out loud.
#
# A TRANSFORM is on the list although there is nothing to draw: loaded, it is
# a node the Transforms module can apply, which is the only way to see what a
# registration did. Off by default for the same reason -- it shows nothing on
# its own.
SHOWABLE = (
    # NOT "Scan". An intraoral scan is a scan, and calling the volume chip
    # that made a reader on an IOS case read the greyed chip as "the scan
    # will not display" -- while the mesh was on screen under `Surfaces`,
    # 122 023 points of it. Two words that cannot both mean one file.
    ("CBCT", index.VOLUME, True),
    ("Surfaces", index.MODEL, True),
    ("Masks", index.LABELMAP, True),
    ("Landmarks", index.MARKUPS, True),
    ("Transforms", index.TRANSFORM, False),
)
_KIND_OF_OPTION = {label: kind for label, kind, _on in SHOWABLE}

# The scan's own placement, which is not a file kind: unlocking it is what
# puts the anchor under a transform with handles.
POSITION = "position"

# What the Modify section offers, and everything starts LOCKED. A landmark
# set is read far more often than it is corrected, and a point nudged by a
# stray drag while scrolling is a correction nobody made and nobody sees.
EDITABLE = (
    ("Landmarks", index.MARKUPS),
    ("Position", POSITION),
)

# Read ahead by one, in a daemon thread, so pressing the arrow does not also
# pay for the disk. It warms the page cache and touches no MRML node: loading
# one is main-thread work whatever we do here, and a 130 MB CBCT that is
# already in memory loads in a fraction of the time it takes off a disk or a
# network share. Bounded, because a cohort folder can hold gigabytes.
_PREFETCH_BUDGET_MB = 400
_PREFETCH_CHUNK = 1 << 20


class SADTVisu(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("VISU")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = ["SADTCore"]
        self.parent.contributors = ["Automated Dental Tools team"]
        self.parent.helpText = _("""
        Step through what a tool produced, one patient at a time. Point it at the folder the
        scans came from and, if you have one, the folder a run wrote, and it pairs them: the
        scan, the masks, the surfaces and the landmarks of one patient are shown together, and
        the arrow keys move to the next.
        <br><br>
        <b>Read the line that says what the landmarks are drawn on.</b> Some tools write their
        points against a scan they oriented rather than the one you sent; the two look alike and
        only one of them is right. The panel names the scan it used and whether it found it
        beside the points or had to assume.
        """)
        self.parent.acknowledgementText = _("This module was supported by NIDCR R01 024450.")


class SceneLoader:
    """Everything that touches the MRML scene, kept in one object.

    Two reasons it is not just three calls inline. The panel must own what it
    put in the scene and nothing else -- a clinician's own data is in there too
    -- so every node created here is remembered and only these are removed.
    And keeping it behind one small object is what lets the panel's navigation
    be exercised without a running Slicer.
    """

    def __init__(self):
        self._owned = []
        # [(artifact, node)], so visibility can be re-decided per kind without
        # anything being read off disk again.
        self._shown = []
        # [(node, observer tag)]. Dropped before the nodes are, or a callback
        # fires on a node that is no longer in the scene.
        self._watched = []

    def watch(self, node, callback) -> None:
        """Tell `callback` whenever this node or its display changes.

        This is the half of the sync that goes the other way. The eye in the
        Markups module and the chips here are two switches on ONE thing, and
        a reader who clicks the eye and then looks at a ticked chip has been
        lied to by whichever of the two did not move.
        """
        for target in (node, node.GetDisplayNode()):
            if target is None:
                continue
            try:
                tag = target.AddObserver(vtk.vtkCommand.ModifiedEvent, callback)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not watch %s: %s", node.GetName(), exc)
                continue
            self._watched.append((target, tag))

    def clear(self) -> None:
        self._shown = []
        for target, tag in self._watched:
            try:
                target.RemoveObserver(tag)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not stop watching a node: %s", exc)
        self._watched = []
        for node in self._owned:
            try:
                slicer.mrmlScene.RemoveNode(node)
            except Exception as exc:  # noqa: BLE001 - a stale node must not wedge the panel
                logger.warning("Could not remove a node VISU loaded: %s", exc)
        self._owned = []

    def load(self, artifact, opened_as: str = ""):
        """Open one artifact and keep the node, or None if it would not open.

        `opened_as` overrides what the file IS with how it should be SHOWN.
        One mask is a label layer; three masks cannot be, a volume having one
        label layer and no more -- so several masks on one scan are opened as
        segmentations, which stack as coloured outlines on the slices and as
        surfaces in 3D. It costs a closed-surface representation the tool did
        not produce, and it is the only way to see a mandible and a maxilla at
        the same time.
        """
        try:
            node = slicer_io.load_result(artifact.path, opened_as or artifact.kind)
        except Exception as exc:  # noqa: BLE001 - one unreadable file is not the case
            logger.warning("Could not open %s: %s", artifact.name, exc)
            return None
        if node is None:
            return None
        node.SetName(artifact.name)
        self._owned.append(node)
        self._draw(node)
        self._shown.append((artifact, node))
        return node

    def shown(self) -> list:
        return list(self._shown)

    @staticmethod
    def set_visible(node, visible: bool) -> None:
        display = node.GetDisplayNode()
        if display is None:
            return
        try:
            display.SetVisibility(bool(visible))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not hide or show %s: %s", node.GetName(), exc)

    @staticmethod
    def _draw(node) -> None:
        """Switch the display node ON, whatever the file says.

        **The chips are the truth about what is on screen.** A ticked chip
        that shows nothing is worse than no chip at all: the reader believes
        they are looking at the landmarks and they are looking at their
        absence. So anything this panel loads because its chip is ticked is
        made visible, and the only way to take it off screen is to untick.

        This is the ONE thing overridden, and it is not a preference.
        `"visibility": false` in a markups file does not mean small or grey --
        it means Slicer builds the node, lists it in the Markups module and
        draws NOTHING. Both original ALI CLIs wrote it; ALI fixed it, and
        every fixture written before that still carries it. Ten such files
        were found in this deployment's own `DATA/`, four of them in the
        first sample this panel offers.

        Size, colour and slice projection stay the file's, because those are
        choices a reader can disagree with. "The reader should see nothing"
        is not one of them.

        A transform has no display node until something asks for one, so it
        is skipped rather than special-cased.
        """
        display = node.GetDisplayNode()
        if display is None:
            return
        try:
            display.SetVisibility(True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not show %s: %s", node.GetName(), exc)

    @staticmethod
    def jump_to(node) -> None:
        """Bring the slices to the first point.

        Measured on the hosted CBCT: the volume spans 230 mm and opens on its
        centre, while its landmarks sit up to 60 mm away -- so every one of
        them is off-slice and the reader presses the chip and sees nothing.
        """
        try:
            logic = slicer.modules.markups.logic()
            if node.GetNumberOfControlPoints():
                logic.JumpSlicesToNthPointInMarkup(node.GetID(), 0, True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not jump to a landmark: %s", exc)

    @staticmethod
    def set_locked(node, locked: bool) -> None:
        """Lock or unlock every point of one markups node.

        LOCKED is the resting state, and deliberately: a landmark set is
        opened to be read far more often than to be changed, and a point
        nudged by a stray drag while scrolling is a correction nobody made
        and nobody sees. Unlocking is one click and it is visible.

        ALI writes each control point with `"locked": true` and the node
        itself unlocked, so both levels are set: a node-level unlock alone
        leaves every point refusing to move. The FILE keeps its own flags --
        only the node in this scene is touched.
        """
        try:
            node.SetLocked(locked)
            for point in range(node.GetNumberOfControlPoints()):
                node.SetNthControlPointLocked(point, locked)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not set the landmark lock: %s", exc)

    @staticmethod
    def draw_on_slices(node) -> None:
        """Show a surface's intersection with the slice planes.

        A model loaded beside a volume is in 3D and NOWHERE on the slices,
        which is where a reader checks whether a mesh sits on the anatomy it
        was registered to. Off by default in Slicer, so it is switched on.
        """
        try:
            display = node.GetDisplayNode()
            if display is not None:
                display.SetVisibility2D(True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not draw a surface on the slices: %s", exc)

    @staticmethod
    def display(anchor, anchor_node, label_node, reframe: bool = True) -> None:
        """Put the case on screen the way it is meant to be read.

        Loading a node is not showing it. A volume lands in the slice views
        and leaves the 3D view EMPTY -- which a clinician reads as "it did not
        work", and which is why every tool panel that returns a scan turns
        volume rendering on. A mesh is the opposite: it is only ever in 3D,
        and the three slice panes around it are dead space.

        So the layout follows the anchor rather than being left where the last
        module put it, and that is the whole difference between a viewer and a
        file loader.
        """
        try:
            if anchor is not None and anchor.kind == index.MODEL:
                if reframe:
                    SceneLoader._layout("SlicerLayoutOneUp3DView")
            else:
                if reframe:
                    SceneLoader._layout("SlicerLayoutFourUpView")
                SceneLoader._set_layers(anchor_node, label_node, fit=reframe)
                if anchor_node is not None and anchor.kind == index.VOLUME:
                    # The same preset the CBCT panels use, with the shift
                    # `slicer_io` measured on a scan out of this pipeline.
                    slicer_io.show_volume_rendering(anchor_node, VOLUME_RENDERING)
            if reframe:
                SceneLoader._frame3D()
        except Exception as exc:  # noqa: BLE001 - a view is never worth a failure
            logger.warning("Could not set the views up: %s", exc)

    @staticmethod
    def show_in_3d(segmentation) -> None:
        """Give a segmentation its surface, which is what the 3D view draws.

        Opened from a labelmap it holds voxels only, and Slicer builds the
        closed surface on demand -- so without asking, the mask was in the
        slices and the 3D view stayed empty.
        """
        try:
            segmentation.CreateClosedSurfaceRepresentation()
        except Exception as exc:  # noqa: BLE001 - the slices still show it
            logger.warning("Could not build the 3D surface of %s: %s", segmentation.GetName(), exc)

    @staticmethod
    def _set_layers(background, label, fit: bool) -> None:
        """Put the scan in the slice views, through the SCENE.

        NOT `slicer.util.setSliceViewerLayers`, and that is the whole point:
        that helper walks the layout manager's current slice WIDGETS, and it
        is called here one line after the layout was changed -- so it writes
        into the views that are on their way out while the new ones come up
        empty. Three grey panes with the volume loaded, visible, and windowed
        correctly, which is what made this so hard to see.

        A composite node lives in the MRML scene and survives any number of
        layout changes, so setting it is order-independent by construction.
        """
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(background.GetID() if background else None)
            composite.SetLabelVolumeID(label.GetID() if label else None)
        if fit:
            # After the layers, and only when the picture changed: fitting is
            # a camera move, and a reader who scrolled somewhere must keep it.
            slicer.util.resetSliceViews()

    @staticmethod
    def _layout(name: str) -> None:
        manager = slicer.app.layoutManager()
        node = getattr(slicer, "vtkMRMLLayoutNode", None)
        if manager is None or node is None or not hasattr(node, name):
            return
        manager.setLayout(getattr(node, name))

    @staticmethod
    def _frame3D() -> None:
        """Point the 3D view at what was just loaded.

        Without it the camera stays where the previous case left it, so
        stepping onto a mesh recorded in another part of the world shows an
        empty view that looks exactly like a failed load.
        """
        manager = slicer.app.layoutManager()
        if manager is None:
            return
        for number in range(manager.threeDViewCount):
            view = manager.threeDWidget(number).threeDView()
            view.resetFocalPoint()
            view.resetCamera()


def prefetch(paths) -> None:
    """Warm the page cache for the next case, in the background."""
    def run():
        budget = _PREFETCH_BUDGET_MB * 1024 * 1024
        for path in paths:
            if budget <= 0 or not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as handle:
                    while budget > 0 and handle.read(_PREFETCH_CHUNK):
                        budget -= _PREFETCH_CHUNK
            except OSError:
                # A file that cannot be read now will report itself when the
                # reader steps onto it. Nothing here is worth a message.
                return

    threading.Thread(target=run, daemon=True).start()


def hosted_choices(found) -> tuple:
    """`(entries, {label: (tool, name, kind)})` for what the server hosts.

    One entry per distinct FILE, however many tool names offer it.
    `deployment.toml` points several tool names at one bundle folder -- ALI,
    ALI_CBCT and ALI_IOS share theirs, and the four AREG names share another
    -- so asking every tool returns the same file up to four times. Measured
    against this deployment: **61 entries for 34 distinct files**, and 8
    distinct files on disk behind those, the bundles being hardlinked.

    The tool is named in the label only when it has to be: two files that
    genuinely differ and happen to share a name. Otherwise the name is the
    name, which is what a reader is looking for.

    `found` is `[(tool, name, kind, size, called), ...]`, and files are the same
    when all three of name, kind and size match -- the most a listing can
    compare without fetching. Two different files agreeing on all three would
    merge; the one that would be lost is reachable under the other's tool.

    `called` is what to show instead of the server's own name, which says
    which regression test a folder belongs to rather than what is in it.
    """
    groups = {}
    for tool, name, kind, size, called in found:
        groups.setdefault((name, kind, size, called), []).append(tool)
    times_named = {}
    for name, _kind, _size, _called in groups:
        times_named[name] = times_named.get(name, 0) + 1

    entries, offered = [], {}
    for (name, kind, size, called), tools in groups.items():
        tool = sorted(tools)[0]
        label = called or (name if times_named[name] == 1
                           else "{} / {}".format(tool, name))
        offered[label] = (tool, name, kind)
        entries.append({"name": label, "kind": kind, "size": size})
    return sorted(entries, key=lambda entry: entry["name"]), offered


def open_for_review(folder: str, on_continue, rewind=None, origin=None, inputs=(),
                    focus=None) -> bool:
    """Bring this module up on `folder`, with a Continue that calls back.

    Here rather than in the caller, and it is the only reason this function
    exists: which Slicer module VISU is, how its panel is reached and what it
    is asked are VISU's to know. A tool panel that stopped mid-run holds a
    folder and a callable and nothing else, so the day this module is renamed
    or its panel grows a second entry point, nothing outside this file moves.

    False rather than an exception when the panel cannot be reached: the
    caller is in the middle of a run that is PAUSED on the server, and it has
    to be able to say so and release it rather than leaving a GPU job waiting
    for a reader who was never shown anything.
    """
    try:
        slicer.util.selectModule("SADTVisu")
        widget = slicer.modules.sadtvisu.widgetRepresentation().self()
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not raised
        logger.warning("Could not open VISU on %s: %s", folder, exc)
        return False
    widget.openForReview(folder, on_continue, rewind=rewind, origin=origin, inputs=inputs,
                         focus=focus)
    return True


def end_review(folder: str) -> bool:
    """Take the pause banner down if VISU is still showing `folder`'s review.

    For a run withdrawn while it waited: its Continue would resume nothing,
    and the banner went on saying a run was paused after the run was gone.
    False when the panel cannot be reached or shows another review.
    """
    try:
        widget = slicer.modules.sadtvisu.widgetRepresentation().self()
    except Exception as exc:  # noqa: BLE001 - a banner is never worth a failure
        logger.warning("Could not reach VISU to end the review of %s: %s", folder, exc)
        return False
    return widget.endReview(folder)


class SADTVisuWidget(ScriptedLoadableModuleWidget):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.cases = []
        self.views = []
        self.position = 0
        self.scene = SceneLoader()
        # Filling a combo fires its own currentIndexChanged, which would
        # re-enter the refresh that is filling it. A flag rather than
        # `blockSignals`, because what has to be suppressed is this panel's
        # reaction and not the widget's signal.
        self._filling = False
        # True between opening the module and the reader's first action.
        # Opening a module must not put somebody's cohort in their scene, so a
        # remembered folder is INDEXED and not shown; the first press of an
        # arrow shows what is already selected rather than moving off it.
        self._waiting = False
        # Set only while `_restore` is driving the input, so `onIndex` can
        # tell a folder the reader just chose -- which is an action, and shows
        # at once -- from one the panel remembered, which must not.
        self._restoring = False
        # {what the dropdown shows: (tool, the server's own name, kind)}. The
        # entries are labelled with their tool because this panel is not one:
        # it borrows every tool's test data, and two tools may host a file of
        # the same name.
        self._hosted = {}
        # Where the last fetched test file was unpacked, removed when the next
        # one lands. A cohort is hundreds of megabytes and a viewer is opened
        # many times in a sitting.
        self._staging = ""
        self._anchorNode = None
        # [(artifact, node)] for the markups on screen: a save has to know
        # which file a node came from, and a node does not carry its path.
        self._points = []
        # The transform "Adjust position" put the anchor under, if any.
        self._adjustment = None
        # True while the panel is writing the scene from its own chips, so
        # the observer that reads the scene back does not chase it.
        self._syncing = False
        # The case keys this reader has marked as needing work, and the
        # folder they belong to. Read back off the folder on every open.
        self._flagged = set()
        self._folder = ""
        # The case keys whose files this panel actually wrote back. Not the
        # same list as the flagged one and not derivable from it: a reader
        # corrects a landmark without flagging the patient, and flags a
        # patient they could not correct at all.
        self._written = set()
        # `{file: {label: RAS position}}` as each landmark file was FIRST
        # read this session -- what the tool produced, before this reader
        # touched it. Kept in memory rather than as a backup file beside the
        # data: a file the reader did not ask for is one more thing to
        # explain, and one more thing a later run would re-ingest. The cost
        # is stated where it is offered: closing Slicer forgets it.
        self._asOpened = {}
        # Set by `openForReview` when somebody else opened this panel. None
        # for a reader who opened it themselves -- who has nothing to
        # continue, and must not be shown a button that says they have.
        self._continue = None
        # Where a flagged patient goes BACK to, when the caller offered
        # somewhere. None is the ordinary case: a reader who opened VISU
        # on a folder has no run behind them.
        self._rewind = None
        # Which run opened this panel, when one did. Empty for a reader
        # who opened VISU on a folder of their own.
        self._origin = {}
        # Everything indexed, before the cohort chips narrow it.
        self._allCases = []

    # -- building the panel ------------------------------------------------

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        # A root of our own, and `design.apply` is the reason for it: the
        # stylesheet goes down a whole widget tree, and `self.parent` also
        # holds Slicer's own Reload & Test box. Styling that would repaint a
        # part of the application this module does not own.
        self.uiWidget = qt.QWidget()
        self.layout.addWidget(self.uiWidget)
        self.panel = qt.QVBoxLayout(self.uiWidget)
        self._buildInput()
        self._buildCase()
        self._buildReview()
        self._buildModify()
        self.panel.addStretch(1)
        self._buildNavigation()
        design.apply(self.uiWidget)
        self._restore()
        self._refresh()

    def _buildInput(self) -> None:
        # Where this panel was opened FROM, when a run opened it. A reader who
        # pressed Apply in ASO and landed here needs to be told that is where
        # they are: without it the panel is a folder browser that appeared,
        # and nothing says which run is waiting on them.
        #
        # A banner rather than a hint line: a stopped run waiting on the reader
        # is the one fact on this panel that must not be read past.
        self.originBanner, self.originLabel, self.originDetail = design.review_banner()
        self.originBanner.setVisible(False)
        self.panel.addWidget(self.originBanner)
        # What a stop shows by default is what its LAST step wrote. This brings
        # back everything the earlier steps left too, for a reader who wants to
        # compare.
        self.earlierStepsCheck = qt.QCheckBox(_("Also show what the earlier steps produced"))
        self.earlierStepsCheck.setVisible(False)
        self.earlierStepsCheck.toggled.connect(lambda _on: self.onIndex())
        self.panel.addWidget(self.earlierStepsCheck)

        self.folderBox = ctk.ctkCollapsibleButton()
        box = self.folderBox
        box.text = _("Folder")
        self.panel.addWidget(box)
        form = qt.QFormLayout(box)

        # The same input row every tool panel uses, in folder-only mode: it
        # browses, it captions what was chosen, and it reports every change --
        # which a ctkPathLineEdit restricted to Dirs does not, its
        # currentPathChanged being swallowed for a folder.
        self.folderInput = formgen.FileOrFolderInput(modes=("folder",))
        self.folderInput.onPathChanged(self.onIndex)

        # The same composite every tool panel puts on a hosted argument: the
        # picker, plus a dropdown of the test data the server holds. Wrapping
        # it is all it takes -- picking an entry is an ACTION, the file is
        # fetched, and the row then holds an ordinary local path.
        self.sources = formgen.ServerFileInput(self.folderInput, hosted_downloads=True)
        # No scene dropdown: every other panel offers one so a volume already
        # open can be SENT to a tool. This panel has nothing to send, and it
        # is the thing that puts volumes in the scene in the first place.
        self.sources.setSceneSupported(False)
        self.sources.setHostedCallback(self.onTestFile)
        self.sources.container.toolTip = _(
            "A folder of scans, of results, or of both. Everything under it is "
            "indexed: a patient's scan, its masks, its surfaces and its landmarks "
            "are shown together."
        )
        form.addRow(_("Folder"), self.sources.container)

        # Which cohorts of this folder to look at. Hidden whenever there is
        # only one, because a single chip that cannot be unticked without
        # emptying the panel is a control with no decision in it.
        # Built empty and filled per folder, so it is connected in
        # `_offerFolders` instead of here: `connect_changed` walks the boxes
        # that EXIST, and a rebuild makes new ones.
        self.foldersGroup = formgen.MultiChoiceGroup({}, layout="chips")
        self.foldersLabel = design.section_title(_("Folders"))
        form.addRow("", self.foldersLabel)
        form.addRow("", self.foldersGroup.container)

        self.countLabel = design.hint_label("")
        form.addRow("", self.countLabel)

    def _buildCase(self) -> None:
        box = ctk.ctkCollapsibleButton()
        box.text = _("Patient")
        self.panel.addWidget(box)
        outer = qt.QVBoxLayout(box)

        self.caseCombo = qt.QComboBox()
        self.caseCombo.toolTip = _("Jump to a patient")
        self.caseCombo.currentIndexChanged.connect(self.onPick)
        outer.addWidget(self.caseCombo)

        self.viewCombo = qt.QComboBox()
        self.viewCombo.toolTip = _(
            "A case can hold more than one picture: the scan you sent, and the one "
            "a tool oriented. They are not interchangeable."
        )
        self.viewCombo.currentIndexChanged.connect(self.onView)
        outer.addWidget(self.viewCombo)

        # The anti-lie line. Never folded into the combo above: what a reader
        # has to notice is not which view is selected but what the points are
        # being drawn against, and whether that was found or assumed.
        #
        # A HINT and not a warning, though it was a warning first. That
        # factory paints in the danger colour and exists for "part of this
        # panel could not be built"; on an ordinary case it read as an error
        # with no error in it. Saying "assumed" is the whole signal, and it
        # says it without crying wolf.
        self.frameLabel = design.hint_label("")
        self.frameLabel.setWordWrap(True)
        outer.addWidget(self.frameLabel)

        # `chips`, so the word IS the control: five short labels, and what is
        # on reads as filled against outlined at a glance rather than as five
        # small ticks to squint at.
        self.showGroup = formgen.MultiChoiceGroup(
            {label: on for label, _kind, on in SHOWABLE}, layout="chips",
        )
        formgen.connect_changed(self.showGroup, self.onShowChanged)
        outer.addWidget(design.section_title(_("Show")))
        outer.addWidget(self.showGroup.container)

        self.contentsLabel = design.hint_label("")
        self.contentsLabel.setWordWrap(True)
        outer.addWidget(self.contentsLabel)

    def _buildReview(self) -> None:
        """What the reader is producing besides corrected files: a list.

        Six cases look fine, two do not, and what happens next -- re-running
        those two, handing them on, coming back tomorrow -- depends on the
        list outliving the session. It is written beside the data, so whoever
        opens the folder next sees what was already judged.
        """
        box = ctk.ctkCollapsibleButton()
        box.text = _("Review")
        box.collapsed = True
        self.panel.addWidget(box)
        column = qt.QVBoxLayout(box)

        self.reviewLabel = design.hint_label("")
        self.reviewLabel.setWordWrap(True)
        column.addWidget(self.reviewLabel)

        row = qt.QHBoxLayout()
        row.setSpacing(design.SPACING_SM)
        self.nextFlaggedButton = design.secondary_button(_("Go to next flagged"))
        self.nextFlaggedButton.connect("clicked()", self.onNextFlagged)
        row.addWidget(self.nextFlaggedButton, 1)
        self.clearFlagsButton = design.secondary_button(_("Clear all flags"))
        self.clearFlagsButton.connect("clicked()", self.onClearFlags)
        row.addWidget(self.clearFlagsButton, 1)
        column.addLayout(row)

    def _buildModify(self) -> None:
        """What may be changed, and nothing else.

        Everything starts locked. The chips here are the same control Slicer
        puts in the Markups module, and they move together: locking a node
        there unticks it here.
        """
        box = ctk.ctkCollapsibleButton()
        box.text = _("Modify")
        box.collapsed = True
        self.panel.addWidget(box)
        column = qt.QVBoxLayout(box)

        column.addWidget(design.hint_label(_(
            "Untick nothing and nothing can move. Unlock Landmarks to drag a "
            "point; unlock Position to drag the scan itself. Save writes what "
            "actually changed."
        )))
        self.unlockGroup = formgen.MultiChoiceGroup(
            {label: False for label, _what in EDITABLE}, layout="chips",
        )
        formgen.connect_changed(self.unlockGroup, self.onUnlockChanged)
        column.addWidget(self.unlockGroup.container)

        buttons = qt.QHBoxLayout()
        column.addLayout(buttons)
        self.revertButton = design.secondary_button(_("Revert to what is on disk"))
        self.revertButton.toolTip = _(
            "Read the files again and throw away whatever has not been saved.")
        self.revertButton.connect("clicked()", self.onRevert)
        buttons.addWidget(self.revertButton, 1)

        self.undoButton = design.secondary_button(_("Undo my changes"))
        self.undoButton.toolTip = _(
            "Write this patient back the way the tool produced it, so a "
            "different correction can be made. Only what was opened in this "
            "session can be put back: closing Slicer forgets it.")
        self.undoButton.enabled = False
        self.undoButton.connect("clicked()", self.onUndo)
        buttons.addWidget(self.undoButton, 1)

        self.modifyLabel = design.hint_label("")
        self.modifyLabel.setWordWrap(True)
        column.addWidget(self.modifyLabel)

    def _buildNavigation(self) -> None:
        """The two steppers, at the very bottom and with nothing between them.

        Below the stretch on purpose: this is the control a reader uses while
        looking at the SCAN, so it wants a fixed place at the edge of the
        panel rather than a position that moves with how much a case has to
        say about itself. Nothing between them either -- a target you reach
        for without looking must not have a drop-down beside it.
        """
        # Which of how many, beside the control that changes it. The panel
        # said it in the Folder box, three sections away from the arrows -- so
        # the one number a reader wants while stepping was the one furthest
        # from where they were looking.
        self.positionLabel = design.section_title("")
        self.panel.addWidget(self.positionLabel)

        row = qt.QHBoxLayout()
        row.setSpacing(design.SPACING_SM)

        self.previousButton = design.nav_button("◀")
        self.previousButton.setShortcut(qt.QKeySequence(qt.Qt.Key_Left))
        self.previousButton.toolTip = _("Previous case (Left arrow)")
        self.previousButton.connect("clicked()", self.onPrevious)
        row.addWidget(self.previousButton, 1)

        self.nextButton = design.nav_button("▶")
        self.nextButton.setShortcut(qt.QKeySequence(qt.Qt.Key_Right))
        self.nextButton.toolTip = _("Next case (Right arrow)")
        self.nextButton.connect("clicked()", self.onNext)
        row.addWidget(self.nextButton, 1)

        self.panel.addLayout(row)

        # Under the arrows because this is where the reader's hands are while
        # they step, and because a correction is worth nothing unsaved. ONE
        # button: what it writes is whatever was unlocked and moved, and the
        # reader does not have to know which file that lands in.
        actions = qt.QHBoxLayout()
        actions.setSpacing(design.SPACING_SM)
        # Marking a patient IS asking for it to be re-done. It was a note
        # before, which was a weaker thing than it looked: a reader who can
        # see a bad result and cannot ask for it to be redone is being asked
        # to keep a list somebody else will act on.
        self.flagButton = design.toggle_button(_("Flag"))
        self.flagButton.toolTip = _(
            "Mark this patient to be done again from an earlier step. The "
            "list is written beside the data, so it is still there tomorrow "
            "and for whoever opens the folder next."
        )
        self.flagButton.connect("clicked()", self.onFlagToggled)
        actions.addWidget(self.flagButton, 1)

        self.saveButton = design.primary_button(_("Save"))
        self.saveButton.toolTip = _(
            "Write what changed: the points that moved, and the scan's "
            "position if it was unlocked and dragged."
        )
        self.saveButton.connect("clicked()", self.onSave)
        actions.addWidget(self.saveButton, 1)
        self.panel.addLayout(actions)

        # Hidden until somebody else opens this panel (see `openForReview`).
        # A reader who opened VISU themselves has nothing to continue, and a
        # button that says otherwise is a button that does nothing.
        #
        # `success_button` rather than `primary_button`: Save is the primary
        # action here and it is pressed per patient, while this one is pressed
        # once and ends the review. Two identical buttons side by side, one of
        # which hands the cohort back to a running job, is the pair that gets
        # pressed by mistake.
        #
        # ONE button, whose words change with what it is about to do: Continue
        # while nothing is marked, and a step BACKWARDS the moment something
        # is. Two buttons were tried first and are the worse shape -- a reader
        # who has just marked three patients is offered "carry on" and "go
        # back" side by side, which is a choice they have already made by
        # marking them. It keeps the same green in both states on purpose:
        # `Save` beside it is the blue one, and a second blue button here is
        # the pair that gets pressed by mistake.
        self.continueButton = design.success_button(_("Continue"))
        self.continueButton.connect("clicked()", self.onContinue)
        self.continueButton.setVisible(False)
        self.panel.addWidget(self.continueButton)

    # -- settings ----------------------------------------------------------

    def _restore(self) -> None:
        remembered = qt.QSettings().value(_KEY_FOLDER, "") or ""
        if not remembered:
            return
        # Setting the path notifies, which indexes. `_waiting` is what keeps
        # that from also LOADING: the panel opens knowing what the folder
        # holds and with the scene untouched.
        self._restoring = True
        try:
            self.folderInput.setCurrentPath(remembered)
        finally:
            self._restoring = False

    def _remember(self) -> None:
        qt.QSettings().setValue(_KEY_FOLDER, self.folderInput.currentPath)

    # -- actions -----------------------------------------------------------

    def onIndex(self) -> None:
        self._remember()
        self._waiting = self._restoring
        folder = self.folderInput.currentPath
        # Everything the folder holds. `self.cases` is the ticked view of it,
        # so unticking a cohort costs a filter rather than another walk.
        supported, supports = getattr(self, "_supports", ("", ()))
        supports = supports if folder and folder == supported else ()
        focused, focus = getattr(self, "_focus", ("", None))
        if not (folder and folder == focused) or self.earlierStepsCheck.isChecked():
            focus = None
        sources = [(SOURCE, folder)] if folder else []
        self._allCases = index.build(sources, supports=supports, focus=focus)
        if focus and not self._allCases:
            # The step wrote nothing a viewer can show -- a report, a
            # transform -- so the earlier steps are all there is to look at.
            self._allCases = index.build(sources, supports=supports)
        self._offerFolders()
        self.cases = self._ticked()
        self.position = 0
        self._folder = folder
        self._flagged = review.load(folder) if folder else set()
        self._syncHandBack()
        self._describeReview()

        self._filling = True
        self.caseCombo.clear()
        for case in self.cases:
            self.caseCombo.addItem(case.label)
        self._filling = False

        if not self.cases:
            self._waiting = False
            self.countLabel.text = _("Nothing to show in that folder.")
        elif self._waiting:
            self.countLabel.text = _(
                "{count} patient(s). Press an arrow to show the first."
            ).format(count=len(self.cases))
        else:
            self.countLabel.text = _("{count} patient(s).").format(count=len(self.cases))
        self._refresh()

    def onPrevious(self) -> None:
        self._step(-1)

    def onNext(self) -> None:
        self._step(1)

    def _leaving(self) -> None:
        """Write what the reader changed on the patient they are leaving.

        The legacy calls this from Previous, Next and Continue alike, and it
        is the right shape: a reviewer moves on by moving on, not by
        remembering a button. Save stays because a reader wants to be TOLD
        it landed, not because anything depends on it being pressed.

        Costs nothing on an ordinary pass: nothing is unlocked, so nothing is
        looked at, so nothing is written.
        """
        if not self.unlocked() or not self._points:
            return
        said = self._saveLandmarks()
        if said:
            self.modifyLabel.text = said

    def _step(self, by: int) -> None:
        if not self.cases:
            return
        if self._waiting:
            # The first press shows what is selected. Moving instead would
            # skip case one of a cohort nobody has seen yet.
            self._waiting = False
            self.countLabel.text = _("{count} patient(s).").format(count=len(self.cases))
            self._refresh()
            return
        # Clamped rather than wrapped: a reader stepping through a cohort wants
        # to be told they are at the end, not silently returned to the start.
        moving = max(0, min(len(self.cases) - 1, self.position + by))
        if moving != self.position:
            self._leaving()
        self.position = moving
        self._refresh()

    def onPick(self, position: int) -> None:
        if self._filling:
            return
        if not (0 <= position < len(self.cases)):
            return
        if position != self.position or self._waiting:
            if position != self.position:
                self._leaving()
            self._waiting = False
            self.position = position
            self._refresh()

    def _offerWhatIsThere(self, case) -> None:
        """Grey the chips for what this patient does not have.

        Greyed rather than removed: a row that changes shape under the reader
        as they step is a row they have to re-read every time, and a chip that
        is there but off tells them this patient has no landmarks -- which is
        worth knowing. Their ticked state is left alone, so it comes back on
        the next patient that does have one.
        """
        present = {artifact.kind for artifact in case.artifacts}
        for label, kind, _on in SHOWABLE:
            box = self.showGroup.boxes.get(label)
            if box is not None:
                box.setEnabled(kind in present)

    def _offerFolders(self) -> None:
        """Rebuild the cohort chips for the folder that was just opened."""
        found = index.folders_in(self._allCases)
        self._filling = True
        try:
            self.foldersGroup.rebuild({name: True for name in found})
            formgen.connect_changed(self.foldersGroup, self.onFoldersChanged)
        finally:
            self._filling = False
        # One cohort is not a choice. Shown from two, where unticking one
        # leaves something to look at.
        offered = len(found) > 1
        self.foldersGroup.container.setVisible(offered)
        self.foldersLabel.setVisible(offered)

    def _ticked(self) -> list:
        wanted = {name for name, on in self.foldersGroup.value().items() if on}
        return [case for case in self._allCases
                if index.folder_of(case) in wanted]

    def onFoldersChanged(self, *_args) -> None:
        """Narrow the cohort without re-reading the folder."""
        if self._filling:
            return
        standing = self.cases[self.position].key if self.cases else None
        self.cases = self._ticked()
        # Stay on the same patient when it survived the change, rather than
        # jumping to the first: a reader unticking a cohort is usually not
        # looking at it.
        keys = [case.key for case in self.cases]
        self.position = keys.index(standing) if standing in keys else 0
        self._waiting = False
        self._refresh()

    def _showOrigin(self, origin) -> None:
        """Say which run is waiting, and stop offering the folder picker.

        A reader who pressed Apply in a tool panel and landed here did not
        choose this folder and must not be invited to change it: repointing
        the picker mid-review is how a correction ends up measured against
        files the run never produced.

        `origin` is `{"tool", "step", "run"}` -- the tool whose panel opened
        this, the checkpoint it stopped at written as it was published
        (`ALI_CBCT`, or `ASO/ALI_CBCT` for one inside a callee), and the
        run's number on that panel.
        """
        self._origin = dict(origin or {})
        opened_by_a_run = bool(self._origin)
        # Hidden rather than disabled: a greyed control still reads as
        # something that could be used, and there is nothing to decide here.
        self.folderBox.setVisible(not opened_by_a_run)
        self.originBanner.setVisible(opened_by_a_run)
        if not opened_by_a_run:
            self.earlierStepsCheck.setVisible(False)
            return
        tool = self._origin.get("tool") or _("a tool")
        step = self._origin.get("step") or ""
        number = self._origin.get("run")
        where = "{} / {}".format(tool, step) if step else tool
        self.originLabel.text = (
            _("Run {number} of {tool} is paused after {step}").format(
                number=number, tool=tool, step=step or _("a checkpoint")) if number
            else _("{where} is paused").format(where=where))
        self.originDetail.text = _(
            "It waits for you. Check what this step produced, correct it if it "
            "needs it, then press Continue below.")

    def _goingBack(self) -> bool:
        """Whether the hand-back button would go BACKWARDS if pressed now.

        Marking a patient is the whole decision: somebody who marked three
        has said those three need an earlier step again, so the one button
        they have left follows what they said rather than asking them again.
        """
        return bool(self._rewind and self._flagged)

    def _syncHandBack(self) -> None:
        """The flag's wording, and what the one button below says it will do.

        The flag's words are the local module's, unchanged: a reader who used
        it reads the same sentence here, and "mark this one" never said what
        marking would DO. The button then says the rest of that sentence.
        """
        somewhere = bool(self._rewind)
        if somewhere and self.flagButton.isChecked():
            self.flagButton.setText(_("Cancel - this patient is fine"))
        elif somewhere:
            self.flagButton.setText(_("Go back and edit this patient"))
        else:
            self.flagButton.setText(_("Flag"))

        if self._goingBack():
            self.continueButton.setText(
                _("Go back to {step} for {count} patient(s)").format(
                    step=self._rewind.get("tool") or _("the previous step"),
                    count=len(self._flagged)))
            self.continueButton.toolTip = _(
                "Write what you corrected, then take the marked patients "
                "back to that step so they can be done again from there. "
                "Everyone else keeps the result they already have."
            )
        else:
            self.continueButton.setText(_("Continue"))
            self.continueButton.toolTip = _(
                "Give this back to the tool that opened it. What you "
                "corrected is written first, then the run carries on from "
                "where it stopped."
            )

    def onFlagToggled(self) -> None:
        if not self.cases:
            self.flagButton.setChecked(False)
            return
        key = self.cases[self.position].key
        if self.flagButton.isChecked():
            self._flagged.add(key)
        else:
            self._flagged.discard(key)
        self._syncHandBack()
        if self._folder and not review.save(self._folder, self._flagged):
            # Said once, where the list is, rather than in a dialog over a
            # scan. A hosted sample is unpacked into a temporary folder the
            # next download deletes; a share can be read-only.
            self.reviewLabel.text = _(
                "Kept for this session only - this folder will not take the list."
            )
            return
        self._describeReview()
        self._describePosition()

    def onNextFlagged(self) -> None:
        """Step to the next marked patient, wrapping once."""
        if not self._flagged or not self.cases:
            return
        keys = [case.key for case in self.cases]
        order = keys[self.position + 1:] + keys[:self.position + 1]
        following = next((key for key in order if key in self._flagged), None)
        if following is None:
            return
        self._leaving()
        self._waiting = False
        self.position = keys.index(following)
        self._refresh()

    def onClearFlags(self) -> None:
        self._flagged = set()
        if self._folder:
            review.save(self._folder, self._flagged)
        self._readFlag()
        self._syncHandBack()
        self._describeReview()
        self._describePosition()

    def _describePosition(self) -> None:
        """Which of how many, and whether this one is marked.

        The mark belongs here and not only in the Review box: a reader
        stepping through a cohort looks at this line and nowhere else.
        """
        if not self.cases:
            self.positionLabel.text = ""
            return
        case = self.cases[self.position]
        self.positionLabel.text = _("{at} of {total} - {patient}{mark}").format(
            at=self.position + 1, total=len(self.cases), patient=case.label,
            mark=_("   FLAGGED") if case.key in self._flagged else "",
        )

    def _readFlag(self) -> None:
        """Put the button where this patient's mark is, without re-saving."""
        marked = bool(self.cases) and self.cases[self.position].key in self._flagged
        if self.flagButton.isChecked() != marked:
            self.flagButton.setChecked(marked)

    def _describeReview(self) -> None:
        self.reviewLabel.text = (
            review.as_text(self._flagged) if self._flagged
            else _("Nothing flagged in this folder.")
        )
        self.nextFlaggedButton.enabled = bool(self._flagged)
        self.clearFlagsButton.enabled = bool(self._flagged)

    def unlocked(self) -> set:
        """What the reader has said may move."""
        return {label for label, on in self.unlockGroup.value().items() if on}

    def onUnlockChanged(self, *_args) -> None:
        if self._filling or self._waiting:
            return
        self._applyLock()
        wanted = self.unlocked()
        if "Position" in wanted:
            self._attachAdjustment()
        else:
            self._detachAdjustment()

    def _applyLock(self) -> None:
        """Push the lock chips onto the nodes.

        Guarded: setting a node's lock fires the observer that reads the
        nodes back into the chips, and the two would chase each other.
        """
        locked = "Landmarks" not in self.unlocked()
        self._syncing = True
        try:
            for _artifact, node in self._points:
                self.scene.set_locked(node, locked)
        finally:
            self._syncing = False

    def onSceneChanged(self, *_args) -> None:
        """Read the scene back into the chips.

        The eye in the Markups module and the Show chips are two switches on
        one thing; so are the padlock there and the Modify chips. Whichever
        the reader touches, the other has to follow, or the panel is telling
        them something that is not on screen.
        """
        if self._syncing or self._filling or self._waiting:
            return
        self._syncing = True
        try:
            self._readVisibility()
            self._readLocks()
        finally:
            self._syncing = False

    def _readVisibility(self) -> None:
        seen, visible = set(), set()
        for artifact, node in self.scene.shown():
            seen.add(artifact.kind)
            display = node.GetDisplayNode()
            if display is not None and display.GetVisibility():
                visible.add(artifact.kind)
        for label, kind, _on in SHOWABLE:
            box = self.showGroup.boxes.get(label)
            if box is None or kind not in seen:
                continue
            if box.isChecked() != (kind in visible):
                box.setChecked(kind in visible)

    def _readLocks(self) -> None:
        if not self._points:
            return
        # Unlocked here means EVERY point of every set can move; one locked
        # node is enough to say the reader is not editing.
        free = all(not node.GetLocked() for _artifact, node in self._points)
        box = self.unlockGroup.boxes.get("Landmarks")
        if box is not None and box.isChecked() != free:
            box.setChecked(free)

    def _untick(self, option: str) -> None:
        """Put an unlock chip back down without re-entering its handler."""
        box = self.unlockGroup.boxes.get(option)
        if box is None:
            return
        self._filling = True
        try:
            box.setChecked(False)
        finally:
            self._filling = False

    def _attachAdjustment(self) -> None:
        """Put the anchor under a transform the reader can drag."""
        if self._adjustment is not None:
            return
        if self._anchorNode is None:
            self._untick("Position")
            self.modifyLabel.text = _("Nothing on screen to move.")
            return
        try:
            self._adjustment = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode", "VISU adjustment")
            self._anchorNode.SetAndObserveTransformNodeID(self._adjustment.GetID())
            self._adjustment.CreateDefaultDisplayNodes()
            display = self._adjustment.GetDisplayNode()
            if display is not None:
                display.SetEditorVisibility(True)
            self.modifyLabel.text = _("Drag the handles, then Save.")
        except Exception as exc:  # noqa: BLE001
            # Half a transform is worse than none: the scan sits under a node
            # with no handles, which moves nothing and unticking cannot undo.
            logger.warning("Could not offer an adjustment: %s", exc)
            self._detachAdjustment()
            self._untick("Position")
            self.modifyLabel.text = _("This scan cannot be moved here.")

    def onShowChanged(self, *_args) -> None:
        """A chip changes what is VISIBLE, and nothing else.

        Not `_show`: that clears the scene and reads every file again, which
        is where the reset came from. Nothing is loaded, unloaded, relaid out
        or recentred here -- a reader who has scrolled to the tooth they were
        checking is still looking at it.
        """
        if not self._filling and not self._waiting:
            self._applyVisibility(reframe=False)

    def _applyVisibility(self, reframe: bool) -> None:
        """Show what the chips ask for, out of what is already loaded."""
        wanted = self.wanted_kinds()
        for artifact, node in self.scene.shown():
            self.scene.set_visible(node, artifact.kind in wanted)

        anchor, anchor_node, label_node = getattr(
            self, "_anchorLayers", (None, None, None))
        on = anchor is not None and anchor.kind in wanted
        self.scene.display(anchor if on else None,
                           anchor_node if on else None,
                           label_node if label_node is not None
                           and index.LABELMAP in wanted else None,
                           reframe=reframe)

    def wanted_kinds(self) -> set:
        """The kinds the check boxes are letting through."""
        return {_KIND_OF_OPTION[label]
                for label, on in self.showGroup.value().items() if on}

    def onView(self, _position: int) -> None:
        if not self._filling and not self._waiting:
            self._show()

    def _refresh(self) -> None:
        has = bool(self.cases)
        self.previousButton.enabled = has and self.position > 0
        self.nextButton.enabled = has and self.position < len(self.cases) - 1
        self._readFlag()
        marked = bool(self.cases) and self.cases[self.position].key in self._flagged
        self.positionLabel.text = (
            _("{at} of {total} - {patient}{mark}").format(
                at=self.position + 1, total=len(self.cases),
                patient=self.cases[self.position].label,
                mark=_("   FLAGGED") if marked else "",
            ) if has else ""
        )
        if not has:
            self.views = []
            self._filling = True
            self.viewCombo.clear()
            self._filling = False
            self.frameLabel.text = ""
            self.contentsLabel.text = ""
            self.scene.clear()
            self.undoButton.enabled = False
            return

        self._filling = True
        if self.caseCombo.currentIndex != self.position:
            self.caseCombo.setCurrentIndex(self.position)

        self._offerWhatIsThere(self.cases[self.position])
        self.views = self.cases[self.position].views(acquisition=SOURCE)
        self.viewCombo.clear()
        for view in self.views:
            self.viewCombo.addItem(view.label)
        # Open on a view that has something ON it. A case can offer the scan
        # as sent and the scan a tool oriented, and only one of them carries
        # the points a reader came to look at.
        carrying = next(
            (n for n, view in enumerate(self.views) if view.overlays), 0
        )
        self.viewCombo.setCurrentIndex(carrying)
        self._filling = False
        self._syncUndo()
        if self._waiting:
            # Controls filled, nothing opened. Reading ahead waits too: it is
            # a courtesy for a reader who is stepping, not for one who has not
            # arrived.
            return
        self._show()
        self._prefetchNeighbour()

    def _show(self, reframe: bool = True) -> None:
        self.scene.clear()
        if not self.views:
            return
        # What VISU did not load is hidden too: the input scan a panel had
        # rendered, an earlier run's results. A pause after ASO showed the
        # oriented CBCT on top of the original, two volumes in one place.
        slicer_io.hide_scene()
        position = max(0, self.viewCombo.currentIndex)
        view = self.views[min(position, len(self.views) - 1)]

        self._points = []
        self._adjustment = None
        # Everything the case holds is LOADED; the chips decide what is
        # SHOWN. Filtering here instead meant a tick tore the volume out of
        # the scene and read it off disk again -- 2.6 s, a brand new node, and
        # every view setting attached to the old one gone. Slicer already has
        # a visibility switch per node; a chip is that switch, put where a
        # reader's hands are.
        anchor = view.anchor
        overlays = list(view.overlays)

        anchor_node = self.scene.load(anchor) if anchor is not None else None
        self._anchorNode = anchor_node
        on_a_scan = anchor is not None and anchor.kind == index.VOLUME
        masks = [o for o in overlays if o.kind == index.LABELMAP]
        # Masks on a scan become segmentations, one as much as several: a
        # label layer is drawn in the slices only, and at an AMASSS stop the
        # reader judged a segmentation with an empty 3D view. Several could
        # never be label layers anyway, a volume having one.
        stack = on_a_scan and bool(masks)

        label_node = None
        points = []
        # Not an overlay: a transform has no geometry, so no view holds one.
        # Taken from the case, which is where it sits.
        for artifact in self.cases[self.position].of_kind(index.TRANSFORM):
            self.scene.load(artifact)
        for overlay in overlays:
            as_segmentation = stack and overlay.kind == index.LABELMAP
            node = self.scene.load(overlay, opened_as=SEGMENTATION if as_segmentation else "")
            if node is None:
                continue
            if as_segmentation:
                SceneLoader.show_in_3d(node)
            if overlay.kind == index.LABELMAP and not stack and label_node is None:
                label_node = node
            if overlay.kind == index.MODEL and on_a_scan:
                self.scene.draw_on_slices(node)
            if overlay.kind == index.MARKUPS:
                points.append(node)
                self._points.append((overlay, node))
                self._rememberAsOpened(overlay, node)
        for _artifact, node in self.scene.shown():
            self.scene.watch(node, self.onSceneChanged)
        # Freshly loaded nodes carry the file's own flags; the reader's choice
        # is applied over them, so stepping to the next patient does not
        # quietly re-lock what they unlocked.
        self._applyLock()
        self._anchorLayers = (anchor, anchor_node, label_node)
        self._applyVisibility(reframe=reframe)
        if reframe and points:
            # The slices open on the volume's centre and the points are not
            # there. One of them has to be, or the chip looks broken.
            self.scene.jump_to(points[0])

        if overlays:
            self.frameLabel.text = _("Drawn on {scan} ({basis})").format(
                scan=view.label, basis=view.basis
            )
        else:
            # Said outright. A blank line here reads as a panel that failed to
            # draw something, and the reader goes looking for the bug: the
            # hosted `cohort_6` is six scans and no landmarks at all, which is
            # what it is rather than what went wrong.
            self.frameLabel.text = _("No landmarks or masks for this patient here.")
        self.contentsLabel.text = "\n".join(
            f"{artifact.kind}: {artifact.name}"
            for artifact in [anchor] + overlays
            if artifact is not None
        )

    def _prefetchNeighbour(self) -> None:
        following = self.position + 1
        if following >= len(self.cases):
            return
        prefetch([artifact.path for artifact in self.cases[following].artifacts])

    # -- modifying -----------------------------------------------------------

    def onSave(self) -> None:
        """Write whatever changed: the points that moved, and the position.

        One button, because the reader corrected a case and not a file. What
        it touches is decided by what was unlocked -- nothing is written for
        a kind that could not have been changed.
        """
        said = []
        if "Landmarks" in self.unlocked():
            said.append(self._saveLandmarks())
        if self._adjustment is not None:
            said.append(self._savePosition())
        self.modifyLabel.text = ("  ".join(s for s in said if s)
                                 or _("Nothing is unlocked, so nothing was written."))
        self._syncUndo()

    @staticmethod
    def _positionsOf(node) -> dict:
        """`{label: RAS position}` for a markups node, or `{}`."""
        positions = {}
        for point in range(node.GetNumberOfControlPoints()):
            place = [0.0, 0.0, 0.0]
            node.GetNthControlPointPosition(point, place)
            positions[node.GetNthControlPointLabel(point)] = place
        return positions

    def _saveLandmarks(self) -> str:
        """Write every point that moved back into the file it came from."""
        moved, files = 0, 0
        for artifact, node in self._points:
            try:
                positions = self._positionsOf(node)
            except Exception as exc:  # noqa: BLE001 - one file is not the save
                logger.warning("Could not read %s back: %s", artifact.name, exc)
                continue
            try:
                count = edits.save_markups(artifact.path, positions)
            except (OSError, ValueError) as exc:
                # ValueError as well as OSError: a landmark file can be
                # truncated or hand-edited, and saving now happens on the way
                # OUT of a patient -- so an unreadable one would take the
                # panel down mid-navigation rather than at a button press.
                slicer.util.errorDisplay(plain_text(
                    _("Could not write {name}: {error}").format(
                        name=artifact.name, error=exc)))
                continue
            moved += count
            files += 1 if count else 0

        if not self._points:
            return _("No landmarks on screen to save.")
        if moved:
            self._written.add(self._currentKey())
            return _("{moved} point(s) written, in {files} file(s).").format(
                moved=moved, files=files)
        # Said, rather than left silent. "Saved" over an unchanged file is a
        # claim the reader cannot check.
        return _("Nothing moved, so nothing was written.")

    def _savePosition(self) -> str:
        """Write the displacement beside the scan, as its own transform.

        Its own file rather than folded into whatever transform the tool
        wrote: composing two matrices is a claim about which order they apply
        in, and getting that backwards is silent. A `.tfm` beside the scan is
        something a reader can load, look at and delete.
        """
        anchor = self._currentAnchor()
        if anchor is None:
            return ""
        stem, _extension = index.split_extension(anchor.name)
        destination = os.path.join(os.path.dirname(anchor.path),
                                   f"{stem}{ADJUSTMENT_SUFFIX}")
        try:
            slicer.util.saveNode(self._adjustment, destination)
        except Exception as exc:  # noqa: BLE001
            slicer.util.errorDisplay(
                _("Could not write the transform: {error}").format(error=exc))
            return ""
        self._written.add(self._currentKey())
        return _("Written: {name}").format(name=os.path.basename(destination))

    def _rememberAsOpened(self, artifact, node) -> None:
        """Keep this file's points as they were the first time it was read.

        `setdefault`, and that is the whole subtlety: a reader who saves and
        steps away comes back to a file that now holds THEIR positions, and a
        second snapshot would quietly make the correction the thing undo
        returns to.
        """
        if artifact.path in self._asOpened:
            return
        try:
            self._asOpened[artifact.path] = self._positionsOf(node)
        except Exception as exc:  # noqa: BLE001 - one file is not the panel
            logger.warning("Could not read %s: %s", artifact.name, exc)

    def _adjustmentFile(self) -> str:
        """The transform this panel wrote beside the scan, if it is there."""
        anchor = self._currentAnchor()
        if anchor is None:
            return ""
        stem, _extension = index.split_extension(anchor.name)
        path = os.path.join(os.path.dirname(anchor.path),
                            f"{stem}{ADJUSTMENT_SUFFIX}")
        return path if os.path.isfile(path) else ""

    def onUndo(self) -> None:
        """Put this patient back to what the tool produced, and rewrite it.

        Not `Revert`, which reloads the file: once a correction has been
        saved, the prediction the reader started from is no longer on disk,
        so there is nothing to reload. This writes the opening positions
        back, which is what lets a reader try a different correction rather
        than having to undo their own by hand.
        """
        restored, failed = 0, 0
        for artifact, _node in self._points:
            opened = self._asOpened.get(artifact.path)
            if not opened:
                continue
            try:
                restored += edits.save_markups(artifact.path, opened)
            except (OSError, ValueError) as exc:
                failed += 1
                slicer.util.errorDisplay(
                    _("Could not put {name} back: {error}").format(
                        name=artifact.name, error=exc))
        dropped = self._adjustmentFile()
        if dropped:
            try:
                os.remove(dropped)
            except OSError as exc:
                dropped = ""
                logger.warning("Could not remove the transform: %s", exc)
        self._detachAdjustment()
        if not failed:
            self._written.discard(self._currentKey())
        self._show(reframe=False)
        self._syncUndo()
        said = []
        if restored:
            said.append(_("{count} point(s) put back.").format(count=restored))
        if dropped:
            said.append(_("Removed {name}.").format(
                name=os.path.basename(dropped)))
        self.modifyLabel.text = ("  ".join(said)
                                 or _("Nothing to undo for this patient."))

    def _syncUndo(self) -> None:
        """Undo is offered only where there is something of ours to undo."""
        key = self._currentKey()
        self.undoButton.enabled = bool(key) and (
            key in self._written or bool(self._adjustmentFile()))

    def onRevert(self) -> None:
        """Throw away every unsaved change by reloading from disk."""
        self._detachAdjustment()
        self._filling = True
        for box in self.unlockGroup.boxes.values():
            box.setChecked(False)
        self._filling = False
        self._show(reframe=False)
        self.modifyLabel.text = _("Reloaded from disk.")

    def _currentKey(self) -> str:
        return self.cases[self.position].key if self.cases else ""

    def _currentAnchor(self):
        if not self.views:
            return None
        view = self.views[min(max(0, self.viewCombo.currentIndex),
                              len(self.views) - 1)]
        return view.anchor

    def _detachAdjustment(self) -> None:
        if self._adjustment is None:
            return
        try:
            if self._anchorNode is not None:
                self._anchorNode.SetAndObserveTransformNodeID(None)
            slicer.mrmlScene.RemoveNode(self._adjustment)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not take the adjustment off: %s", exc)
        self._adjustment = None

    # -- opened by somebody else ---------------------------------------------

    def openForReview(self, folder: str, on_continue=None, rewind=None,
                      origin=None, inputs=(), focus=None) -> None:
        """Show `folder`, and give the caller a way to be told when to carry on.

        This is the whole of what VISU learns about the thing that opened it.
        It is handed a directory and a callable; it does not know there is a
        run, a server, or a step that stopped -- and it must not, or the next
        caller with a folder and a question has to be added to it.

        `on_continue` is what puts the Continue button on the panel. A reader
        who opened VISU themselves passes none and sees none, because they
        have nothing to hand back to.

        `rewind` is the step a reader may send flagged patients BACK to, as
        `{"slot", "tool", "kind"}`, or None when there is nowhere to go --
        which is the ordinary case and is why it defaults to none. VISU still
        learns nothing about runs: it is handed a NAME to show and hands the
        flags back, and what that name means is the caller's.
        """
        self._continue = on_continue
        self._rewind = rewind
        # The run's own files, for landmarks a step handed back without their
        # scan. Tied to THIS folder: repointing the picker drops them.
        self._supports = (folder, tuple(inputs or ()))
        # What the step that stopped wrote, shown first (see `index.build`).
        self._focus = (folder, focus or None)
        self._showOrigin(origin)
        self.earlierStepsCheck.setVisible(bool(focus))
        self.earlierStepsCheck.setChecked(False)
        self.continueButton.setVisible(on_continue is not None)
        self._syncHandBack()
        # Reviewing this folder starts now, whatever an earlier pass over it
        # wrote: the caller wants what this reader changes, not the union.
        self._written = set()
        before = self.folderInput.currentPath
        self.folderInput.setCurrentPath(folder)
        if self.folderInput.currentPath == before:
            # `setCurrentPath` notifies only on a CHANGE, and the same folder
            # is opened twice whenever one run stops at a second checkpoint
            # under one output directory. Indexing again is what puts the
            # files the first resume produced on screen.
            self.onIndex()

    def endReview(self, folder=None) -> bool:
        """Leave review mode: no banner, no Continue, the folder picker back.

        Called when the reader hands the review back and when the run is
        withdrawn. Without it the banner kept saying which run was paused and
        where after that run had resumed and finished. `folder` limits it to
        that review, so withdrawing one run never ends another's.
        """
        reviewing = self._supports[0] if getattr(self, "_supports", None) else None
        if folder is not None and folder != reviewing:
            return False
        self._continue = None
        self._rewind = None
        self._supports = (None, ())
        self._focus = (None, None)
        self._showOrigin(None)
        self.earlierStepsCheck.setVisible(False)
        self.continueButton.setVisible(False)
        self._syncHandBack()
        return True

    def reviewed(self, rewind_to=None) -> dict:
        """What this pass produced, for whoever asked for it.

        A dict rather than arguments in an order. This crosses the seam
        between two modules that ship together and are read apart, and the
        next thing a caller will want -- a note per patient, which
        `SADTVisuLib.review` already versioned its file for -- must be one key
        added here rather than a signature both sides change on the same day.

        Three things, and each is something only the panel can answer:

        * `folder`, because the reader can repoint the picker, so what was
          reviewed is not necessarily what the caller opened;
        * `flagged`, the reader's verdict, which exists nowhere else;
        * `written`, the patients whose files this panel actually changed.
          Without it a caller must send a whole cohort back -- hundreds of
          megabytes -- on behalf of a reader who corrected nothing.

        `rewind_to` is the step the reader asked the FLAGGED patients to be
        taken back to, or None for an ordinary Continue. It is the caller's
        own name for that step, handed straight back: VISU is told a name and
        repeats it, which is what keeps it ignorant of runs.
        """
        return {
            "folder": self._folder,
            "flagged": set(self._flagged),
            "written": set(self._written),
            "rewind_to": rewind_to,
        }

    def onContinue(self) -> None:
        """Write what is pending, then hand control back. Once."""
        if self._continue is None:
            return
        # The reader pressed Continue rather than Save, and the point they
        # just dragged is exactly what the caller is about to collect. Same
        # call the arrows make on the way out of a patient.
        self._leaving()
        # Taken before it is called: the handler will start an upload and may
        # well come back through this panel, and a second Continue would
        # resume one run twice.
        handler, self._continue = self._continue, None
        # Read BEFORE the button is hidden and while the flags are still
        # standing: this is the one place the two destinations part company.
        backwards = self._rewind.get("slot") if self._goingBack() else None
        reviewed = self.reviewed(rewind_to=backwards)
        # Before the handler: a run that stops again reopens the review from
        # inside it, and that new review must not be ended behind it.
        self.endReview()
        handler(reviewed)

    # -- leaving -----------------------------------------------------------

    def enter(self) -> None:
        # Re-read on every visit rather than once at build: the server may
        # have been down when the module was first opened, or bundles may have
        # been fetched since.
        self._refreshTestFiles()

    def _refreshTestFiles(self) -> None:
        """Offer every tool's hosted test files, fetched off the main thread."""
        def work(_progress):
            client = get_client()
            called = {(tool, name): label for tool, name, label in SAMPLE_DATA}
            wanted = {}
            for tool, name, _label in SAMPLE_DATA:
                wanted.setdefault(tool, set()).add(name)
            found = []
            for tool in sorted(wanted):
                try:
                    data = client.list_tool_data(tool)
                except Exception as exc:  # noqa: BLE001 - one tool is not the list
                    logger.info("No hosted data for %s: %s", tool, exc)
                    continue
                for entry in testfile_entries(data):
                    name = entry.get("name", "")
                    if name in wanted[tool]:
                        found.append((tool, name, entry.get("kind"),
                                      entry.get("size"), called[(tool, name)]))
            return hosted_choices(found)

        def done(result):
            entries, offered = result
            self._hosted = offered
            self.sources.setChoices(entries)

        def failed(exc):
            # A server that is away costs the dropdown, not the panel: every
            # local folder still opens.
            logger.warning("Could not list the hosted test files: %s", exc)

        BackgroundJob(work, on_success=done, on_error=failed).start()

    def onTestFile(self, label: str) -> None:
        """Fetch the hosted entry the reader picked, then point the panel at it."""
        found = self._hosted.get(label)
        if found is None:
            logger.warning("No hosted test file called %r", label)
            return
        tool, name, kind = found

        def work(progress):
            return self._fetch(tool, name, kind, progress)

        def done(path):
            # Writing the local path resets the dropdown to its prompt and
            # notifies, which indexes and shows. The row then holds a local
            # folder like any other.
            formgen.set_local_path(self.sources, path)

        def failed(exc):
            slicer.util.errorDisplay(
                _("Could not fetch {name}: {error}").format(name=name, error=exc)
            )

        BackgroundJob(work, on_success=done, on_error=failed).start()

    def _fetch(self, tool: str, name: str, kind, progress) -> str:
        """Download one hosted entry and hand back a FOLDER to index.

        A hosted folder arrives as a zip the server built and is unpacked. A
        hosted single file is left as it is and its staging directory is
        returned instead -- a lone scan is a cohort of one, and the index
        walks directories.
        """
        if self._staging:
            shutil.rmtree(self._staging, ignore_errors=True)
        self._staging = tempfile.mkdtemp(prefix="VISU_")
        payload = os.path.join(self._staging, os.path.basename(name) or "testfile")
        get_client().download_testfile(tool, name, payload, progress)
        if kind != "folder":
            return self._staging
        unpacked = os.path.join(self._staging, "unpacked")
        slicer_io.unzip_folder(payload, unpacked)
        os.remove(payload)
        return unpacked

    def exit(self) -> None:
        """Leaving the panel leaves the scene alone, deliberately.

        It used to empty it, on the reasoning that what this panel loaded is
        this panel's. That is wrong for a VIEWER: Models, Volume Rendering
        and Segment Editor are where a reader goes to work on exactly what
        VISU just put on screen, and switching to one of them wiped it. The
        panel was clearing the scene at the precise moment its work became
        useful.

        What still bounds the scene is unchanged: every case load clears what
        the previous one owned, so stepping never accumulates, and `cleanup`
        empties it when the module itself goes away.
        """

    def cleanup(self) -> None:
        self.scene.clear()
        if self._staging:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = ""
