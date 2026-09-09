"""Pausing a run so a clinician can look at a step's result, and fix it.

A step declares what it produced and whether the user may change it; this
module turns that into nodes on screen, and writes back only what actually
moved. The widget owns the buttons and the wording, nothing here touches the
interface.
"""

import logging
import os
import re

import slicer
import vtk

logger = logging.getLogger(__name__)

VOLUME_EXT = (".nrrd", ".nii.gz", ".nii", ".nrrd.gz", ".gipl.gz", ".gipl")
MODEL_EXT = (".vtk", ".vtp", ".stl")

# What a step lets the user do with its result.
VIEW = "view"                # look only
LANDMARKS = "landmarks"      # drag the points, saved back to their file
REGISTRATION = "registration"  # drag the scan, folded into its matrix


def patientIdFromFileName(basename: str) -> str:
    """Patient id shared by a scan, its landmarks and its matrix.

    Every module of the pipeline appends its own marker to the scan name, so
    the id is what is left once those are stripped. The order matters: the
    longest markers have to go first or a shorter one cuts inside them.
    """
    for token in ["_SegOr", "_Scan", "_scan", "_Or", "_OR", "_MAND", "_MD",
                  "_MAX", "_MX", "_CB", "_lm", "_Pred", "_T1", "_T2", "_Cl",
                  "_Center", "_left", "_Left", "_right", "_Right", "_U", "_L",
                  "."]:
        basename = basename.split(token)[0]
    return basename


class ReviewSession:
    """The patients waiting to be reviewed for one finished step."""

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self):
        """Forget the review in progress, scene included."""
        self.clearNodes()
        self.queue = []
        self.index = 0
        self.pending = False
        self.step = {}

    def clearNodes(self):
        """Take the previous patient's nodes back out of the scene."""
        for node in getattr(self, "nodes", []):
            try:
                slicer.mrmlScene.RemoveNode(node)
            except Exception as e:
                logger.warning(f"Could not remove a reviewed node: {e}")
        self.nodes = []
        self.markups_nodes = []
        self.markups_start = {}
        self.transform = None
        self.transform_item = None

    @property
    def total(self):
        return len(self.queue)

    @property
    def current(self):
        if 0 <= self.index < len(self.queue):
            return self.queue[self.index]
        return None

    @property
    def remaining(self):
        return max(0, len(self.queue) - (self.index + 1))

    # --------------------------------------------------------------- building

    def build(self, step: dict, expected=None) -> list:
        """One review item per patient for the step that just finished.

        Args:
            step: the finished step's dictionary, carrying its Review* keys
            expected: patient ids this run is about. Output folders are reused
                between runs, so without it the review walks patients the run
                never touched. Identity rather than file date: a module that
                skips a patient whose output is already there leaves a file
                weeks old, and dating the file drops a patient that really is
                part of the run.

        Returns:
            list: the items, which is also kept as this session's queue
        """
        wanted_ids = {self._normalisedId(p) for p in (expected or ())} or None
        self.step = step
        folder = step.get("ReviewFolder")
        if not folder or not os.path.isdir(folder):
            logger.warning(f"Nothing to review: {folder} not found")
            self.queue = []
            return []

        # A step carries its id; what that id lets the user do is the
        # catalogue's business, so a mode never has to repeat it.
        kind = step.get("ReviewKind") or CATALOGUE.get(
            step.get("ReviewId", ""), {}
        ).get("kind", VIEW)
        if kind == LANDMARKS:
            wanted = (".json",)
        elif kind == REGISTRATION:
            wanted = VOLUME_EXT + MODEL_EXT
        else:
            wanted = VOLUME_EXT + MODEL_EXT

        by_patient = {}
        skipped = set()
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                if not name.endswith(wanted):
                    continue
                patient = patientIdFromFileName(name)
                if wanted_ids is not None and self._normalisedId(patient) not in wanted_ids:
                    skipped.add(patient)
                    continue
                by_patient.setdefault(patient, []).append(os.path.join(root, name))
        if skipped:
            logger.info(
                f"{len(skipped)} patient(s) left out of the review, not part of this "
                f"run: {sorted(skipped)}"
            )

        # Landmarks mean nothing without the scan they were placed on, and a
        # registration can only be judged against what it was registered to.
        references = self._indexAll(step.get("ReviewReferenceFolder"), VOLUME_EXT + MODEL_EXT)

        # An adjusted registration is only worth saving if its matrix can be
        # updated with it: what comes after follows the matrix, not the voxels.
        matrices = self._index(step.get("ReviewMatrixFolder") or folder, (".tfm",))

        def matrixFor(patient):
            return matrices.get(patient) or self._matchAcrossNaming(patient, matrices)

        queue = []
        for patient in sorted(by_patient):
            if kind == LANDMARKS and patient not in references:
                logger.warning(f"No scan found for {patient}, its landmarks lose their context")
            matrix = matrixFor(patient)
            adjustable = kind == REGISTRATION and matrix is not None
            if kind == REGISTRATION and not adjustable:
                logger.warning(
                    f"No matrix for {patient}: its registration can be looked at, not moved"
                )
            found = references.get(patient) or self._matchAcrossNaming(
                patient, references
            )
            queue.append({
                "patient": patient,
                "files": by_patient[patient],
                "references": list(found or []),
                "kind": kind,
                "editable": kind == LANDMARKS,
                "adjustable": adjustable,
                "matrix": matrix,
            })

        self.queue = queue
        self.index = 0
        return queue

    @staticmethod
    def _normalisedId(patient_id):
        """The id AREG_IOSCBCT reduces a patient to when it pairs the modalities.

        Underscores go, then leading zeros, so P_0001, P001 and P1 all land on
        the same patient.
        """
        compact = (patient_id or "").replace("_", "")
        match = re.match(r"([A-Za-z]*)([0-9]*)", compact)
        if not match:
            return compact
        letters, digits = match.group(1), match.group(2)
        if digits:
            digits = str(int(digits))
        return letters + digits

    @classmethod
    def _matchAcrossNaming(cls, patient, index):
        """Find this patient's file when the two sides name it differently.

        A CBCT keeps the name its dataset gave it while the registered IOS
        carries the id AREG_IOSCBCT normalised, so P1_T2_Reg_U.vtk and
        P_0001_T2_Or.nii.gz never match as strings even though they are the
        same patient - and the review would show the IOS alone, with nothing
        to judge it against.
        """
        target = cls._normalisedId(patient)
        if not target:
            return None
        matches = [path for other, path in index.items()
                   if cls._normalisedId(other) == target]
        if not matches:
            return None
        if len(matches) > 1:
            # Two patients whose ids collide once normalised. Which one belongs
            # to this scan is not recoverable here, so say so rather than pick
            # one quietly and show the wrong anatomy underneath.
            logger.warning(
                f"{patient}: several candidates match once the id is normalised "
                f"({[os.path.basename(str(m)) for m in matches]}), taking the first"
            )
        return matches[0]

    @staticmethod
    def _index(folder, extensions):
        """First file of each patient in a folder, by patient id."""
        found = {}
        for patient, paths in ReviewSession._indexAll(folder, extensions).items():
            found[patient] = paths[0]
        return found

    @staticmethod
    def _indexAll(folder, extensions):
        """Every file of each patient in a folder, by patient id.

        An IOS is two files, one arch each, and keeping only the first left the
        upper arch off the screen while its landmarks were shown on it.
        """
        found = {}
        if not folder or not os.path.isdir(folder):
            return found
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                if name.endswith(extensions):
                    found.setdefault(patientIdFromFileName(name), []).append(
                        os.path.join(root, name)
                    )
        return found

    # ---------------------------------------------------------------- loading

    def loadCurrent(self) -> bool:
        """Put the current patient on screen. True if anything could be shown."""
        item = self.current
        if item is None:
            return False

        self.clearNodes()
        loaded = False

        loaded_references = []
        for path in item.get("references", []):
            node = self._load(path)
            if node is not None:
                loaded_references.append(node)
                loaded = True

        moving = None
        for path in item["files"]:
            node = self._load(path)
            if node is not None:
                moving = node
                loaded = True

        if not loaded:
            return False

        # A volume is what the slice views can show behind everything else; an
        # IOS reference is a pair of surfaces and has no such role.
        reference = next(
            (n for n in loaded_references if n.IsA("vtkMRMLScalarVolumeNode")),
            loaded_references[0] if loaded_references else None,
        )

        if item["adjustable"] and moving is not None:
            self._setUpAdjustment(item, reference, moving)
        else:
            self._showTogether(reference, moving)

        self._layout(item)
        return True

    def _load(self, path):
        """Load one file, dispatching on what it is."""
        try:
            if path.endswith(".json"):
                node = self._loadEditableMarkups(path)
            elif path.endswith(MODEL_EXT):
                node = slicer.util.loadModel(path)
            else:
                node = slicer.util.loadVolume(path)
        except Exception as e:
            logger.error(f"Could not load {path}: {e}")
            return None

        if node is None:
            logger.warning(f"Nothing loaded from {path}")
            return None

        self.nodes.append(node)
        return node

    def _loadEditableMarkups(self, path):
        """Load landmarks so their points can actually be dragged.

        ALI writes every control point with "locked": true and the display
        hidden, so loading one as it comes shows an empty view holding points
        that refuse to move. Both are forced on the node only: the file keeps
        its own flags unless a position really changes.
        """
        node = slicer.util.loadMarkups(path)
        if node is None:
            return None

        node.SetLocked(False)
        for i in range(node.GetNumberOfControlPoints()):
            node.SetNthControlPointLocked(i, False)

        display = node.GetDisplayNode()
        if display:
            display.SetVisibility(True)
            display.SetPointLabelsVisibility(True)

        self.markups_nodes.append(node)
        self.markups_start[node.GetID()] = self.markupsPositions(node)
        return node

    def _setUpAdjustment(self, item, reference, moving):
        """Let the user drag the registered scan onto the one it targets.

        The displacement is held in a transform node, and read back on
        Continue. Nothing is asked of the user beyond moving the image.
        """
        transform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", f"{item['patient']} manual adjustment"
        )
        moving.SetAndObserveTransformNodeID(transform.GetID())
        self.nodes.append(transform)
        self.transform = transform
        self.transform_item = item

        self._showTogether(reference, moving)
        try:
            display = transform.GetDisplayNode()
            if display is None:
                transform.CreateDefaultDisplayNodes()
                display = transform.GetDisplayNode()
            if display is not None:
                display.SetEditorVisibility(True)
        except Exception as e:
            logger.warning(f"Could not show the interaction handles: {e}")

    @staticmethod
    def _showTogether(reference, moving):
        """Half-opaque over the reference is what makes a misalignment visible."""
        if reference is not None and reference.IsA("vtkMRMLScalarVolumeNode"):
            if moving is not None and moving.IsA("vtkMRMLScalarVolumeNode"):
                slicer.util.setSliceViewerLayers(
                    background=reference, foreground=moving, foregroundOpacity=0.5
                )
            else:
                slicer.util.setSliceViewerLayers(background=reference)
        elif moving is not None and moving.IsA("vtkMRMLScalarVolumeNode"):
            slicer.util.setSliceViewerLayers(background=moving)

    def _layout(self, item):
        """A layout suited to what is on screen."""
        manager = slicer.app.layoutManager()
        if manager is None:
            return
        paths = list(item["files"]) + list(item.get("references", []))
        # Landmarks sitting on a pair of surfaces belong in the 3D view: the
        # slice views have no volume to draw them against.
        no_volume = paths and not any(p.endswith(VOLUME_EXT) for p in paths)
        if no_volume:
            manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
            widget = manager.threeDWidget(0)
            if widget:
                widget.threeDView().resetFocalPoint()
        else:
            manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            slicer.util.resetSliceViews()

    # ----------------------------------------------------------------- saving

    def saveEdits(self) -> None:
        """Write back whatever the user actually changed, and nothing else."""
        self._saveMarkups()
        self._saveAdjustment()

    def _saveMarkups(self):
        """Rewrite the landmark files whose points moved.

        A file left untouched is not rewritten, so a run where the user only
        looked leaves ALI's output byte for byte as it was.
        """
        for node in self.markups_nodes:
            storage = node.GetStorageNode()
            path = storage.GetFileName() if storage else None
            if not path:
                logger.warning(f"No file to save {node.GetName()} back to")
                continue

            before = self.markups_start.get(node.GetID())
            after = self.markupsPositions(node)
            if before == after:
                logger.info(f"{os.path.basename(path)} unchanged, not rewritten")
                continue

            moved = sum(1 for a, b in zip(before, after) if a != b)
            try:
                if slicer.util.saveNode(node, path):
                    logger.info(f"{moved} landmark(s) adjusted, saved to {path}")
                else:
                    logger.error(f"Could not save adjusted landmarks to {path}")
            except Exception as e:
                logger.error(f"Could not save adjusted landmarks to {path}: {e}")

    def _saveAdjustment(self):
        """Fold the user's displacement into the matrix it corrects.

        What runs after a registration applies one matrix per patient and
        cannot chain two, so composing here keeps that contract: the file it
        already reads carries the registration and the correction together, and
        nothing downstream has to know an adjustment happened.
        """
        transform = self.transform
        item = self.transform_item
        if transform is None or not item or not item.get("matrix"):
            return

        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(matrix)
        if self.isIdentityMatrix(matrix):
            logger.info(f"{item['patient']}: registration left as it was produced")
            return

        import numpy as np
        import SimpleITK as sitk

        path = item["matrix"]
        try:
            original = sitk.ReadTransform(path)
        except Exception as e:
            logger.error(f"Could not read {os.path.basename(path)}: {e}")
            return

        ras = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
        # Slicer holds the displacement in RAS; a .tfm is LPS, and the two
        # differ by a flip of the first two axes.
        flip = np.diag([-1.0, -1.0, 1.0, 1.0])
        lps = flip @ ras @ flip

        nudge = sitk.AffineTransform(3)
        nudge.SetMatrix(lps[:3, :3].flatten().tolist())
        nudge.SetTranslation(lps[:3, 3].tolist())

        # A transform maps the output point back to the input, so the
        # correction goes in inverted for the pair to read as "register, then
        # nudge" when it is applied.
        composed = sitk.CompositeTransform([original, nudge.GetInverse()])

        try:
            sitk.WriteTransform(composed, path)
            logger.info(f"{item['patient']}: adjustment folded into {os.path.basename(path)}")
        except Exception as e:
            logger.error(f"Could not write the adjusted matrix to {path}: {e}")
            return

        self._saveAdjustedScan(item, transform)

    def _saveAdjustedScan(self, item, transform):
        """Write the moved scan back, so the file matches its matrix."""
        for node in self.nodes:
            if node.GetTransformNodeID() != transform.GetID():
                continue
            storage = node.GetStorageNode()
            path = storage.GetFileName() if storage else None
            if not path:
                continue
            try:
                node.HardenTransform()
                if slicer.util.saveNode(node, path):
                    logger.info(
                        f"{item['patient']}: adjusted scan saved to {os.path.basename(path)}"
                    )
                else:
                    logger.error(f"Could not save the adjusted scan to {path}")
            except Exception as e:
                logger.error(f"Could not save the adjusted scan to {path}: {e}")
            return

    # ------------------------------------------------------------------ utils

    @staticmethod
    def markupsPositions(node) -> list:
        """Control point positions of a markups node, in order."""
        positions = []
        for i in range(node.GetNumberOfControlPoints()):
            position = [0.0, 0.0, 0.0]
            node.GetNthControlPointPosition(i, position)
            positions.append(tuple(position))
        return positions

    @staticmethod
    def isIdentityMatrix(matrix, tolerance: float = 1e-9) -> bool:
        """Whether a 4x4 holds no displacement at all."""
        for row in range(4):
            for col in range(4):
                expected = 1.0 if row == col else 0.0
                if abs(matrix.GetElement(row, col) - expected) > tolerance:
                    return False
        return True


# ---------------------------------------------------------------------------
# What each mode can offer to review.
#
# Ids are what the widget stores and what a step carries, so they outlive any
# change of wording. Several modes run the same kind of step - a mode lists the
# ids it offers and the wording stays identical everywhere.
# ---------------------------------------------------------------------------

CBCT = "CBCT"
IOS = "IOS"
REG = "Registration"

LOOK = "Nothing to edit here - look at the result, then continue."

CATALOGUE = {
    "cbct_resampled": {
        "label": "Resampled CBCT",
        "group": CBCT,
        "kind": VIEW,
        "hint": "Check the CBCT came through resampling intact. " + LOOK,
    },
    "cbct_landmarks_orientation": {
        "label": "CBCT landmarks - orientation",
        "group": CBCT,
        "kind": LANDMARKS,
        "hint": "These points decide how the CBCT is oriented. Drag any that sits "
                "off its anatomy. Your changes are saved when you continue - you "
                "do not need to save in Slicer.",
    },
    "cbct_oriented": {
        "label": "Oriented CBCT",
        "group": CBCT,
        "kind": VIEW,
        "hint": "Check the scan is oriented on the reference planes. " + LOOK,
    },
    "cbct_masks": {
        "label": "Registration masks",
        "group": CBCT,
        "kind": VIEW,
        "hint": "These masks decide what the registration matches on. Check they "
                "follow the bone. " + LOOK,
    },
    "cbct_centered_t2": {
        "label": "Centered T2",
        "group": CBCT,
        "kind": VIEW,
        "hint": "Check the T2 scan is centered before it is registered. " + LOOK,
    },
    "cbct_landmarks_registration": {
        "label": "CBCT landmarks - registration",
        "group": CBCT,
        "kind": LANDMARKS,
        "hint": "These points are what the IOS is registered onto. Drag any that "
                "sits off its tooth. Your changes are saved when you continue - "
                "you do not need to save in Slicer.",
    },
    "cbct_segmentation": {
        "label": "Final CBCT segmentations",
        "group": CBCT,
        "kind": VIEW,
        "hint": "Check the segmentations produced at the end of the run. " + LOOK,
    },
    "ios_segmented": {
        "label": "Segmented teeth",
        "group": IOS,
        "kind": VIEW,
        "hint": "Check every tooth carries its label. " + LOOK,
    },
    "ios_oriented": {
        "label": "Oriented IOS",
        "group": IOS,
        "kind": VIEW,
        "hint": "Check the scan is oriented before registration. " + LOOK,
    },
    "ios_oriented_t1": {
        "label": "Oriented IOS - T1",
        "group": IOS,
        "kind": VIEW,
        "hint": "Check the T1 scan is oriented before registration. " + LOOK,
    },
    "ios_oriented_t2": {
        "label": "Oriented IOS - T2",
        "group": IOS,
        "kind": VIEW,
        "hint": "Check the T2 scan is oriented before registration. " + LOOK,
    },
    "ios_landmarks": {
        "label": "IOS landmarks",
        "group": IOS,
        "kind": LANDMARKS,
        "hint": "These points are what the IOS is registered by. Drag any that "
                "sits off its tooth. Your changes are saved when you continue - "
                "you do not need to save in Slicer.",
    },
    "ios_landmarks_t1": {
        "label": "IOS landmarks - T1",
        "group": IOS,
        "kind": LANDMARKS,
        "hint": "Drag any point that sits off its tooth on the T1 scan. Your "
                "changes are saved when you continue.",
    },
    "ios_landmarks_t2": {
        "label": "IOS landmarks - T2",
        "group": IOS,
        "kind": LANDMARKS,
        "hint": "Drag any point that sits off its tooth on the T2 scan. Your "
                "changes are saved when you continue.",
    },
    "cbct_registration": {
        "label": "Registered CBCT",
        "group": REG,
        "kind": REGISTRATION,
        "hint": "The registered scan is shown over the one it targets. Drag it "
                "into place if it is off; the correction is folded into its "
                "matrix when you continue. Leave it alone to keep the result "
                "as computed.",
    },
    "ios_registration": {
        "label": "Registered IOS",
        "group": REG,
        "kind": REGISTRATION,
        "hint": "The registered scan is shown over the one it targets. Drag it "
                "into place if it is off; the correction is folded into its "
                "matrix when you continue. Leave it alone to keep the result "
                "as computed.",
    },
    "ioscbct_registration": {
        "label": "Registered IOS on CBCT",
        "group": REG,
        "kind": VIEW,
        "hint": "Check the IOS sits correctly in the CBCT. This step writes no "
                "matrix, so the result cannot be moved here. " + LOOK,
    },
}


def describe(review_id: str) -> dict:
    """Catalogue entry of a pause, or an empty dict if the id is unknown."""
    return CATALOGUE.get(review_id, {})


def stepsFor(ids) -> list:
    """Catalogue entries for the ids a mode offers, in the order given."""
    steps = []
    for review_id in ids:
        entry = CATALOGUE.get(review_id)
        if entry is None:
            logger.warning(f"Unknown review step '{review_id}', ignored")
            continue
        steps.append(dict(entry, id=review_id))
    return steps
