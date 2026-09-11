"""Which steps of a VFACE run can be paused on, and what the user may do there.

Kept apart from the pipeline itself: the module panel needs this list as soon
as a mode is picked, long before CreateListProcess runs and starts making
folders. Ids are what the panel stores and what a step carries, so they outlive
any change of wording.
"""

import logging
import os
import shutil

logger = logging.getLogger("VFACE")

VIEW = "view"                   # look only
LANDMARKS = "landmarks"         # drag the points, saved back to their file
REGISTRATION = "registration"   # drag the scan, folded into its matrix

PREPARATION = "Preparation"
MIRROR = "Mirroring"
REGISTRATION_GROUP = "Registration"
MEASUREMENT = "Measurements"
VISUALIZATION = "Visualization"

LOOK = "Nothing to edit here - look at the result, then click Continue."

CATALOGUE = {
    "t1_landmarks_orientation_max": {
        "label": "Maxilla orientation landmarks",
        "group": PREPARATION,
        "kind": LANDMARKS,
    },
    "t1_landmarks_orientation_cb": {
        "label": "Cranial base orientation landmarks",
        "group": PREPARATION,
        "kind": LANDMARKS,
    },
    "t1_oriented_max": {
        "label": "Maxilla orientation",
        "group": PREPARATION,
        "kind": VIEW,
    },
    "t1_oriented_cb": {
        "label": "Cranial base orientation",
        "group": PREPARATION,
        "kind": VIEW,
    },
    "t1_masks": {
        "label": "Bone segmentation",
        "group": PREPARATION,
        "kind": VIEW,
    },
    "mirror_masks": {
        "label": "Mirrored segmentation",
        "group": MIRROR,
        "kind": VIEW,
    },
    "mirror_scans": {
        "label": "Mirrored scans",
        "group": MIRROR,
        "kind": VIEW,
    },
    "registration_cb": {
        "label": "Registered on the cranial base",
        "group": REGISTRATION_GROUP,
        "kind": REGISTRATION,
    },
    "registration_max": {
        "label": "Registered on the maxilla",
        "group": REGISTRATION_GROUP,
        "kind": REGISTRATION,
    },
    "registration_mand": {
        "label": "Registered on the mandible",
        "group": REGISTRATION_GROUP,
        "kind": REGISTRATION,
    },
    "t1_landmarks": {
        "label": "T1 landmarks",
        "group": MEASUREMENT,
        "kind": LANDMARKS,
    },
    "bone_surfaces": {
        "label": "Bone surfaces",
        "group": VISUALIZATION,
        "kind": VIEW,
    },
}

# The order the run reaches them, which is the order the panel lists them in.
ORDER = [
    "t1_landmarks_orientation_max",
    "t1_oriented_max",
    "t1_landmarks_orientation_cb",
    "t1_oriented_cb",
    "t1_masks",
    "mirror_masks",
    "mirror_scans",
    "registration_cb",
    "registration_max",
    "registration_mand",
    "t1_landmarks",
    "bone_surfaces",
]


def describe(review_id):
    """Catalogue entry of a pause, or an empty dict if the id is unknown."""
    return CATALOGUE.get(review_id, {})


def availableSteps(mode, mode2, reg_type, visualization, quantification):
    """The pauses a run with these settings will actually reach.

    A mode that skips a step must not offer it: ticking a pause that never
    happens reads as a broken feature the first time the run goes straight past
    it.

    Args:
        mode: "Full pipeline", "File already Oriented" or "File already Registered"
        mode2: "Asymmetry Assesment" or "Longitudinal studies"
        reg_type: "AREG" or "CMFReg"
        visualization: whether the heatmap branch runs
        quantification: whether the measurement branch runs

    Returns:
        list: catalogue entries, each carrying its id, in run order
    """
    full_pipeline = mode == "Full pipeline"
    registered = mode == "File already Registered"
    asymmetry = mode2 == "Asymmetry Assesment"

    available = {
        # These points decide the orientation everything downstream inherits:
        # a bad one throws ASO off, which throws the masks and the registration
        # off in turn. They are only placed when the run does its own
        # orientation.
        "t1_landmarks_orientation_max": full_pipeline,
        "t1_landmarks_orientation_cb": full_pipeline,
        "t1_oriented_max": full_pipeline,
        "t1_oriented_cb": full_pipeline,
        "t1_masks": not registered,
        "mirror_masks": not registered and asymmetry and reg_type == "CMFReg",
        "mirror_scans": not registered and asymmetry,
        "registration_cb": not registered,
        "registration_max": not registered,
        "registration_mand": not registered,
        "t1_landmarks": bool(quantification),
        "bone_surfaces": bool(visualization),
    }

    steps = []
    for review_id in ORDER:
        if available.get(review_id):
            steps.append(dict(CATALOGUE[review_id], id=review_id))
    return steps


# Parameters that name a folder of patient files. A replay narrowed to a few
# patients has to point these somewhere smaller; everything else in a step -
# models, spacings, output paths - is left alone.
INPUT_KEYS = ("input", "input_patient", "input_matrix")


def restrictStepToPatients(step, patients, tempdir_factory=None, id_of=None):
    """A copy of this step that only reads the patients given.

    Args:
        step: the step dictionary to narrow
        patients: patient ids to keep
        tempdir_factory: callable returning a fresh empty directory, so a
            caller can control where the links go
        id_of: callable turning a file name into a patient id

    Returns:
        tuple: (narrowed step, folders created). The step is returned unchanged
            when nothing can be narrowed, which is the safe outcome: replaying
            every patient wastes time, replaying none loses work.
    """
    wanted = set(patients or ())
    if not wanted:
        return step, []

    if tempdir_factory is None:
        import slicer
        tempdir_factory = slicer.util.tempDirectory
    if id_of is None:
        from VFACE_utils.functionaq3dc import patientIdFromFileName as id_of

    narrowed = dict(step)
    parameters = dict(step.get("Parameter") or {})
    created = []
    changed = False

    for key in INPUT_KEYS:
        folder = parameters.get(key)
        if not isinstance(folder, str) or not os.path.isdir(folder):
            continue

        linked = tempdir_factory()
        kept = 0
        # Walked, and the tree kept: VFACE writes per-structure results into
        # sub-folders, and the modules it calls walk those to find their input.
        # A flat listing would leave them with nothing.
        for root, _, names in os.walk(folder):
            for name in sorted(names):
                source = os.path.join(root, name)
                if not os.path.isfile(source):
                    continue
                if id_of(name) not in wanted:
                    continue
                relative = os.path.relpath(source, folder)
                target = os.path.join(linked, relative)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                # slicer.util.tempDirectory() names its folder to the millisecond,
                # so two narrowings close together can be handed the same one. The
                # file is then already linked, and copying onto a link that points
                # at its own source raises SameFileError.
                if os.path.lexists(target):
                    kept += 1
                    continue
                try:
                    os.symlink(source, target)
                    kept += 1
                except OSError as e:
                    # A filesystem without links is no reason to lose the replay.
                    logger.warning(f"Could not link {relative}, copying instead: {e}")
                    shutil.copy(source, target)
                    kept += 1

        # Shared inputs - the mirror matrix is one file for the whole run - match
        # no patient and must be left exactly as they were, or the step loses
        # what it needs to run at all.
        if kept == 0:
            logger.warning(
                f"None of {sorted(wanted)} found in {folder}; leaving '{key}' as it was"
            )
            continue

        parameters[key] = linked
        created.append(linked)
        changed = True
        logger.info(f"'{key}' narrowed to {kept} file(s) for {sorted(wanted)}")

    if not changed:
        return step, []

    narrowed["Parameter"] = parameters
    return narrowed, created
