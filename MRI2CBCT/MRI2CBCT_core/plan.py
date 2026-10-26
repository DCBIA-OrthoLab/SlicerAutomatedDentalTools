"""The run, step by step: which CLI, with which arguments, in which order.

Defined once so the widget and the command-line test run the same thing. ASO
and AMASSS are called exactly as their own modules call them (fully automatic
ASO: PRE_ASO -> ALI landmarks -> SEMI_ASO onto the Frankfort / midsagittal
reference), except that every temporary folder is a fresh one under the
output, never ALI's shared ~/Documents temp, which reuses stale files by name.
"""
import os
from collections import OrderedDict

from . import layout

ASO_LANDMARKS = ["Ba", "LPo", "N", "RPo", "S", "LOr", "ROr"]

# id, title, what the clinician checks at the pause
REVIEW_STEPS = [
    ("prepare", "MRI joints and sides",
     "Each MRI joint as it will be segmented. Check the side; swap it if it is wrong."),
    ("orient", "CBCT oriented (ASO)",
     "The CBCT turned to Frankfort horizontal and the midsagittal plane."),
    ("cbct_seg", "CBCT cranial base (AMASSS)",
     "The cranial base found on the CBCT, drawn over it."),
    ("mri_seg", "MRI cranial base (Dataset516)",
     "The cranial base found on each MRI joint, drawn over it."),
    ("register", "Registration",
     "Each MRI on its CBCT, with the QC verdict."),
]


def model_paths(downloads):
    """downloads: the Slicer downloads folder (~/Documents/SlicerDownloads)."""
    aso = os.path.join(downloads, "ASO", "ASO_CBCT")
    return {
        "preaso": os.path.join(aso, "Models", "PreASOModels"),
        "gold": os.path.join(aso, "Reference", "Frankfurt Horizontal and Midsagittal Plane"),
        "ali": os.path.join(downloads, "ALI", "ALI_CBCT"),
        "amasss": os.path.join(downloads, "AMASSS", "Models", "Segmentation", "AMASSS_Models"),
    }


def missing_models(paths, dataset516):
    """Human-readable list of what is not on disk."""
    miss = []
    if not os.path.isfile(os.path.join(paths["gold"], "MAMP_0002_T1.mrk.json")):
        miss.append(f"ASO Frankfort reference ({paths['gold']}) - download it once from the ASO module")
    for lm in ASO_LANDMARKS:
        if not os.path.isdir(os.path.join(paths["ali"], lm)):
            miss.append(f"ALI landmark model {lm} ({paths['ali']}) - download it from the ASO module")
    if not os.path.isdir(os.path.join(paths["amasss"], "CB")):
        miss.append(f"AMASSS cranial base model ({paths['amasss']}) - download it from the AMASSS module")
    if not os.path.isfile(os.path.join(dataset516, "fold_0", "checkpoint_final.pth")):
        miss.append(f"Dataset516 model ({dataset516})")
    return miss


def build(mri_folder, cbct_folder, output, resample, separate, acquisition_spacing,
          folds, on_cbct_grid, workers, downloads, dataset516):
    m = model_paths(downloads)
    tmp = lambda name: os.path.join(output, layout.STEPS, layout.TMP, name)
    s = lambda name: os.path.join(output, layout.STEPS, name)

    steps = []

    def add(cli, title, params, review=None):
        steps.append({"cli": cli, "title": title, "review": review,
                      "params": OrderedDict(params)})

    add("MRI2CBCT_PREPARE", "Preparing the MRIs", [
        ("mri_folder", mri_folder), ("cbct_folder", cbct_folder), ("output_folder", output),
        ("resample", str(bool(resample))), ("separate", str(bool(separate))),
        ("acquisition_spacing", f"{float(acquisition_spacing):g}")], review="prepare")
    add("PRE_ASO_CBCT", "Orienting the CBCT: centring", [
        ("input", s(layout.CBCT_IN)), ("output_folder", tmp("aso")),
        ("model_folder", m["preaso"]), ("SmallFOV", "False"),
        ("temp_folder", tmp("preaso")), ("DCMInput", False)])
    add("ALI_CBCT", "Orienting the CBCT: landmarks", [
        ("input", tmp("aso")), ("dir_models", m["ali"]),
        ("lm_type", ", ".join(f"'{x}'" for x in ASO_LANDMARKS)),
        ("output_dir", tmp("aso")), ("temp_fold", tmp("ali")), ("DCMInput", False),
        ("spacing", "[1,0.3]"), ("speed_per_scale", "[1,1]"),
        ("agent_FOV", "[64,64,64]"), ("spawn_radius", "10")])
    add("SEMI_ASO_CBCT", "Orienting the CBCT: Frankfort / midsagittal", [
        ("input", tmp("aso")), ("gold_folder", m["gold"]),
        ("output_folder", s(layout.CBCT_OR)), ("add_inname", "Or"),
        ("list_landmark", " ".join(ASO_LANDMARKS))], review="orient")
    add("AMASSS_CLI", "CBCT cranial base (AMASSS)", [
        ("inputVolume", s(layout.CBCT_OR)), ("modelDirectory", m["amasss"]),
        ("skullStructure", "CB"), ("merge", "MERGE"), ("genVtk", False),
        ("save_in_folder", False), ("output_folder", s(layout.CBCT_SEG)),
        ("vtk_smooth", 5), ("prediction_ID", "Pred"), ("temp_fold", tmp("amasss")),
        ("SegmentInput", "False"), ("DCMInput", "False")], review="cbct_seg")
    add("MRI2CBCT_SEGMENT_MRI", "MRI cranial base (Dataset516)", [
        ("output_folder", output), ("model_folder", dataset516),
        ("folds", ",".join(str(f) for f in folds))], review="mri_seg")
    add("MRI2CBCT_REGISTER", "Registering", [
        ("output_folder", output), ("on_cbct_grid", str(bool(on_cbct_grid))),
        ("workers", str(int(workers)))], review="register")
    return steps
