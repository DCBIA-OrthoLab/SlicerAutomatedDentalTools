"""Where every step reads and writes, and the run manifest.

    <output>/
        _steps/                     working files, replaced at each run
            0_cbct_in/<PID>.nii.gz      links to the clinician's CBCTs
            1_mri/<PID>_<S>_MRI.nii.gz  prepared joints (S = L, R or B)
            2_cbct_oriented/            ASO: <PID>_Or.nii.gz, <PID>_Or_transform.tfm
            3_cbct_seg/                 AMASSS: <PID>_Or_Pred_CB.nii.gz
            4_mri_seg/<PID>_<S>_MRI_seg.nii.gz
            manifest.json
        <PID>/<PID>_<S>_MRI_REG.nii.gz, _MASK.nii.gz, _transform.tfm
                                    the results, in the ORIGINAL CBCT's frame
        summary.csv
    _steps is deleted after a successful run unless the clinician keeps it.
"""
import json
import os
import sys

STEPS = "_steps"
CBCT_IN, MRI, CBCT_OR, CBCT_SEG, MRI_SEG, TMP = (
    "0_cbct_in", "1_mri", "2_cbct_oriented", "3_cbct_seg", "4_mri_seg", "tmp")


def step_dir(output, name, create=True):
    d = os.path.join(output, STEPS, name)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def manifest_path(output):
    return os.path.join(output, STEPS, "manifest.json")


def read_manifest(output):
    with open(manifest_path(output)) as f:
        return json.load(f)


def write_manifest(output, manifest):
    p = manifest_path(output)
    tmp = p + ".part"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, p)


def joint_mri(output, pid, side):
    return os.path.join(output, STEPS, MRI, f"{pid}_{side}_MRI.nii.gz")


def joint_seg(output, pid, side):
    return os.path.join(output, STEPS, MRI_SEG, f"{pid}_{side}_MRI_seg.nii.gz")


def cbct_oriented(output, pid):
    return os.path.join(output, STEPS, CBCT_OR, f"{pid}_Or.nii.gz")


def aso_transform(output, pid):
    return os.path.join(output, STEPS, CBCT_OR, f"{pid}_Or_transform.tfm")


def cbct_seg(output, pid):
    return os.path.join(output, STEPS, CBCT_SEG, f"{pid}_Or_Pred_CB.nii.gz")


def patient_dir(output, pid):
    d = os.path.join(output, pid)
    os.makedirs(d, exist_ok=True)
    return d


def progress(fraction):
    """Slicer's CLI progress protocol."""
    print(f"<filter-progress>{max(0.0, min(1.0, fraction)):.4f}</filter-progress>", flush=True)


def say(*parts):
    print(*parts, flush=True)


def as_bool(text):
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def add_core_to_path(cli_file):
    """CLIs live in cli-modules/, the package in qt-scripted-modules/ next to it;
    in the source tree, in MRI2CBCT_CLI/<name>/, two levels below MRI2CBCT/."""
    here = os.path.dirname(os.path.realpath(cli_file))
    for cand in (os.path.join(here, "..", "qt-scripted-modules"),
                 os.path.join(here, "..", "..", "MRI2CBCT"), here):
        cand = os.path.abspath(cand)
        if os.path.isdir(os.path.join(cand, "MRI2CBCT_core")) and cand not in sys.path:
            sys.path.insert(0, cand)
