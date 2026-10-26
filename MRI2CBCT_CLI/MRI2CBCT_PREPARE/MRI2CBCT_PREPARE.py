#!/usr/bin/env python-real
"""Step 1: pair the two folders, prepare every MRI joint, write the manifest."""
import argparse
import os
import shutil
import sys

here = os.path.dirname(os.path.realpath(__file__))
# Installed: cli-modules/ next to qt-scripted-modules/. Source tree:
# MRI2CBCT_CLI/<name>/ two levels below MRI2CBCT/.
for _p in (os.path.join(here, "..", "qt-scripted-modules"),
           os.path.join(here, "..", "..", "MRI2CBCT"), here):
    if os.path.isdir(os.path.join(_p, "MRI2CBCT_core")):
        sys.path.insert(0, os.path.abspath(_p))

import SimpleITK as sitk
from MRI2CBCT_core import layout, pairing, mri_prep


def main(a):
    out = a.output_folder
    steps = os.path.join(out, layout.STEPS)
    if os.path.isdir(steps):                      # this module's own scratch, rebuilt each run
        shutil.rmtree(steps)
    patients, problems = pairing.pair_folders(a.mri_folder, a.cbct_folder)
    for p in problems:
        layout.say("SKIPPED:", p)
    if not patients:
        layout.say("No patient has both an MRI and a CBCT. Nothing to do.")
        sys.exit(1)

    resample = layout.as_bool(a.resample)
    separate = layout.as_bool(a.separate)
    acq = float(a.acquisition_spacing)
    cbct_in = layout.step_dir(out, layout.CBCT_IN)
    layout.step_dir(out, layout.MRI)

    manifest = {"options": {"resample": resample, "separate": separate,
                            "acquisition_spacing": acq},
                "skipped": problems, "patients": {}}
    for i, (pid, files) in enumerate(patients.items()):
        layout.say(f"\n{pid}")
        ext = ".nii.gz" if files["cbct"].lower().endswith(".nii.gz") else os.path.splitext(files["cbct"])[1]
        link = os.path.join(cbct_in, pid + ext)
        try:
            os.symlink(os.path.abspath(files["cbct"]), link)
        except OSError:
            shutil.copyfile(files["cbct"], link)
        joints = []
        taken = set()
        for mri_path in files["mri"]:
            image = sitk.ReadImage(mri_path)
            parts, note = mri_prep.plan_joints(image, separate, acq)
            layout.say(f"  {os.path.basename(mri_path)}: {note}")
            for side, img in parts.items():
                if side in taken:
                    layout.say(f"  SKIPPED: a second {side} joint for {pid} ({os.path.basename(mri_path)})")
                    continue
                taken.add(side)
                if resample:
                    img = mri_prep.resample_iso(img)
                path = layout.joint_mri(out, pid, side)
                sitk.WriteImage(img, path, True)
                joints.append({"side": side, "source": mri_path, "note": note,
                               "mri": os.path.basename(path)})
        manifest["patients"][pid] = {"cbct": files["cbct"], "joints": joints}
        layout.progress((i + 1) / len(patients))
    layout.write_manifest(out, manifest)
    n = sum(len(p["joints"]) for p in manifest["patients"].values())
    layout.say(f"\n{len(manifest['patients'])} patient(s), {n} joint(s) prepared")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mri_folder")
    ap.add_argument("cbct_folder")
    ap.add_argument("output_folder")
    ap.add_argument("resample")
    ap.add_argument("separate")
    ap.add_argument("acquisition_spacing")
    main(ap.parse_args())
