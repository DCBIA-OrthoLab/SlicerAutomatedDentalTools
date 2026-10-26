#!/usr/bin/env python-real
"""Step 4: Dataset516 cranial base on every prepared MRI joint."""
import argparse
import os
import sys

here = os.path.dirname(os.path.realpath(__file__))
# Installed: cli-modules/ next to qt-scripted-modules/. Source tree:
# MRI2CBCT_CLI/<name>/ two levels below MRI2CBCT/.
for _p in (os.path.join(here, "..", "qt-scripted-modules"),
           os.path.join(here, "..", "..", "MRI2CBCT"), here):
    if os.path.isdir(os.path.join(_p, "MRI2CBCT_core")):
        sys.path.insert(0, os.path.abspath(_p))

from MRI2CBCT_core import layout, seg516


def main(a):
    out = a.output_folder
    m = layout.read_manifest(out)
    layout.step_dir(out, layout.MRI_SEG)
    ins, outs = [], []
    for pid, p in m["patients"].items():
        for j in p["joints"]:
            ins.append(layout.joint_mri(out, pid, j["side"]))
            outs.append(layout.joint_seg(out, pid, j["side"]))
    folds = tuple(int(f) for f in a.folds.split(",") if f.strip())
    layout.say(f"{len(ins)} joint(s), model {a.model_folder}")
    layout.progress(0.05)
    seg516.segment(ins, outs, a.model_folder, folds=folds, log=layout.say)
    missing = [o for o in outs if not os.path.exists(o)]
    if missing:
        layout.say("Segmentation missing for: " + ", ".join(os.path.basename(x) for x in missing))
        sys.exit(1)
    layout.progress(1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output_folder")
    ap.add_argument("model_folder")
    ap.add_argument("folds")
    main(ap.parse_args())
