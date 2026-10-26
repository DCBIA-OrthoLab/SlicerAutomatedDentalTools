#!/usr/bin/env python-real
"""Step 5: register every joint and write three files per joint, in the ORIGINAL CBCT frame.

    <output>/<PID>/<PID>_<S>_MRI_REG<TAG>.nii.gz     the MRI, registered onto the CBCT
    <output>/<PID>/<PID>_<S>_MRI_MASK<TAG>.nii.gz    its cranial base (Dataset516), registered
    <output>/<PID>/<PID>_<S>_MRI_transform<TAG>.tfm  MRI -> CBCT
    <output>/<PID>/<PID>_CBCT_MASK.nii.gz            once per patient: the CBCT cranial base (AMASSS)
    <output>/summary.csv                             one row per joint: QC verdict and notes

TAG is "", "_MIRRORED" or "_FLAGGED". By default the volumes keep the MRI's own voxels
and only their header is moved (nothing interpolated); with on_cbct_grid they are
resampled onto the CBCT's voxel grid instead.
"""
import argparse
import csv
import os
import sys

here = os.path.dirname(os.path.realpath(__file__))
# Installed: cli-modules/ next to qt-scripted-modules/. Source tree:
# MRI2CBCT_CLI/<name>/ two levels below MRI2CBCT/.
for _p in (os.path.join(here, "..", "qt-scripted-modules"),
           os.path.join(here, "..", "..", "MRI2CBCT"), here):
    if os.path.isdir(os.path.join(_p, "MRI2CBCT_core")):
        sys.path.insert(0, os.path.abspath(_p))

import numpy as np
from MRI2CBCT_core import layout, frames, ladder

FIELDS = ["patient", "side", "status", "qc", "tag", "fitness", "rotation_deg", "mask_cm3",
          "resolved_by", "ladder", "note"]


def outputs(out, pid, side, tag):
    stem = os.path.join(out, pid, f"{pid}_{side}_MRI")
    return f"{stem}_REG{tag}.nii.gz", f"{stem}_MASK{tag}.nii.gz", f"{stem}_transform{tag}.tfm"


def main(a):
    out = a.output_folder
    m = layout.read_manifest(out)
    on_grid = layout.as_bool(a.on_cbct_grid)
    workers = int(a.workers) if int(a.workers) > 0 else None
    rows = []
    total = sum(len(p["joints"]) for p in m["patients"].values())
    done = 0
    for pid, p in m["patients"].items():
        layout.say(f"\n{pid}")
        try:
            surf = ladder.cbct_surface(layout.cbct_seg(out, pid))
            T_aso = frames.aso_original_to_oriented_ras(layout.aso_transform(out, pid))
        except Exception as e:
            layout.say(f"  FAILED before registration: {e}")
            rows += [dict(patient=pid, side=j["side"], status="failed", note=str(e)) for j in p["joints"]]
            done += len(p["joints"])
            continue
        layout.patient_dir(out, pid)
        # AMASSS ran on the oriented CBCT; bring its cranial base back to the original
        cbct_mask = os.path.join(out, pid, f"{pid}_CBCT_MASK.nii.gz")
        if on_grid:
            frames.resample(layout.cbct_seg(out, pid), p["cbct"], np.linalg.inv(T_aso),
                            cbct_mask, label=True)
        else:
            frames.move_header(layout.cbct_seg(out, pid), np.linalg.inv(T_aso), cbct_mask, label=True)

        for j in p["joints"]:
            s = j["side"]
            layout.say(f"  joint {s}")
            try:
                r = ladder.register_joint(layout.joint_seg(out, pid, s), surf,
                                          s if s in ("L", "R") else None,
                                          workers=workers, log=layout.say)
            except Exception as e:
                r = {"status": "failed", "reason": str(e)}
            done += 1
            if r.get("status") != "ok":
                layout.say(f"  FAILED: {r.get('reason')}")
                rows.append(dict(patient=pid, side=s, status="failed", note=r.get("reason", "")))
                layout.progress(done / total)
                continue

            side = r["side"] or s
            note = j["note"]
            if side != s:
                if any(o["side"] == side for o in p["joints"]):
                    layout.say(f"  registered on side {side}, but {pid} already has a {side} "
                               f"joint: files keep the name {s}")
                    note += f"; registered on side {side}, name kept {s}"
                    side = s
                else:
                    note += f"; side relabelled {s} -> {side}"
            tag = "_FLAGGED" if r["flagged"] else ("_MIRRORED" if r["mirrored"] else "")
            T = np.linalg.inv(T_aso) @ np.asarray(r["T"], float)     # MRI -> original CBCT
            reg, mask, tfm = outputs(out, pid, side, tag)
            mri, seg = layout.joint_mri(out, pid, s), layout.joint_seg(out, pid, s)
            frames.write_tfm(T, tfm)
            if on_grid:
                frames.resample(mri, p["cbct"], T, reg)
                frames.resample(seg, p["cbct"], T, mask, label=True)
            else:
                frames.move_header(mri, T, reg)
                frames.move_header(seg, T, mask, label=True)
            tried = [x["rung"] for x in r["rungs"][1:]]
            rows.append(dict(patient=pid, side=side, status="ok", qc=r["qc"], tag=tag.strip("_"),
                             fitness=r["fitness"], rotation_deg=r["rotation_deg"],
                             mask_cm3=r["mask_cm3"], resolved_by=r["resolved_by"],
                             ladder=" > ".join(tried), note=note))
            layout.say(f"  -> {r['qc']}{(' ' + tag) if tag else ''}, fitness {r['fitness']}, "
                       f"rotation {r['rotation_deg']} deg")
            layout.progress(done / total)

    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    ok = sum(r["status"] == "ok" for r in rows)
    layout.say(f"\n{ok}/{len(rows)} joint(s) registered. Summary: {os.path.join(out, 'summary.csv')}")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output_folder")
    ap.add_argument("on_cbct_grid")
    ap.add_argument("workers")
    main(ap.parse_args())
