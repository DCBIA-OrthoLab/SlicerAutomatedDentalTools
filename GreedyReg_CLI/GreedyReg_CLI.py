#!/usr/bin/env python-real

import os
import re
import sys
import shutil
import tempfile
import argparse
import subprocess

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("GreedyReg_CLI requires the 'nibabel' Python package, which is not "
          "installed in Slicer's Python environment. Open the GreedyReg module "
          "and re-run registration (it will offer to install it), or install it "
          "manually with: PythonSlicer -m pip install nibabel",
          file=sys.stderr, flush=True)
    sys.exit(1)

# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.progress_protocol import emit_fraction

logger = get_logger("GreedyReg_CLI")

# Matches a leading letter(s)+number(s) patient ID, ignoring everything after
ID_PATTERN = re.compile(r'^([A-Za-z]+\d+)', re.IGNORECASE)


def findNiftiFiles(folder):
    ids = {}
    if not folder or not os.path.isdir(folder):
        return ids
    for fname in os.listdir(folder):
        if fname.endswith('.nii.gz') or fname.endswith('.nii'):
            m = ID_PATTERN.match(fname)
            if m:
                ids[m.group(1).upper()] = os.path.join(folder, fname)
    return ids


def findMatFiles(folder):
    ids = {}
    if not folder or not os.path.isdir(folder):
        return ids
    for fname in os.listdir(folder):
        if fname.endswith('.mat'):
            m = ID_PATTERN.match(fname)
            if m:
                ids[m.group(1).upper()] = os.path.join(folder, fname)
    return ids


def findPairs(t1_folder, t2_folder, maskFolder, initFolder):
    t1s = findNiftiFiles(t1_folder)
    t2s = findNiftiFiles(t2_folder)
    masks = findNiftiFiles(maskFolder) if maskFolder else {}
    inits = findMatFiles(initFolder) if initFolder else {}

    pairs = []
    for patient_id in sorted(set(t1s.keys()) & set(t2s.keys())):
        pairs.append((
            patient_id,
            t1s[patient_id],
            t2s[patient_id],
            masks.get(patient_id),
            inits.get(patient_id),
        ))
    return pairs


def writeIdentityInit(initPath):
    """Write a Greedy-format .mat init file holding identity, nudging the
    zero translation slightly so Greedy doesn't treat it as identity."""
    matrix = np.eye(4)
    matrix[0, 3] = 0.001
    with open(initPath, 'w') as f:
        for row in matrix:
            f.write(' '.join(str(v) for v in row) + '\n')


def binarizeMaskFile(src_path, dest_path):
    mask_img = nib.load(src_path)
    mask_data = (mask_img.get_fdata() > 0).astype(np.float32)
    new_mask = nib.Nifti1Image(mask_data, mask_img.affine)
    new_mask.header.set_data_dtype(np.float32)
    nib.save(new_mask, dest_path)


def buildRegistrationCommand(greedy_binary, fixedPath, movingPath, warpPath, initPath,
                              metric, transform_type, maskPath=None):
    dof = "6" if transform_type == "Rigid" else "12"
    if metric == "NMI":
        metric_args = ["-m", "NMI"]
    elif metric == "NCC":
        metric_args = ["-m", "NCC", "4x4x4"]
    else:
        metric_args = ["-m", "SSD"]
    cmd = [greedy_binary]
    cmd.extend(["-d", "3", "-a"])
    cmd.extend(metric_args)
    cmd.extend(["-i", fixedPath, movingPath])
    cmd.extend(["-o", warpPath])
    cmd.extend(["-n", "100x100x50x25"])
    cmd.extend(["-e", "0.5"])
    cmd.extend(["-search", "100", "10", "20"])
    cmd.extend(["-dof", dof])
    cmd.extend(["-ia", initPath])
    if maskPath:
        cmd += ["-gm", maskPath]
    return cmd


def greedyMatToTfm(mat_path, tfm_path):
    """Convert a Greedy affine (.mat: 4x4 RAS matrix mapping fixed-space points
    to moving-space points) into an ITK transform file (.tfm) that Slicer loads
    directly. ITK stores the same fixed->moving direction but in LPS, so only
    the X and Y axes flip. Loaded in Slicer and applied to the moving (T2)
    volume, it moves T2 onto T1."""
    ras_matrix = np.loadtxt(mat_path)
    flip_xy = np.diag([-1.0, -1.0, 1.0, 1.0])
    lps_matrix = flip_xy @ ras_matrix @ flip_xy
    parameters = list(lps_matrix[:3, :3].ravel()) + list(lps_matrix[:3, 3])
    with open(tfm_path, 'w') as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("#Transform 0\n")
        f.write("Transform: AffineTransform_double_3_3\n")
        f.write("Parameters: " + " ".join(f"{v:.17g}" for v in parameters) + "\n")
        f.write("FixedParameters: 0 0 0\n")


def runGreedyCase(greedy_binary, fixedPath, movingPath, outputPath, warpPath, initPath,
                   metric, transform_type, maskPath, timeout=600):
    cmd = buildRegistrationCommand(
        greedy_binary, fixedPath, movingPath, warpPath, initPath, metric, transform_type, maskPath)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Greedy affine registration failed")

    resample_cmd = [greedy_binary, "-d", "3",
                   "-rf", fixedPath,
                   "-rm", movingPath, outputPath,
                   "-r", warpPath]
    result2 = subprocess.run(resample_cmd, capture_output=True, text=True, timeout=timeout)
    if result2.returncode != 0:
        raise RuntimeError(result2.stderr.strip() or "Greedy resampling failed")


def main(args):
    if not os.path.isfile(args.greedyBinary):
        logger.error(f"Greedy binary not found: {args.greedyBinary}")
        sys.exit(1)

    os.makedirs(args.outputFolder, exist_ok=True)

    pairs = findPairs(args.t1Folder, args.t2Folder, args.maskFolder, args.initFolder)
    if not pairs:
        logger.error(f"No matching T1/T2 pairs found between {args.t1Folder} and {args.t2Folder}")
        sys.exit(1)

    total = len(pairs)
    logger.info(f"Found {total} pair(s): {', '.join(p[0] for p in pairs)}")

    for i, (patient_id, fixed_path, moving_path, mask_path, init_path) in enumerate(pairs):
        progress = i / total
        emit_fraction(progress)
        print(f"<filter-comment>Registering {patient_id} ({i + 1}/{total})...</filter-comment>", flush=True)
        logger.info(f"Processing {patient_id} ({i + 1}/{total})")

        case_tmp_dir = tempfile.mkdtemp(prefix=f"greedyreg_{patient_id}_")
        try:
            output_path = os.path.join(args.outputFolder, f"{patient_id}_registered.nii.gz")
            tfm_path = os.path.join(args.outputFolder, f"{patient_id}_transform.tfm")
            # Greedy's own .mat is only an intermediate; the .tfm is the output
            warp_path = os.path.join(case_tmp_dir, "warp.mat")

            resolved_init_path = init_path
            if not resolved_init_path:
                resolved_init_path = os.path.join(case_tmp_dir, "init.mat")
                writeIdentityInit(resolved_init_path)

            resolved_mask_path = None
            if mask_path:
                resolved_mask_path = os.path.join(case_tmp_dir, "mask.nii.gz")
                binarizeMaskFile(mask_path, resolved_mask_path)

            runGreedyCase(
                args.greedyBinary, fixed_path, moving_path, output_path, warp_path,
                resolved_init_path, args.metric, args.transformType, resolved_mask_path)
            greedyMatToTfm(warp_path, tfm_path)

            logger.info(f"{patient_id} done -> {output_path}, {tfm_path}")
        except Exception as e:
            logger.error(f"FAILED on {patient_id}: {e}")
            sys.exit(1)
        finally:
            shutil.rmtree(case_tmp_dir, ignore_errors=True)

    emit_fraction(1.00)
    print(f"<filter-comment>Batch complete! {total} case(s) registered.</filter-comment>", flush=True)
    logger.info(f"Batch complete! {total} case(s) registered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('t1Folder', type=str)
    parser.add_argument('t2Folder', type=str)
    parser.add_argument('--maskFolder', type=str, default="")
    parser.add_argument('--initFolder', type=str, default="")
    parser.add_argument('outputFolder', type=str)
    parser.add_argument('greedyBinary', type=str)
    parser.add_argument('metric', type=str)
    parser.add_argument('transformType', type=str)

    main(parser.parse_args())
