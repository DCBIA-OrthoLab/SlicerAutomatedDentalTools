"""Pair the MRI folder with the CBCT folder by patient ID.

The modality comes from the folder the clinician chose, never from the file
name. The patient ID is the leading letters + number of the file name: "P01",
"B038", "A3_scan"... Leading zeros are not significant, so P1 and P01 are the
same patient.
"""
import os
import re

EXTENSIONS = (".nii.gz", ".nii", ".nrrd")
_ID = re.compile(r"^([A-Za-z]+)[_\-\s]*0*(\d+)")


def is_image(name):
    return name.lower().endswith(EXTENSIONS)


def stem(path):
    name = os.path.basename(path)
    for ext in EXTENSIONS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def patient_key(filename):
    """("P", 1) for "P01_MRI.nii.gz"; None when no ID can be read."""
    m = _ID.match(os.path.basename(filename))
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))


def written_id(filename):
    """The ID as the clinician wrote it ("B038", "P1"), separators dropped."""
    m = _ID.match(os.path.basename(filename))
    return re.sub(r"[_\-\s]", "", m.group(0)) if m else None


def list_images(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if is_image(f) and not f.startswith("."):
                out.append(os.path.join(root, f))
    return sorted(out)


def looks_like(path):
    """"CBCT", "MRI" or None, from the header alone (no voxels read).

    A CBCT is isotropic and at least 120 mm across in every direction; an MRI
    slab is anisotropic or thin in one direction. A whole-head isotropic MRI
    reads as neither, which is why this only ever feeds a warning.
    """
    import SimpleITK as sitk
    try:
        r = sitk.ImageFileReader()
        r.SetFileName(path)
        r.ReadImageInformation()
    except Exception:
        return None
    sp = [float(s) for s in r.GetSpacing()]
    ext = [s * n for s, n in zip(sp, r.GetSize())]
    if max(sp) / min(sp) > 2.0 or min(ext) < 90.0:
        return "MRI"
    if max(sp) / min(sp) < 1.05 and min(ext) >= 120.0:
        return "CBCT"
    return None


def folders_swapped(mri_folder, cbct_folder):
    """True when the MRI folder holds only CBCT-looking files and the CBCT folder
    only MRI-looking ones."""
    m = [looks_like(p) for p in list_images(mri_folder)[:4]]
    c = [looks_like(p) for p in list_images(cbct_folder)[:4]]
    return bool(m) and bool(c) and all(x == "CBCT" for x in m) and all(x == "MRI" for x in c)


def pair_folders(mri_folder, cbct_folder):
    """Returns (patients, problems).

    patients: {label: {"cbct": path, "mri": [path, ...]}}, sorted by label.
    problems: human-readable reasons a file or patient was left out.
    """
    problems = []
    cbcts, mris = {}, {}
    for path in list_images(cbct_folder):
        key = patient_key(path)
        if key is None:
            problems.append(f"CBCT {os.path.basename(path)}: no patient ID (expected e.g. P01) - skipped")
            continue
        cbcts.setdefault(key, []).append(path)
    for path in list_images(mri_folder):
        key = patient_key(path)
        if key is None:
            problems.append(f"MRI {os.path.basename(path)}: no patient ID (expected e.g. P01) - skipped")
            continue
        mris.setdefault(key, []).append(path)

    patients = {}
    for key in sorted(set(cbcts) | set(mris)):
        label = written_id((cbcts.get(key) or mris[key])[0])
        if key not in cbcts:
            problems.append(f"{label}: MRI found but no CBCT - skipped")
            continue
        if key not in mris:
            problems.append(f"{label}: CBCT found but no MRI - skipped")
            continue
        if len(cbcts[key]) > 1:
            names = ", ".join(os.path.basename(p) for p in cbcts[key])
            problems.append(f"{label}: {len(cbcts[key])} CBCT files ({names}); one per patient expected - skipped")
            continue
        patients[label] = {"cbct": cbcts[key][0], "mri": mris[key]}
    return patients, problems
