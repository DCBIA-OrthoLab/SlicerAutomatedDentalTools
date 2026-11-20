"""The crop of a whole batch, on what the legacy chain already produced.

SADT-VISOR's TMJ_Crop (tools/TMJ_Crop/src/sadt_tmj_crop/dispatch.py, commit
11f3cb8) calls ASO and AMASSS itself through a supervisor. In this extension the
module's interface runs the steps one after the other (PRE_ASO_CBCT, ALI_CBCT,
SEMI_ASO_CBCT, AMASSS_CLI), as ASO and the other chain modules do, and this
function takes over from their folders. What follows `crop_scans` there follows
here unchanged: the per-scan reports, the quick view, the extra outputs, the run
report and the case statuses. What differs is where each input comes from:

- the reference landmarks are read from the reference folder, because the legacy
  ASO publishes no report;
- the landmarks ASO kept are not published either, so they are worked out here
  the way `ICP()` of ASO_CBCT_utils works them out, with its own functions;
- a scan that was not oriented is found by comparing the oriented folder with
  the input folder, and its reason is not known (ASO_CBCT writes none).

Messages carry the position in the batch ("scan 3 of 40"), never a file name:
the reports are shared, and a name is patient metadata.
"""

import json
import logging
import os
import re
import shutil

from . import catalogs, crop

logger = logging.getLogger(__name__)

SCAN_EXTENSIONS = (".nii.gz", ".nii", ".nrrd", ".nrrd.gz", ".gipl.gz", ".gipl")
_PATH_LIKE = re.compile(r"\S*[/\\]\S*|\S+\.(?:nii|nrrd|gz|json|vtk|stl|gipl)\S*", re.IGNORECASE)

# The run report every tool of the SADT-VISOR catalogue writes beside its results.
REPORT_NAME = "run_report.json"


class ToolInputError(ValueError):
    """An input the tool cannot work with, phrased for whoever sent it."""


def safe_reason(reason, limit: int = 120) -> str:
    """A reason from another step, with anything that looks like a path replaced."""
    text = str(reason or "").strip()
    found = _PATH_LIKE.search(text)
    if found:
        text = text[: found.start()] + "<file>"
    return " ".join(text.split())[:limit]


def strip_extension(name: str) -> str:
    for extension in SCAN_EXTENSIONS:
        if name.lower().endswith(extension):
            return name[: -len(extension)]
    return name


def _discard(output_dir, name: str) -> None:
    """Remove what a case wrote before it failed, so no file is left that the report disowns."""
    for file_name in (f"{name}_R.nii.gz", f"{name}_L.nii.gz", f"{name}_qc.png",
                      f"{name}_oriented.nii.gz", f"{name}_mandible.nii.gz"):
        (output_dir / file_name).unlink(missing_ok=True)


def _write_failure(output_dir, name: str, reason: str) -> None:
    (output_dir / f"{name}.json").write_text(
        json.dumps({"scan": name, "status": reason}, indent=1)
    )


def find_oriented_scans(oriented_dir, suffix):
    """{stem: oriented scan} of what SEMI_ASO_CBCT wrote under `oriented_dir`."""
    end = f"_{suffix}.nii.gz"
    found = {}
    for folder, _, file_names in os.walk(oriented_dir):
        for file_name in sorted(file_names):
            if file_name.endswith(end) and "_lm_" not in file_name:
                found.setdefault(file_name[: -len(end)], os.path.join(folder, file_name))
    return found


def input_stems(input_dir):
    """The patient id ASO would give each scan under `input_dir`, in the order found."""
    from ADTLib.naming import patient_id as read_patient_id, ASO_CBCT_CLI_MARKERS

    stems = []
    for folder, _, file_names in os.walk(input_dir):
        for file_name in sorted(file_names):
            if file_name.lower().endswith(SCAN_EXTENSIONS):
                stems.append(read_patient_id(file_name, ASO_CBCT_CLI_MARKERS))
    return stems


def kept_landmarks(pre_json, gold_json, requested):
    """(used, dropped) for one scan, as ASO's registration worked them out.

    The same decisions as `ICP()` in ASO_CBCT_utils, taken with its own
    `GetLandmarkToRemove`: a requested landmark absent from the scan or from the
    reference is out, and so is one rejected as an outlier. `used` is sorted,
    as the registration sorts what it keeps.
    """
    from ASO_CBCT_utils.utils import GetLandmarkToRemove, LoadJsonLandmarks

    scan = LoadJsonLandmarks(pre_json)
    reference = LoadJsonLandmarks(gold_json)
    outliers = GetLandmarkToRemove(pre_json, gold_json)
    used, dropped = [], {}
    for name in requested:
        if name not in scan:
            dropped[name] = "not in the input landmark file"
        elif name not in reference:
            dropped[name] = "not in the reference landmark file"
        elif name in outliers:
            dropped[name] = "rejected as an outlier"
        else:
            used.append(name)
    return sorted(used), dropped


def crop_batch(oriented_dir, landmarks_dir, mask_dir, reference_dir, output_dir,
               input_dir=None, sides="Left and Right", outputs=(), scale_by="none",
               scale=0.0, landmarks=catalogs.ORIENTATION_LANDMARKS,
               suffix=catalogs.ORIENTED_SUFFIX, prediction_id=catalogs.PREDICTION_ID,
               on_scan_done=None, tell=None) -> str:
    """Crop the requested joints of every oriented CBCT; returns `output_dir`.

    `oriented_dir` holds SEMI_ASO_CBCT's output, `landmarks_dir` the landmarks
    that went into it (ALI_CBCT's, merged by SEMI_ASO_CBCT), `mask_dir`
    AMASSS_CLI's output and `reference_dir` the reference SEMI_ASO_CBCT
    registered onto. `input_dir`, when given, is the folder the scans came from:
    it is how a scan that was never oriented is told apart from one that was not
    asked for. `on_scan_done(finished, total)` is called after each scan, `tell(message)`
    for each message meant for the clinician.
    """
    from pathlib import Path

    from ASO_CBCT_utils.utils import ExtractFilesFromFolder, GetPatients

    if sides not in catalogs.SIDES:
        raise ToolInputError(f"sides must be one of {', '.join(catalogs.SIDES)}.")
    outputs = tuple(outputs)
    unknown = sorted(set(outputs) - set(catalogs.OUTPUT_HELP))
    if unknown:
        raise ToolInputError(f"Unknown extra output(s): {', '.join(unknown)}.")
    try:
        settings = crop.CropSettings(scale_by=scale_by, scale=scale)
    except ValueError as error:
        raise ToolInputError(str(error)) from None

    tell = tell or (lambda message: None)
    on_scan_done = on_scan_done or (lambda finished, total: None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, gold_files = ExtractFilesFromFolder(reference_dir, [".nii.gz", ".nii", ".nrrd", ".gipl"], [".json"])
    if not gold_files:
        raise ToolInputError("The reference folder holds no landmark file.")
    gold_json = gold_files[0]
    reference = crop.read_landmarks(gold_json)
    for needed in crop.PORION.values():
        if needed not in reference:
            raise ToolInputError(
                f"The reference carries no {needed} landmark, so the crop cannot be placed. "
                "Use the Frankfurt horizontal reference."
            )

    scans = find_oriented_scans(oriented_dir, suffix)
    stems = sorted(scans)
    if len({stem.casefold() for stem in stems}) != len(stems):
        raise ToolInputError("Two scans share a name, so their results would overwrite each other.")
    pre_landmarks = GetPatients(landmarks_dir)

    entries = {}
    finished = 0
    expected = input_stems(input_dir) if input_dir else stems
    total = len(set(expected) | set(stems))

    def finish():
        nonlocal finished
        finished += 1
        on_scan_done(finished, total)

    for stem in sorted(set(expected) - set(stems)):
        reason = "no oriented scan was written (see the log of the orientation step)"
        _write_failure(output_dir, stem, f"orientation failed: {reason}")
        entries[stem] = {"status": f"orientation failed: {reason}", "produced": [f"{stem}.json"]}
        logger.warning("a scan was not oriented")
        tell("a scan was not oriented")
        finish()
    logger.info("TMJ_Crop: %d of %d scan(s) oriented", len(stems), total)
    if not stems:
        raise ToolInputError(
            f"The orientation produced none of the {total} scan(s), so nothing was cropped."
        )

    done = partial = 0
    for index, name in enumerate(stems, start=1):
        scan = scans[name]
        landmarks_file = os.path.join(os.path.dirname(scan), f"{name}_lm_{suffix}.mrk.json")
        pre_json = pre_landmarks.get(name, {}).get("json")
        mask = None
        for folder, _, file_names in os.walk(mask_dir):
            if f"{name}_{suffix}_{prediction_id}_MAND.nii.gz" in file_names:
                mask = os.path.join(folder, f"{name}_{suffix}_{prediction_id}_MAND.nii.gz")
                break
        if not os.path.isfile(landmarks_file) or pre_json is None:
            reason = "the landmarks of the scan are missing"
            _write_failure(output_dir, name, f"orientation failed: {reason}")
            entries[name] = {"status": f"orientation failed: {reason}", "produced": [f"{name}.json"]}
            tell(f"scan {index} of {len(stems)}: orientation failed ({reason})")
            finish()
            continue
        if mask is None:
            logger.warning("scan %d of %d: AMASSS wrote no mandible mask", index, len(stems))
            _write_failure(output_dir, name, "no mandible mask")
            entries[name] = {"status": "no mandible mask", "produced": [f"{name}.json"]}
            tell(f"scan {index} of {len(stems)}: no mandible mask")
            finish()
            continue
        try:
            used, dropped = kept_landmarks(pre_json, gold_json, landmarks)
            result = crop.crop_scan(
                scan, landmarks_file, mask, reference,
                used=used, dropped=dropped,
                settings=settings, name=name, output_dir=str(output_dir),
                sides=catalogs.SIDES[sides],
            )
            crop.write_quality_image(result, reference, name, str(output_dir))
        except ValueError as error:
            reason = safe_reason(error)
            logger.warning("scan %d of %d: crop refused: %s", index, len(stems), reason)
            _discard(output_dir, name)
            _write_failure(output_dir, name, f"crop refused: {reason}")
            entries[name] = {"status": f"crop refused: {reason}", "produced": [f"{name}.json"]}
            tell(f"scan {index} of {len(stems)}: crop refused ({reason})")
            finish()
            continue
        except Exception as error:  # noqa: BLE001 - one damaged scan must not cost the batch
            # Only the class: the message of an I/O or library error can quote a path.
            logger.warning("scan %d of %d: crop failed (%s)", index, len(stems), type(error).__name__)
            _discard(output_dir, name)
            _write_failure(output_dir, name, f"crop failed: {type(error).__name__}")
            entries[name] = {"status": f"crop failed: {type(error).__name__}", "produced": [f"{name}.json"]}
            tell(f"scan {index} of {len(stems)}: crop failed ({type(error).__name__})")
            finish()
            continue
        produced = [f"{name}_{side}.nii.gz" for side in ("R", "L") if side in result.crops]
        produced += [f"{name}.json", f"{name}_qc.png"]
        try:
            if catalogs.OUTPUT_ORIENTED_SCAN in outputs:
                shutil.copyfile(scan, output_dir / f"{name}_oriented.nii.gz")
                produced.append(f"{name}_oriented.nii.gz")
            if catalogs.OUTPUT_MANDIBLE in outputs:
                shutil.copyfile(mask, output_dir / f"{name}_mandible.nii.gz")
                produced.append(f"{name}_mandible.nii.gz")
        except OSError as error:
            logger.warning("scan %d of %d: extra output failed (%s)", index, len(stems), type(error).__name__)
            _discard(output_dir, name)
            _write_failure(output_dir, name, f"crop failed: {type(error).__name__}")
            entries[name] = {"status": f"crop failed: {type(error).__name__}", "produced": [f"{name}.json"]}
            tell(f"scan {index} of {len(stems)}: crop failed ({type(error).__name__})")
            finish()
            continue
        problems = [
            f"{side}: {result.report['sides'][side]['status']}"
            for side in catalogs.SIDES[sides]
            if result.report["sides"].get(side, {}).get("status") != "ok"
        ]
        if not result.crops:
            entries[name] = {"status": "no requested joint found", "produced": produced}
            tell(f"scan {index} of {len(stems)}: no requested joint found")
        elif problems:
            done += 1
            partial += 1
            entries[name] = {"status": "partial: " + "; ".join(problems), "produced": produced}
            tell(f"scan {index} of {len(stems)}: partial ({'; '.join(problems)})")
        else:
            done += 1
            entries[name] = {"status": "ok", "produced": produced}
        finish()

    logger.info("TMJ_Crop: %d of %d scan(s) cropped", done, total)
    if done == 0:
        raise ToolInputError(
            f"None of the {len(stems)} oriented scan(s) could be cropped. "
            "The per-scan reports in the output folder say why."
        )
    (output_dir / REPORT_NAME).write_text(json.dumps({
        "tool": "TMJ_Crop",
        "sides": sides,
        "outputs": sorted(outputs),
        "summary": {"cases": total, "cropped": done, "partial": partial,
                    "failed": total - done},
        "cases": entries,
    }, indent=1))
    return str(output_dir)
