#!/usr/bin/env python3
import os
import sys
import time
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

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

# --- LOGGING CONFIGURATION ---
from ADTLib.logging_setup import get_logger
from ADTLib.progress_protocol import emit

logger = get_logger("ALI_CBCT")


# --- DYNAMIC IMPORTS ---
try:
    # Add the script's own directory to sys.path for local imports.
    # realpath, not __file__: a CLI registered through a symlink - the flat
    # dev folder of links into the source tree - leaves __file__ on the link,
    # whose directory holds no ALI_CBCT_utils. Resolving first lands beside it.
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    
    from ALI_CBCT_utils import (
        Agent, GetAgentLst, Brain, DNet, Environment, GenEnvironmentLst,
        GetBrain, CorrectHisto, SetSpacing, convertdicom2nifti,
        MOVEMENTS, DEVICE
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def update_slicer_progress(value):
    """Send a value on the progress channel.

    WARNING -- what this CLI sends arrives nowhere. It passes percentages
    (5, 20, then 20 to 100) while Slicer multiplies by a hundred what it
    reads: the window therefore receives 500, 2000, up to 10000. But
    `DisplayALICBCT.isProgress`, a l'autre bout, ne reagit qu'a 100 et a 200 --
    that is, to the values 1 and 2. **ALI CBCT's progress bar and its landmark
    counter therefore never move.**

    Mesure a l'appui : un CLI qui imprime 0.42 donne `GetProgress() == 42`, 1
    donne 100, 2 donne 200, 20 donne 2000. Voir
    `DEBUG/adt-validation/probe_progress_scale/`.

    Fixing it means deciding what the bar should show -- a fraction of
    progress, or one event per patient as the four other CLIs do
    (`emit_event(PATIENT_DONE)`). That is a decision, not a cleanup, so
    nothing is changed here: the bytes emitted are the ones from before.
    """
    emit(value)
    time.sleep(0.05)

def _report_missing_landmarks(patient_id, missing, out_dir):
    """Make visible what the output file does not say.

    When `Search` returns -1, no `AddPredictedLandmark` is made: the landmark
    is simply ABSENT from the `.mrk.json`, and nothing tells a landmark nobody
    asked for apart from one the search did not find. The only sign was a
    warning line lost in the middle
    of the CLI log.

    Here the list goes into a file placed beside the predictions -- same
    folder, same patient prefix, so you run into it on your way to your
    results -- and a boxed block goes to the CLI output,
    ou Slicer l'affiche.
    """
    if not missing:
        return None

    stem = str(patient_id).split(".")[0]
    report = {
        "patient": str(patient_id),
        "not_found": [{"landmark": lm, "reason": reason}
                      for lm, reason in sorted(missing.items())],
    }

    file_path = None
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            file_path = os.path.join(out_dir, f"{stem}_lm_NotFound.json")
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=4)
        except OSError as e:
            logger.error(f"Could not write the not-found report for "
                         f"{patient_id}: {e}")
            file_path = None

    logger.warning("=" * 70)
    logger.warning(f"{len(missing)} LANDMARK(S) NOT PLACED for {patient_id} "
                   "-- they are absent from the output files:")
    for lm, reason in sorted(missing.items()):
        logger.warning(f"    {lm} : {reason}")
    if file_path:
        logger.warning(f"  listed in {file_path}")
    logger.warning("=" * 70)
    return file_path


def _predict_one_patient(agent_lst, args, brain_weights, env_idx, environment, environment_lst, fails, scale_keys, tot_step, transition_layer_size):
    """Move the agents over one scan until they settle."""
    logger.info(f"Processing patient: {environment.patient_id}")
    missing = {}

    for agent in agent_lst:
        try:
            # Initialize Brain for the specific landmark
            brain = Brain(
                network_type=DNet,
                network_scales=scale_keys,
                device=DEVICE,
                in_channels=transition_layer_size,
                out_channels=len(MOVEMENTS["id"]),
                batch_size=1,
                generate_tensorboard=False,
                verbose=False
            )

            # Load weights
            if agent.target in brain_weights:
                try:
                    brain.LoadModels(brain_weights[agent.target])
                    agent.SetBrain(brain)
                    agent.SetEnvironment(environment)

                    # Execute Deep RL Search
                    search_result = agent.Search()

                    if search_result == -1:
                        fails[agent.target] = fails.get(agent.target, 0) + 1
                        missing[agent.target] = (
                            agent.failure_reason or "the search did not place it")
                        logger.warning(f"Agent failed to find {agent.target}")
                    else:
                        tot_step += search_result
                except Exception as e:
                    logger.error(f"Error loading model weights for {agent.target}: {e}")
                    fails[agent.target] = fails.get(agent.target, 0) + 1
                    missing[agent.target] = f"could not load its model: {e}"
            else:
                # Counted as a failure like the others: without it the
                # end-of-run summary stayed silent about a landmark that was
                # asked for and never even searched.
                logger.error(f"No model found for landmark: {agent.target}")
                fails[agent.target] = fails.get(agent.target, 0) + 1
                missing[agent.target] = "no model for it in the model folder"

        except Exception as e:
            logger.error(f"Error during agent search for {agent.target}: {e}")
            missing[agent.target] = f"the search raised: {e}"
        finally:
            # Cleanup to free GPU memory
            agent.SetBrain(None)
            if 'brain' in locals(): del brain
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save results for this patient
    try:
        unplaceable = environment.SavePredictedLandmarks(scale_keys[-1], args.output_dir)
        for landmark in unplaceable or ():
            missing[landmark] = (
                "found, but no group is declared for this name, so it could not "
                "be written to the output file")
    except Exception as e:
        logger.error(f"Failed to save predictions for patient {environment.patient_id}: {e}")

    _report_missing_landmarks(environment.patient_id, missing, args.output_dir)

    # Update Slicer Progress
    progress = 20 + int((env_idx + 1) / len(environment_lst) * 80)
    update_slicer_progress(progress)
    return tot_step

def _prepare_one_patient(data, p_name, patients, scale_spacing, temp_fold):
    """Correct the histogram and resample one scan at every scale."""
    try:
        scan_path = data["scan"]
        # Correct Histogram
        temp_patient_path = temp_fold / p_name
        if not temp_patient_path.exists():
            logger.info(f"Correcting histogram for {p_name}")
            try:
                CorrectHisto(scan_path, str(temp_patient_path), 0.01, 0.99)
            except Exception as e:
                logger.error(f"Histogram correction failed for {p_name}: {e}")
                return

        # Resample for each scale
        for sp in scale_spacing:
            try:
                spac_key = str(sp).replace(".", "-")
                # Construct new filename: name_scan_sp1-0.nii.gz
                resampled_name = f"{temp_patient_path.stem}_sp{spac_key}{''.join(temp_patient_path.suffixes)}"
                out_resampled = temp_fold / resampled_name

                if not out_resampled.exists():
                    logger.debug(f"Setting spacing {sp} for {p_name}")
                    SetSpacing(str(temp_patient_path), [sp, sp, sp], str(out_resampled))

                patients[p_name]["scans"][spac_key] = str(out_resampled)
            except Exception as e:
                logger.error(f"Spacing resampling failed for {p_name} at scale {sp}: {e}")
                continue
    except Exception as e:
        logger.error(f"Pre-processing failed for patient {p_name}: {e}")
        return

def main(args):
    # 1. PARAMETERS PARSING
    try:
        scale_spacing = ast.literal_eval(args.spacing)
        speed_per_scale = ast.literal_eval(args.speed_per_scale)
        agent_fov = ast.literal_eval(args.agent_fov)
        lm_type = ast.literal_eval(f"[{args.lm_type}]")
        spawn_radius = int(args.spawn_radius)
        logger.info(f"Initialized with spacings: {scale_spacing}")
    except ValueError as e:
        logger.error(f"Error parsing arguments - Invalid value format: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        sys.exit(1)

    # 2. DICOM CONVERSION
    if args.dcm_input.lower() == "true":
        logger.info("Converting DICOM input to NIFTI...")
        try:
            convertdicom2nifti(args.input)
            logger.info("DICOM conversion completed successfully")
        except Exception as e:
            logger.error(f"Failed to convert DICOM files: {e}")
            sys.exit(1)

    # 3. FILE DISCOVERY
    patients = {}
    input_path = Path(args.input)
    temp_fold = Path(args.temp_fold)
    
    try:
        temp_fold.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create temporary folder '{args.temp_fold}': {e}")
        sys.exit(1)

    extensions = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
    
    try:
        if input_path.is_file():
            patients[input_path.name] = {"scan": str(input_path), "scans": {}}
        else:
            for ext in extensions:
                for file in input_path.rglob(f"*{ext}"):
                    if file.name not in patients:
                        patients[file.name] = {"scan": str(file), "scans": {}}
    except Exception as e:
        logger.error(f"Error discovering input files: {e}")
        sys.exit(1)

    if not patients:
        logger.error("No valid medical imaging files found. Use these formats: .nrrd,.nrrd.gz,.nii,.nii.gz,.gipl,.gipl.gz")
        sys.exit(1)

    # 4. PRE-PROCESSING (HISTOGRAM & SPACING)
    update_slicer_progress(5)
    for p_name, data in patients.items():
        _prepare_one_patient(data, p_name, patients, scale_spacing, temp_fold)

    update_slicer_progress(20)

    # 5. ENVIRONMENT & AGENT INIT
    scale_keys = [str(s).replace('.', '-') for s in scale_spacing]
    
    try:
        environment_lst = GenEnvironmentLst(
            patient_dic=patients,
            env_type=Environment,
            padding=np.array(agent_fov) / 2 + 1,
            device=DEVICE,
            scale_keys=scale_keys
        )
    except Exception as e:
        logger.error(f"Failed to generate environments: {e}")
        sys.exit(1)

    try:
        agent_params = {
            "type": Agent,
            "FOV": agent_fov,
            "movements": MOVEMENTS,
            "scale_keys": scale_keys,
            "spawn_rad": spawn_radius,
            "speed_per_scale": speed_per_scale,
            "verbose": False,
            "landmarks": lm_type,
        }

        agent_lst = GetAgentLst(agent_params)
    except Exception as e:
        logger.error(f"Failed to generate agents: {e}")
        sys.exit(1)

    try:
        brain_weights = GetBrain(args.dir_models)
    except Exception as e:
        logger.error(f"Failed to load brain models from '{args.dir_models}': {e}")
        sys.exit(1)
    
    transition_layer_size = 1024

    # 6. INFERENCE LOOP
    logger.info(f"Starting prediction on {len(environment_lst)} patients")
    start_time = time.time()
    tot_step = 0
    fails = {}

    for env_idx, environment in enumerate(environment_lst):
        tot_step = _predict_one_patient(agent_lst, args, brain_weights, env_idx, environment, environment_lst, fails, scale_keys, tot_step, transition_layer_size)

    # 7. FINAL LOGS
    end_time = time.time()
    logger.info("--- Execution Summary ---")
    logger.info(f"Total steps taken: {tot_step}")
    logger.info(f"Execution time: {end_time - start_time:.2f}s")
    
    if fails:
        logger.warning(
            f"{len(fails)} landmark(s) were not placed on at least one scan. "
            "They are ABSENT from the output files, not misplaced in them; "
            "each scan concerned has a <patient>_lm_NotFound.json next to "
            "its predictions saying which and why.")
        for lm, count in sorted(fails.items()):
            logger.warning(f"Landmark '{lm}': {count}/{len(environment_lst)} failures")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="ALI-CBCT: Automatic Landmark Identification")
        parser.add_argument("input", type=str, help="Input folder or file")
        parser.add_argument("dir_models", type=str, help="Directory of the models")
        parser.add_argument("lm_type", type=str, help="Type of landmarks (e.g., 'Sella')")
        parser.add_argument("output_dir", type=str, help="Output directory")
        parser.add_argument("temp_fold", type=str, help="Temporary folder")
        parser.add_argument("dcm_input", type=str, help="Is input DICOM? (true/false)")
        parser.add_argument("spacing", type=str, help="Spacings list")
        parser.add_argument("speed_per_scale", type=str, help="Agent speed")
        parser.add_argument("agent_fov", type=str, help="Agent FOV")
        parser.add_argument("spawn_radius", type=str, help="Agent spawn radius")
        
        args = parser.parse_args()
        main(args)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)