#!/usr/bin/env python3
import os
import sys
import time
import glob
import logging
import argparse
import ast
import shutil
from pathlib import Path

import numpy as np
import torch

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("ALI_CBCT")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- DYNAMIC IMPORTS ---
try:
    # Add parent directory to sys.path for local imports
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
    from ALI_CBCT_utils import (
        Agent, GetAgentLst, Brain, DNet, Environment, GenEnvironmentLst,
        GetBrain, CorrectHisto, SetSpacing, convertdicom2nifti,
        MOVEMENTS, DEVICE
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def update_slicer_progress(value):
    """Utility to update Slicer progress bar."""
    print(f"<filter-progress>{value}</filter-progress>", flush=True)
    time.sleep(0.05)

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
                    continue

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
            continue

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
        logger.info(f"Processing patient: {environment.patient_id}")
        
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
                            logger.warning(f"Agent failed to find {agent.target}")
                        else:
                            tot_step += search_result
                    except Exception as e:
                        logger.error(f"Error loading model weights for {agent.target}: {e}")
                        fails[agent.target] = fails.get(agent.target, 0) + 1
                else:
                    logger.error(f"No model found for landmark: {agent.target}")

            except Exception as e:
                logger.error(f"Error during agent search for {agent.target}: {e}")
            finally:
                # Cleanup to free GPU memory
                agent.SetBrain(None)
                if 'brain' in locals(): del brain
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Save results for this patient
        try:
            environment.SavePredictedLandmarks(scale_keys[-1], args.output_dir)
        except Exception as e:
            logger.error(f"Failed to save predictions for patient {environment.patient_id}: {e}")
        
        # Update Slicer Progress
        progress = 20 + int((env_idx + 1) / len(environment_lst) * 80)
        update_slicer_progress(progress)

    # 7. FINAL LOGS
    end_time = time.time()
    logger.info("--- Execution Summary ---")
    logger.info(f"Total steps taken: {tot_step}")
    logger.info(f"Execution time: {end_time - start_time:.2f}s")
    
    for lm, count in fails.items():
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