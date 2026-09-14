#!/usr/bin/env python-real

import os
import sys
import glob
import shutil
import argparse
import platform
import logging

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("AREG_IOS")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ===== DEPENDENCY CHECK =====
# Check and fix torch/torchvision compatibility before any imports
try:
    # Add parent for deps check
    areg_ios_path = os.path.dirname(__file__)
    if areg_ios_path not in sys.path:
        sys.path.insert(0, areg_ios_path)
    
    from AREG_IOS_utils.check_deps import ensure_compatible
    ensure_compatible()
    logger.debug("Dependency check passed")
except ImportError as e:
    logger.warning("[WARNING] Could not import dependency checker: {}".format(e))
# ===== END DEPENDENCY CHECK =====

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

def check_platform():
    if platform.system() == 'Windows':
        return "Windows"
    elif platform.system() == 'Linux':
        if 'Microsoft' in platform.release():
            return "WSL"
        else:
            return "Linux"
    else:
        return "Unknown"

if check_platform()=="WSL":
    from AREG_IOS_utils.dataset import DatasetPatch, SortLower
    from AREG_IOS_utils.PredPatch import PredPatch
    from AREG_IOS_utils.vtkSegTeeth import vtkMeshTeeth
    from AREG_IOS_utils.ICP import vtkICP
    from AREG_IOS_utils.ICP import ICP
    from AREG_IOS_utils.utils import WriteSurf, ReadSurf, LoadJsonLandmarks
    from AREG_IOS_utils.transformation import TransformSurf
    from AREG_IOS.AREG_IOS_utils.transformation import saveMatrixAsTfm
    from AREG_IOS_utils.mgl_patch import (
        MGLPatch, DropDoubtfulLandmarks, DEFAULT_RADIUS, MGL_ARRAY_NAME)

else :
    from AREG_IOS_utils import (
        DatasetPatch,
        SortLower,
        PredPatch,
        vtkMeshTeeth,
        vtkICP,
        ICP,
        WriteSurf,
        ReadSurf,
        LoadJsonLandmarks,
        TransformSurf,
        saveMatrixAsTfm,
        MGLPatch,
        DropDoubtfulLandmarks,
        DEFAULT_RADIUS,
        MGL_ARRAY_NAME,
    )


def FindLandmarkFile(folder, surf_path):
    """Locate the MG landmark json that goes with `surf_path`.

    ALI_IOS names its output '<scan>_Lower_MG_Pred.json', which is tried first;
    a folder produced by hand is then searched for any json carrying the scan
    name, so users are not forced into that convention.
    """
    stem = os.path.splitext(os.path.basename(surf_path))[0]

    expected = os.path.join(folder, f"{stem}_Lower_MG_Pred.json")
    if os.path.isfile(expected):
        return expected

    candidates = sorted(f for f in glob.glob(os.path.join(folder, "*.json"))
                        if stem in os.path.basename(f))
    if candidates:
        if len(candidates) > 1:
            logger.warning(f"Several landmark files match {stem}, using {os.path.basename(candidates[0])}")
        return candidates[0]

    raise FileNotFoundError(f"No MG landmark json found for {stem} in {folder}")


def RunMGL(args, icp):
    """Register the lower arches on the band around the mucogingival line.

    Mirrors the palatal flow: a stable region is painted on both timepoints,
    the ICP runs on that region only, and T2 is written transformed. The upper
    arches are left untouched, the MG model covering the mandible only.
    """
    pairs = SortLower(args.T1, args.T2)
    if not pairs:
        raise ValueError("MGL registration needs lower arches, none were paired between the input folders")

    logger.info(f"MGL registration on {len(pairs)} lower pair(s), patch radius {args.patch_radius} mm")

    processed, failed = 0, []
    for idx, pair in enumerate(pairs):
        context = f"sample {idx + 1}/{len(pairs)}"
        try:
            surfaces = {}
            for time in ("T1", "T2"):
                surf = ReadSurf(pair[time])
                folder = args.lm_T1 if time == "T1" else args.lm_T2
                landmark_path = FindLandmarkFile(folder, pair[time])
                landmarks = DropDoubtfulLandmarks(
                    LoadJsonLandmarks(landmark_path), landmark_path)
                surfaces[time] = MGLPatch(surf, landmarks, radius=args.patch_radius)

            output_icp = icp.run(surfaces["T2"], surfaces["T1"])

            WriteSurf(surfaces["T1"], args.output, os.path.basename(pair["T1"]), args.suffix)
            WriteSurf(output_icp["source_Or"], args.output,
                      os.path.basename(pair["T2"]), args.suffix)

            with open(args.log_path, "w") as log_f:
                log_f.write(str(idx + 1))

            processed += 1
            logger.info(f"Successfully processed {context}")
        except Exception as e:
            logger.error(f"Failed to process {context}: {e}")
            failed.append((idx, str(e)))
            continue

    logger.info(f"MGL registration completed: {processed}/{len(pairs)} pair(s) processed successfully")
    for idx, error in failed:
        logger.warning(f"  Pair {idx}: {error}")

    if processed == 0:
        # Every pair failed for the same reason more often than not, and a run
        # that reports success while leaving the output folder empty sends the
        # user looking for the answer in the wrong place. Fail with the first
        # reason instead.
        raise RuntimeError(
            f"No pair could be registered out of {len(pairs)}. First error: "
            f"{failed[0][1] if failed else 'unknown'}"
        )


def main(args):
    """Main function for IOS alignment registration with comprehensive error handling."""
    try:
        logger.info("Starting AREG_IOS registration pipeline")
        
        # ===== LOG FILE SETUP =====
        try:
            logger.debug(f"Setting up log file: {args.log_path}")
            log_dir = os.path.split(args.log_path)[0]
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            with open(args.log_path, "w") as log_f:
                log_f.truncate(0)
            logger.debug("Log file ready")
        except Exception as e:
            logger.error(f"Error setting up log file: {e}")
            raise

        # ===== REGISTRATION SETUP =====
        try:
            logger.debug("Initializing registration method (vtkICP)")
            Method = [vtkICP()]
            option = vtkMeshTeeth(list_teeth=[1], property="Butterfly")
            icp = ICP(Method, option=option)
            logger.debug("Registration method initialized")
        except Exception as e:
            logger.error(f"Error initializing registration method: {e}")
            raise

        # ===== MGL MODE =====
        # The lower patch comes from the landmarks, so neither the palatal model
        # nor the upper arches are involved, and the dataset of upper pairs is
        # not built at all: the input may hold lower scans only. Its patch lives
        # in its own array, so the ICP is pointed at that one.
        if args.reg_type == "MGL":
            mgl_icp = ICP([vtkICP()],
                          option=vtkMeshTeeth(list_teeth=[1], property=MGL_ARRAY_NAME))
            RunMGL(args, mgl_icp)
            return

        # ===== DATASET INITIALIZATION =====
        try:
            logger.debug(f"Loading dataset from T1: {args.T1}, T2: {args.T2}")
            dataset = DatasetPatch(args.T1, args.T2, "Universal_ID")
            logger.info(f"Dataset loaded with {len(dataset)} sample(s)")
            if not dataset:
                logger.error("Error: The dataset to process is empty for AREG_IOS, check files names")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

        # ===== MODEL INITIALIZATION =====
        try:
            logger.debug(f"Loading prediction model: {args.model}")
            Patched = PredPatch(args.model)
            logger.debug("Prediction model loaded")
        except Exception as e:
            logger.error(f"Error loading prediction model: {e}")
            raise

        # ===== CHECK DENTITION TYPE =====
        try:
            logger.debug("Checking dentition type (upper/lower)")
            lower = False
            if dataset.isLower():
                lower = True
            logger.debug(f"Dentition type: {'lower' if lower else 'upper'}")
        except Exception as e:
            logger.warning(f"Error checking dentition type: {e}, assuming upper only")
            lower = False

        # ===== MAIN PROCESSING LOOP =====
        processed_samples = 0
        failed_samples = []

        for idx in range(len(dataset)):
            sample_context = f"sample {idx+1}/{len(dataset)}"
            logger.info(f"Processing {sample_context}")
            
            try:
                # ===== UPPER SURFACE T1 =====
                try:
                    logger.debug(f"Processing upper T1 surface")
                    name_t1 = os.path.basename(dataset.getUpperPath(idx, "T1"))
                    surf_T1 = dataset.getUpperSurf(idx, "T1")
                    
                    if surf_T1 is None:
                        logger.warning(f"Upper T1 surface is None, skipping")
                        raise ValueError("Upper T1 surface not found")
                    
                    surf_T1 = Patched(dataset[idx, "T1"], surf_T1)
                    WriteSurf(surf_T1, args.output, name_t1, args.suffix)
                    logger.debug(f"Saved upper T1 surface")
                except Exception as e:
                    logger.error(f"Error processing upper T1 surface: {e}")
                    raise

                # ===== UPDATE LOG =====
                try:
                    with open(args.log_path, "w") as log_f:
                        log_f.write(str(1))
                except Exception as e:
                    logger.warning(f"Error updating log file: {e}")

                # ===== UPPER SURFACE T2 =====
                try:
                    logger.debug(f"Processing upper T2 surface")
                    name_t2 = os.path.basename(dataset.getUpperPath(idx, "T2"))
                    surf_T2 = dataset.getUpperSurf(idx, "T2")
                    
                    if surf_T2 is None:
                        logger.warning(f"Upper T2 surface is None, skipping")
                        raise ValueError("Upper T2 surface not found")
                    
                    surf_T2 = Patched(dataset[idx, "T2"], surf_T2)
                    logger.debug(f"Predicted upper T2 surface")
                except Exception as e:
                    logger.error(f"Error processing upper T2 surface: {e}")
                    raise

                # ===== UPDATE LOG =====
                try:
                    with open(args.log_path, "w") as log_f:
                        log_f.write(str(1))
                except Exception as e:
                    logger.warning(f"Error updating log file: {e}")

                # ===== RUN ICP REGISTRATION =====
                try:
                    logger.debug(f"Running ICP registration")
                    output_icp = icp.run(surf_T2, surf_T1)
                    logger.info(f"ICP registration completed")
                except Exception as e:
                    logger.error(f"Error running ICP registration: {e}")
                    raise

                # ===== SAVE REGISTERED UPPER T2 =====
                try:
                    logger.debug(f"Saving registered upper T2 surface")
                    WriteSurf(output_icp["source_Or"], args.output, name_t2, args.suffix)
                    logger.debug(f"Saved registered upper T2 surface")
                except Exception as e:
                    logger.error(f"Error saving registered upper T2: {e}")
                    raise

                # ===== HANDLE TFM FILES (FOR Auto_IOS MODE) =====
                if args.areg_mode == "Auto_IOS":
                    try:
                        logger.debug(f"Processing transformation files for Auto_IOS mode")
                        # TIMEPOINT-SUFFIX: this assumes the T2 scan is named _T2; a _T4 input keeps
                        # the suffix in patient_id and stops matching its T1. See the full note above
                        # GetPatients in AREG_CBCT/AREG_CBCT_utils/utils.py.
                        patient_id = name_t2.split("_T2")[0]
                        patient_id_short = patient_id.split("_")[0] if "_" in patient_id else patient_id
                        
                        aso_tfm_path_T1 = os.path.join(args.T1, f"{patient_id_short}_SegOr.tfm")
                        aso_tfm_path_T2 = os.path.join(args.T2, f"{patient_id_short}_SegOr.tfm")
                        out_tfm_T1 = os.path.join(args.output, f"{patient_id_short}_T1_SegOr.tfm")
                        
                        # Copy T1 matrix
                        try:
                            if os.path.exists(aso_tfm_path_T1):
                                shutil.copy(aso_tfm_path_T1, out_tfm_T1)
                                logger.debug(f"Saved T1 matrix: {out_tfm_T1}")
                            else:
                                logger.warning(f"T1 tfm file not found at {aso_tfm_path_T1}")
                        except Exception as e:
                            logger.error(f"Error copying T1 matrix: {e}")
                        
                        # Save T2 matrix
                        try:
                            saveMatrixAsTfm(output_icp["matrix"], aso_tfm_path_T2, args.output, patient_id_short, args.suffix, args.areg_mode)
                            logger.debug(f"Saved T2 transformation matrix")
                        except Exception as e:
                            logger.error(f"Error saving T2 matrix: {e}")
                    except Exception as e:
                        logger.warning(f"Error handling TFM files: {e}")

                # ===== LOWER SURFACES (IF PRESENT) =====
                if lower:
                    try:
                        logger.debug(f"Processing lower surfaces")
                        
                        # Lower T2
                        try:
                            surf_lower_t2 = dataset.getLowerSurf(idx, "T2")
                            if surf_lower_t2 is not None:
                                surf_lower_t2 = TransformSurf(surf_lower_t2, output_icp["matrix"])
                                name_lower_t2 = os.path.basename(dataset.getLowerPath(idx, "T2"))
                                WriteSurf(surf_lower_t2, args.output, name_lower_t2, args.suffix)
                                logger.debug(f"Saved registered lower T2 surface")
                        except Exception as e:
                            logger.warning(f"Error processing lower T2: {e}")
                        
                        # Lower T1
                        try:
                            surf_lower_t1 = dataset.getLowerSurf(idx, "T1")
                            if surf_lower_t1 is not None:
                                name_lower_t1 = os.path.basename(dataset.getLowerPath(idx, "T1"))
                                WriteSurf(surf_lower_t1, args.output, name_lower_t1, args.suffix)
                                logger.debug(f"Saved lower T1 surface")
                        except Exception as e:
                            logger.warning(f"Error processing lower T1: {e}")
                    except Exception as e:
                        logger.warning(f"Error processing lower surfaces: {e}")

                # ===== UPDATE FINAL LOG =====
                try:
                    with open(args.log_path, "w") as log_f:
                        log_f.write(str(idx + 1))
                    logger.debug(f"Log file updated")
                except Exception as e:
                    logger.warning(f"Error updating final log: {e}")

                processed_samples += 1
                logger.info(f"Successfully processed {sample_context}")
            
            except Exception as e:
                logger.error(f"Failed to process {sample_context}: {e}")
                failed_samples.append((idx, str(e)))
                continue

        # ===== FINAL REPORT =====
        try:
            logger.info(f"Registration pipeline completed: {processed_samples}/{len(dataset)} sample(s) processed successfully")
            if failed_samples:
                logger.warning(f"Failed to process {len(failed_samples)} sample(s):")
                for idx, error in failed_samples:
                    logger.warning(f"  Sample {idx}: {error}")
        except Exception as e:
            logger.error(f"Error generating final report: {e}")

    except Exception as e:
        logger.error(f"Fatal error in main(): {e}")
        raise


if __name__ == "__main__":
    try:
        logger.info("AREG_IOS entry point initiated")
        
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("T1", type=str)
            parser.add_argument("T2", type=str)
            parser.add_argument("output", type=str)
            parser.add_argument("model", type=str)
            parser.add_argument("suffix", type=str)
            parser.add_argument("log_path", type=str)
            parser.add_argument("areg_mode", type=str)
            # Appended after the historical arguments so the positional order
            # the GUI relies on stays valid.
            parser.add_argument("reg_type", type=str, nargs="?", default="Butterfly",
                                help="'Butterfly' (palatal patch, upper arch) or "
                                     "'MGL' (band along the mucogingival line, lower arch)")
            parser.add_argument("patch_radius", type=float, nargs="?", default=DEFAULT_RADIUS,
                                help="MGL only: half-height of the band around the curve, in mm. "
                                     "0 leaves no band: the registration runs on the landmarks alone")
            parser.add_argument("lm_T1", type=str, nargs="?", default="None",
                                help="MGL only: folder holding the T1 MG landmark json files")
            parser.add_argument("lm_T2", type=str, nargs="?", default="None",
                                help="MGL only: folder holding the T2 MG landmark json files")

            args = parser.parse_args()
            logger.debug(f"Arguments parsed successfully: T1={args.T1}, T2={args.T2}")
        except Exception as e:
            logger.error(f"Error parsing command line arguments: {e}")
            raise

        try:
            logger.info("Calling main() function")
            main(args)
            logger.info("AREG_IOS completed successfully")
        except Exception as e:
            logger.error(f"Error in main() execution: {e}")
            raise

    except SystemExit as e:
        logger.info(f"Script exited with code: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.critical(f"Fatal error in entry point: {e}")
        sys.exit(f"Fatal error: {e}")
