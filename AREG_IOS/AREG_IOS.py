#!/usr/bin/env python-real

import os
import sys
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
    from AREG_IOS_utils.dataset import DatasetPatch
    from AREG_IOS_utils.PredPatch import PredPatch
    from AREG_IOS_utils.vtkSegTeeth import vtkMeshTeeth
    from AREG_IOS_utils.ICP import vtkICP
    from AREG_IOS_utils.ICP import ICP
    from AREG_IOS_utils.utils import WriteSurf
    from AREG_IOS_utils.transformation import TransformSurf
    from AREG_IOS.AREG_IOS_utils.transformation import saveMatrixAsTfm

else : 
    from AREG_IOS_utils import (
        DatasetPatch,
        PredPatch,
        vtkMeshTeeth,
        vtkICP,
        ICP,
        WriteSurf,
        TransformSurf,
        saveMatrixAsTfm,
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
