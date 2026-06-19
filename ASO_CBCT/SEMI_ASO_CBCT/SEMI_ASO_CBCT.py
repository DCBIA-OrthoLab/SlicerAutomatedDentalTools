#!/usr/bin/env python-real

import sys
import os
import time
import argparse
import logging
import SimpleITK as sitk

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("SEMI_ASO_CBCT")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from ASO_CBCT_utils import (
    ICP,
    ExtractFilesFromFolder,
    MergeJson,
    WriteJson,
    GetPatients,
)


def main(args):
    """Main function for SEMI_ASO_CBCT registration with comprehensive error handling."""
    try:
        logger.info("Starting SEMI_ASO_CBCT registration pipeline")
        
        # ===== ARGUMENT PARSING =====
        try:
            logger.debug("Parsing arguments")
            scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
            lm_extension = [".json"]

            list_landmark = args.list_landmark[0].split(" ")
            input_dir, gold_dir, out_dir = (
                args.input[0],
                args.gold_folder[0],
                args.output_folder[0],
            )
            logger.debug(f"Arguments parsed: input_dir={input_dir}, gold_dir={gold_dir}")
        except Exception as e:
            logger.error(f"Error parsing arguments: {e}")
            raise

        # ===== MERGE JSON =====
        try:
            logger.debug("Merging JSON files")
            MergeJson(input_dir)
            logger.debug("JSON merge completed")
        except Exception as e:
            logger.warning(f"Error merging JSON files: {e}")

        # ===== OUTPUT SETUP =====
        try:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            logger.debug(f"Output directory ready: {out_dir}")
        except Exception as e:
            logger.error(f"Error setting up output directory: {e}")
            raise

        # ===== PATIENT DISCOVERY =====
        try:
            logger.debug("Discovering patients")
            patients = GetPatients(input_dir)
            logger.info(f"Found {len(patients)} patient(s)")
        except Exception as e:
            logger.error(f"Error discovering patients: {e}")
            raise

        # ===== GOLD REFERENCE =====
        try:
            logger.debug("Loading gold reference files")
            gold_file, gold_json_file = ExtractFilesFromFolder(
                gold_dir, scan_extension, lm_extension, gold=True
            )
            logger.debug("Gold reference loaded")
        except Exception as e:
            logger.error(f"Error loading gold reference: {e}")
            raise

        # ===== MAIN PROCESSING LOOP =====
        processed_patients = 0
        failed_patients = []

        for patient, data in patients.items():
            patient_context = f"patient: {patient}"
            logger.info(f"Processing {patient_context}")

            try:
                # ===== EXTRACT PATIENT DATA =====
                try:
                    logger.debug(f"Extracting patient data")
                    input_file, input_json_file, input_transform = data["scan"], data["json"], data["tfm"]
                    logger.debug(f"Patient data extracted")
                except KeyError as e:
                    logger.warning(f"Patient {patient} missing required files: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error extracting patient data: {e}")
                    raise

                # ===== RUN ICP REGISTRATION =====
                try:
                    logger.debug(f"Running ICP registration")
                    output, source_transformed, TransformSITK = ICP(
                        input_file, input_json_file, gold_file, gold_json_file, list_landmark, input_transform
                    )
                    
                    if output is None:
                        logger.error(f"ICP registration failed")
                        raise RuntimeError("ICP registration returned None")
                    
                    logger.info(f"ICP registration completed")
                except Exception as e:
                    logger.error(f"Error during ICP registration: {e}")
                    raise

                # ===== SAVE JSON =====
                try:
                    logger.debug(f"Saving landmark JSON")
                    dir_json = os.path.dirname(input_json_file.replace(input_dir, out_dir))
                    if not os.path.exists(dir_json):
                        os.makedirs(dir_json)
                    json_path = os.path.join(
                        dir_json, patient + "_lm_" + args.add_inname[0] + ".mrk.json"
                    )

                    if not os.path.exists(json_path):
                        WriteJson(source_transformed, json_path)
                        logger.info(f"Saved landmark JSON")
                except Exception as e:
                    logger.error(f"Error saving landmark JSON: {e}")
                    raise

                # ===== SAVE SCAN =====
                try:
                    logger.debug(f"Saving registered scan")
                    dir_scan = os.path.dirname(input_file.replace(input_dir, out_dir))
                    if not os.path.exists(dir_scan):
                        os.makedirs(dir_scan)

                    file_outpath = os.path.join(
                        dir_scan, patient + "_" + args.add_inname[0] + ".nii.gz"
                    )
                    if not os.path.exists(file_outpath):
                        sitk.WriteImage(output, file_outpath)
                        logger.info(f"Saved registered scan")
                except Exception as e:
                    logger.error(f"Error saving registered scan: {e}")
                    raise

                # ===== SAVE TRANSFORM =====
                try:
                    logger.debug(f"Saving transformation")
                    transform_outpath = os.path.join(
                        dir_scan, patient + "_" + args.add_inname[0] + "_transform.tfm"
                    )
                    if not os.path.exists(transform_outpath):
                        sitk.WriteTransform(TransformSITK, transform_outpath)
                        logger.info(f"Saved transformation")
                except Exception as e:
                    logger.error(f"Error saving transformation: {e}")
                    raise

                # ===== PROGRESS REPORTING =====
                try:
                    print(f"""<filter-progress>{0}</filter-progress>""")
                    sys.stdout.flush()
                    time.sleep(0.2)
                    print(f"""<filter-progress>{2}</filter-progress>""")
                    sys.stdout.flush()
                    time.sleep(0.2)
                    print(f"""<filter-progress>{0}</filter-progress>""")
                    sys.stdout.flush()
                    time.sleep(0.2)
                    logger.debug("Progress reported")
                except Exception as e:
                    logger.warning(f"Error reporting progress: {e}")

                processed_patients += 1
                logger.info(f"Successfully processed {patient_context}")

            except Exception as e:
                logger.error(f"Failed to process {patient_context}: {e}")
                failed_patients.append((patient, str(e)))
                continue

        # ===== FINAL REPORT =====
        try:
            logger.info(f"Registration completed: {processed_patients}/{len(patients)} patient(s) processed successfully")
            if failed_patients:
                logger.warning(f"Failed to process {len(failed_patients)} patient(s):")
                for patient, error in failed_patients:
                    logger.warning(f"  {patient}: {error}")
        except Exception as e:
            logger.error(f"Error generating final report: {e}")

    except Exception as e:
        logger.error(f"Fatal error in main(): {e}")
        raise


if __name__ == "__main__":
    try:
        logger.info("SEMI_ASO_CBCT entry point initiated")
        
        try:
            parser = argparse.ArgumentParser()

            parser.add_argument("input", nargs=1)
            parser.add_argument("gold_folder", nargs=1)
            parser.add_argument("output_folder", nargs=1)
            parser.add_argument("add_inname", nargs=1)
            parser.add_argument("list_landmark", nargs=1)

            args = parser.parse_args()
            logger.debug("Arguments parsed successfully")
        except Exception as e:
            logger.error(f"Error parsing command line arguments: {e}")
            raise

        try:
            logger.info("Calling main() function")
            main(args)
            logger.info("SEMI_ASO_CBCT completed successfully")
        except Exception as e:
            logger.error(f"Error in main() execution: {e}")
            raise

    except SystemExit as e:
        logger.info(f"Script exited with code: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.critical(f"Fatal error in entry point: {e}")
        sys.exit(f"Fatal error: {e}")
