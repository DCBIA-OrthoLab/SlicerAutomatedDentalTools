#!/usr/bin/env python-real
import argparse
import sys, os, time
import logging
import numpy as np
import slicer
import SimpleITK as sitk

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("AREG_CBCT")
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

from AREG_CBCT_utils import (
    GetDictPatients,
    VoxelBasedRegistration,
    LoadOnlyLandmarks,
    applyTransformLandmarks,
    WriteJson,
    translate,
    convertdicom2nifti,
)


def main(args):
    """Main function for CBCT registration with comprehensive error handling."""
    try:
        logger.info("Starting AREG_CBCT registration pipeline")
        
        # ===== ARGUMENT PARSING AND VALIDATION =====
        try:
            logger.debug("Parsing and validating arguments")
            (
                t1_folder,
                t2_folder,
                output_dir,
                reg_type,
                add_name,
                SegLabel,
                temp_folder,
                Approx,
                mask_folder_t1,
            ) = (
                args.t1_folder[0],
                args.t2_folder[0],
                args.output_folder[0],
                args.reg_type[0],
                args.add_name[0],
                int(args.SegmentationLabel[0]),
                args.temp_folder[0],
                True if args.ApproxReg[0] == "true" else False,
                None if args.mask_folder_t1[0] == "None" else args.mask_folder_t1[0],
            )
            if SegLabel == 0:
                SegLabel = None
            logger.debug(f"Arguments parsed: reg_type={reg_type}, SegLabel={SegLabel}, Approx={Approx}")
        except (IndexError, ValueError) as e:
            logger.error(f"Error parsing arguments: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during argument parsing: {e}")
            raise

        # ===== INPUT VALIDATION =====
        try:
            logger.debug("Validating input paths")
            if not os.path.exists(t1_folder):
                logger.error(f"T1 folder not found: {t1_folder}")
                raise FileNotFoundError(f"T1 folder does not exist: {t1_folder}")
            if not os.path.exists(t2_folder):
                logger.error(f"T2 folder not found: {t2_folder}")
                raise FileNotFoundError(f"T2 folder does not exist: {t2_folder}")
            if mask_folder_t1 and not os.path.exists(mask_folder_t1):
                logger.error(f"Mask folder not found: {mask_folder_t1}")
                raise FileNotFoundError(f"Mask folder does not exist: {mask_folder_t1}")
            logger.debug("Input paths validated")
        except Exception as e:
            logger.error(f"Error validating input paths: {e}")
            raise

        # ===== OUTPUT DIRECTORY SETUP =====
        try:
            logger.debug(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            logger.debug("Output directory ready")
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            raise

        # ===== DICOM CONVERSION =====
        if args.DCMInput[0] == "true":
            try:
                logger.debug("Converting DICOM files for T1 folder")
                convertdicom2nifti(t1_folder)
                logger.debug("T1 DICOM conversion completed")
            except Exception as e:
                logger.error(f"Error converting T1 DICOM files: {e}")
                raise
            
            try:
                logger.debug("Converting DICOM files for T2 folder")
                convertdicom2nifti(t2_folder)
                logger.debug("T2 DICOM conversion completed")
            except Exception as e:
                logger.error(f"Error converting T2 DICOM files: {e}")
                raise

        # ===== PATIENT DATA DISCOVERY =====
        try:
            logger.debug("Discovering patient data")
            patients = GetDictPatients(
                folder_t1_path=t1_folder,
                folder_t2_path=t2_folder,
                segmentationType=reg_type,
                mask_folder_t1=mask_folder_t1,
            )
            logger.info(f"Found {len(patients)} patient(s)")
        except Exception as e:
            logger.error(f"Error discovering patient data: {e}")
            raise

        if not patients:
            logger.error(f"Error: There is no patient to process. Check the files names.")

        # ===== REGISTRATION PROCESSING LOOP =====
        logger.info(f"Starting {translate(reg_type)} registration")
        
        processed_patients = 0
        failed_patients = []
        
        for patient, data in patients.items():
            patient_context = f"patient: {patient}"
            logger.info(f"Processing {patient_context}")
            
            try:
                # ===== OUTPUT PATH SETUP =====
                try:
                    outpath = os.path.join(output_dir, translate(reg_type), patient + "_OutReg")
                    ScanOutPath = os.path.join(
                        outpath, patient + "_" + reg_type + "Scan" + add_name + ".nii.gz"
                    )
                    TransOutPath = os.path.join(
                        outpath, patient + "_" + reg_type + add_name + "_matrix.tfm"
                    )
                    logger.debug(f"Output paths set for {patient}")
                except Exception as e:
                    logger.error(f"Error setting output paths for {patient}: {e}")
                    raise

                # ===== REGISTRATION EXECUTION =====
                try:
                    logger.debug(f"Starting registration for {patient}")
                    transform, resample_t2 = VoxelBasedRegistration(
                        fixed_image_path=data["scanT1"],
                        moving_image_path=data["scanT2"],
                        fixed_seg_path=data["segT1"],
                        temp_folder=temp_folder,
                        approx=Approx,
                        SegLabel=SegLabel,
                    )
                    logger.info(f"Registration completed for {patient}")
                except Exception as e:
                    logger.error(f"Error during registration for {patient}: {e}")
                    raise

                # ===== OUTPUT SAVING =====
                try:
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)
                    logger.debug(f"Saving transform to {TransOutPath}")
                    sitk.WriteTransform(transform, TransOutPath)
                    logger.debug(f"Saving resampled image to {ScanOutPath}")
                    sitk.WriteImage(resample_t2, ScanOutPath)
                    logger.info(f"Output files saved for {patient}")
                except Exception as e:
                    logger.error(f"Error saving output files for {patient}: {e}")
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
            logger.info(f"Registration pipeline completed: {processed_patients}/{len(patients)} patients processed successfully")
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
        logger.info("AREG_CBCT entry point initiated")
        
        try:
            parser = argparse.ArgumentParser()

            parser.add_argument("t1_folder", nargs=1)
            parser.add_argument("t2_folder", nargs=1)
            parser.add_argument("reg_type", nargs=1)
            parser.add_argument("output_folder", nargs=1)
            parser.add_argument("add_name", nargs=1)
            parser.add_argument("DCMInput", nargs=1)
            parser.add_argument("SegmentationLabel", nargs=1)
            parser.add_argument("temp_folder", nargs=1)
            parser.add_argument("ApproxReg", nargs=1)
            parser.add_argument("mask_folder_t1", nargs=1)

            args = parser.parse_args()
            logger.debug(f"Arguments parsed successfully")
        except Exception as e:
            logger.error(f"Error parsing command line arguments: {e}")
            raise

        try:
            logger.info("Calling main() function")
            main(args)
            logger.info("AREG_CBCT completed successfully")
        except Exception as e:
            logger.error(f"Error in main() execution: {e}")
            raise

    except SystemExit as e:
        logger.info(f"Script exited with code: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.critical(f"Fatal error in entry point: {e}")
        sys.exit(f"Fatal error: {e}")
