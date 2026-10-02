#!/usr/bin/env python-real

import argparse
import SimpleITK as sitk
import sys, os
import numpy as np

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
from ADTLib.progress_protocol import PATIENT_DONE, emit_event

logger = get_logger("PRE_ASO_CBCT")

# realpath, not __file__: registering the CLI through a symlink (a flat dev
# folder of links into the source tree) leaves __file__ on the link, whose
# parent holds no ASO_CBCT_utils. Resolving first lands in ASO_CBCT
# either way. In a built install the package is already on sys.path.
fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(fpath)


from ASO_CBCT_utils import (
    ExtractFilesFromFolder,
    convertdicom2nifti,
)


def ResampleImage(image, transform):
    """
    Resample image using SimpleITK

    Parameters
    ----------
    image : SimpleITK.Image
        Image to be resampled
    target : SimpleITK.Image
        Target image
    transform : SimpleITK transform
        Transform to be applied to the image.

    Returns
    -------
    SimpleITK image
        Resampled image.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    resample.SetTransform(transform)
    resample.SetInterpolator(sitk.sitkLinear)
    orig_size = np.array(image.GetSize(), dtype=int)
    ratio = 1
    new_size = orig_size * ratio
    new_size = np.ceil(new_size).astype(int)  #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetDefaultPixelValue(0)

    # Set New Origin
    orig_origin = np.array(image.GetOrigin())
    # apply transform to the origin
    orig_center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0)
    )
    new_origin = orig_origin - orig_center
    resample.SetOutputOrigin(new_origin)

    return resample.Execute(image)


def _preprocess_one_file(failed_files, i, input_dir, input_files, out_dir, processed_files):
    """Orient and resample one scan before the registration."""
    input_file = input_files[i]
    file_context = f"file {i+1}/{len(input_files)}: {os.path.basename(input_file)}"
    logger.info(f"Processing {file_context}")

    try:
        # ===== READ IMAGE =====
        try:
            logger.debug(f"Reading image")
            img = sitk.ReadImage(input_file)
            logger.debug("Image read successfully")
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            raise

        # ===== TRANSLATION TRANSFORM =====
        try:
            logger.debug("Computing center translation")
            T = -np.array(
                img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0)
            )
            translation = sitk.TranslationTransform(3)
            translation.SetOffset(T.tolist())
            logger.debug(f"Translation transform created")
        except Exception as e:
            logger.error(f"Error creating translation transform: {e}")
            raise

        # ===== RESAMPLE IMAGE =====
        try:
            logger.debug("Resampling image")
            img_trans = ResampleImage(img, translation.GetInverse())
            img_out = img_trans
            logger.debug("Image resampled successfully")
        except Exception as e:
            logger.error(f"Error resampling image: {e}")
            raise

        # ===== PREPARE OUTPUT DIRECTORY =====
        try:
            logger.debug("Preparing output directory")
            dir_scan = os.path.dirname(input_file.replace(input_dir, out_dir))
            if not os.path.exists(dir_scan):
                os.makedirs(dir_scan)
            logger.debug(f"Output directory ready: {dir_scan}")
        except Exception as e:
            logger.error(f"Error preparing output directory: {e}")
            raise

        # ===== SAVE PROCESSED IMAGE =====
        try:
            file_outpath = os.path.join(dir_scan, os.path.basename(input_file))
            if not os.path.exists(file_outpath):
                logger.debug(f"Saving processed image to {file_outpath}")
                sitk.WriteImage(img_out, file_outpath)
                logger.info(f"Saved processed image")
            else:
                logger.debug(f"Output file already exists, skipping")
        except Exception as e:
            logger.error(f"Error saving processed image: {e}")
            raise

        # ===== SAVE TRANSFORMATION =====
        try:
            MED_EXTS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")

            def strip_ext(name):
                for ext in MED_EXTS:
                    if name.endswith(ext):
                        return name[: -len(ext)]
                return os.path.splitext(name)[0]

            translation_outpath = os.path.join(dir_scan, strip_ext(os.path.basename(input_file)) + ".tfm")

            if not os.path.exists(translation_outpath):
                logger.debug(f"Saving transformation to {translation_outpath}")
                sitk.WriteTransform(translation, translation_outpath)
                logger.info(f"Saved transformation")
            else:
                logger.debug(f"Transform file already exists, skipping")
        except Exception as e:
            logger.error(f"Error saving transformation: {e}")
            raise

        # ===== PROGRESS REPORTING =====
        try:
            emit_event(PATIENT_DONE)
            logger.debug("Progress reported")
        except Exception as e:
            logger.warning(f"Error reporting progress: {e}")

        processed_files += 1
        logger.info(f"Successfully processed {file_context}")

    except Exception as e:
        logger.error(f"Failed to process {file_context}: {e}")
        failed_files.append((i, os.path.basename(input_file), str(e)))
        return processed_files
    return processed_files

def main(args):
    """Main function for PRE_ASO_CBCT preprocessing with comprehensive error handling."""
    try:
        logger.info("Starting PRE_ASO_CBCT preprocessing pipeline")
        
        # ===== ARGUMENT PARSING =====
        try:
            logger.debug("Parsing arguments")
            input_dir, out_dir, small_fov, is_dcm_input = (
                os.path.normpath(args.input[0]),
                os.path.normpath(args.output_folder[0]),
                args.SmallFOV[0] == "true",
                args.DCMInput[0] == "true",
            )
            logger.debug(f"Arguments parsed: input_dir={input_dir}, out_dir={out_dir}, SmallFOV={small_fov}")
        except Exception as e:
            logger.error(f"Error parsing arguments: {e}")
            raise

        # ===== INPUT VALIDATION =====
        try:
            if not os.path.exists(input_dir):
                logger.error(f"Input directory not found: {input_dir}")
                raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
            logger.debug("Input directory validated")
        except Exception as e:
            logger.error(f"Error validating input: {e}")
            raise

        # ===== DICOM CONVERSION =====
        if is_dcm_input:
            try:
                logger.debug("Converting DICOM files")
                convertdicom2nifti(input_dir)
                logger.info("DICOM conversion completed")
            except Exception as e:
                logger.error(f"Error converting DICOM files: {e}")
                raise

        # ===== OUTPUT DIRECTORY SETUP =====
        try:
            scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            logger.debug(f"Output directory ready: {out_dir}")
        except Exception as e:
            logger.error(f"Error setting up output directory: {e}")
            raise

        # ===== FILE DISCOVERY =====
        try:
            logger.debug("Discovering input files")
            input_files, _ = ExtractFilesFromFolder(input_dir, scan_extension)
            logger.info(f"Found {len(input_files)} input file(s)")
        except Exception as e:
            logger.error(f"Error discovering input files: {e}")
            raise

        # ===== MAIN PROCESSING LOOP =====
        processed_files = 0
        failed_files = []

        for i in range(len(input_files)):
            processed_files = _preprocess_one_file(failed_files, i, input_dir, input_files, out_dir, processed_files)

        # ===== FINAL REPORT =====
        try:
            logger.info(f"Preprocessing completed: {processed_files}/{len(input_files)} file(s) processed successfully")
            if failed_files:
                logger.warning(f"Failed to process {len(failed_files)} file(s):")
                for idx, filename, error in failed_files:
                    logger.warning(f"  {filename}: {error}")
        except Exception as e:
            logger.error(f"Error generating final report: {e}")

    except Exception as e:
        logger.error(f"Fatal error in main(): {e}")
        raise


if __name__ == "__main__":
    try:
        logger.info("PRE_ASO_CBCT entry point initiated")

        try:
            parser = argparse.ArgumentParser()

            parser.add_argument("input", nargs=1)
            parser.add_argument("output_folder", nargs=1)
            parser.add_argument("model_folder", nargs=1)
            parser.add_argument("SmallFOV", nargs=1)
            parser.add_argument("temp_folder", nargs=1)
            parser.add_argument("DCMInput", nargs=1)

            args = parser.parse_args()
            logger.debug("Arguments parsed successfully")
        except Exception as e:
            logger.error(f"Error parsing command line arguments: {e}")
            raise

        try:
            logger.info("Calling main() function")
            main(args)
            logger.info("PRE_ASO_CBCT completed successfully")
        except Exception as e:
            logger.error(f"Error in main() execution: {e}")
            raise

    except SystemExit as e:
        logger.info(f"Script exited with code: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.critical(f"Fatal error in entry point: {e}")
        sys.exit(f"Fatal error: {e}")
