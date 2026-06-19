import os
import logging
import sys
import numpy as np
import itk
import SimpleITK as sitk
import dicom2nifti

from ALI_CBCT_utils.io import search

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("ALI_CBCT_preprocess")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def CorrectHisto(filepath, outpath, min_porcent=0.01, max_porcent=0.95, i_min=-1500, i_max=4000):
    """
    Correct histogram of medical image with error handling.
    
    Parameters
    ----------
    filepath : str
        Path to input image file
    outpath : str
        Path to save output image
    min_porcent : float
        Minimum percentile
    max_porcent : float
        Maximum percentile
    i_min : int
        Minimum intensity value
    i_max : int
        Maximum intensity value
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Input file not found: {filepath}")
            raise FileNotFoundError(f"Input file does not exist: {filepath}")
        
        logger.debug(f"Reading image from {filepath}")
        input_img = sitk.ReadImage(filepath)
        input_img = sitk.Cast(input_img, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(input_img)

        img_min = np.min(img)
        img_max = np.max(img)
        img_range = img_max - img_min

        if img_range == 0:
            logger.warning(f"Image has zero range for {filepath}")
        
        definition = 1000
        histo = np.histogram(img, definition)
        cum = np.cumsum(histo[0])
        cum = cum - np.min(cum)
        cum = cum / np.max(cum)

        res_high = list(map(lambda i: i > max_porcent, cum)).index(True)
        res_max = (res_high * img_range) / definition + img_min

        res_low = list(map(lambda i: i > min_porcent, cum)).index(True)
        res_min = (res_low * img_range) / definition + img_min

        res_min = max(res_min, i_min)
        res_max = min(res_max, i_max)

        img = np.where(img > res_max, res_max, img)
        img = np.where(img < res_min, res_min, img)

        output = sitk.GetImageFromArray(img)
        output.SetSpacing(input_img.GetSpacing())
        output.SetDirection(input_img.GetDirection())
        output.SetOrigin(input_img.GetOrigin())
        output = sitk.Cast(output, sitk.sitkInt16)

        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(outpath)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(outpath)
        writer.Execute(output)
        logger.info(f"Histogram correction completed for {filepath}")
        return output
        
    except Exception as e:
        logger.error(f"Error correcting histogram for {filepath}: {e}")
        raise

def ResampleImage(input, size, spacing, origin, direction, interpolator, VectorImageType):
    """Resample image with error handling."""
    try:
        logger.debug("Starting image resampling")
        ResampleType = itk.ResampleImageFilter[VectorImageType, VectorImageType]

        resampleImageFilter = ResampleType.New()
        resampleImageFilter.SetOutputSpacing(spacing.tolist())
        resampleImageFilter.SetOutputOrigin(origin)
        resampleImageFilter.SetOutputDirection(direction)
        resampleImageFilter.SetInterpolator(interpolator)
        resampleImageFilter.SetSize(size)
        resampleImageFilter.SetInput(input)
        resampleImageFilter.Update()

        resampled_img = resampleImageFilter.GetOutput()
        logger.debug("Image resampling completed successfully")
        return resampled_img
    except Exception as e:
        logger.error(f"Error during image resampling: {e}")
        raise

def SetSpacing(filepath, output_spacing=[0.5, 0.5, 0.5], outpath=-1):
    """
    Set the spacing of the image at the wanted scale with error handling.

    Parameters
    ----------
    filepath : str
        Path of the image file
    output_spacing : list
        Wanted spacing of the new image file (default : [0.5, 0.5, 0.5])
    outpath : str
        Path to save the new image
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Input file not found: {filepath}")
            raise FileNotFoundError(f"Input file does not exist: {filepath}")
        
        logger.debug(f"Setting spacing for {filepath}")
        img = itk.imread(filepath)
        spacing = np.array(img.GetSpacing())
        output_spacing = np.array(output_spacing)

        if not np.array_equal(spacing, output_spacing):
            size = itk.size(img)
            scale = spacing / output_spacing
            output_size = (np.array(size) * scale).astype(int).tolist()
            output_origin = img.GetOrigin()

            # Find new origin
            output_physical_size = np.array(output_size) * np.array(output_spacing)
            input_physical_size = np.array(size) * spacing
            output_origin = np.array(output_origin) - (output_physical_size - input_physical_size) / 2.0

            img_info = itk.template(img)[1]
            pixel_type = img_info[0]
            pixel_dimension = img_info[1]

            VectorImageType = itk.Image[pixel_type, pixel_dimension]

            if True in [seg in os.path.basename(filepath) for seg in ["seg", "Seg"]]:
                InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]
            else:
                InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]

            interpolator = InterpolatorType.New()
            resampled_img = ResampleImage(img, output_size, output_spacing, output_origin, img.GetDirection(), interpolator, VectorImageType)

            if outpath != -1:
                out_dir = os.path.dirname(outpath)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                itk.imwrite(resampled_img, outpath)
                logger.info(f"Resampled image saved to {outpath}")
            return resampled_img
        else:
            logger.debug(f"Spacing already matches target for {filepath}")
            if outpath != -1:
                out_dir = os.path.dirname(outpath)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                itk.imwrite(img, outpath)
            return img
    except Exception as e:
        logger.error(f"Error setting spacing for {filepath}: {e}")
        raise

def convertdicom2nifti(input_folder, output_folder=None):
    """
    Convert DICOM files to NIFTI format with error handling.
    
    Parameters
    ----------
    input_folder : str
        Path to input folder containing DICOM files
    output_folder : str, optional
        Path to output folder (default: NIFTI subfolder in input_folder)
    """
    try:
        if not os.path.exists(input_folder):
            logger.error(f"Input folder not found: {input_folder}")
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        
        logger.info(f"Starting DICOM to NIFTI conversion from {input_folder}")
        
        patients_folders = [
            folder for folder in os.listdir(input_folder) 
            if os.path.isdir(os.path.join(input_folder, folder)) and folder != 'NIFTI'
        ]

        if not patients_folders:
            logger.warning(f"No patient folders found in {input_folder}")
            return

        if output_folder is None:
            output_folder = os.path.join(input_folder, 'NIFTI')

        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                logger.info(f"Created output folder: {output_folder}")
            except Exception as e:
                logger.error(f"Failed to create output folder {output_folder}: {e}")
                raise

        for patient in patients_folders:
            try:
                output_file = os.path.join(output_folder, patient + ".nii.gz")
                if os.path.exists(output_file):
                    logger.debug(f"Output file already exists, skipping: {output_file}")
                    continue
                
                current_directory = os.path.join(input_folder, patient)
                logger.debug(f"Converting patient folder: {current_directory}")
                
                try:
                    reader = sitk.ImageSeriesReader()
                    sitk.ProcessObject_SetGlobalWarningDisplay(False)
                    dicom_names = reader.GetGDCMSeriesFileNames(current_directory)
                    
                    if not dicom_names:
                        logger.warning(f"No DICOM files found in {current_directory}, trying alternative method")
                        dicom2nifti.convert_directory(current_directory, output_folder)
                        nifti_file = search(output_folder, 'nii.gz')['nii.gz'][0]
                        os.rename(nifti_file, output_file)
                    else:
                        reader.SetFileNames(dicom_names)
                        image = reader.Execute()
                        sitk.ProcessObject_SetGlobalWarningDisplay(True)
                        sitk.WriteImage(image, output_file)
                    
                    logger.info(f"Successfully converted: {patient}")
                    
                except Exception as e:
                    logger.warning(f"Standard conversion failed for {patient}, trying alternative method: {e}")
                    try:
                        dicom2nifti.convert_directory(current_directory, output_folder)
                        nifti_file = search(output_folder, 'nii.gz')['nii.gz'][0]
                        os.rename(nifti_file, output_file)
                        logger.info(f"Successfully converted with alternative method: {patient}")
                    except Exception as alt_e:
                        logger.error(f"Failed to convert patient {patient}: {alt_e}")
                        
            except Exception as e:
                logger.error(f"Error processing patient {patient}: {e}")
                continue
        
        logger.info("DICOM to NIFTI conversion completed")
        
    except Exception as e:
        logger.error(f"Fatal error during DICOM conversion: {e}")
        raise