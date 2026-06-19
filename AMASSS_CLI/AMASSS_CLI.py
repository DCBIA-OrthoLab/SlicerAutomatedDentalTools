#!/usr/bin/env python3
"""
AMASSS_CLI.py – Adaptation for nnUNet v2 (MAX, MAND, CB)
"""
import argparse
import time, os, sys, glob, subprocess, shutil
import numpy as np
import torch, itk, cc3d, dicom2nifti
import SimpleITK as sitk
import vtk
import re
import vtk
import logging

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("AMASSS_CLI")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSLATE = {
  "Mandible":"MAND","Maxilla":"MAX","Cranial-base":"CB",
  "Cervical-vertebra":"CV","Root-canal":"RC","Mandibular-canal":"MCAN",
  "Upper-airway":"UAW","Skin":"SKIN","Teeth":"TEETH",
  "Cranial Base (Mask)":"CBMASK","Mandible (Mask)":"MANDMASK","Maxilla (Mask)":"MAXMASK",
}
NTRANSLATE = {v:v for v in TRANSLATE.values()}

LABELS = {
    "LARGE":{"MAND":1,"CB":2,"UAW":3,"MAX":4,"CV":5,"SKIN":6,"CBMASK":7,"MANDMASK":8,"MAXMASK":9},
    "SMALL":{"MAND":1,"RC":2,"MAX":4},
}
LABEL_COLORS = {1:[216,101,79],2:[128,174,128],3:[0,0,0],4:[230,220,70],5:[111,184,210],6:[172,122,101]}
NAMES_FROM_LABELS = {"LARGE":{}, "SMALL":{}}
for g,d in LABELS.items():
    for k,v in d.items():
        NAMES_FROM_LABELS[g][v] = k

def wait_for_stable_output(process, output_path, timeout_s=900, poll_s=1.0, stable_checks=3, min_size_bytes=262144):
    """Wait until process exits or output file exists with stable, significant size."""
    start_t = time.time()
    last_size = -1
    stable_count = 0

    while True:
        rc = process.poll()

        if os.path.isfile(output_path):
            try:
                cur_size = os.path.getsize(output_path)
            except OSError:
                cur_size = -1

            if cur_size >= min_size_bytes:
                if cur_size == last_size:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_size = cur_size

                if stable_count >= stable_checks:
                    logger.info(
                        f"Stable output detected ({cur_size} bytes) at {output_path}"
                    )
                    return True
            else:
                stable_count = 0
                last_size = cur_size

        if rc is not None:
            if rc != 0:
                raise subprocess.CalledProcessError(rc, process.args)

            if os.path.isfile(output_path):
                try:
                    final_size = os.path.getsize(output_path)
                except OSError:
                    final_size = -1
                logger.info(
                    f"Prediction process exited cleanly; output found ({final_size} bytes)"
                )
                return True

            raise FileNotFoundError(f"Process exited but output file is missing: {output_path}")

        if (time.time() - start_t) > timeout_s:
            raise TimeoutError(
                f"Timeout waiting for stable output file: {output_path}"
            )

        time.sleep(poll_s)

MODELS_GROUP = {
    "LARGE":{
        "FF":     {"MAND":1,"CB":2,"UAW":3,"MAX":4,"CV":5},
        "SKIN":   {"SKIN":1},
        "CBMASK": {"CBMASK":1},
        "MANDMASK":{"MANDMASK":1},
        "MAXMASK":{"MAXMASK":1},
    },
    "SMALL":{
        "HD-MAND":{"MAND":1},
        "HD-MAX": {"MAX":1},
        "RC":     {"RC":1},
    },
}

def CorrectHisto(filepath, outpath, min_porcent=0.01, max_porcent=0.95, i_min=-1500, i_max=4000):
    """Correct histogram of image with error handling."""
    try:
        if not os.path.exists(filepath):
            logger.error(f"Input file not found: {filepath}")
            raise FileNotFoundError(f"File does not exist: {filepath}")
        
        logger.debug(f"Correcting scan contrast: {filepath}")
        
        img = sitk.Cast(sitk.ReadImage(filepath), sitk.sitkFloat32)
        logger.debug(f"Successfully corrected histogram for {filepath}")
        return img
    except Exception as e:
        logger.error(f"Error correcting histogram: {e}")
        raise

def Write(vtkdata, output_name):
    """Write VTK data with error handling."""
    try:
        if not vtkdata:
            logger.error("VTK data is None")
            raise ValueError("Invalid VTK data")
        
        logger.debug(f"Writing VTK file: {output_name}")
        polydatawriter = vtk.vtkPolyDataWriter()
        polydatawriter.SetFileName(output_name)
        polydatawriter.SetInputData(vtkdata)
        polydatawriter.Write()
        
        logger.info(f"Successfully wrote VTK file: {output_name}")
    except Exception as e:
        logger.error(f"Error writing VTK file {output_name}: {e}")
        raise

def SavePredToVTK(file_path, temp_folder, smoothing, vtk_output_path, model_size="LARGE"):
    """Save prediction to VTK with error handling."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        logger.debug(f"Converting prediction to VTK: {file_path}")

        img = sitk.ReadImage(file_path)
        arr = sitk.GetArrayFromImage(img)

        base = os.path.basename(file_path)
        for ext in ('.nii.gz', '.nrrd.gz', '.nii', '.nrrd'):
            if base.endswith(ext):
                base = base[:-len(ext)]
                break

        is_merged = base.endswith("_MERGED")

        output_is_dir = vtk_output_path.endswith(os.sep) or os.path.isdir(vtk_output_path)
        if output_is_dir:
            os.makedirs(vtk_output_path, exist_ok=True)

        def write_poly(poly, outvtk):
            try:
                w = vtk.vtkPolyDataWriter()
                w.SetFileName(outvtk)
                w.SetInputData(poly)
                w.Write()
                logger.info(f"Written VTK: {outvtk}")
            except Exception as e:
                logger.error(f"Error writing VTK file: {e}")
                raise

        def mesh_from_nrrd(nrrd, iters, color_rgb):
            try:
                r = vtk.vtkNrrdReader()
                r.SetFileName(nrrd)
                r.Update()
                dmc = vtk.vtkDiscreteMarchingCubes()
                dmc.SetInputConnection(r.GetOutputPort())
                dmc.GenerateValues(1, 1, 1)
                s = vtk.vtkSmoothPolyDataFilter()
                s.SetInputConnection(dmc.GetOutputPort())
                s.SetNumberOfIterations(iters)
                s.Update()
                poly = s.GetOutput()
                cols = vtk.vtkUnsignedCharArray()
                cols.SetName("Colors")
                cols.SetNumberOfComponents(3)
                cols.SetNumberOfTuples(poly.GetNumberOfCells())
                for i in range(poly.GetNumberOfCells()):
                    cols.SetTuple(i, color_rgb)
                poly.GetCellData().SetScalars(cols)
                return poly
            except Exception as e:
                logger.error(f"Error creating mesh from NRRD: {e}")
                raise

        # MODE MERGED
        if is_merged:
            logger.debug("Creating merged segmentation")
            append = vtk.vtkAppendPolyData()
            for label in sorted(np.unique(arr)):
                if label == 0:
                    continue
                try:
                    struct = NAMES_FROM_LABELS[model_size][label]

                    tmp_nrrd = os.path.join(temp_folder, f"temp.nrrd")
                    mask = (arr == label).astype(np.uint8)
                    img2 = sitk.GetImageFromArray(mask)
                    img2.CopyInformation(img)
                    sitk.WriteImage(img2, tmp_nrrd)
                    color = LABEL_COLORS.get(label, [255, 255, 255])
                    mesh = mesh_from_nrrd(tmp_nrrd, smoothing, color)
                    append.AddInputData(mesh)
                except Exception as e:
                    logger.warning(f"Error processing label {label}: {e}")
                    continue
            
            append.Update()
            merged_poly = append.GetOutput()

            outname = f"{base}.vtk"
            if output_is_dir:
                outvtk = os.path.join(vtk_output_path, outname)
            else:
                root, _ = os.path.splitext(vtk_output_path)
                outvtk = f"{root}_{outname}"
                os.makedirs(os.path.dirname(outvtk), exist_ok=True)
            write_poly(merged_poly, outvtk)
            return

        # MODE SEPARATE
        logger.debug("Creating separate segmentation")
        struct = base.split('_')[-1]

        tmp_nrrd = os.path.join(temp_folder, f"temp.nrrd")
        m = (arr > 0).astype(np.uint8)
        i2 = sitk.GetImageFromArray(m)
        i2.CopyInformation(img)
        sitk.WriteImage(i2, tmp_nrrd)

        label_index = LABELS[model_size][struct]
        color = LABEL_COLORS.get(label_index, [255, 255, 255])

        poly = mesh_from_nrrd(tmp_nrrd, smoothing, color)
        outname = f"{base}.vtk"
        outvtk = os.path.join(vtk_output_path, outname) if output_is_dir else os.path.join(os.path.dirname(vtk_output_path), outname)
        write_poly(poly, outvtk)
        logger.info("VTK export completed successfully")
    except Exception as e:
        logger.error(f"Error in SavePredToVTK: {e}")
        raise

def SetSpacingFromRef(filepath, refFile, interpolator="NearestNeighbor", outpath=-1):
    """Set spacing from reference file with error handling."""
    try:
        if not os.path.exists(filepath):
            logger.error(f"Input file not found: {filepath}")
            raise FileNotFoundError(f"File does not exist: {filepath}")
        
        if not os.path.exists(refFile):
            logger.error(f"Reference file not found: {refFile}")
            raise FileNotFoundError(f"Reference file does not exist: {refFile}")
        
        logger.debug(f"Setting spacing from reference: {refFile}")
        
        img = itk.imread(filepath)
        ref = itk.imread(refFile)
        sp_i, sz_i = np.array(img.GetSpacing()), np.array(itk.size(img))
        sp_r, sz_r = np.array(ref.GetSpacing()), np.array(itk.size(ref))
        
        if not np.allclose(sp_i, sp_r) or not np.array_equal(sz_i, sz_r):
            PixelType = itk.template(img)[1][0]
            Dim = 3
            IVec = itk.Image[PixelType, Dim]
            interp = itk.NearestNeighborInterpolateImageFunction[IVec, itk.D].New() \
                     if interpolator == "NearestNeighbor" else itk.LinearInterpolateImageFunction[IVec, itk.D].New()
            res = itk.ResampleImageFilter[IVec, IVec].New(
                Input=img,
                OutputSpacing=sp_r.tolist(), 
                OutputOrigin=ref.GetOrigin(),
                OutputDirection=ref.GetDirection(), 
                Interpolator=interp,
                Size=sz_r.tolist()
            )
            res.Update()
            out = sitk.GetImageFromArray(itk.GetArrayFromImage(res.GetOutput()))
            out.CopyInformation(sitk.ReadImage(refFile))
        else:
            out = sitk.ReadImage(filepath)
        
        out = sitk.Cast(out, sitk.sitkInt16)
        if outpath != -1:
            sitk.WriteImage(out, outpath)
            logger.debug(f"Spacing set and saved to {outpath}")
        return out
    except Exception as e:
        logger.error(f"Error setting spacing: {e}")
        raise

def CleanArray(seg_arr, radius):
    """Clean segmentation array using morphological operations with error handling."""
    try:
        if seg_arr is None:
            logger.error("Input segmentation array is None")
            raise ValueError("Segmentation array is None")
        
        if seg_arr.size == 0:
            logger.error("Input segmentation array is empty")
            raise ValueError("Segmentation array is empty")
        
        logger.debug(f"Cleaning array with radius: {radius}")
        
        try:
            img = sitk.GetImageFromArray(seg_arr.astype(np.uint8))
            logger.debug("Converted array to SimpleITK image")
        except Exception as e:
            logger.error(f"Error converting array to SimpleITK image: {e}")
            raise
        
        try:
            img = sitk.BinaryDilate(img, [radius]*3)
            logger.debug("Completed binary dilation")
        except Exception as e:
            logger.error(f"Error during binary dilation: {e}")
            raise
        
        try:
            img = sitk.BinaryFillhole(img)
            logger.debug("Completed binary fill hole")
        except Exception as e:
            logger.error(f"Error during binary fill hole: {e}")
            raise
        
        try:
            img = sitk.BinaryErode(img, [radius]*3)
            logger.debug("Completed binary erosion")
        except Exception as e:
            logger.error(f"Error during binary erosion: {e}")
            raise
        
        try:
            arr = sitk.GetArrayFromImage(img)
            logger.debug("Converted SimpleITK image back to array")
        except Exception as e:
            logger.error(f"Error converting SimpleITK image to array: {e}")
            raise
        
        try:
            cc, n = cc3d.connected_components(arr, return_N=True)
            logger.debug(f"Found {n} connected components")
            
            if n > 1:
                sizes = [(cc==i).sum() for i in range(1, n+1)]
                max_idx = int(np.argmax(sizes))
                arr = (cc == (1 + max_idx)).astype(np.uint8)
                logger.debug(f"Selected largest connected component with {sizes[max_idx]} voxels")
        except Exception as e:
            logger.error(f"Error during connected components analysis: {e}")
            raise
        
        logger.debug("Array cleaning completed successfully")
        return arr
    except Exception as e:
        logger.error(f"Error in CleanArray: {e}")
        raise

def CropSkin(skin_seg_arr,thickness):
    img = sitk.GetImageFromArray(skin_seg_arr.astype(np.uint8))
    fill = sitk.BinaryFillhole(img)
    ero = sitk.BinaryErode(fill,[thickness]*3)
    arr = sitk.GetArrayFromImage(fill)
    earr = sitk.GetArrayFromImage(ero)
    crop = np.where(earr==1,0,arr)
    cc, n = cc3d.connected_components(crop,return_N=True)
    if n>1:
        sizes=[(cc==i).sum() for i in range(1,n+1)]
        crop=(cc==(1+int(np.argmax(sizes)))).astype(np.uint8)
    return crop

def SavePrediction(img,ref_filepath,outpath,output_spacing):
    ref = sitk.ReadImage(ref_filepath)
    out = sitk.GetImageFromArray(img.astype(np.int16))
    out.SetSpacing(output_spacing)
    out.SetDirection(ref.GetDirection())
    out.SetOrigin(ref.GetOrigin())
    sitk.WriteImage(out, outpath)

def SaveSeg(file_path, spacing, seg_arr, input_path, temp_path, outputdir, temp_folder, save_vtk, smoothing=5, model_size="LARGE"):
    """Save segmentation with error handling for prediction, spacing correction, and VTK conversion."""
    try:
        logger.debug(f"Saving segmentation for: {file_path}")
        
        # Step 1: Save prediction
        try:
            logger.debug(f"Step 1: Saving prediction with spacing {spacing}")
            SavePrediction(seg_arr, input_path, temp_path, output_spacing=spacing)
            logger.info("Prediction saved successfully")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            raise
        
        # Step 2: Set spacing from reference
        try:
            logger.debug(f"Step 2: Setting spacing from reference image: {input_path}")
            SetSpacingFromRef(
                temp_path,
                input_path,
                outpath=file_path
            )
            logger.info(f"Spacing corrected and saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error setting spacing: {e}")
            raise
        
        # Step 3: Save VTK mesh if requested
        if save_vtk:
            try:
                logger.debug(f"Step 3: Converting to VTK with smoothing: {smoothing}")
                SavePredToVTK(file_path, temp_folder, smoothing, vtk_output_path=outputdir)
                logger.info("VTK mesh saved successfully")
            except Exception as e:
                logger.error(f"Error saving VTK mesh: {e}")
                raise
        
        logger.info(f"Segmentation saving completed successfully for: {file_path}")
    except Exception as e:
        logger.error(f"Error in SaveSeg: {e}")
        raise
# -- Main adapt for nnUNet v2 ---
def main(args):
    try:
        logger.info("Starting AMASSS_CLI with nnUNet v2 backend")
        
        # ===== SETUP PHASE =====
        try:
            logger.debug("Initializing temporary folder and output setup")
            
            tmp = args["temp_fold"]
            base_output = args["output_folder"]
            
            try:
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp, exist_ok=True)
                logger.debug(f"Temporary folder created: {tmp}")
            except Exception as e:
                logger.error(f"Error creating temporary folder: {e}")
                raise
        except Exception as e:
            logger.error(f"Error during setup phase: {e}")
            raise
        
        # ===== INPUT FILE DISCOVERY PHASE =====
        try:
            logger.debug("Discovering input files")
            input_path = args["inputVolume"]
            extensions = (".nii", ".nii.gz", ".nrrd", ".nrrd.gz")
            
            if not os.path.exists(input_path):
                logger.error(f"Input path does not exist: {input_path}")
                raise FileNotFoundError(f"Input path not found: {input_path}")
            
            if os.path.isdir(input_path):
                logger.debug(f"Input is directory, scanning for volume files")
                input_files = []
                for f in os.listdir(input_path):
                    file = os.path.join(input_path, f)
                    if f.lower().endswith(extensions):
                        if 'MASK' not in f:
                            input_files.append(file)
                            logger.debug(f"Found input file: {file}")
            else:
                if not input_path.lower().endswith(extensions):
                    logger.warning(f"Input file has unexpected extension: {input_path}")
                input_files = [input_path]
                logger.debug(f"Single input file: {input_path}")
            
            scan_count = len(input_files)
            if scan_count == 0:
                logger.error("No valid input files found")
                sys.exit(1)
            
            logger.info(f"Found {scan_count} input file(s)")
        except Exception as e:
            logger.error(f"Error during input file discovery: {e}")
            raise

        start_time = time.time()
        print("<filter-start><filter-name>AMASSS</filter-name></filter-start>", flush=True)
        sys.stdout.flush()

        # ===== MAIN PROCESSING LOOP =====
        processed_scans = 0
        failed_scans = []
        
        for scan_idx, volume_file in enumerate(input_files, start=1):
            scan_context = f"scan {scan_idx}/{scan_count}: {os.path.basename(volume_file)}"
            logger.info(f"Starting processing of {scan_context}")
            
            try:
                # --- FILE PREPARATION ---
                try:
                    case_id = f"{scan_idx:03d}"
                    basename = os.path.basename(volume_file)
                    base, ext = os.path.splitext(basename)
                    if ext == ".gz":
                        base, ext2 = os.path.splitext(base)
                        ext = ext2 + ext
                    logger.debug(f"File parsed: base={base}, ext={ext}, case_id={case_id}")
                except Exception as e:
                    logger.error(f"Error parsing filename for {volume_file}: {e}")
                    raise

                # --- OUTPUT DIRECTORY ---
                try:
                    if args.get("save_in_folder"):
                        outdir = os.path.join(base_output, f"{base}_{args['prediction_ID']}_SegOut")
                    else:
                        outdir = base_output
                    os.makedirs(outdir, exist_ok=True)
                    logger.debug(f"Output directory set to: {outdir}")
                except Exception as e:
                    logger.error(f"Error creating output directory: {e}")
                    raise

                # --- INPUT FILE COPY ---
                try:
                    tmp_name = f"p_{case_id}_0000.nii.gz"
                    input_vol = os.path.join(tmp, tmp_name)
                    shutil.copy(volume_file, input_vol)
                    logger.debug(f"Input volume copied to: {input_vol}")
                except Exception as e:
                    logger.error(f"Error copying input volume: {e}")
                    raise

                # --- MODEL DISCOVERY ---
                try:
                    logger.debug("Searching for nnUNet models")
                    nnunet_models = {}
                    for struct in args["skullStructure"].split(","):
                        try:
                            root = os.path.join(args["modelDirectory"], struct)
                            if not os.path.exists(root):
                                logger.warning(f"Model directory not found for structure {struct}: {root}")
                                continue
                            
                            pattern = os.path.join(root, "**", "*__nnUNetPlans__3d_fullres")
                            plans = glob.glob(pattern, recursive=True)
                            if plans:
                                nnunet_models[struct] = plans[0]
                                logger.debug(f"Found model for {struct}: {plans[0]}")
                            else:
                                logger.warning(f"No model found for structure {struct}")
                        except Exception as e:
                            logger.warning(f"Error searching model for {struct}: {e}")
                            continue
                    
                    if not nnunet_models:
                        logger.error("No models found for any structure")
                        raise FileNotFoundError("No nnUNet models found for specified structures")
                    
                    logger.info(f"Found {len(nnunet_models)} models to process")
                except Exception as e:
                    logger.error(f"Error during model discovery: {e}")
                    raise

                # --- PREDICTIONS LOOP ---
                total_struct = len(nnunet_models)
                total_steps = scan_count * total_struct
                prediction_segmentation = {}

                logger.debug("Starting predictions for all structures")
                
                for struct_idx, (struct, plans_dir) in enumerate(nnunet_models.items(), start=1):
                    struct_context = f"structure {struct_idx}/{total_struct}: {struct}"
                    logger.info(f"Processing {struct_context}")
                    
                    try:
                        # Setup nnUNet environment
                        try:
                            dataset_name = os.path.basename(os.path.dirname(plans_dir))
                            os.environ['nnUNet_results'] = os.path.dirname(os.path.dirname(plans_dir))
                            outp = os.path.join(tmp, f"pred_{struct}")
                            os.makedirs(outp, exist_ok=True)
                            logger.debug(f"nnUNet output directory: {outp}")
                        except Exception as e:
                            logger.error(f"Error setting up nnUNet environment: {e}")
                            raise

                        # Find checkpoint
                        try:
                            checkpoint = os.path.join(plans_dir, "fold_0", "checkpoint_final.pth")
                            if not os.path.isfile(checkpoint):
                                logger.error(f"Checkpoint not found for {struct}: {checkpoint}")
                                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
                            logger.debug(f"Checkpoint found: {checkpoint}")
                        except Exception as e:
                            logger.error(f"Error locating checkpoint: {e}")
                            raise

                        # Run nnUNet prediction
                        try:
                            if torch.cuda.is_available():
                                device = "cuda"
                            else:
                                device = "cpu"
                                
                            logger.info(f"Predicting {struct} on device: {device}")
                            
                            cmd = [
                                "nnUNetv2_predict",
                                "-i", tmp,
                                "-o", outp,
                                "-d", dataset_name,
                                "-c", "3d_fullres",
                                "-f", "0",
                                "-device", device,
                                "--disable_tta",
                            ]
                            print(cmd)

                            nifti_pred = os.path.join(outp, f"p_{case_id}.nii.gz")

                            # Start prediction and keep checking output stability to avoid waiting only on process signal.
                            proc = subprocess.Popen(cmd, stdout=None, stderr=None, close_fds=True)
                            wait_for_stable_output(
                                proc,
                                nifti_pred,
                                timeout_s=3600,
                                poll_s=1.0,
                                stable_checks=3,
                                min_size_bytes=100,
                            )

                            if proc.poll() is None:
                                logger.info("Stable output reached before process exit; stopping predictor process")
                                proc.terminate()
                                try:
                                    proc.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait(timeout=5)

                            logger.info(f"Prediction for {struct} completed")
                        except subprocess.CalledProcessError as e:
                            logger.error(f"nnUNet prediction failed for {struct}: {e}")
                            raise
                        except Exception as e:
                            logger.error(f"Error executing nnUNet prediction: {e}")
                            raise

                        # Report progress
                        try:
                            step = (scan_idx - 1) * total_struct + struct_idx
                            fraction = step / total_steps
                            print(f"<filter-progress>{fraction:.4f}</filter-progress>", flush=True)
                            sys.stdout.flush()
                            logger.debug(f"Progress: {fraction:.4f}")
                        except Exception as e:
                            logger.warning(f"Error reporting progress: {e}")

                        # Load prediction
                        try:
                            if not os.path.isfile(nifti_pred):
                                logger.error(f"Prediction output not found: {nifti_pred}")
                                raise FileNotFoundError(f"nnUNet output file not found: {nifti_pred}")
                            
                            img = sitk.ReadImage(nifti_pred)
                            arr = sitk.GetArrayFromImage(img)
                            mask = (arr > 0).astype(np.uint8)
                            prediction_segmentation[struct] = mask
                            logger.debug(f"Loaded prediction mask for {struct}")
                        except Exception as e:
                            logger.error(f"Error loading prediction for {struct}: {e}")
                            raise
                    
                    except Exception as e:
                        logger.error(f"Error processing {struct_context}: {e}")
                        raise

                # --- SEGMENTATION SAVING PHASE ---
                try:
                    logger.debug("Starting segmentation saving")
                    
                    try:
                        spacing = list(sitk.ReadImage(volume_file).GetSpacing())
                        logger.debug(f"Image spacing: {spacing}")
                    except Exception as e:
                        logger.error(f"Error reading image spacing: {e}")
                        raise

                    # SEPARATE mode
                    if "SEPARATE" in args["merge"] or len(prediction_segmentation) == 1:
                        try:
                            logger.debug("Saving separate segmentations")
                            for struct, mask in prediction_segmentation.items():
                                try:
                                    outfn = os.path.join(outdir, f"{base}_{args['prediction_ID']}_{struct}{ext}")
                                    SaveSeg(
                                        outfn, spacing, mask, volume_file,
                                        os.path.join(tmp, "tmp.nii.gz"),
                                        outdir, tmp, args["genVtk"], args["vtk_smooth"], "LARGE"
                                    )
                                    logger.info(f"Saved segmentation for {struct}")
                                except Exception as e:
                                    logger.error(f"Error saving segmentation for {struct}: {e}")
                                    raise
                        except Exception as e:
                            logger.error(f"Error in separate segmentation mode: {e}")
                            raise

                    # MERGE mode
                    if "MERGE" in args["merge"] and len(prediction_segmentation) > 1:
                        try:
                            logger.debug("Merging segmentations")
                            
                            shape = next(iter(prediction_segmentation.values())).shape
                            merged = np.zeros(shape, dtype=np.int16)
                            
                            for struct in args["merging_order"]:
                                if struct in prediction_segmentation:
                                    lbl = LABELS["LARGE"].get(struct, 1)
                                    merged = np.where(prediction_segmentation[struct] == 1, lbl, merged)
                                    logger.debug(f"Merged {struct} with label {lbl}")
                            
                            outfn = os.path.join(outdir, f"{base}_{args['prediction_ID']}_MERGED{ext}")
                            SaveSeg(
                                outfn, spacing, merged, volume_file,
                                os.path.join(tmp, "tmp.nii.gz"),
                                outdir, tmp, args["genVtk"], args["vtk_smooth"], "LARGE"
                            )
                            logger.info("Merged segmentation saved")
                        except Exception as e:
                            logger.error(f"Error in merge segmentation mode: {e}")
                            raise

                    logger.info(f"Segmentation saving completed for {scan_context}")
                except Exception as e:
                    logger.error(f"Error during segmentation saving: {e}")
                    raise

                # --- CLEANUP ---
                try:
                    shutil.rmtree(tmp, ignore_errors=True)
                    os.makedirs(tmp, exist_ok=True)
                    logger.debug("Temporary files cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary files: {e}")

                processed_scans += 1
                logger.info(f"Successfully processed {scan_context}")
            
            except Exception as e:
                logger.error(f"Failed to process {scan_context}: {e}")
                failed_scans.append((scan_idx, volume_file, str(e)))
                if scan_idx < scan_count:
                    logger.info("Continuing with next scan")
                    continue
                else:
                    raise

        # --- FINAL REPORT ---
        try:
            elapsed = time.time() - start_time
            logger.info(f"Processing completed in {elapsed:.2f}s")
            print(f"<filter-end><filter-name>AMASSS</filter-name><filter-time>{elapsed:.2f}</filter-time></filter-end>", flush=True)
            sys.stdout.flush()
            
            logger.info(f"Processed {processed_scans}/{scan_count} scans successfully")
            if failed_scans:
                logger.warning(f"Failed to process {len(failed_scans)} scan(s)")
                for idx, path, err in failed_scans:
                    logger.warning(f"  Scan {idx} ({os.path.basename(path)}): {err}")
            
            if processed_scans == scan_count:
                logger.info("All scans processed successfully.")
            else:
                logger.warning(f"Processing completed with {len(failed_scans)} failure(s).", flush=True)
        except Exception as e:
            logger.error(f"Error generating final report: {e}")

    except Exception as e:
        logger.error(f"Fatal error in main(): {e}")
        sys.exit(f"Processing failed: {e}")

if __name__=="__main__":
    try:
        logger.info("AMASSS_CLI entry point initiated")
        logger.debug(f"Command line arguments: {sys.argv}")
        
        try:
            argv = sys.argv
            if len(argv) < 13:
                logger.error(f"Insufficient arguments provided: {len(argv)} (expected 13)")
                raise ValueError(f"Expected 13 arguments, got {len(argv)-1}")
            
            logger.debug("Parsing command line arguments")
            args = {
                "inputVolume":    argv[1],
                "modelDirectory": argv[2],
                "skullStructure": argv[3],
                "merge":          re.split(r'[, ]+', argv[4].strip()),  
                "genVtk":         argv[5].lower()=="true",
                "save_in_folder": argv[6].lower()=="true",
                "output_folder":  argv[7],
                "vtk_smooth":     int(argv[8]),
                "prediction_ID":  argv[9],        
                "temp_fold":      argv[10],
                "isSegmentInput": argv[11].lower()=="true",
                "isDCMInput":     argv[12].lower()=="true",
                "merging_order":  ["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC","CBMASK","MANDMASK","MAXMASK"],
            }
            logger.info("Arguments parsed successfully")
            logger.debug(f"Parsed arguments: {args}")
        except ValueError as e:
            logger.error(f"Argument parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing arguments: {e}")
            raise
        
        try:
            logger.info("Calling main() function")
            main(args)
            logger.info("AMASSS_CLI completed successfully")
        except Exception as e:
            logger.error(f"Error in main() execution: {e}")
            raise
    
    except SystemExit as e:
        logger.info(f"Script exited with code: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.critical(f"Fatal error in entry point: {e}")
        sys.exit(f"Fatal error: {e}")