#!/usr/bin/env python3
"""
AMASSS_CLI.py – Adaptation for nnUNet v2 (MAX, MAND, CB)
"""
import argparse
import inspect
import time, os, sys, glob, shutil
import numpy as np
import torch, cc3d, dicom2nifti
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

# ---------------------------------------------------------------------------
# nnUNet v2 inference, called through the Python API instead of the
# `nnUNetv2_predict` command line. Ported from the cloud version of AMASSS
# (sadt_amasss/nnunet_runner.py), which fixes three defects of the subprocess
# version this file used to have:
#
# 1. No `nnUNet_results` environment variable. The CLI set it before spawning
#    `nnUNetv2_predict`, and `os.environ` is process-global: two overlapping
#    AMASSS runs would overwrite each other's model path.
#    `initialize_from_trained_model_folder` takes an explicit path instead.
# 2. No output-file polling. The CLI killed the predictor once the output file
#    stopped growing for three seconds, which could interrupt nnUNet
#    mid-postprocessing. The Python API simply returns when it is done.
# 3. The resampling runs on the GPU too (see EnableGpuResampling), which is
#    where the run time actually went -- the network was an eighth of it.
#
# In-process also means the checkpoint is loaded once per structure rather than
# once per (scan x structure): no process start-up and no model load per scan.
#
# nnunetv2 is imported lazily inside these functions: the CPU path never needs
# the resampler module, and an installation without nnunetv2 should fail where
# it is used, with a message about inference, rather than at import time.
# ---------------------------------------------------------------------------

CHECKPOINT_NAME = "checkpoint_final.pth"
PLANS_FOLDER_PATTERN = "*__nnUNetPlans__3d_fullres"


def ResolveDevice():
    """Return the device to actually use, falling back to CPU when needed."""
    if torch.cuda.is_available():
        return "cuda"
    logger.info("No CUDA device available; running nnUNet on CPU")
    return "cpu"


def FindModelFolder(model_root, structure_code):
    """Locate the trained nnUNet folder for one structure, or None.

    Layout expected under the model bundle:
        <model_root>/<CODE>/**/<Dataset...>__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth

    A candidate is only accepted once its fold_0 checkpoint is confirmed
    present, so a half-copied bundle degrades to "this structure is
    unavailable" rather than crashing minutes into the run.
    """
    structure_root = os.path.join(model_root, structure_code)
    if not os.path.isdir(structure_root):
        return None

    pattern = os.path.join(structure_root, "**", PLANS_FOLDER_PATTERN)
    for candidate in sorted(glob.glob(pattern, recursive=True)):
        if os.path.isfile(os.path.join(candidate, "fold_0", CHECKPOINT_NAME)):
            return candidate

    # Also accept the plans folder being the structure folder itself.
    if os.path.isfile(os.path.join(structure_root, "fold_0", CHECKPOINT_NAME)):
        return structure_root
    return None


def BuildPredictor(device, tile_step_size=0.5):
    """Instantiate an nnUNetPredictor, tolerating nnUNet's renamed kwargs.

    nnUNet 2.x renamed `perform_everything_on_gpu` to
    `perform_everything_on_device` mid-series; passing whichever the installed
    version declares keeps this working across the range.
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    options = {
        "tile_step_size": float(tile_step_size),
        "use_gaussian": True,
        # Equivalent to the command line's --disable_tta: no test-time mirroring.
        "use_mirroring": False,
        "device": torch.device(device),
        "verbose": False,
        "verbose_preprocessing": False,
        "allow_tqdm": False,
    }
    accepted = set(inspect.signature(nnUNetPredictor.__init__).parameters)
    for name in ("perform_everything_on_device", "perform_everything_on_gpu"):
        if name in accepted:
            options[name] = device.startswith("cuda")
            break

    return nnUNetPredictor(**{k: v for k, v in options.items() if k in accepted})


# The resampler nnUNet's own plans name by default, and the only one we are
# willing to substitute. A bundle asking for anything else (no_resampling, a
# custom function) was configured that way deliberately and its geometry is not
# ours to reinterpret.
_STOCK_RESAMPLER = "resample_data_or_seg_to_shape"
_RESAMPLING_KEYS = ("resampling_fn_data", "resampling_fn_probabilities")


def EnableGpuResampling(predictor, device):
    """Point this predictor's resamplers at the GPU. Returns whether it applied.

    Resampling, not inference, is what makes AMASSS slow: nnUNet's defaults are
    scipy splines on one core and outweigh the network by roughly seven to one.
    nnUNet ships torch equivalents, so there is nothing to reimplement, only to
    select.

    Selected by NAME: nnUNet resolves both resampling functions out of the
    configuration dict via `recursive_find_resampling_fn_by_name`, so rewriting
    the two names redirects both ends. No monkeypatching.

    Mutating that dict is safe because PlansManager hands out a `deepcopy`: it
    touches neither the shared plans nor a concurrent run, and the
    `torch.device` put in here never reaches the `plans.json` nnUNet writes
    beside its output (which `json.dump` could not serialize).
    """
    if not device.startswith("cuda"):
        return False

    try:
        from nnunetv2.preprocessing.resampling.resample_torch import (  # noqa: F401
            resample_torch_fornnunet,
        )
    except ImportError:
        logger.info("This nnUNet has no torch resampler; keeping the scipy one")
        return False

    configuration_manager = predictor.configuration_manager
    configuration = configuration_manager.configuration

    if any(configuration.get(key) != _STOCK_RESAMPLER for key in _RESAMPLING_KEYS):
        logger.info("Model plans request a non-default resampler; leaving it alone")
        return False

    for key in _RESAMPLING_KEYS:
        configuration[key] = "resample_torch_fornnunet"
        # 'linear' is order 1, already what the plans ask for on the
        # probabilities. The input data drops from order 3 to order 1 (torch
        # has no 3D cubic interpolation): that is the whole numerical
        # difference.
        configuration[f"{key}_kwargs"] = {
            "is_seg": False,
            "device": torch.device(device),
            "mode": "linear",
        }

    # Both are `@property @lru_cache`, so a value read before this point would
    # otherwise outlive the swap.
    manager_class = type(configuration_manager)
    for key in _RESAMPLING_KEYS:
        getattr(manager_class, key).fget.cache_clear()

    return True


def PredictFolder(model_folder, input_dir, output_dir, device, tile_step_size=0.5):
    """Segment every `*_0000.nii.gz` in `input_dir`, writing masks to `output_dir`.

    A whole folder per call is deliberate: the model is loaded once per
    structure rather than once per (scan x structure), which on a batch run is
    the difference between N*S and S checkpoint loads.
    """
    os.makedirs(output_dir, exist_ok=True)

    def run(gpu_resampling):
        predictor = BuildPredictor(device, tile_step_size)
        # Explicit path: no nnUNet_results env var, hence no cross-run race.
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(0,),
            checkpoint_name=CHECKPOINT_NAME,
        )

        on_gpu = gpu_resampling and EnableGpuResampling(predictor, device)
        logger.info(
            f"nnUNet predicting on {device} (GPU resampling: {'on' if on_gpu else 'off'})"
        )

        # `predict_from_files` fans preprocessing and export out to SPAWNED
        # processes, each of which would need its own CUDA context to run a GPU
        # resampler -- and under Slicer a spawned worker re-imports this module
        # and pays the whole torch import again. Everything therefore runs in
        # this process, trading away the CPU/GPU overlap on multi-scan batches
        # -- a smaller loss than the resampling win.
        predictor.predict_from_files_sequential(
            input_dir,
            output_dir,
            save_probabilities=False,
            overwrite=True,
        )

    try:
        run(True)
    except torch.cuda.OutOfMemoryError as e:
        # Resampling a whole CBCT on the GPU needs a few GB on top of the
        # network, which a small card may not have. The cloud version has a
        # `gpu_resampling` argument for this; the CLI's parameter list is fixed
        # by AMASSS_CLI.xml and its callers, so the fallback is automatic
        # instead: give the memory back and redo this structure with nnUNet's
        # own scipy resampler rather than failing the run.
        logger.warning(f"GPU out of memory ({e}); retrying with CPU resampling")
        torch.cuda.empty_cache()
        run(False)


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

def MatchReferenceGeometry(mask_img, ref_img):
    """Put a predicted mask back onto the reference scan's exact grid.

    nnUNet already returns its prediction on the input grid, so the resample is
    a no-op in the normal case and only fires when a model bundle hands back a
    different geometry.
    """
    same_geometry = (
        mask_img.GetSize() == ref_img.GetSize()
        and np.allclose(mask_img.GetSpacing(), ref_img.GetSpacing())
        and np.allclose(mask_img.GetOrigin(), ref_img.GetOrigin())
        and np.allclose(mask_img.GetDirection(), ref_img.GetDirection())
    )
    if not same_geometry:
        logger.debug("Prediction geometry differs from the scan; resampling")
        mask_img = sitk.Resample(
            mask_img, ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
            mask_img.GetPixelID(),
        )
    return sitk.Cast(mask_img, sitk.sitkInt16)


def SaveSeg(file_path, seg_arr, ref_img, outputdir, temp_folder, save_vtk, smoothing=5, model_size="LARGE"):
    """Write one segmentation array beside the scan it came from.

    Ported from the cloud version's `_write_segmentation`: one write, not two.
    The old path saved the mask to a temporary volume and read it back through
    ITK only to force the geometry -- two extra gzip round-trips of a full CBCT
    per structure, for a resample that in practice had nothing to do.

    `useCompression` covers the formats whose extension does not already imply
    it (.nrrd); for a .nii.gz the gz is applied from the name either way.
    """
    try:
        logger.debug(f"Saving segmentation to: {file_path}")

        out = sitk.GetImageFromArray(seg_arr.astype(np.int16))
        out.SetSpacing(ref_img.GetSpacing())
        out.SetDirection(ref_img.GetDirection())
        out.SetOrigin(ref_img.GetOrigin())
        sitk.WriteImage(MatchReferenceGeometry(out, ref_img), file_path, useCompression=True)
        logger.info(f"Segmentation saved: {file_path}")

        if save_vtk:
            try:
                logger.debug(f"Converting to VTK with smoothing: {smoothing}")
                SavePredToVTK(file_path, temp_folder, smoothing, vtk_output_path=outputdir)
                logger.info("VTK mesh saved successfully")
            except Exception as e:
                logger.error(f"Error saving VTK mesh: {e}")
                raise
    except Exception as e:
        logger.error(f"Error in SaveSeg: {e}")
        raise

def PrepareScanForNnunet(scan_path, destination):
    """Put one scan into the folder nnUNet reads, as a real NIfTI.

    Ported fix: the original did `shutil.copy(volume_file, "p_XXX_0000.nii.gz")`
    -- an NRRD renamed to .nii.gz, handed to a reader that picks its format from
    the extension. NRRD is Slicer's own default format, so "supported" input was
    in practice only reliable for NIfTI. A read + write actually converts.

    A file that is already a gzipped NIfTI is copied instead: converting it
    would be a gunzip + gzip of a full CBCT for no change. The voxel type is
    left alone -- nnUNet's reader casts to float32 itself.
    """
    if scan_path.lower().endswith(".nii.gz"):
        shutil.copy(scan_path, destination)
        return
    sitk.WriteImage(sitk.ReadImage(scan_path), destination)


def AssembleScanOutputs(record, predictions, args, temp_folder):
    """Turn one scan's per-structure nnUNet masks into its final files."""
    ref_img = sitk.ReadImage(record["path"])

    masks = {}
    for struct, pred_dir in predictions.items():
        predicted_file = os.path.join(pred_dir, f"p_{record['case_id']}.nii.gz")
        if not os.path.isfile(predicted_file):
            logger.warning(f"No {struct} prediction for {record['name']}")
            continue
        arr = sitk.GetArrayFromImage(sitk.ReadImage(predicted_file))
        masks[struct] = (arr > 0).astype(np.uint8)

    if not masks:
        raise FileNotFoundError(f"nnUNet produced no prediction for {record['name']}")

    outdir = record["outdir"]
    base = record["base"]
    ext = record["ext"]
    pid = args["prediction_ID"]
    written = []

    # SEPARATE mode: requested, or unavoidable when there is only one structure
    # (a "merged" volume of one structure is just that structure).
    if "SEPARATE" in args["merge"] or len(masks) == 1:
        for struct, mask in masks.items():
            outfn = os.path.join(outdir, f"{base}_{pid}_{struct}{ext}")
            SaveSeg(
                outfn, mask, ref_img, outdir, temp_folder,
                args["genVtk"], args["vtk_smooth"], "LARGE",
            )
            written.append(outfn)

    # MERGE mode
    if "MERGE" in args["merge"] and len(masks) > 1:
        shape = next(iter(masks.values())).shape
        merged = np.zeros(shape, dtype=np.int16)
        for struct in args["merging_order"]:
            if struct in masks:
                lbl = LABELS["LARGE"].get(struct, 1)
                merged = np.where(masks[struct] == 1, lbl, merged)
                logger.debug(f"Merged {struct} with label {lbl}")

        outfn = os.path.join(outdir, f"{base}_{pid}_MERGED{ext}")
        SaveSeg(
            outfn, merged, ref_img, outdir, temp_folder,
            args["genVtk"], args["vtk_smooth"], "LARGE",
        )
        written.append(outfn)

    if not written:
        raise RuntimeError(
            f"No segmentation was written for {record['name']} "
            f"(merge modes: {', '.join(args['merge'])})."
        )

    record["structures"] = sorted(masks)
    return written


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
                # Walk sub-directories: AREG writes its registered scans to
                # <output>/<Region>/<patient>_OutReg/, so a flat listing finds
                # none of them and the segmentation silently produces nothing.
                skip_own_output = not args["isSegmentInput"]
                own_suffix = "_{}_".format(args["prediction_ID"])
                input_files = []
                for root, _, files in os.walk(input_path):
                    for f in sorted(files):
                        if not f.lower().endswith(extensions):
                            continue
                        if 'MASK' in f:
                            continue
                        # An earlier pass may have written its segmentations
                        # into this same tree. Re-segmenting one as if it were
                        # a scan yields an empty mesh, so skip our own output
                        # unless the caller really is feeding us segmentations.
                        if skip_own_output and own_suffix in f:
                            logger.debug(f"Skipping previously generated segmentation: {f}")
                            continue
                        file = os.path.join(root, f)
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

        device = ResolveDevice()

        # ===== MODEL DISCOVERY =====
        # Once for the whole run, not once per scan: the bundle is the same for
        # every scan, and a missing model must be found out before inference.
        try:
            logger.debug("Searching for nnUNet models")
            nnunet_models = {}
            missing_structures = []
            for struct in args["skullStructure"].split(","):
                struct = struct.strip()
                if not struct:
                    continue
                model_folder = FindModelFolder(args["modelDirectory"], struct)
                if model_folder is None:
                    logger.warning(
                        f"No usable model for structure '{struct}' in {args['modelDirectory']}"
                    )
                    missing_structures.append(struct)
                else:
                    nnunet_models[struct] = model_folder
                    logger.debug(f"Found model for {struct}: {model_folder}")

            if not nnunet_models:
                logger.error("No models found for any structure")
                raise FileNotFoundError(
                    f"No nnUNet model found in '{args['modelDirectory']}' for any of the "
                    f"requested structures ({args['skullStructure']}). Expected "
                    f"<bundle>/<CODE>/**/*__nnUNetPlans__3d_fullres/fold_0/{CHECKPOINT_NAME}"
                )

            logger.info(f"Found {len(nnunet_models)} model(s) to process on {device}")
        except Exception as e:
            logger.error(f"Error during model discovery: {e}")
            raise

        # ===== SCAN PREPARATION =====
        # Every scan is converted ONCE into the single folder nnUNet reads, so
        # each structure is one folder-wide prediction and each checkpoint is
        # loaded once instead of once per (scan x structure).
        nnunet_input = os.path.join(tmp, "nnunet_input")
        os.makedirs(nnunet_input, exist_ok=True)

        scan_records = []
        for scan_idx, volume_file in enumerate(input_files, start=1):
            case_id = f"{scan_idx:03d}"
            basename = os.path.basename(volume_file)
            base, ext = os.path.splitext(basename)
            if ext == ".gz":
                base, ext2 = os.path.splitext(base)
                ext = ext2 + ext

            if args.get("save_in_folder"):
                outdir = os.path.join(base_output, f"{base}_{args['prediction_ID']}_SegOut")
            else:
                outdir = base_output
            os.makedirs(outdir, exist_ok=True)

            record = {
                "case_id": case_id,
                "name": basename,
                "path": volume_file,
                "base": base,
                "ext": ext,
                "outdir": outdir,
                "status": "pending",
            }
            try:
                PrepareScanForNnunet(
                    volume_file, os.path.join(nnunet_input, f"p_{case_id}_0000.nii.gz")
                )
                logger.debug(f"Prepared {basename} as case p_{case_id}")
            except Exception as e:
                logger.error(f"Could not read scan {volume_file}: {e}")
                record["status"] = "failed"
                record["error"] = f"Unreadable input: {e}"
            scan_records.append(record)

        readable = [r for r in scan_records if r["status"] != "failed"]
        if not readable:
            raise RuntimeError("None of the input scans could be read as a medical volume.")

        # ===== INFERENCE: one model load per structure =====
        total_struct = len(nnunet_models)
        total_steps = scan_count * total_struct
        predictions = {}
        failed_structures = {}

        for struct_idx, (struct, model_folder) in enumerate(nnunet_models.items(), start=1):
            logger.info(f"Processing structure {struct_idx}/{total_struct}: {struct}")
            outp = os.path.join(tmp, f"pred_{struct}")
            try:
                PredictFolder(model_folder, nnunet_input, outp, device)
                predictions[struct] = outp
                logger.info(f"Prediction for {struct} completed")
            except Exception as e:
                # One structure failing must not lose the others.
                logger.error(f"Prediction failed for structure {struct}: {e}")
                failed_structures[struct] = str(e)

            # Progress is still counted in (scan x structure) steps, so the
            # scale the Slicer modules were written against is unchanged.
            try:
                step = struct_idx * scan_count
                fraction = step / total_steps
                print(f"<filter-progress>{fraction:.4f}</filter-progress>", flush=True)
                sys.stdout.flush()
                logger.debug(f"Progress: {fraction:.4f}")
            except Exception as e:
                logger.warning(f"Error reporting progress: {e}")

        if not predictions:
            raise RuntimeError(
                "Every structure failed to predict: "
                + "; ".join(f"{s}: {err}" for s, err in failed_structures.items())
            )

        # The converted copies are a full CBCT each and inference is done with
        # them; a batch would otherwise sit on all of them until the run ends.
        shutil.rmtree(nnunet_input, ignore_errors=True)

        # ===== PER-SCAN OUTPUT ASSEMBLY =====
        processed_scans = 0
        failed_scans = []

        for record in readable:
            scan_context = f"scan {record['case_id']}: {record['name']}"
            logger.info(f"Saving outputs for {scan_context}")
            try:
                AssembleScanOutputs(record, predictions, args, tmp)
                record["status"] = "ok"
                processed_scans += 1
                logger.info(f"Successfully processed {scan_context}")
            except Exception as e:
                # A failure here is recorded and the run continues: the original
                # re-raised on the LAST scan only, so a batch could abort at the
                # very end and lose everything already produced.
                logger.error(f"Failed to process {scan_context}: {e}")
                record["status"] = "failed"
                failed_scans.append((record["case_id"], record["path"], str(e)))

            # This scan's predictions are no longer needed.
            for pred_dir in predictions.values():
                try:
                    os.remove(os.path.join(pred_dir, f"p_{record['case_id']}.nii.gz"))
                except OSError:
                    pass

        for record in scan_records:
            if record["status"] == "failed" and "error" in record:
                failed_scans.append((record["case_id"], record["path"], record["error"]))

        # --- CLEANUP ---
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp, exist_ok=True)
            logger.debug("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")

        if processed_scans == 0:
            raise RuntimeError(
                "AMASSS produced no output for any scan. First error: "
                + (failed_scans[0][2] if failed_scans else "unknown")
            )

        # --- FINAL REPORT ---
        try:
            elapsed = time.time() - start_time
            logger.info(f"Processing completed in {elapsed:.2f}s")
            print(f"<filter-end><filter-name>AMASSS</filter-name><filter-time>{elapsed:.2f}</filter-time></filter-end>", flush=True)
            sys.stdout.flush()

            logger.info(f"Processed {processed_scans}/{scan_count} scans successfully")
            if missing_structures:
                logger.warning(
                    f"No model for structure(s): {', '.join(missing_structures)}"
                )
            if failed_structures:
                for struct, err in failed_structures.items():
                    logger.warning(f"  Structure {struct} failed: {err}")
            if failed_scans:
                logger.warning(f"Failed to process {len(failed_scans)} scan(s)")
                for idx, path, err in failed_scans:
                    logger.warning(f"  Scan {idx} ({os.path.basename(path)}): {err}")

            if processed_scans == scan_count:
                logger.info("All scans processed successfully.")
            else:
                logger.warning(f"Processing completed with {len(failed_scans)} failure(s).")
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