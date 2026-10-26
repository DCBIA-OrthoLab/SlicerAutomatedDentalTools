"""Dataset516: cranial base on the MRI, through nnU-Net's Python API.

Called in-process rather than through the nnUNetv2_predict console script,
whose python-real shebang cannot find libpython outside Slicer's launcher.
Settings as in the published run: 3d_fullres, no test-time mirroring.
"""
import os

os.environ.setdefault("nnUNet_compile", "f")      # torch.compile breaks in Slicer's Python
for _k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ.setdefault(_k, "/unused")           # must be set; never read for inference


def segment(inputs, outputs, model_folder, folds=(0, 1, 2, 3, 4), log=print):
    """inputs/outputs: matching lists of .nii.gz paths. Writes label 1 = cranial base."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log(f"Dataset516 on {device.type}, folds {list(folds)}")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=device.type == "cuda", device=device,
        verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=tuple(folds), checkpoint_name="checkpoint_final.pth")
    truncated = [o[: -len(".nii.gz")] if o.endswith(".nii.gz") else o for o in outputs]
    for o in outputs:
        os.makedirs(os.path.dirname(o), exist_ok=True)
    predictor.predict_from_files_sequential(
        [[i] for i in inputs], truncated, save_probabilities=False, overwrite=True)
