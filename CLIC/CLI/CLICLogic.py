# CLICLogic.py ────────────────────────────────────────────────────────────────
# Mask-R-CNN 2D-slice segmentation – logique « pure »
# Patch 2025-07-15 : finished_evt, suppression sync_event.wait()
# ─────────────────────────────────────────────────────────────────────────────
import os, glob, numpy as np, nibabel as nib, torch, scipy.ndimage
from pathlib import Path
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class CLICLogic(ScriptedLoadableModuleLogic):
    CLASS_NAMES = {1: "buccal", 2: "bicortical", 3: "palatal"}

    # ────────────────────────── init ─────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.seg_files = []
        self.cancelRequested = False
        self._cached_model = None      # reuse même modèle sur appels multiples
        self._cached_path  = None

    # ──────────────────── modèle Mask-R-CNN ─────────────────────────────────
    def _get_blank_model(self, num_classes: int):
        model = maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 256, num_classes
        )
        return model

    def _load_model(self, model_path: str, num_classes: int, device):
        """Charge le modèle (avec cache mémoire pour les appels ultérieurs)."""
        if self._cached_model and self._cached_path == model_path:
            return self._cached_model
        model = self._get_blank_model(num_classes)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        self._cached_model, self._cached_path = model, model_path
        return model

    # ────────────────────────── slice utils ─────────────────────────────────
    @staticmethod
    def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
        mn, mx = slice_2d.min(), slice_2d.max()
        return ((slice_2d - mn) / (mx - mn)) if (mx - mn) > 1e-8 else np.zeros_like(slice_2d)

    # ───────────────────────── processing core ──────────────────────────────
    def _process_nii_file(
        self,
        model,
        nii_path: str,
        device,
        progress_cb=None,
        log_cb=None,
        score_th: float = 0.7,
    ):
        nib_vol = nib.load(nii_path)
        vol_data = nib_vol.get_fdata(dtype=np.float32)
        H, W, Z = vol_data.shape

        if log_cb: log_cb(f"  • Slices : {Z}")

        all_detections = []
        slice_counts = {n: {"left": 0, "right": 0} for n in self.CLASS_NAMES.values()}

        for z in range(Z):
            if self.cancelRequested:
                log_cb and log_cb(f"[CANCEL] arrêt à la slice {z}")  # noqa: E501
                break

            slice_norm = self._normalize_slice(vol_data[..., z])
            slice_tensor = (
                torch.from_numpy(slice_norm)
                .unsqueeze(0)
                .repeat(3, 1, 1)
                .float()
                .to(device)
            )

            with torch.no_grad():
                preds = model([slice_tensor])[0]

            keep = preds["scores"] >= score_th
            if keep.sum() == 0:
                progress_cb and progress_cb((z + 1) * 100 / Z)
                continue

            labels = preds["labels"][keep].cpu().numpy()
            masks  = (preds["masks"][keep] > 0.5).squeeze(1).cpu().numpy()

            for i, mk in enumerate(masks):
                l = int(labels[i])
                com = scipy.ndimage.center_of_mass(mk)
                side = "left" if com[0] < (H / 2) else "right"
                slice_counts[self.CLASS_NAMES[l]][side] += 1
                all_detections.append({"label": l, "slice_z": z, "mask_2d": mk})

            progress_cb and progress_cb((z + 1) * 100 / Z)

        return vol_data, nib_vol, all_detections, slice_counts

    # ───────────────────────── main entrypoint ──────────────────────────────
    def process(
        self,
        parameters: dict,
        progress_callback=None,
        log_callback=None,
        display_callback=None,
        finished_evt=None,          # <─ événement passé par CLIC.py
    ):
        """
        Segmente UN scan (params['input_path']) et signale toujours finished_evt.set().
        """
        try:
            print("torch.cuda.is_available():", torch.cuda.is_available())
            print("torch path:", torch.__file__)
            print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
            self.cancelRequested = False

            input_path   = parameters["input_path"]
            model_folder = parameters["model_folder"]
            output_dir   = parameters.get("output_dir", os.path.dirname(input_path))
            suffix       = parameters.get("suffix", "seg") or "seg"

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_files = glob.glob(os.path.join(model_folder, "*.pth"))
            if not model_files:
                raise FileNotFoundError("No .pth files found in model folder.")
            model = self._load_model(model_files[0], num_classes=4, device=device)

            # ----------------------------------------------------------------
            # 1. Lecture unique du fichier (CLICWidget fournit 1 scan / appel)
            # ----------------------------------------------------------------
            if log_callback: log_callback(f"Processing file: {input_path}")
            if display_callback: display_callback("loadScan", input_path)

            vol, nib_ref, dets, _ = self._process_nii_file(
                model, input_path, device, progress_callback, log_callback
            )

            # ----------------------------------------------------------------
            # 2. Construction du volume de segmentation en 3D
            # ----------------------------------------------------------------
            seg = np.zeros_like(vol, dtype=np.int16)
            for d in dets:
                seg[..., d["slice_z"]][d["mask_2d"]] = d["label"]

            base_name = Path(input_path).stem
            out_dir   = Path(output_dir) / base_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path  = out_dir / f"{base_name}_{suffix}.nii.gz"

            nib.save(
                nib.Nifti1Image(seg.astype(np.int16), nib_ref.affine, nib_ref.header),
                str(out_path),
            )
            self.seg_files = [str(out_path)]

            display_callback and display_callback("segmentation", str(out_path))
            log_callback and log_callback("✓ Finished.")

        except Exception as e:
            log_callback and log_callback(f"[ERROR] {e}")
        finally:
            # ─── réveille toujours le thread appelant ───────────────────────
            if finished_evt is not None:
                finished_evt.set()
