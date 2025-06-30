#!/usr/bin/env python
"""
clic_runner.py
==============

Exécuté par Slicer via self.conda.condaRunFilePython.
– Charge un scan .nii(.gz)
– Applique un Mask-R-CNN 2D slice-wise
– Sauvegarde la segmentation
– Envoie la progression & les logs sur stdout.

Les messages stdout doivent commencer par :
  [PROGRESS] <0-100>
  [LOG]      <texte>
  [SEG]      <chemin_nii_gz>
"""

import argparse, json, glob
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ───────────────────────── helpers ──────────────────────────────────────────
def _blank_model(nc: int):
    m = maskrcnn_resnet50_fpn(weights=None)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, nc)
    in_fm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, nc)
    return m


def _norm(x2d: np.ndarray) -> np.ndarray:
    lo, hi = x2d.min(), x2d.max()
    return (x2d - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(x2d)


log      = lambda s: print(f"[LOG] {s}",      flush=True)
progress = lambda p: print(f"[PROGRESS] {p}", flush=True)
seg_out  = lambda p: print(f"[SEG] {p}",      flush=True)


# ───────────────────────── main ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_json", required=True)
    P = json.loads(Path(ap.parse_args().params_json).read_text())

    inp         = Path(P["input_path"]).expanduser()
    model_dir   = Path(P["model_folder"]).expanduser()
    out_root    = Path(P.get("output_dir", inp.parent)).expanduser()
    suffix      = P.get("suffix", "seg")
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Modèle ───────────────────────────────────────────────────────────────
    state = sorted(model_dir.glob("*.pth"))[0]
    model = _blank_model(4)
    model.load_state_dict(torch.load(state, map_location=device))
    model.to(device).eval()
    log(f"Model: {state.name}")

    # 2. Volume ───────────────────────────────────────────────────────────────
    nii  = nib.load(str(inp))
    vol  = nii.get_fdata(dtype=np.float32)
    Z    = vol.shape[2]
    seg  = np.zeros_like(vol, dtype=np.int16)

    for z in range(Z):
        sl = _norm(vol[..., z])
        t  = torch.from_numpy(sl).unsqueeze(0).repeat(3, 1, 1).float().to(device)
        with torch.no_grad():
            pr = model([t])[0]
        keep = pr["scores"] >= 0.7
        for mk, lb in zip((pr["masks"][keep] > .5).squeeze(1).cpu().numpy(),
                          pr["labels"][keep].cpu().numpy()):
            seg[..., z][mk] = int(lb)
        progress(int((z + 1) * 100 / Z))

    # 3. Sauvegarde ───────────────────────────────────────────────────────────
    out_dir  = out_root / inp.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{inp.stem}_{suffix}.nii.gz"
    nib.save(nib.Nifti1Image(seg.astype(np.int16), nii.affine, nii.header),
             str(out_path))
    seg_out(str(out_path))
    log("✓ Finished")


if __name__ == "__main__":
    main()
