#!/usr/bin/env python-real

"""
AUTOMATIC LANDMARK IDENTIFICATION IN INTRAORAL SCANS (ALI_CBCT)

Authors :
- Maxime Gillot (UoM)
- Baptiste Baquero (UoM)
"""
#pytorch3d : need version 0.6.2
#monai : need version 0.7.0
#IMPORT DE BASE
import time
import os
import glob
import sys
import tempfile
import shutil
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import platform
import argparse
import numpy as np
import torch
import logging

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("ALI_IOS")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from monai.networks.nets import UNet
from monai.transforms import AsDiscrete
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

# realpath, not __file__: a CLI registered through a symlink - the flat dev
# folder of links into the source tree - leaves __file__ on the link, whose
# parent holds no ALI_IOS_utils. Resolving first lands beside the package.
fpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(fpath)

def check_platform():
    if platform.system() == 'Windows':
        return "Windows"
    elif platform.system() == 'Linux':
        if 'microsoft' in platform.release().lower():
            return "WSL"
        else:
            return "Linux"
    else:
        return "Unknown"

# Import from utils
if check_platform()=="WSL":
    from ALI_IOS_utils.render import GenPhongRenderer
    from ALI_IOS_utils.surface import (
        ReadSurf, ScaleSurf, GetSurfProp, RemoveExtraFaces, Upscale, UnifyArchLabels)
    from ALI_IOS_utils.model import dic_cam, dic_label, MODELS_DICT
    from ALI_IOS_utils.io import (
        GenControlPoint, WriteJson, TradLabel, TradLabelMG, JawFromFileName)
    from ALI_IOS_utils.orientation import (
        LowerArchMatrix, TransformSurf, TransformPoint, ArchScale)
    from ALI_IOS_utils.segmentation import IsSegmented, SegmentSurface
    from ALI_IOS_utils.fill_gaps import FillGaps
    from ALI_IOS_utils.complete_line import CompleteLine, SnapAll
    from ALI_IOS_utils.smooth import SmoothAlongArch, DEFAULT_STRENGTH as SMOOTH_STRENGTH
    from ALI_IOS_utils.pick_patch import (
        PickNearAim, ToothPitch, ResolveCollisions, OFF_AIM_NOTE)
    from ALI_IOS_utils.paint_scan import PaintScan
    from ALI_IOS_utils.agent import Agent

else :
    from ALI_IOS_utils import (
        GenPhongRenderer, ReadSurf, ScaleSurf,
        GetSurfProp, RemoveExtraFaces, Upscale, UnifyArchLabels,
        dic_cam, dic_label, MODELS_DICT,
        GenControlPoint, WriteJson, TradLabel, TradLabelMG, JawFromFileName, Agent,
        LowerArchMatrix, TransformSurf, TransformPoint, ArchScale,
        IsSegmented, SegmentSurface, FillGaps, CompleteLine, SnapAll,
        SmoothAlongArch,
        PickNearAim, ToothPitch
    )
    from ALI_IOS_utils.pick_patch import ResolveCollisions, OFF_AIM_NOTE
    from ALI_IOS_utils.paint_scan import PaintScan
    from ALI_IOS_utils.smooth import DEFAULT_STRENGTH as SMOOTH_STRENGTH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# How far the second look at a tooth may move its landmark before the first
# answer is kept instead, in millimetres. On scans like the ones the network
# was trained on it hardly ever bites -- 4% of points -- and costs nothing.
# On scans unlike them a third of the points cross it, some by tens of
# millimetres, and without it the share of landmarks landing somewhere
# anatomically impossible goes from 3% to 13%.
REFINE_LIMIT = 3.0

# Surfaces ReadSurf opens. Only .vtk and .vtp can carry the teeth labels the
# cameras are aimed with; the others are segmented on the way in, which is
# also what converts them.
SURFACES = (".vtk", ".vtp", ".stl", ".obj", ".off")


def EstimateMissingArchPositions(lst_teeth, RI, V):
    """Centroid and arch tangent of the teeth missing from the segmentation.

    The MG cameras are aimed with the centroid of each tooth and the local
    direction of the arch, both read from the segmentation. When a tooth has
    no label, both can still be estimated: the lower labels are consecutive
    along the arch (19 to 31), so the centroids of the segmented teeth trace
    the arch and a quadratic fit of each coordinate against the label index
    fills the gaps.

    Needs at least 4 segmented teeth spread over a span of at least 4 labels,
    otherwise the extrapolation is not trustworthy and {} is returned, leaving
    the caller to skip those teeth as before.

    Returns {label: (position, tangent)} as float32 tensors in unit-sphere
    space, for the missing labels only.
    """
    ids = RI.squeeze(0)
    present, centroids = [], []
    for label in lst_teeth:
        idx = (ids == int(label)).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            present.append(int(label))
            centroids.append(V[0][idx].mean(dim=0).cpu().numpy())
    missing = [int(label) for label in lst_teeth if int(label) not in present]
    if not missing:
        return {}
    if len(present) < 4 or (max(present) - min(present)) < 4:
        span = max(present) - min(present) if present else 0
        logger.warning(
            f"Only {len(present)} teeth segmented over a span of {span} labels: "
            f"not enough to estimate the position of teeth {missing}")
        return {}

    present_arr = np.array(present, dtype=float)
    centroids = np.array(centroids)
    fits = [np.polyfit(present_arr, centroids[:, axis], deg=2) for axis in range(3)]

    def at(label_value):
        return np.array([np.polyval(fit, label_value) for fit in fits])

    estimated = {}
    for label in missing:
        estimated[label] = (
            torch.tensor(at(label), dtype=torch.float32),
            torch.tensor(at(label + 0.5) - at(label - 0.5), dtype=torch.float32),
        )
    logger.info(
        f"Teeth {missing} are not in the segmentation: their positions were "
        f"estimated from the arch traced by the {len(present)} segmented teeth")
    return estimated


def main(args):
    """Main function with comprehensive error handling."""
    logger.info(f"Starting ALI_IOS with args: {args}")
    
    # Setup log file
    try:
        log_dir = os.path.split(args.log_path)[0]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        with open(args.log_path, "w") as log_f:
            log_f.truncate(0)
    except Exception as e:
        logger.error(f"Failed to setup log file: {e}")
        sys.exit(1)
    
    # Parse arguments
    try:
        def clean_list(raw):
            items = [item.strip().replace("'", "").replace('"', '') for item in raw.split(" ")]
            return [item for item in items if item and item != "None"]

        lm_types = clean_list(args.lm_type)
        teeth = clean_list(args.teeth)
        teeth_mg = clean_list(args.teeth_mg)

        if not (lm_types and teeth) and not teeth_mg:
            logger.error("Nothing to process: give teeth + landmark types and/or MG teeth")
            raise ValueError("Invalid landmark types or teeth list")

        landmarks_selected = [tooth + lm_type for tooth in teeth for lm_type in lm_types]
        # MG points are already named with their output labels (LL6MG...L0MG...LR6MG)
        landmarks_selected += teeth_mg
        logger.info(f"Processing landmarks: {landmarks_selected}")
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        sys.exit(1)

    # Translate labels
    try:
        dic_teeth = TradLabel(teeth)
        dic_teeth_mg = TradLabelMG(teeth_mg)
        logger.debug(f"Tooth labels translated: {dic_teeth}, MG: {dic_teeth_mg}")
    except Exception as e:
        logger.error(f"Failed to translate tooth labels: {e}")
        sys.exit(1)
    
    # Find available models in folder
    try:
        available_models = {}
        models_to_use = {}
        
        if not os.path.exists(args.dir_models):
            logger.error(f"Models directory not found: {args.dir_models}")
            raise FileNotFoundError(f"Directory does not exist: {args.dir_models}")
        
        normpath = os.path.normpath("/".join([args.dir_models, '**', '']))
        # sorted() so that two checkpoints competing for the same slot resolve the
        # same way every run: an older model left in the folder must not silently
        # win a coin toss, since weights only work with the geometry they were
        # trained on
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            basename = os.path.basename(img_fn)
            if basename.endswith(".pth"):
                try:
                    model_id = basename.split("_")[1]
                    if model_id not in available_models.keys():
                        available_models[model_id] = {}
                    jaw = 'Lower' if 'Lower' in basename else 'Upper'
                    if jaw in available_models[model_id]:
                        logger.warning(
                            f"Several {jaw} '{model_id}' models found, using {basename} and "
                            f"ignoring {os.path.basename(available_models[model_id][jaw])}. "
                            "Keep only one to be sure which is used."
                        )
                    available_models[model_id][jaw] = (img_fn)
                except Exception as e:
                    logger.warning(f"Error processing model file {basename}: {e}")
                    continue

        logger.info(f'Available models: {available_models}')

        for model_id in MODELS_DICT.keys():
            if model_id not in available_models:
                continue
            if model_id == "MG":
                # The MG model is driven by its own teeth list, not by lm_types
                if dic_teeth_mg['Lower'] and 'Lower' in available_models[model_id]:
                    models_to_use[model_id] = available_models[model_id]
                continue
            for lmtype in lm_types:
                if lmtype in MODELS_DICT[model_id].keys():
                    if model_id not in models_to_use.keys():
                        models_to_use[model_id] = available_models[model_id]

        logger.info(f'Models to use: {models_to_use}')

        if dic_teeth_mg['Lower'] and "MG" not in models_to_use:
            logger.error("MG teeth selected but no mucogingival model found (expected a 'Lower_MG_*.pth' file in the models directory)")
            raise RuntimeError("No MG model found")

        if not models_to_use:
            logger.error("No suitable models found for the specified landmark types")
            raise RuntimeError("No matching models found")
            
    except Exception as e:
        logger.error(f"Error discovering models: {e}")
        sys.exit(1)


    dic_patients = {}
    
    try:
        if not os.path.exists(args.input):
            logger.error(f"Input path not found: {args.input}")
            raise FileNotFoundError(f"Input path does not exist: {args.input}")
        
        if os.path.isfile(args.input):
            logger.info(f"Loading single scan: {args.input}")
            basename = os.path.basename(args.input).split('.')[0]
            if basename not in dic_patients.keys():
                dic_patients[basename] = args.input

        else:
            logger.info(f"Loading data from directory: {args.input}")
            normpath = os.path.normpath("/".join([args.input, '**', '']))
            for vtkfile in sorted(glob.iglob(normpath, recursive=True)):
                if os.path.isfile(vtkfile) and os.path.splitext(vtkfile)[1].lower() in SURFACES:
                    basename = os.path.basename(vtkfile).split('.')[0]
                    if basename not in dic_patients.keys():
                        dic_patients[basename] = vtkfile
        
        if not dic_patients:
            # A .stl is a common thing to point this at, and the reason it is
            # not read is worth saying: the landmarks are placed tooth by
            # tooth, and only a .vtk can carry the segmentation that names
            # them. Segment the scans first, which also converts them.
            others = [name for name in os.listdir(args.input)
                      if os.path.splitext(name)[1].lower() in (".stl", ".obj", ".vtp", ".off")] \
                if os.path.isdir(args.input) else []
            if others:
                logger.error(
                    f"No .vtk found in {args.input}, but {len(others)} other surface(s) "
                    f"are there ({', '.join(sorted(others)[:3])}...). Those formats hold "
                    "no teeth segmentation, which the landmarks are placed from: run the "
                    "crown segmentation on them first, it writes the .vtk this needs")
            else:
                logger.error("No valid medical imaging files found. Use .vtk format")
            raise FileNotFoundError("No .vtk files found in input path")
        
        logger.info(f'Loaded {len(dic_patients)} patient(s)')
        
    except Exception as e:
        logger.error(f"Error loading patient data: {e}")
        sys.exit(1)

    total_landmarks = 0
    for jaw_teeth in dic_teeth.values():
        total_landmarks += len(jaw_teeth)
    total_landmarks *= len(dic_patients)


    for idx, (patient_id, patient_path) in enumerate(dic_patients.items()):
        logger.info(f"Processing patient {idx + 1}/{len(dic_patients)}: {patient_id}")
        
        for models_type in models_to_use.keys():
            try:
                LABEL = dic_label[models_type]
                sphere_radius = 0.2 if models_type in ("O", "MG") else 0.3

                logger.debug(f"Processing model type: {models_type}")

                teeth_for_model = dic_teeth_mg if models_type == "MG" else dic_teeth
                for jaw, lst_teeth in teeth_for_model.items():
                    if models_type == "MG" and jaw != "Lower":
                        continue
                    if not lst_teeth:
                        continue

                    group_data = {}

                    try:
                        path_vtk = patient_path

                        # The cameras are aimed tooth by tooth, off a
                        # Universal_ID array. A scan that has none gets it here
                        # rather than being turned away, which is also what
                        # turns an .stl into the .vtk the rest of this reads.
                        # The segmentation leaves the points where they are, so
                        # the landmarks stay valid in the file the user gave.
                        segmented_folder = None
                        segmented_path = None
                        unified_folder = None
                        if not IsSegmented(path_vtk):
                            segmented_folder = tempfile.mkdtemp(prefix="ALI_IOS_segmented_")
                            segmented = SegmentSurface(path_vtk, folder=segmented_folder)
                            if segmented is None:
                                shutil.rmtree(segmented_folder, ignore_errors=True)
                                logger.error(f"{patient_id} cannot be segmented, no landmark "
                                             "can be placed on it")
                                continue
                            path_vtk = segmented
                            segmented_path = segmented

                        # The segmentation names each point on its own, and
                        # nothing in a neighbourhood says which jaw the scan is:
                        # a maxilla and a mirrored mandible have the same shape,
                        # only the palate tells them apart. On an arch it cannot
                        # place, it splits every tooth between its own number and
                        # the same rank in the other arch -- and the occlusal cap,
                        # which is exactly where the occlusal landmark sits, tends
                        # to take the upper number. RemoveExtraFaces then finds no
                        # face carrying the label it was asked for and the landmark
                        # is never written, while the wrong model answers on the
                        # same scan and names its file after both.
                        unified_folder = None
                        scan_jaw = JawFromFileName(patient_path)
                        surf_labels = ReadSurf(path_vtk)
                        if UnifyArchLabels(surf_labels, scan_jaw):
                            unified_folder = tempfile.mkdtemp(prefix="ALI_IOS_unified_")
                            unified = os.path.join(unified_folder, os.path.basename(path_vtk))
                            writer = vtk.vtkPolyDataWriter()
                            writer.SetFileName(unified)
                            writer.SetInputData(surf_labels)
                            writer.SetFileTypeToBinary()
                            writer.Write()
                            path_vtk = unified
                        del surf_labels

                        model = models_to_use[models_type]['Lower'] if jaw == 'Lower' else models_to_use[models_type]['Upper']
                        camera_position = dic_cam[models_type]['L'] if jaw == 'Lower' else dic_cam[models_type]['U']

                        # The MG cameras are built on a vertical axis taken to
                        # be Z, so the scan is brought into that frame before
                        # anything is predicted on it and the landmarks are
                        # sent back to the coordinates of the file afterwards.
                        # A scan left as it came off the scanner puts its arch
                        # on another axis, and the cameras then frame the
                        # crowns instead of the gingival margin.
                        back_to_file = None
                        if models_type == "MG":
                            matrix = LowerArchMatrix(ReadSurf(path_vtk))
                            if matrix is not None:
                                oriented = os.path.join(
                                    tempfile.mkdtemp(prefix="ALI_IOS_oriented_"),
                                    os.path.basename(path_vtk))
                                writer = vtk.vtkPolyDataWriter()
                                writer.SetFileName(oriented)
                                writer.SetInputData(TransformSurf(ReadSurf(path_vtk), matrix))
                                writer.SetFileTypeToBinary()
                                writer.Write()
                                path_vtk = oriented
                                back_to_file = np.linalg.inv(matrix)
                                logger.info(f"{patient_id}: oriented on its four lower teeth "
                                            "for the mucogingival prediction")

                        # The cameras are placed at a fixed distance in the
                        # normalised space, so what that space measures decides
                        # how they frame the gum. Read off the arch, it follows
                        # the jaw; read off the scan extent, it follows how much
                        # vestibule was captured.
                        mg_scale = None
                        if models_type == "MG" and args.arch_scale:
                            mg_scale = (ArchScale(ReadSurf(path_vtk), args.arch_ratio)
                                        if args.arch_ratio else ArchScale(ReadSurf(path_vtk)))
                            if mg_scale is not None:
                                logger.info(f"{patient_id}: scaled on the arch "
                                            f"(1 unit = {1 / mg_scale:.1f} mm)")

                        # The MG cameras need a position per tooth. For the
                        # teeth the segmentation does not know, estimate one
                        # from the arch of the teeth it does know, instead of
                        # skipping their landmark.
                        mg_estimated = {}
                        if models_type == "MG" and args.estimate_missing:
                            surf_est = ReadSurf(path_vtk)
                            unit_est, mean_est, scale_est = ScaleSurf(surf_est, scale_factor=mg_scale)
                            (V_est, _f_est, _cn_est, RI_est) = GetSurfProp(unit_est, mean_est, scale_est)
                            mg_estimated = EstimateMissingArchPositions(lst_teeth, RI_est, V_est)

                        # A second look, aimed at what the first one found. The
                        # cameras are otherwise pointed at an anatomical prior,
                        # the same offset for every patient, so the landmark
                        # sits off-centre whenever that prior is off -- and the
                        # network was taught on images where it is centred.
                        refine = models_type == "MG" and args.refine
                        first_aim, first_point = {}, {}
                        mg_pitch = {}      # tooth spacing, measured once per scan
                        mg_aims = {}       # where each landmark's cameras were aimed
                        mg_scale_factor = None
                        rounds = ([(0, label) for label in lst_teeth]
                                  + ([(1, label) for label in lst_teeth] if refine else []))

                        for pass_index, label in rounds:
                            try:
                                logger.debug(f"Loading model for patient {patient_id}, label {label}, jaw {jaw}")
                                
                                phong_renderer, mask_renderer = GenPhongRenderer(
                                    int(args.image_size), int(args.blur_radius), int(args.faces_per_pixel), DEVICE
                                )

                                agent = Agent(
                                    renderer=phong_renderer,
                                    renderer2=mask_renderer,
                                    radius=sphere_radius,
                                    camera_position=camera_position,
                                    lm_type=models_type
                                )

                                SURF = ReadSurf(path_vtk)
                                surf_unit, mean_arr, scale_factor = ScaleSurf(
                                    SURF, scale_factor=mg_scale if models_type == "MG" else None)
                                (V, F, CN, RI) = GetSurfProp(surf_unit, mean_arr, scale_factor)

                                estimated = mg_estimated.get(int(label)) if models_type == "MG" else None
                                if int(label) in RI.squeeze(0) or estimated is not None:
                                    if estimated is None:
                                        agent.position_agent(RI, V, label)
                                    else:
                                        agent.position_agent_estimated(V, estimated[0], estimated[1], label)
                                        logger.info(
                                            f"Label {label} is not in the segmentation of {patient_id}: "
                                            "cameras aimed at its position estimated from the arch")
                                    if pass_index == 1 and int(label) in first_aim:
                                        agent.aim_points = first_aim[int(label)]

                                    textures = TexturesVertex(verts_features=CN)
                                    meshe = Meshes(verts=V, faces=F, textures=textures).to(DEVICE)

                                    try:
                                        images_model, tens_pix_to_face_model = agent.get_view_rasterize(meshe)
                                        tens_pix_to_face_model = tens_pix_to_face_model.permute(1, 0, 4, 2, 3)

                                        if models_type == "MG":
                                            # MG network: 12 channels (3 cameras x RGB+Z) stacked
                                            # in a single input, 3 output classes
                                            net = UNet(
                                                spatial_dims=2,
                                                in_channels=12,
                                                out_channels=3,
                                                channels=(16, 32, 64, 128, 256, 512),
                                                strides=(2, 2, 2, 2, 2),
                                                num_res_units=4
                                            ).to(DEVICE)

                                            b, cam, c, h, w = images_model.shape
                                            inputs = images_model.reshape(b, cam * c, h, w).to(dtype=torch.float32).to(DEVICE)
                                        else:
                                            net = UNet(
                                                spatial_dims=2,
                                                in_channels=4,
                                                out_channels=4,
                                                channels=(16, 32, 64, 128, 256, 512),
                                                strides=(2, 2, 2, 2, 2),
                                                num_res_units=4
                                            ).to(DEVICE)

                                            inputs = torch.cat([batch.to(DEVICE) for batch in images_model], dim=0).float()

                                        net.load_state_dict(torch.load(model, map_location=DEVICE))
                                        images_pred = net(inputs)

                                        if models_type != "MG":
                                            post_pred = AsDiscrete(argmax=True, to_onehot=4)

                                            val_pred = torch.empty((0)).to(DEVICE)
                                            for image in images_pred:
                                                val_pred = torch.cat((val_pred, post_pred(image).unsqueeze(0).to(DEVICE)), dim=0)

                                        if models_type == "MG":
                                            # argmax on the raw scores, NOT on an int16 cast of them:
                                            # truncating the logits to integers merges classes that are
                                            # close and, on a tie, argmax falls back to index 0
                                            # (background), silently dropping pixels
                                            logits = images_pred.detach().cpu().float()
                                            pred_data = torch.argmax(logits, dim=1).unsqueeze(0).unsqueeze(2)
                                        else:
                                            pred_data = images_pred.detach().cpu().unsqueeze(0).type(torch.int16)
                                            pred_data = torch.argmax(pred_data, dim=2).unsqueeze(2)

                                        # recover where there is the landmark in the image
                                        index_label_land_r = (pred_data == 1.).nonzero(as_tuple=False)

                                        # Fallback: the landmark class won nowhere, so nothing would be
                                        # written for this tooth. Keep the pixels where that class is the
                                        # most likely anyway, so a point is always placed. Confidence is
                                        # reported and stored in the json, because a forced point is
                                        # markedly less accurate than a won one.
                                        # How sure the network was, over the pixels it chose. It
                                        # costs nothing to keep and it is the only thing that says,
                                        # on a dataset with no annotation to check against, whether
                                        # the model recognises what it is looking at.
                                        won_conf = None
                                        if models_type == "MG":
                                            prob1_all = torch.softmax(logits, dim=1)[:, 1]
                                            if len(index_label_land_r) > 0:
                                                chosen = [float(prob1_all[idx[1], idx[3], idx[4]])
                                                          for idx in index_label_land_r]
                                                won_conf = float(np.mean(chosen))

                                        forced_conf = None
                                        if models_type == "MG" and args.force_landmarks and len(index_label_land_r) == 0:
                                            prob1 = torch.softmax(logits, dim=1)[:, 1]
                                            k = min(args.force_topk, prob1.numel())
                                            if k > 0:
                                                conf, flat_idx = prob1.reshape(-1).topk(k)
                                                cam, yy, xx = np.unravel_index(flat_idx.numpy(), tuple(prob1.shape))
                                                index_label_land_r = torch.tensor(
                                                    [[0, int(c), 0, int(y), int(x)] for c, y, x in zip(cam, yy, xx)])
                                                forced_conf = float(conf.max())
                                                logger.info(f"FORCED landmark for label {label} | max confidence {forced_conf:.3f}")

                                        def collect_faces(index_list):
                                            return [tens_pix_to_face_model[idx[0], idx[1], idx[2], idx[3], idx[4]] for idx in index_list]

                                        # recover the face in my mesh
                                        num_faces_r = collect_faces(index_label_land_r)

                                        dico_rgb = {}
                                        off_aim = False
                                        if models_type == "MG":
                                            # The MG landmark lies on the gingiva, not on the tooth
                                            # crown: keep every rendered face instead of filtering by
                                            # the tooth region id (RemoveExtraFaces would drop them all)
                                            last_num_faces_r = [face for face in num_faces_r if int(face.item()) >= 0]
                                            # Which is why MG needs its own filter: with none at all,
                                            # a neighbour's mucogingival point marked in the same
                                            # picture is averaged into this tooth's answer.
                                            if args.pick_patch and getattr(agent, "aim_points", None) is not None:
                                                vertices_np = V[0].detach().cpu().numpy()
                                                if patient_id not in mg_pitch:
                                                    mg_pitch[patient_id] = ToothPitch(
                                                        RI.squeeze(0).detach().cpu().numpy(), vertices_np)
                                                pitch = mg_pitch[patient_id]
                                                last_num_faces_r, off_aim = PickNearAim(
                                                    last_num_faces_r,
                                                    F[0].detach().cpu().numpy(),
                                                    vertices_np,
                                                    agent.aim_points[0].detach().cpu().numpy(),
                                                    radius=pitch / 2 if pitch else None,
                                                    name=f"label {label}")
                                            dico_rgb[LABEL[str(label)][MODELS_DICT['MG']['MG']]] = last_num_faces_r
                                        else:
                                            index_label_land_g = (pred_data == 2.).nonzero(as_tuple=False)
                                            index_label_land_b = (pred_data == 3.).nonzero(as_tuple=False)

                                            num_faces_g = collect_faces(index_label_land_g)
                                            num_faces_b = collect_faces(index_label_land_b)

                                            last_num_faces_r = RemoveExtraFaces(F, num_faces_r, RI, int(label))
                                            last_num_faces_g = RemoveExtraFaces(F, num_faces_g, RI, int(label))
                                            last_num_faces_b = RemoveExtraFaces(F, num_faces_b, RI, int(label))

                                            if models_type == "O":
                                                logger.debug(f"Processing Occlusal model, label: {LABEL[str(label)]}")
                                                dico_rgb[LABEL[str(label)][MODELS_DICT['O']['O']]] = last_num_faces_r
                                                dico_rgb[LABEL[str(label)][MODELS_DICT['O']['MB']]] = last_num_faces_g
                                                dico_rgb[LABEL[str(label)][MODELS_DICT['O']['DB']]] = last_num_faces_b

                                            else:
                                                dico_rgb[LABEL[str(label)][MODELS_DICT['C']['CL']]] = last_num_faces_r
                                                dico_rgb[LABEL[str(label)][MODELS_DICT['C']['CB']]] = last_num_faces_g

                                        locator = vtk.vtkOctreePointLocator()
                                        locator.SetDataSet(surf_unit)
                                        locator.BuildLocator()

                                        for land_name, face_ids in dico_rgb.items():
                                            logger.debug(f'Processing landmark: {land_name}')
                                            try:
                                                all_verts = [int(F[0][int(face.item())][i].item()) for face in face_ids for i in range(3)]
                                                if all_verts:
                                                    if models_type == "MG":
                                                        # Tensor mean, matching the MG reference implementation
                                                        # (sequential summation rounds differently in float32)
                                                        landmark_pos = V[0][all_verts].mean(dim=0)
                                                    else:
                                                        vert_coord = sum(V[0][v] for v in all_verts)
                                                        landmark_pos = vert_coord / len(all_verts)
                                                    pid = locator.FindClosestPoint(landmark_pos.cpu().numpy())
                                                    closest_pos = torch.tensor(surf_unit.GetPoint(pid))
                                                    if refine and pass_index == 0:
                                                        first_aim[int(label)] = closest_pos.view(1, 3).to(
                                                            DEVICE, dtype=torch.float32)
                                                    upscale_pos = Upscale(closest_pos, mean_arr, scale_factor)
                                                    final = upscale_pos.detach().cpu().numpy()

                                                    # The second look is only worth having when it agrees
                                                    # roughly with the first. Where the first prediction was
                                                    # already off, aiming the cameras at it sends the second
                                                    # further astray -- on scans unlike the training ones a
                                                    # third of the points jumped more than this, some of them
                                                    # clean off the arch. Past the limit the first answer stands.
                                                    if pass_index == 1 and int(label) in first_point:
                                                        jumped = float(np.linalg.norm(final - first_point[int(label)]))
                                                        if jumped > REFINE_LIMIT:
                                                            logger.info(
                                                                f"{land_name}: the second look moved it {jumped:.1f} mm, "
                                                                f"further than {REFINE_LIMIT:.0f} mm, keeping the first")
                                                            continue
                                                    if refine and pass_index == 0:
                                                        first_point[int(label)] = final

                                                    if models_type == "MG" and getattr(agent, "aim_points", None) is not None:
                                                        # Where this tooth's cameras were aimed, in the same
                                                        # frame as the point, for deciding later which of two
                                                        # landmarks on one spot is really this one's. Recorded
                                                        # after the point is settled and guarded on its own:
                                                        # this is a note about the answer, and failing to take
                                                        # it must never cost the answer.
                                                        try:
                                                            aim = Upscale(agent.aim_points[0].detach().cpu(),
                                                                          mean_arr, scale_factor)
                                                            mg_aims[land_name] = np.asarray(aim, dtype=float)
                                                            mg_scale_factor = scale_factor
                                                        except Exception as error:
                                                            logger.warning(f"Could not record the aim of {land_name}: {error}")

                                                    entry = {"x": final[0], "y": final[1], "z": final[2]}
                                                    # Flag degraded points in the json: they need a clinical review
                                                    notes = []
                                                    if estimated is not None:
                                                        notes.append("cameras aimed from an arch fit, tooth not segmented")
                                                    if forced_conf is not None:
                                                        notes.append(f"forced (confidence {forced_conf:.3f})")
                                                    elif won_conf is not None:
                                                        notes.append(f"confidence {won_conf:.3f}")
                                                    if off_aim:
                                                        # The network marked a mucogingival point, but
                                                        # not one this tooth's cameras were aimed at:
                                                        # its neighbour's, in the same picture. Said
                                                        # here so nothing downstream reads it as a
                                                        # measurement of this tooth.
                                                        notes.append(OFF_AIM_NOTE)
                                                    if notes:
                                                        entry["desc"] = "; ".join(notes)
                                                    group_data[land_name] = entry
                                                elif models_type == "MG" and args.force_landmarks:
                                                    # Last resort: not one predicted pixel landed on the mesh.
                                                    # Anchor the point on the tooth itself, lowered toward the
                                                    # gum by the offset the cameras already aim at (0.2 in
                                                    # unit-sphere space), then snapped to the surface.
                                                    anchor = agent.positions.view(-1, 3)[0].clone()
                                                    anchor[2] -= 0.2
                                                    pid = locator.FindClosestPoint(anchor.detach().cpu().numpy())
                                                    pos = Upscale(torch.tensor(surf_unit.GetPoint(pid)), mean_arr, scale_factor).detach().cpu().numpy()
                                                    group_data[land_name] = {
                                                        "x": pos[0], "y": pos[1], "z": pos[2],
                                                        "desc": "fallback (nothing predicted on the mesh)"}
                                                    logger.info(f"FALLBACK landmark for label {label} anchored on the tooth")
                                                else:
                                                    logger.warning(f"No vertices found for landmark {land_name}")
                                            except Exception as e:
                                                logger.error(f"Error processing landmark {land_name}: {e}")
                                                continue
                                    except Exception as e:
                                        logger.error(f"Error during neural network inference for label {label}: {e}")
                                        continue
                                else:
                                    reason = ("too few teeth are segmented to estimate it"
                                              if args.estimate_missing else
                                              "a point aimed at a guessed position lands 4 to 21 mm "
                                              "away, so it is left out (--estimate_missing places it)")
                                    logger.warning(
                                        f"Label {label} is not in the segmentation of {patient_id} "
                                        f"and {reason}: the landmark(s) {LABEL[str(label)]} "
                                        "are not placed")
                                    
                            except Exception as e:
                                logger.error(f"Error processing label {label} for patient {patient_id}: {e}")
                                continue
                        
                        if models_type == "MG":
                            # A skipped tooth is easy to miss in the log stream:
                            # say in one line how much of the line is missing.
                            requested = [LABEL[str(label)][MODELS_DICT['MG']['MG']] for label in lst_teeth]
                            missing = [name for name in requested if name not in group_data]
                            if missing:
                                logger.warning(
                                    f"{patient_id}: only {len(requested) - len(missing)} of the "
                                    f"{len(requested)} requested MG landmarks were placed. Missing: "
                                    f"{', '.join(missing)} — their teeth are not in the segmentation "
                                    "(Universal_ID / PredictedID). The curve spans the gap; pass "
                                    "--estimate_missing to place a point there anyway, 4 to 21 mm "
                                    "off in the scans this was measured on")

                        if models_type == "MG" and args.pick_patch:
                            # Two landmarks on one spot: the network found the
                            # same mucogingival point twice. Decide which tooth
                            # it belongs to and record the other as doubtful,
                            # so the hole-filling below rebuilds it instead of
                            # leaving a duplicate.
                            pitch = next(iter(mg_pitch.values()), None)
                            ResolveCollisions(group_data, mg_aims,
                                              pitch / mg_scale_factor
                                              if pitch and mg_scale_factor else None,
                                              note=OFF_AIM_NOTE)

                        # A hole in the line is awkward to work with, and the
                        # curve through the points that are trusted goes as
                        # close to a missing one as the network itself does.
                        if models_type == "MG" and args.fill_gaps:
                            FillGaps(group_data)

                        if models_type == "MG" and args.pick_patch:
                            # A point that is on its neighbour's spot and could
                            # not be rebuilt is left out rather than written
                            # where it is: it is a duplicate of the landmark
                            # beside it, and the same rule already applies to
                            # the tooth a scan does not have -- a point nobody
                            # can stand behind is worse than a hole, which the
                            # curve spans.
                            stranded = [name for name, entry in group_data.items()
                                        if OFF_AIM_NOTE in (entry.get("desc") or "")]
                            for name in stranded:
                                del group_data[name]
                            if stranded:
                                logger.info(
                                    f"{patient_id}: leaving out {len(stranded)} landmark(s) "
                                    f"found on a neighbouring tooth's point and impossible to "
                                    f"rebuild: {', '.join(sorted(stranded))}")

                        if models_type == "MG" and args.complete_line:
                            # Whatever is still missing gets a position, marked
                            # as extrapolated. It is there so the line always
                            # has its 13 points; the note keeps it out of the
                            # band AREG registers on.
                            #
                            # The scan goes with it: an end of the arch is then
                            # placed off its own tooth's gingival collar rather
                            # than by running the spline past its support, and
                            # every answer is put back on the mesh. Measured on
                            # scans held back from the choosing, that is 1.44 mm
                            # instead of 5.11, and 70% within 2 mm instead of 6%.
                            # It is `path_vtk`, not the file the user gave: the
                            # landmarks are still in the oriented frame here,
                            # and the back-transform below is what moves them.
                            CompleteLine(group_data, surf=ReadSurf(path_vtk))

                        # last, once every point is there: the line as a whole
                        # knows more about any one point than that point does
                        if models_type == "MG" and args.smooth:
                            SmoothAlongArch(group_data, args.smooth_strength
                                            if args.smooth_strength is not None
                                            else SMOOTH_STRENGTH)

                        if models_type == "MG" and args.snap_to_surface:
                            # Last of all, and after the smoothing: a
                            # mucogingival point is on the mucosa, and the
                            # spline through a hole's neighbours and the pull
                            # toward the line both leave the surface.
                            SnapAll(group_data, ReadSurf(path_vtk))

                        if back_to_file is not None:
                            # The prediction ran on the oriented copy; what is
                            # written has to be in the coordinates of the file
                            # the user gave, or nothing lines up with it.
                            for entry in group_data.values():
                                x, y, z = TransformPoint(
                                    (entry["x"], entry["y"], entry["z"]), back_to_file)
                                entry["x"], entry["y"], entry["z"] = x, y, z

                        if models_type == "MG" and args.paint_scan and group_data:
                            # On the file the user gave, in its own coordinates:
                            # the landmarks have just been transformed back into
                            # them. Only two arrays are added, so the scan stays
                            # the scan -- and a failure here costs the drawing,
                            # never the landmarks.
                            labels = None
                            if segmented_path is not None:
                                try:
                                    labelled = ReadSurf(segmented_path)
                                    array = labelled.GetPointData().GetArray("Universal_ID")
                                    if array is not None:
                                        labels = vtk_to_numpy(array)
                                except Exception as error:
                                    logger.warning(f"Could not read the segmentation back: {error}")
                            positions = {name: (entry["x"], entry["y"], entry["z"])
                                         for name, entry in group_data.items()}
                            descriptions = {name: entry.get("desc")
                                            for name, entry in group_data.items()}
                            PaintScan(patient_path, positions, labels=labels,
                                      descriptions=descriptions)

                        if len(group_data.keys()) > 0:
                            try:
                                lm_lst = GenControlPoint(group_data, landmarks_selected)
                                output_file = os.path.join(args.output_dir, f"{patient_id}_{jaw}_{models_type}_Pred.json")
                                WriteJson(lm_lst, output_file)
                                logger.info(f"Saved predictions to {output_file}")
                            except Exception as e:
                                logger.error(f"Error saving predictions for {patient_id}_{jaw}_{models_type}: {e}")

                        if back_to_file is not None:
                            # The oriented copy has served its purpose.
                            shutil.rmtree(os.path.dirname(path_vtk), ignore_errors=True)
                        if unified_folder is not None:
                            shutil.rmtree(unified_folder, ignore_errors=True)
                        if segmented_folder is not None:
                            shutil.rmtree(segmented_folder, ignore_errors=True)
                                
                    except Exception as e:
                        logger.error(f"Error processing jaw {jaw} for patient {patient_id}, model {models_type}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing model type {models_type} for patient {patient_id}: {e}")
                continue
        
        # Update log file with progress
        try:
            with open(args.log_path, "w+") as log_f:
                log_f.write(str(idx + 1))
        except Exception as e:
            logger.error(f"Failed to update log file: {e}")


if __name__ == "__main__":
    try:
        logger.info("Starting ALI_IOS application")
        logger.info(f"Command line arguments: {sys.argv}")
        
        parser = argparse.ArgumentParser(description="Automatic Landmark Identification for Intraoral Scans")
        parser.add_argument("input", type=str, help="Input VTK file or folder containing VTK files")
        parser.add_argument("dir_models", type=str, help="Directory containing trained models")
        parser.add_argument("lm_type", type=str, help="Type of landmarks to identify")
        parser.add_argument("teeth", type=str, help="Teeth to process")
        parser.add_argument("teeth_mg", type=str, help="Lower teeth for mucogingival (MG) landmarks, 'None' to disable")
        parser.add_argument("output_dir", type=str, help="Output directory for predictions")
        parser.add_argument("image_size", default="224", type=str, help="Image size for neural network")
        parser.add_argument("blur_radius", default="0", type=str, help="Blur radius for rendering")
        parser.add_argument("faces_per_pixel", default="1", type=str, help="Faces per pixel in rasterization")
        parser.add_argument("log_path", type=str, help="Path to log file")
        parser.add_argument("--force_landmarks", dest="force_landmarks", action="store_true", default=True,
                            help="always place an MG point, even when the landmark class wins no pixel: "
                                 "the most likely pixels for that class are used instead. Forced points "
                                 "are ~5 mm off instead of ~1.2 mm and are marked 'forced' in the json "
                                 "description. Use --no-force_landmarks to leave the landmark out instead")
        parser.add_argument("--no-force_landmarks", dest="force_landmarks", action="store_false",
                            help="leave an MG landmark out when the network predicts nothing")
        parser.add_argument("--force_topk", type=int, default=50,
                            help="number of most likely pixels averaged when an MG landmark is forced")
        parser.add_argument("--estimate_missing", dest="estimate_missing", action="store_true",
                            default=False,
                            help="place an MG point for a tooth the segmentation does not have, by "
                                 "aiming the cameras at a position fitted through the arch. Measured "
                                 "against hand annotations those points land 4 to 21 mm away, where a "
                                 "point aimed at a real tooth lands within 0.5 mm, so they are left out "
                                 "by default: the curve simply spans the gap")
        parser.add_argument("--no-estimate_missing", dest="estimate_missing", action="store_false",
                            help="leave out the MG landmark of a tooth absent from the segmentation")
        parser.add_argument("--arch_scale", dest="arch_scale", action="store_true", default=True,
                            help="normalise the MG scan on the distance between its first molars "
                                 "rather than on its bounding box, so the cameras frame the gum "
                                 "the same way whatever amount of vestibule was scanned. Measured "
                                 "better on the training corpus and better still outside it")
        parser.add_argument("--no-arch_scale", dest="arch_scale", action="store_false",
                            help="normalise the MG scan on its bounding box, as before")
        parser.add_argument("--smooth", dest="smooth", action="store_true", default=True,
                            help="pull each mucogingival landmark part of the way onto the curve "
                                 "its neighbours draw, so a point that strays comes back")
        parser.add_argument("--no-smooth", dest="smooth", action="store_false",
                            help="leave every landmark exactly where the network put it")
        parser.add_argument("--smooth_strength", type=float, default=None,
                            help="how far along, 0 to 1. Defaults to 0.3")
        parser.add_argument("--refine", dest="refine", action="store_true", default=True,
                            help="look twice at each tooth: the second time with the cameras aimed "
                                 "at where the first look found the landmark, instead of at the "
                                 "anatomical prior. Costs about 11 s more per arch")
        parser.add_argument("--no-refine", dest="refine", action="store_false",
                            help="look at each tooth once")
        parser.add_argument("--arch_ratio", type=float, default=None,
                            help="with --arch_scale, how wide the framing is: the scan extent the "
                                 "normalisation pretends to see, as a multiple of the distance "
                                 "between the first molars. Defaults to the training corpus value")
        parser.add_argument("--snap_to_surface", dest="snap_to_surface", action="store_true",
                            default=True,
                            help="put every mucogingival landmark back on the mesh once the "
                                 "line is finished. The network's own points are already on it; "
                                 "a point rebuilt from its neighbours sits a median 0.60 mm above "
                                 "it and up to 4.89 mm, because a spline does not follow a surface")
        parser.add_argument("--no-snap_to_surface", dest="snap_to_surface",
                            action="store_false",
                            help="leave the landmarks where the curve put them")
        parser.add_argument("--complete_line", dest="complete_line", action="store_true", default=True,
                            help="always write the 13 landmarks: whatever the prediction could not "
                                 "give is extrapolated from the curve through the rest and marked "
                                 "'extrapolated'. Measured at an end of the arch, such a point is "
                                 "9.7 mm from the annotation, so it is a position to look at rather "
                                 "than a measurement -- AREG leaves it out of the band it builds")
        parser.add_argument("--no-complete_line", dest="complete_line", action="store_false",
                            help="leave the line short where nothing could be measured")
        parser.add_argument("--paint_scan", dest="paint_scan", action="store_true", default=True,
                            help="add the mucogingival band (Bottom_MGL) and the landmarks "
                                 "(MG_landmarks) to the scan itself, as two 0/1 point arrays, "
                                 "so the line can be looked at by opening that file. Nothing "
                                 "else in it is touched, and only .vtk and .vtp can carry them")
        parser.add_argument("--no-paint_scan", dest="paint_scan", action="store_false",
                            help="leave the scan exactly as it was given")
        parser.add_argument("--pick_patch", dest="pick_patch", action="store_true", default=True,
                            help="when the network marks several separate spots in one picture -- "
                                 "this tooth's mucogingival point and a neighbour's -- keep only the "
                                 "one the cameras were aimed at, instead of averaging them into a "
                                 "position that belongs to neither")
        parser.add_argument("--no-pick_patch", dest="pick_patch", action="store_false",
                            help="average every marked spot into one landmark, as before")
        parser.add_argument("--fill_gaps", dest="fill_gaps", action="store_true", default=True,
                            help="rebuild an MG landmark the prediction could not give -- its tooth "
                                 "absent, or the network unsure of it -- by following the curve "
                                 "through the points that are trusted, when there are trusted "
                                 "points on both sides. Measured at 1.53 mm from the annotation "
                                 "against 1.20 mm for a predicted point. Marked as rebuilt")
        parser.add_argument("--no-fill_gaps", dest="fill_gaps", action="store_false",
                            help="leave a hole where a landmark could not be predicted")

        args = parser.parse_args()
        
        # Validate output directory
        if not os.path.exists(args.output_dir):
            try:
                os.makedirs(args.output_dir)
                logger.info(f"Created output directory: {args.output_dir}")
            except Exception as e:
                logger.error(f"Failed to create output directory: {e}")
                sys.exit(1)
        
        main(args)
        logger.info("ALI_IOS completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in ALI_IOS: {e}")
        sys.exit(1)