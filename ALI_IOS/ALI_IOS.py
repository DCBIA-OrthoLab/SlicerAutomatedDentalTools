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
import vtk
import platform
import argparse
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

fpath = os.path.join(os.path.dirname(__file__), "..")
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
    from ALI_IOS_utils.surface import ReadSurf, ScaleSurf, GetSurfProp, RemoveExtraFaces, Upscale
    from ALI_IOS_utils.model import dic_cam, dic_label, MODELS_DICT
    from ALI_IOS_utils.io import GenControlPoint, WriteJson, TradLabel
    from ALI_IOS_utils.agent import Agent
    
else :
    from ALI_IOS_utils import (
        GenPhongRenderer, ReadSurf, ScaleSurf,
        GetSurfProp, RemoveExtraFaces, Upscale,
        dic_cam, dic_label, MODELS_DICT,
        GenControlPoint, WriteJson, TradLabel, Agent
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        lm_types = args.lm_type.replace("'", "").replace('"', '').split(" ")
        teeth = [tooth.strip().replace("'", "").replace('"', '') for tooth in args.teeth.split(" ")]
        
        if not lm_types or not teeth:
            logger.error("Landmark types or teeth list is empty")
            raise ValueError("Invalid landmark types or teeth list")
        
        landmarks_selected = [tooth + lm_type for tooth in teeth for lm_type in lm_types]
        logger.info(f"Processing landmarks: {landmarks_selected}")
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        sys.exit(1)
    
    # Translate labels
    try:
        dic_teeth = TradLabel(teeth)
        logger.debug(f"Tooth labels translated: {dic_teeth}")
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
        for img_fn in glob.iglob(normpath, recursive=True):
            basename = os.path.basename(img_fn)
            if basename.endswith(".pth"):
                try:
                    model_id = basename.split("_")[1]
                    if model_id not in available_models.keys():
                        available_models[model_id] = {}
                    if 'Lower' in basename:
                        available_models[model_id]['Lower'] = (img_fn)
                    else:
                        available_models[model_id]['Upper'] = (img_fn)
                except Exception as e:
                    logger.warning(f"Error processing model file {basename}: {e}")
                    continue
        
        logger.info(f'Available models: {available_models}')

        for model_id in MODELS_DICT.keys():
            if model_id in available_models:
                for lmtype in lm_types:
                    if lmtype in MODELS_DICT[model_id].keys():
                        if model_id not in models_to_use.keys():
                            models_to_use[model_id] = available_models[model_id]

        logger.info(f'Models to use: {models_to_use}')
        
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
                if os.path.isfile(vtkfile) and True in [ext in vtkfile for ext in [".vtk"]]:
                    basename = os.path.basename(vtkfile).split('.')[0]
                    if basename not in dic_patients.keys():
                        dic_patients[basename] = vtkfile
        
        if not dic_patients:
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
                sphere_radius = 0.2 if models_type == "O" else 0.3

                logger.debug(f"Processing model type: {models_type}")
                
                for jaw, lst_teeth in dic_teeth.items():
                    group_data = {}

                    try:
                        path_vtk = patient_path
                        model = models_to_use[models_type]['Lower'] if jaw == 'Lower' else models_to_use[models_type]['Upper']
                        camera_position = dic_cam[models_type]['L'] if jaw == 'Lower' else dic_cam[models_type]['U']
                        
                        for label in lst_teeth:
                            try:
                                logger.debug(f"Loading model for patient {patient_id}, label {label}, jaw {jaw}")
                                
                                phong_renderer, mask_renderer = GenPhongRenderer(
                                    int(args.image_size), int(args.blur_radius), int(args.faces_per_pixel), DEVICE
                                )

                                agent = Agent(
                                    renderer=phong_renderer,
                                    renderer2=mask_renderer,
                                    radius=sphere_radius,
                                    camera_position=camera_position
                                )

                                SURF = ReadSurf(path_vtk)
                                surf_unit, mean_arr, scale_factor = ScaleSurf(SURF)
                                (V, F, CN, RI) = GetSurfProp(surf_unit, mean_arr, scale_factor)

                                if int(label) in RI.squeeze(0):
                                    agent.position_agent(RI, V, label)
                                    textures = TexturesVertex(verts_features=CN)
                                    meshe = Meshes(verts=V, faces=F, textures=textures).to(DEVICE)

                                    try:
                                        images_model, tens_pix_to_face_model = agent.get_view_rasterize(meshe)
                                        tens_pix_to_face_model = tens_pix_to_face_model.permute(1, 0, 4, 2, 3)

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

                                        post_pred = AsDiscrete(argmax=True, to_onehot=4)

                                        val_pred = torch.empty((0)).to(DEVICE)
                                        for image in images_pred:
                                            val_pred = torch.cat((val_pred, post_pred(image).unsqueeze(0).to(DEVICE)), dim=0)

                                        pred_data = images_pred.detach().cpu().unsqueeze(0).type(torch.int16)
                                        pred_data = torch.argmax(pred_data, dim=2).unsqueeze(2)

                                        # recover where there is the landmark in the image
                                        index_label_land_r = (pred_data == 1.).nonzero(as_tuple=False)
                                        index_label_land_g = (pred_data == 2.).nonzero(as_tuple=False)
                                        index_label_land_b = (pred_data == 3.).nonzero(as_tuple=False)

                                        def collect_faces(index_list):
                                            return [tens_pix_to_face_model[idx[0], idx[1], idx[2], idx[3], idx[4]] for idx in index_list]

                                        # recover the face in my mesh
                                        num_faces_r = collect_faces(index_label_land_r)
                                        num_faces_g = collect_faces(index_label_land_g)
                                        num_faces_b = collect_faces(index_label_land_b)

                                        last_num_faces_r = RemoveExtraFaces(F, num_faces_r, RI, int(label))
                                        last_num_faces_g = RemoveExtraFaces(F, num_faces_g, RI, int(label))
                                        last_num_faces_b = RemoveExtraFaces(F, num_faces_b, RI, int(label))

                                        dico_rgb = {}
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
                                                    vert_coord = sum(V[0][v] for v in all_verts)
                                                    landmark_pos = vert_coord / len(all_verts)
                                                    pid = locator.FindClosestPoint(landmark_pos.cpu().numpy())
                                                    closest_pos = torch.tensor(surf_unit.GetPoint(pid))
                                                    upscale_pos = Upscale(closest_pos, mean_arr, scale_factor)
                                                    final = upscale_pos.detach().cpu().numpy()
                                                    
                                                    group_data[land_name] = {"x": final[0], "y": final[1], "z": final[2]}
                                                else:
                                                    logger.warning(f"No vertices found for landmark {land_name}")
                                            except Exception as e:
                                                logger.error(f"Error processing landmark {land_name}: {e}")
                                                continue
                                    except Exception as e:
                                        logger.error(f"Error during neural network inference for label {label}: {e}")
                                        continue
                                else:
                                    logger.debug(f"Label {label} not found in surface")
                                    
                            except Exception as e:
                                logger.error(f"Error processing label {label} for patient {patient_id}: {e}")
                                continue
                        
                        if len(group_data.keys()) > 0:
                            try:
                                lm_lst = GenControlPoint(group_data, landmarks_selected)
                                output_file = os.path.join(args.output_dir, f"{patient_id}_{jaw}_{models_type}_Pred.json")
                                WriteJson(lm_lst, output_file)
                                logger.info(f"Saved predictions to {output_file}")
                            except Exception as e:
                                logger.error(f"Error saving predictions for {patient_id}_{jaw}_{models_type}: {e}")
                                
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
        parser.add_argument("output_dir", type=str, help="Output directory for predictions")
        parser.add_argument("image_size", default="224", type=str, help="Image size for neural network")
        parser.add_argument("blur_radius", default="0", type=str, help="Blur radius for rendering")
        parser.add_argument("faces_per_pixel", default="1", type=str, help="Faces per pixel in rasterization")
        parser.add_argument("log_path", type=str, help="Path to log file")

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