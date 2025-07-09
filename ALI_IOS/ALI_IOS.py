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
    print("args : ",args)
    
    if not os.path.exists(os.path.split(args.log_path)[0]):
        os.mkdir(os.path.split(args.log_path)[0])
        
    with open(args.log_path, "w") as log_f:
        log_f.truncate(0)
        
    lm_types = args.lm_type.replace("'", "").replace('"', '').split(" ")
    teeth = [tooth.strip().replace("'", "").replace('"', '') for tooth in args.teeth.split(" ")]
    
    landmarks_selected = [tooth + lm_type for tooth in teeth for lm_type in lm_types]
    dic_teeth = TradLabel(teeth)

    # Find available models in folder
    available_models = {}
    models_to_use = {}
    print("Loading models from", args.dir_models)
    normpath = os.path.normpath("/".join([args.dir_models, '**', '']))
    for img_fn in glob.iglob(normpath, recursive=True):
        basename = os.path.basename(img_fn)
        if basename.endswith(".pth"):
            model_id = basename.split("_")[1]
            if model_id not in available_models.keys():
                available_models[model_id] = {}
            if 'Lower' in basename:
                available_models[model_id]['Lower'] = (img_fn)
            else:
                available_models[model_id]['Upper'] = (img_fn)
    print('available_models :',available_models)

    for model_id in MODELS_DICT.keys():
        if model_id in available_models:
            for lmtype in lm_types:
                if lmtype in MODELS_DICT[model_id].keys():
                    if model_id not in models_to_use.keys():
                        models_to_use[model_id] = available_models[model_id]

    print('models_to_use :',models_to_use)


    dic_patients = {}
    if os.path.isfile(args.input):
        print("Loading scan :", args.input)
        basename = os.path.basename(args.input).split('.')[0]
        if basename not in dic_patients.keys():
            dic_patients[basename] = args.input

    else:
        print("Loading data from", args.input)
        normpath = os.path.normpath("/".join([args.input, '**', '']))
        for vtkfile in sorted(glob.iglob(normpath, recursive=True)):
            if os.path.isfile(vtkfile) and True in [ext in vtkfile for ext in [".vtk"]]:
                basename = os.path.basename(vtkfile).split('.')[0]
                if basename not in dic_patients.keys():
                    dic_patients[basename] = vtkfile

    print('dic_patients :',dic_patients)

    total_landmarks = 0
    for jaw_teeth in dic_teeth.values():
        total_landmarks += len(jaw_teeth)
    total_landmarks *= len(dic_patients)
    
    
    for patient_id,patient_path in dic_patients.items():

        print(f"prediction for patient {patient_id}")
        for models_type in models_to_use.keys():
            LABEL = dic_label[models_type]
            sphere_radius = 0.2 if models_type == "O" else 0.3

            print(dic_teeth)
            for jaw, lst_teeth in dic_teeth.items():
                group_data = {}

                path_vtk = patient_path
                model = models_to_use[models_type]['Lower'] if jaw == 'Lower' else models_to_use[models_type]['Upper']
                camera_position = dic_cam[models_type]['L'] if jaw == 'Lower' else dic_cam[models_type]['U']
                
                for label in lst_teeth:
                    print("Loading model :", model, "for patient :", patient_id, "label :", label)
                    phong_renderer, mask_renderer = GenPhongRenderer(
                        int(args.image_size), int(args.blur_radius), int(args.faces_per_pixel), DEVICE
                    )

                    agent = Agent(
                        renderer=phong_renderer,
                        renderer2=mask_renderer,
                        radius=sphere_radius,
                        camera_position = camera_position
                    )

                    SURF = ReadSurf(path_vtk)
                    surf_unit, mean_arr, scale_factor= ScaleSurf(SURF)
                    (V, F, CN, RI) = GetSurfProp(surf_unit, mean_arr, scale_factor)

                    if int(label) in RI.squeeze(0):
                        agent.position_agent(RI, V, label)
                        textures = TexturesVertex(verts_features=CN)
                        meshe = Meshes(verts=V, faces=F, textures=textures).to(DEVICE)

                        images_model, tens_pix_to_face_model = agent.get_view_rasterize(meshe) #[batch,num_ima,channels,size,size] torch.Size([1, 2, 4, 224, 224])
                        tens_pix_to_face_model = tens_pix_to_face_model.permute(1,0,4,2,3) #tens_pix_to_face : torch.Size([1, 2, 1, 224, 224])

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
                            val_pred = torch.cat((val_pred,post_pred(image).unsqueeze(0).to(DEVICE)),dim=0)

                        pred_data = images_pred.detach().cpu().unsqueeze(0).type(torch.int16) #torch.Size([1, 2, 2, 224, 224])
                        pred_data = torch.argmax(pred_data, dim=2).unsqueeze(2)

                        # recover where there is the landmark in the image
                        index_label_land_r = (pred_data==1.).nonzero(as_tuple=False) #torch.Size([6252, 5])
                        index_label_land_g = (pred_data==2.).nonzero(as_tuple=False) #torch.Size([6252, 5])
                        index_label_land_b = (pred_data==3.).nonzero(as_tuple=False) #torch.Size([6252, 5])

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
                            print(LABEL[str(label)])
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
                            print('land_name :',land_name)
                            all_verts = [int(F[0][int(face.item())][i].item()) for face in face_ids for i in range(3)]
                            vert_coord = sum(V[0][v] for v in all_verts)
                            
                            if all_verts:
                                landmark_pos = vert_coord / len(all_verts)
                                pid = locator.FindClosestPoint(landmark_pos.cpu().numpy())
                                closest_pos = torch.tensor(surf_unit.GetPoint(pid))
                                upscale_pos = Upscale(closest_pos, mean_arr, scale_factor)
                                final = upscale_pos.detach().cpu().numpy()
                                
                                if jaw not in group_data.keys():
                                    group_data[jaw] = {}

                                group_data[land_name] = {"x": final[0], "y": final[1], "z": final[2]}

                    # print(f"""<filter-progress>{1}</filter-progress>""")
                    # sys.stdout.flush()
                    # time.sleep(0.5)
                    # print(f"""<filter-progress>{0}</filter-progress>""")

                if len(group_data.keys()) > 0:
                    lm_lst = GenControlPoint(group_data, landmarks_selected)
                    WriteJson(lm_lst,os.path.join(args.output_dir,f"{patient_id}_{jaw}_{models_type}_Pred.json"))
                    
                    with open(args.log_path, "r+") as log_f:
                        log_f.write(patient_id)


if __name__ == "__main__":

    print("Starting")
    print(sys.argv)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("dir_models", type=str)
    parser.add_argument("lm_type", type=str)
    parser.add_argument("teeth", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("image_size", default="224",type=str)
    parser.add_argument("blur_radius", default="0",type=str)
    parser.add_argument("faces_per_pixel", default="1",type=str)
    parser.add_argument("log_path", type=str)

    args = parser.parse_args()
    main(args)