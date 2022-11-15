#!/usr/bin/env python-real

"""
AUTOMATIC LANDMARK IDENTIFICATION IN INTRAORAL SCANS (ALI_CBCT)

Authors :
- Maxime Gillot (UoM)
- Baptiste Baquero (UoM)
"""


import time
import os
import glob
import sys
import json
import vtk
import numpy as np

# try:
#     import argparse
# except ImportError:
#     pip_install('argparse')
#     import argparse


# print(sys.argv)


from slicer.util import pip_install

# from slicer.util import pip_uninstall
# # pip_uninstall('torch torchvision torchaudio') 

# pip_uninstall('monai')

try:
    import torch
except ImportError:
    pip_install('torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
    import torch

try :
    from monai.networks.nets import UNETR
except ImportError:
    pip_install('monai==0.7.0')
    from monai.networks.nets import UNETR

from platform import system # to know which OS is used

if system() == 'Darwin':  # MACOS
    try:
        import pytorch3d
    except ImportError:
        pip_install('pytorch3d')
        import pytorch3d

else: # Linux or Windows
    try:
        import pytorch3d
        if pytorch3d.__version__ != '0.6.2':
            raise ImportError
    except ImportError:
        try:
        #   import torch
            pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
            version_str="".join([f"py3{sys.version_info.minor}_cu",torch.version.cuda.replace(".",""),f"_pyt{pyt_version_str}"])
            pip_install('--upgrade pip')
            pip_install('fvcore==0.1.5.post20220305')
            pip_install('--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
        except: # install correct torch version
            pip_install('--no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113') 
            pip_install('--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html')


import torch.nn as nn
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import Materials
from typing import Optional
from pytorch3d.renderer.blending import (hard_rgb_blend,BlendParams)
from pytorch3d.renderer.mesh.rasterizer import (Fragments)
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.lighting import PointLights
# from pytorch3d.common.types import Device

from vtk.util.numpy_support import vtk_to_numpy
from monai.networks.nets import UNet
from monai.data import decollate_batch
from monai.transforms import (AsDiscrete,ToTensor)
from scipy import linalg
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,look_at_rotation,TexturesVertex,blending

)

dic_cam = { 'O':{
                'L' : ([0,0,1],
                    np.array([0.5,0.,1.0])/linalg.norm([0.5,0.5,1.0]),
                    np.array([-0.5,0.,1.0])/linalg.norm([-0.5,-0.5,1.0]),
                    np.array([0,0.5,1])/linalg.norm([1,0,1]),
                    np.array([0,-0.5,1])/linalg.norm([0,1,1])
                    ),

                'U' : ([0,0,-1],
                    np.array([0.5,0.,-1])/linalg.norm([0.5,0.5,-1]),
                    np.array([-0.5,0.,-1])/linalg.norm([-0.5,-0.5,-1]),
                    np.array([0,0.5,-1])/linalg.norm([1,0,-1]),
                    np.array([0,-0.5,-1])/linalg.norm([0,1,-1])
                    )
                },

            'C' : {
                'L' : (np.array([1,0,0])/linalg.norm([1,0,0]),np.array([-1,0,0])/linalg.norm([-1,0,0]),
                    np.array([1,-1,0])/linalg.norm([1,-1,0]),np.array([-1,-1,0])/linalg.norm([-1,-1,0]),
                    np.array([1,1,0])/linalg.norm([1,1,0]),np.array([-1,1,0])/linalg.norm([-1,1,0]),

                    np.array([1,0,0.5])/linalg.norm([1,0,0.5]),np.array([-1,0,0.5])/linalg.norm([-1,0,0.5]),
                    np.array([1,-1,0.5])/linalg.norm([1,-1,0.5]),np.array([-1,-1,0.5])/linalg.norm([-1,-1,0.5]),
                    np.array([1,1,0.5])/linalg.norm([1,1,0.5]),np.array([-1,1,0.5])/linalg.norm([-1,1,0.5])
                    ),
                
                'U' : (np.array([1,0,0])/linalg.norm([1,0,0]),np.array([-1,0,0])/linalg.norm([-1,0,0]),
                    np.array([1,-1,0])/linalg.norm([1,-1,0]),np.array([-1,-1,0])/linalg.norm([-1,-1,0]),
                    np.array([1,1,0])/linalg.norm([1,1,0]),np.array([-1,1,0])/linalg.norm([-1,1,0]),

                    np.array([1,0,-0.5])/linalg.norm([1,0,-0.5]),np.array([-1,0,-0.5])/linalg.norm([-1,0,-0.5]),
                    np.array([1,-1,-0.5])/linalg.norm([1,-1,-0.5]),np.array([-1,-1,-0.5])/linalg.norm([-1,-1,-0.5]),
                    np.array([1,1,-0.5])/linalg.norm([1,1,-0.5]),np.array([-1,1,-0.5])/linalg.norm([-1,1,-0.5])
                )
            }

    }   

LOWER_DENTAL = ['LL7','LL6','LL5','LL4','LL3','LL2','LL1','LR1','LR2','LR3','LR4','LR5','LR6','LR7']

UPPER_DENTAL = ['UL7','UL6','UL5','UL4','UL3','UL2','UL1','UR1','UR2','UR3','UR4','UR5','UR6','UR7']

TYPE_LM = ['O','MB','DB','CL','CB']


Lower = []
Upper = []

for tooth in LOWER_DENTAL:
    for lmtype in TYPE_LM:
        Lower.append(tooth+lmtype)   

for tooth in UPPER_DENTAL:
    for lmtype in TYPE_LM:
        Upper.append(tooth+lmtype)

LANDMARKS = {"L":Lower,"U":Upper}


dic_label = {
    'O' : {
            "15" : LANDMARKS["U"][0:3],
            "14" : LANDMARKS["U"][5:8],
            "13" : LANDMARKS["U"][10:13],
            "12" : LANDMARKS["U"][15:18],
            "11" : LANDMARKS["U"][20:23],
            "10" : LANDMARKS["U"][25:28],
            "9" : LANDMARKS["U"][30:33],
            "8" : LANDMARKS["U"][35:38],
            "7" : LANDMARKS["U"][40:43],
            "6" : LANDMARKS["U"][45:48],
            "5" : LANDMARKS["U"][50:53],
            "4" : LANDMARKS["U"][55:58],
            "3" : LANDMARKS["U"][60:63],
            "2" : LANDMARKS["U"][65:68],

            "18" : LANDMARKS["L"][0:3],
            "19" : LANDMARKS["L"][5:8],
            "20" : LANDMARKS["L"][10:13],
            "21" : LANDMARKS["L"][15:18],
            "22" : LANDMARKS["L"][20:23],
            "23" : LANDMARKS["L"][25:28],
            "24" : LANDMARKS["L"][30:33],
            "25" : LANDMARKS["L"][35:38],
            "26" : LANDMARKS["L"][40:43],
            "27" : LANDMARKS["L"][45:48],
            "28" : LANDMARKS["L"][50:53],
            "29" : LANDMARKS["L"][55:58],
            "30" : LANDMARKS["L"][60:63],
            "31" : LANDMARKS["L"][65:68]
            
        },

    'C' : {
        
        "15" : LANDMARKS["U"][3:5],
        "14" : LANDMARKS["U"][8:10],
        "13" : LANDMARKS["U"][13:15],
        "12" : LANDMARKS["U"][18:20],
        "11" : LANDMARKS["U"][23:25],
        "10" : LANDMARKS["U"][28:30],
        "9" : LANDMARKS["U"][33:35],
        "8" : LANDMARKS["U"][38:40],
        "7" : LANDMARKS["U"][43:45],
        "6" : LANDMARKS["U"][48:50],
        "5" : LANDMARKS["U"][53:55],
        "4" : LANDMARKS["U"][58:60],
        "3" : LANDMARKS["U"][63:65],
        "2" : LANDMARKS["U"][68:70],

        "18" : LANDMARKS["L"][3:5],
        "19" : LANDMARKS["L"][8:10],
        "20" : LANDMARKS["L"][13:15],
        "21" : LANDMARKS["L"][18:20],
        "22" : LANDMARKS["L"][23:25],
        "23" : LANDMARKS["L"][28:30],
        "24" : LANDMARKS["L"][33:35],
        "25" : LANDMARKS["L"][38:40],
        "26" : LANDMARKS["L"][43:45],
        "27" : LANDMARKS["L"][48:50],
        "28" : LANDMARKS["L"][53:55],
        "29" : LANDMARKS["L"][58:60],
        "30" : LANDMARKS["L"][63:65],
        "31" : LANDMARKS["L"][68:70]
        }
        
    }


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

LABEL_L = ["18","19","20","21","22","23","24","25","26","27","28","29","30","31"]

LABEL_U = ["2","3","4","5","6","7","8","9","10","11","12","13","14","15"]

MODELS_DICT = {
                'O':{
                    'O':0,
                    'MB':1,
                    'DB':2
                },
                'C':{
                    'CL':0,
                    'CB':1
                }
            }

def GenPhongRenderer(image_size,blur_radius,faces_per_pixel,device):
    
    cameras = FoVPerspectiveCameras(znear=0.01,zfar = 10, fov= 90, device=device) # Initialize a perspective camera.

    raster_settings = RasterizationSettings(        
        image_size=image_size, 
        blur_radius=blur_radius, 
        faces_per_pixel=faces_per_pixel, 
    )

    lights = PointLights(device=device) # light in front of the object. 

    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    
    b = blending.BlendParams(background_color=(0,0,0))
    phong_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights,blend_params=b)
    )
    mask_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=MaskRenderer(device=device, cameras=cameras, lights=lights,blend_params=b)
    )
    return phong_renderer,mask_renderer

def ReadSurf(fileName):

    fname, extension = os.path.splitext(fileName)
    extension = extension.lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()    
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".off":
        reader = OFFReader()
        reader.SetFileName(fileName)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".obj":
        if os.path.exists(fname + ".mtl"):
            obj_import = vtk.vtkOBJImporter()
            obj_import.SetFileName(fileName)
            obj_import.SetFileNameMTL(fname + ".mtl")
            textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
            if os.path.exists(textures_path):
                obj_import.SetTexturePath(textures_path)
            obj_import.Read()

            actors = obj_import.GetRenderer().GetActors()
            actors.InitTraversal()
            append = vtk.vtkAppendPolyData()

            for i in range(actors.GetNumberOfItems()):
                surfActor = actors.GetNextActor()
                append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())
            
            append.Update()
            surf = append.GetOutput()
            
        else:
            reader = vtk.vtkOBJReader()
            reader.SetFileName(fileName)
            reader.Update()
            surf = reader.GetOutput()

    return surf

def ScaleSurf(surf, mean_arr = None, scale_factor = None):
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    shapedatapoints = surf.GetPoints()

    #calculate bounding box
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    mean_v[0] = (bounds[0] + bounds[1])/2.0
    mean_v[1] = (bounds[2] + bounds[3])/2.0
    mean_v[2] = (bounds[4] + bounds[5])/2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = []
    for i in range(shapedatapoints.GetNumberOfPoints()):
        p = shapedatapoints.GetPoint(i)
        shape_points.append(p)
    shape_points = np.array(shape_points)
    
    #centering points of the shape
    if mean_arr is None:
        mean_arr = np.array(mean_v)
    # print("Mean:", mean_arr)
    shape_points = shape_points - mean_arr

    #Computing scale factor if it is not provided
    if(scale_factor is None):
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1/np.linalg.norm(bounds_max_arr - mean_arr)

    #scale points of the shape by scale factor
    # print("Scale:", scale_factor)
    shape_points_scaled = np.multiply(shape_points, scale_factor)

    #assigning scaled points back to shape
    for i in range(shapedatapoints.GetNumberOfPoints()):
       shapedatapoints.SetPoint(i, shape_points_scaled[i])    

    surf.SetPoints(shapedatapoints)

    return surf, mean_arr, scale_factor

def ComputeNormals(surf):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf);
    normals.ComputeCellNormalsOff();
    normals.ComputePointNormalsOn();
    normals.SplittingOff();
    normals.Update()
    
    return normals.GetOutput()

def GetSurfProp(surf_unit, surf_mean, surf_scale):     
    surf = ComputeNormals(surf_unit)
    color_normals = ToTensor(dtype=torch.float32, device=DEVICE)(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0)
    verts = ToTensor(dtype=torch.float32, device=DEVICE)(vtk_to_numpy(surf.GetPoints().GetData()))
    faces = ToTensor(dtype=torch.int64, device=DEVICE)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])

    try :
        region_id = torch.tensor((vtk_to_numpy(surf.GetPointData().GetScalars("PredictedID"))),dtype=torch.int64)
    except AttributeError :
        try :
            region_id = torch.tensor((vtk_to_numpy(surf.GetPointData().GetScalars("predictedId"))),dtype=torch.int64)
        except AttributeError:
            region_id = torch.tensor((vtk_to_numpy(surf.GetPointData().GetScalars("Universal_ID"))),dtype=torch.int64)

    region_id = torch.clamp(region_id, min=0)

    #print("type(surf.GetPointData()) :",type(surf.GetPointData()))
    #print("type(...GetScalars) :",type(surf.GetPointData().GetScalars("PredictedID")))
    
    '''
    With a file that works
    type(...GetScalars) : <class 'vtkmodules.vtkCommonCore.vtkTypeInt64Array'>
    
    with a file that isn't working
    type(...GetScalars) : <class 'NoneType'>

    AttributeError: 'NoneType' object has no attribute 'GetDataType'
    '''
        
    return verts.unsqueeze(0), faces.unsqueeze(0), color_normals.unsqueeze(0), region_id.unsqueeze(0)

def GetColorArray(surf, array_name):
    colored_points = vtk.vtkUnsignedCharArray()
    colored_points.SetName('colors')
    colored_points.SetNumberOfComponents(3)

    normals = surf.GetPointData().GetArray(array_name)

    for pid in range(surf.GetNumberOfPoints()):
        normal = np.array(normals.GetTuple(pid))
        rgb = (normal*0.5 + 0.5)*255.0
        colored_points.InsertNextTuple3(rgb[0], rgb[1], rgb[2])
    return colored_points

def RemoveExtraFaces(F,num_faces,RI,label):
    last_num_faces =[]
    for face in num_faces:
        vertex_color = F.squeeze(0)[int(face.item())]
        for vert in vertex_color:
            if RI.squeeze(0)[vert] == label:
                last_num_faces.append(face)
    return last_num_faces

def Upscale(landmark_pos,mean_arr,scale_factor):
    new_pos_center = (landmark_pos.cpu()/scale_factor) + mean_arr
    return new_pos_center

# def GenControlePoint(dic_points,landmarks_selected):
#     lm_lst = []
#     false = False
#     true = True
#     id = 0
#     dic_lower = {}
#     dic_upper = {}
#     for patient_id,dic_U_L in dic_points.items():
#         for jaw,dic_landmarks in dic_U_L.items():
#             for landmark in dic_landmarks.keys():
#                 if landmark in landmarks_selected:
#                     id+=1
#                     controle_point = {
#                         "id": str(id),
#                         "label": landmark,
#                         "description": "",
#                         "associatedNodeID": "",
#                         "position": [float(dic_landmarks[landmark]["x"]), float(dic_landmarks[landmark]["y"]), float(dic_landmarks[landmark]["z"])],
#                         "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
#                         "selected": true,
#                         "locked": true,
#                         "visibility": true,
#                         "positionStatus": "preview"
#                     }
#                     # lm_lst.append(controle_point)
#                     if patient_id not in dic_lower.keys():
#                         dic_lower[patient_id] = {}
#                     if jaw not in dic_lower[patient_id].keys():
#                         dic_lower[patient_id][jaw] = {}
#                         if jaw == 'Lower':
#                             dic_lower[patient_id][jaw] = controle_point 
#                         else:
#                             dic_upper[patient_id][jaw] = controle_point

#     return dic_lower,dic_upper


def GenControlePoint(groupe_data,landmarks_selected):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark,data in groupe_data.items():
        if landmark in landmarks_selected:            
            id+=1
            controle_point = {
                "id": str(id),
                "label": landmark,
                "description": "",
                "associatedNodeID": "",
                "position": [data["x"], data["y"], data["z"]],
                "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "selected": true,
                "locked": true,
                "visibility": true,
                "positionStatus": "defined"
            }
            lm_lst.append(controle_point)

    return lm_lst



def WriteJson(lm_lst,out_path):
    false = False
    true = True
    file = {
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
    "markups": [
        {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": false,
            "labelFormat": "%N-%d",
            "controlPoints": lm_lst,
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.0],
                "color": [0.5, 0.5, 0.5],
                "selectedColor": [0.26666666666666669, 0.6745098039215687, 0.39215686274509806],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 2.0,
                "glyphType": "Sphere3D",
                "glyphScale": 2.0,
                "glyphSize": 5.0,
                "useGlyphScale": true,
                "sliceProjection": false,
                "sliceProjectionUseFiducialColor": true,
                "sliceProjectionOutlinedBehindSlicePlane": false,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": false,
                "snapMode": "toVisibleSurface"
            }
        }
    ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=4)

    f.close

def TradLabel(lst_teeth):
    dico_trad ={'LL7':18,'LL6':19,'LL5':20,'LL4':21,'LL3':22,'LL2':23,'LL1':24,'LR1':25,'LR2':26,'LR3':27,'LR4':28,'LR5':29,'LR6':30,'LR7':31,
                'UL7':15,'UL6':14,'UL5':13,'UL4':12,'UL3':11,'UL2':10,'UL1':9,'UR1':8,'UR2':7,'UR3':6,'UR4':5,'UR5':4,'UR6':3,'UR7':2
                }
    dic_teeth = {'Lower':[],'Upper':[]}
    for tooth in lst_teeth:
        if tooth in dico_trad.keys():
            if tooth[0] == 'L':
                dic_teeth['Lower'].append(dico_trad[tooth])
            else:
                dic_teeth['Upper'].append(dico_trad[tooth])
        
    return dic_teeth




class Agent:
    def __init__(
        self,
        renderer, 
        renderer2,
        camera_position,
        radius = 1,
        verbose = True,
        ):
        super(Agent, self).__init__()
        self.renderer = renderer
        self.renderer2=renderer2
        self.camera_points = torch.tensor(camera_position).type(torch.float32).to(DEVICE)
        self.scale = 0
        self.radius = radius
        self.verbose = verbose


    def position_agent(self, text, vert, label):
   
        final_pos = torch.empty((0)).to(DEVICE)
        
        for mesh in range(len(text)):
            if int(label) in text[mesh]:
                index_pos_land = (text[mesh]==int(label)).nonzero(as_tuple=True)[0]
                lst_pos = []
                for index in index_pos_land:
                    lst_pos.append(vert[mesh][index])
                position_agent = sum(lst_pos)/len(lst_pos)
                final_pos = torch.cat((final_pos,position_agent.unsqueeze(0).to(DEVICE)),dim=0)
            else:
                final_pos = torch.cat((final_pos,torch.zeros((1,3)).to(DEVICE)),dim=0)
        # print(final_pos.shape)
        self.positions = final_pos
        # print(self.positions)
        return self.positions

    
    def GetView(self,meshes,rend=False):
        spc = self.positions
        img_lst = torch.empty((0)).to(DEVICE)
        seuil = 0.5

        for sp in self.camera_points:
            sp_i = sp*self.radius
            # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
            current_cam_pos = spc + sp_i
            R = look_at_rotation(current_cam_pos, at=spc, device=DEVICE)  # (1, 3, 3)
            # print( 'R shape :',R.shape)
            # print(R)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)

            if rend:
                renderer = self.renderer2
                images = renderer(meshes_world=meshes.clone(), R=R, T=T.to(DEVICE))
                y = images[:,:,:,:-1]

                # yd = torch.where(y[:,:,:,:]<=seuil,0.,0.)
                yr = torch.where(y[:,:,:,0]>seuil,1.,0.).unsqueeze(-1)
                yg = torch.where(y[:,:,:,1]>seuil,2.,0.).unsqueeze(-1)
                yb = torch.where(y[:,:,:,2]>seuil,3.,0.).unsqueeze(-1)

                y = ( yr + yg + yb).to(torch.float32)

                y = y.permute(0,3,1,2)
              
            else:
                renderer = self.renderer
                images = self.renderer(meshes_world=meshes.clone(), R=R, T=T.to(DEVICE))
                images = images.permute(0,3,1,2)
                images = images[:,:-1,:,:]

                pix_to_face, zbuf, bary_coords, dists = self.renderer.rasterizer(meshes.clone())
                zbuf = zbuf.permute(0, 3, 1, 2)
                y = torch.cat([images, zbuf], dim=1)

            img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
        img_batch =  img_lst.permute(1,0,2,3,4)
        
        return img_batch
    
    def get_view_rasterize(self,meshes):
        spc = self.positions
        img_lst = torch.empty((0)).to(DEVICE)
        tens_pix_to_face = torch.empty((0)).to(DEVICE)

        for sp in self.camera_points:
            sp_i = sp*self.radius
            current_cam_pos = spc + sp_i
            R = look_at_rotation(current_cam_pos, at=spc, device=DEVICE)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)
              
            renderer = self.renderer
            images = renderer(meshes_world=meshes.clone(), R=R, T=T.to(DEVICE))
            images = images.permute(0,3,1,2)
            images = images[:,:-1,:,:]            
            #pix_to_face, zbuf, bary_coords, dists = renderer.rasterizer(meshes.clone())
            temp = renderer.rasterizer(meshes.clone())
            pix_to_face, zbuf = temp.pix_to_face, temp.zbuf
            
            '''< Class : pytorch3d.renderer.mesh.rasterizer.Fragments >'''
            '''TypeError: cannot unpack non-iterable Fragments object'''
            zbuf = zbuf.permute(0, 3, 1, 2)
            y = torch.cat([images, zbuf], dim=1)

            img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
            tens_pix_to_face = torch.cat((tens_pix_to_face,pix_to_face.unsqueeze(0)),dim=0)
        img_batch =  img_lst.permute(1,0,2,3,4)
    
        return img_batch , tens_pix_to_face  

class MaskRenderer(nn.Module):

    def __init__(
        self,
        device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardFlatShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = texels   
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images

def main(args):

    landmarks_selected = []
    for tooth in args['teeth']:
        for lm_type in args['lm_type']:
            landmarks_selected.append(tooth+lm_type)
    print(landmarks_selected)



    
    
    print(LANDMARKS)
    
    # print(dic_label['O'])
    

    # print(MODELS_DICT['O']['O'])
    dic_teeth = TradLabel(args["teeth"])
    # print(dic_teeth)


    # Find available models in folder
    available_models = {}
    models_to_use = {}
    print("Loading models from", args["dir_models"])
    normpath = os.path.normpath("/".join([args["dir_models"], '**', '']))
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
    
    # for model_id in MODELS_DICT.keys():
    #     if model_id in args['lm_type']:
    #         if model_id not in models_to_use.keys():
    #             models_to_use[model_id] = {} 
    #         if 'Lower' in dic_teeth.keys():
    #             models_to_use[model_id]['Lower'] = available_models[model_id]['Lower']
    #         if 'Upper' in dic_teeth.keys():
    #             models_to_use[model_id]['Upper'] = available_models[model_id]['Upper']

    for model_id in MODELS_DICT.keys():
        if model_id in available_models:
            for lmtype in args["lm_type"]:
                if lmtype in MODELS_DICT[model_id].keys():
                    if model_id not in models_to_use.keys():
                        models_to_use[model_id] = available_models[model_id]
                # if model_id not in models_to_use.keys():
                #     models_to_use[model_id] = {} 
                # if 'Lower' in dic_teeth.keys():
                #     models_to_use[model_id]['Lower'] = available_models[model_id]['Lower']
                # if 'Upper' in dic_teeth.keys():
                #     models_to_use[model_id]['Upper'] = available_models[model_id]['Upper']

    print('models_to_use :',models_to_use)

    
    # lst_label = args['landmarks']
    data = args['input']
    dic_patients = {}
    if os.path.isfile(data):  
        print("Loading scan :", data)
        vtkfile = data
        basename = os.path.basename(data).split('.')[0]
        if basename not in dic_patients.keys():
            dic_patients[basename] = vtkfile
        # patient_id = basename[0].split('_')[0]+'_'+basename[0].split('_')[1]
        # if patient_id not in dic_patients.keys():
        #     dic_patients[patient_id] = {}
        # if '_L_' in basename:
        #     dic_patients[basename]["Lower"] = vtkfile
        # else:
        #     dic_patients[basename]["Upper"] = vtkfile
    else:
        scan_dir = data
        print("Loading data from",scan_dir)
        normpath = os.path.normpath("/".join([scan_dir, '**', '']))
        for vtkfile in sorted(glob.iglob(normpath, recursive=True)):
            if os.path.isfile(vtkfile) and True in [ext in vtkfile for ext in [".vtk"]]:
                basename = os.path.basename(vtkfile).split('.')[0]
                if basename not in dic_patients.keys():
                    dic_patients[basename] = vtkfile
                # patient_id = basename[0].split('_')[0]+'_'+basename[0].split('_')[1]
                # if patient_id not in dic_patients.keys():
                #     dic_patients[patient_id] = {}
                # if '_L_' in vtkfile:
                #     dic_patients[patient_id]["Lower"] = vtkfile
                # else:
                #     dic_patients[patient_id]["Upper"] = vtkfile

    print('dic_patients :',dic_patients)




    for patient_id,patient_path in dic_patients.items():
        # num_patient = patient_id.split('_')[1]

        print(f"prediction for patient {patient_id}")
        dic_points = {}
        for models_type in models_to_use.keys():
            LABEL = dic_label[models_type]
            if models_type == "O":
                sphere_radius = 0.2
            else:
                sphere_radius = 0.3
            print(dic_teeth)
            for jaw,lst_teeth in dic_teeth.items():
                group_data = {}

                path_vtk = patient_path
                if jaw == 'Lower':
                    model = models_to_use[models_type]['Lower']
                    camera_position = dic_cam[models_type]['L']
                else:
                    model = models_to_use[models_type]['Upper']
                    camera_position = dic_cam[models_type]['U']
                
                for label in lst_teeth:         
                    print("Loading model :", model, "for patient :", patient_id, "label :", label)
                    phong_renderer,mask_renderer = GenPhongRenderer(args['image_size'],args['blur_radius'],args['faces_per_pixel'],DEVICE)

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
                        agent.position_agent(RI,V,label)
                        textures = TexturesVertex(verts_features=CN)
                        meshe = Meshes(
                                    verts=V,   
                                    faces=F, 
                                    textures=textures
                                    ).to(DEVICE)

                        images_model , tens_pix_to_face_model=  agent.get_view_rasterize(meshe) #[batch,num_ima,channels,size,size] torch.Size([1, 2, 4, 224, 224])
                        tens_pix_to_face_model = tens_pix_to_face_model.permute(1,0,4,2,3) #tens_pix_to_face : torch.Size([1, 2, 1, 224, 224])
                            
                        net = UNet(
                            spatial_dims=2,
                            in_channels=4,
                            out_channels=4,
                            channels=( 16, 32, 64, 128, 256, 512),
                            strides=(2, 2, 2, 2, 2),
                            num_res_units=4
                        ).to(DEVICE)
                        
                        inputs = torch.empty((0)).to(DEVICE)
                        for i,batch in enumerate(images_model):
                            inputs = torch.cat((inputs,batch.to(DEVICE)),dim=0) #[num_im*batch,channels,size,size]

                        inputs = inputs.to(dtype=torch.float32)
                        net.load_state_dict(torch.load(model, map_location=DEVICE))
                        images_pred = net(inputs)

                        post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=4)


                        val_pred_outputs_list = decollate_batch(images_pred)                
                        val_pred_outputs_convert = [
                            post_pred(val_pred_outputs_tensor) for val_pred_outputs_tensor in val_pred_outputs_list
                        ]
                        val_pred = torch.empty((0)).to(DEVICE)
                        for image in images_pred:
                            val_pred = torch.cat((val_pred,post_pred(image).unsqueeze(0).to(DEVICE)),dim=0)
                                
                        
                        pred_data = images_pred.detach().cpu().unsqueeze(0).type(torch.int16) #torch.Size([1, 2, 2, 224, 224])
                        pred_data = torch.argmax(pred_data, dim=2).unsqueeze(2)
                        
                        
            
                        # recover where there is the landmark in the image
                        index_label_land_r = (pred_data==1.).nonzero(as_tuple=False) #torch.Size([6252, 5])
                        index_label_land_g = (pred_data==2.).nonzero(as_tuple=False) #torch.Size([6252, 5])
                        index_label_land_b = (pred_data==3.).nonzero(as_tuple=False) #torch.Size([6252, 5])

                        # recover the face in my mesh 
                        num_faces_r = []
                        num_faces_g = []
                        num_faces_b = []
                    
                        for index in index_label_land_r:
                            num_faces_r.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]]) 
                        for index in index_label_land_g:
                            num_faces_g.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]])
                        for index in index_label_land_b:
                            num_faces_b.append(tens_pix_to_face_model[index[0],index[1],index[2],index[3],index[4]]) 
                        
                        
                        last_num_faces_r = RemoveExtraFaces(F,num_faces_r,RI,int(label))
                        last_num_faces_g = RemoveExtraFaces(F,num_faces_g,RI,int(label))
                        last_num_faces_b = RemoveExtraFaces(F,num_faces_b,RI,int(label))       

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
                        
                        for land_name,list_face_ids in dico_rgb.items():
                            print('land_name :',land_name)
                            list_face_id=[]
                            for faces in list_face_ids:
                                faces_int = int(faces.item())
                                juan = F[0][faces_int]
                                list_face_id += [int(juan[0].item()) , int(juan[1].item()) , int(juan[2].item())]
                            
                            vert_coord = 0
                            for vert in list_face_id:
                                vert_coord += V[0][vert]

                            if len(list_face_id) != 0:
                                landmark_pos = vert_coord/len(list_face_id)
                                pid = locator.FindClosestPoint(landmark_pos.cpu().numpy())
                                closest_landmark_pos = torch.tensor(surf_unit.GetPoint(pid))

                                upscale_landmark_pos = Upscale(closest_landmark_pos,mean_arr,scale_factor)
                                final_landmark_pos = upscale_landmark_pos.detach().cpu().numpy()
                                
                                coord_dic = {"x":final_landmark_pos[0],"y":final_landmark_pos[1],"z":final_landmark_pos[2]}
                                
                                if jaw not in group_data.keys():
                                    group_data[jaw] = {}
                                
                                group_data[land_name]=coord_dic

                    print(f"""<filter-progress>{1}</filter-progress>""")
                    sys.stdout.flush()
                    time.sleep(0.5)
                    print(f"""<filter-progress>{0}</filter-progress>""")
                    sys.stdout.flush()
                    
                # print("GROUP_DATA")
                # print(group_data)
                if len(group_data.keys()) > 0:
                    lm_lst = GenControlePoint(group_data,landmarks_selected)
                    # print("ControlPoints")
                    # print(lm_lst)
                    # print(jaw)

                    out_path = args["output_dir"]

                    if args["save_in_folder"]:
                        outputdir = out_path + "/" + patient_id + "_landmarks"
                        # print("Output dir :",outputdir)
                        if not os.path.exists(outputdir):
                            os.makedirs(outputdir)
                    
                    else:
                        outputdir = out_path    

                    WriteJson(lm_lst,os.path.join(outputdir,f"{patient_id}_{jaw}_{models_type}_Pred.json"))

        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)


        # dic_points[patient_id] = group_data

        # print(group_data)
        # print(dic_points)
        # dic_lower,dic_upper = GenControlePoint(group_data,landmarks_selected)
        # print(dic_lower)
        # print(dic_upper)

        # out_path = args["output_dir"]

        # if args["save_in_folder"]:
        #     outputdir = out_path + "/" + patient_id + "_landmarks"
        #     # print("Output dir :",outputdir)
        #     if not os.path.exists(outputdir):
        #         os.makedirs(outputdir)
        
        # else:
        #     outputdir = out_path


        # print(outputdir)

        # path_num_patient = os.path.join(args['output_dir'],patient_id)
        # if not os.path.exists(path_num_patient):
        #     os.makedirs(path_num_patient)
        
        # if args["jaw"] == "L":
        #     path_jaw = os.path.join(path_num_patient,'Lower')
        #     landmark_path = os.path.join(os.path.dirname(path_vtk),f"{num_patient}_L_Pred.json")
            
        # else:
        #     path_jaw = os.path.join(outputdir,'Upper')
        #     landmark_path = os.path.join(os.path.dirname(path_vtk),f"{num_patient}_U.json")
    

        # if not os.path.exists(path_jaw):
        #         os.makedirs(path_jaw)
  

        # copy_file = os.path.join(path_jaw,os.path.basename(path_vtk))
        # shutil.copy(path_vtk,copy_file)
        # copy_json_file =  os.path.join(out_path_jaw,os.path.basename(landmark_path))
        # final_outpath_json = shutil.copy(landmark_path,copy_json_file)
        
        # print('out_path :',outputdir)
        # print('out_path_jaw :',out_path_jaw)
        # print('landmark_path :',landmark_path)
        
        # final_out_path = shutil.copytree(path_vtk,out_path_L)

        # if args["jaw"] == "L":
        #     WriteJson(lm_lst,os.path.join(path_num_patient,f"{patient_id}_L_Pred.json"))
        # else:
        # WriteJson(lm_lst,os.path.join(path_num_patient,f"{patient_id}_U_Pred.json"))



if __name__ == "__main__":


    print("Starting")
    print(sys.argv)


    args = {
        "input": sys.argv[1],
        "dir_models": sys.argv[2],
        "lm_type": sys.argv[3].split(" "),
        "teeth": sys.argv[4].split(" "),
        "save_in_folder": sys.argv[5] == "true",
        "output_dir": sys.argv[6],

        "image_size": 224,
        "blur_radius": 0,
        "faces_per_pixel": 1,
        # "sphere_radius": 0.3,
    }

    
    # args = {
    #         "input": '/home/luciacev-admin/Desktop/data_cervical/T1_14_L_segmented.vtk',
    #         "dir_models": '/home/luciacev-admin/Desktop/Data_allios_cli/Models',
    #         "teeth": ['LL7','LL6','LL5','LL4','LL3','LL2','LL1','LR1','LR2','LR3','LR4','LR5','LR6','LR7'],
    #         "lm_type": ["C"],
    #         # "save_in_folder": sys.argv[4] == "true",
    #         "output_dir": '/home/luciacev-admin/Desktop/data_cervical/test',
            
    #         "image_size": 224,
    #         "blur_radius": 0,
    #         "faces_per_pixel": 1,
    #         "sphere_radius": 0.3,

    #     }

    main(args)