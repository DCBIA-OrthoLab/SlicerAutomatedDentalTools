#!/usr/bin/env python-real
import json
import os
import argparse
from urllib import request
import subprocess
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

import shapeaxi
from shapeaxi.saxi_dataset import SaxiDataset
from shapeaxi.saxi_transforms import TrainTransform, EvalTransform

from shapeaxi.saxi_gradcam import gradcam_process 


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import vtk

from shapeaxi import saxi_nets_lightning, post_process as psp, utils
from captum.attr import LayerGradCam
import cv2
import numpy as np

def scale_cam_image(cam, target_size=None):
    ## adapted from https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/image.py#L162
    result = []
    for img in cam:
      if target_size is not None:

        img = cv2.resize(np.float32(img), target_size)

        new_max = np.percentile(img.flatten(),q=99)
        new_min = np.percentile(img.flatten(),q=1)
        img = np.clip(img,new_min,new_max)

        img =  2*((img - np.min(img)) / (np.max(img) -np.min(img))) -1 

      result.append(img)
    result = np.float32(result)

    return result

def gradcam_save(args, gradcam_path, surf_path, surf):
    '''
    Function to save the GradCAM on the surface

    Args : 
        gradcam_path : path to save the GradCAM
        surf_path : path to the surface
        surf : surface read by utils.ReadSurf
    '''

    if not os.path.exists(gradcam_path):
        os.makedirs(gradcam_path)
    
    out_surf_path = os.path.join(gradcam_path, os.path.basename(surf_path))

    subprocess.call(["cp", surf_path, out_surf_path])

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_surf_path)
    writer.SetInputData(surf)
    writer.Write()

class MultiHead(nn.Module):
    def __init__(self, mha_fb):
        super().__init__()
        self.mha_fb = mha_fb

    def forward(self, x):
        x, score = self.mha_fb(x,x,x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, x):
        x, score = self.attn(x,x)
        return x

def csv_edit(args):
    """
    Check if the surfaces files are present in the input directory and edit csv file with surface path
    Args: Arguments from the command line
    """
    surf_dir =args.input_dir
    for surf in os.listdir(surf_dir):
      surf_path = os.path.join(surf_dir, surf)
      if os.path.splitext(surf)[1] == '.vtk':
        if not os.path.exists(surf_path):
          print(f"Missing files: {surf}")
        else:
          with open(args.input_csv, 'a') as f:
              f.write(f"{surf}\n")

def download_model(model_name, output_path):
    json_path = os.path.join(os.path.dirname(__file__), "model_path.json")
    with open(json_path, 'r') as file:
        model_info = json.load(file)
    model_url = model_info[model_name]["url"]
    request.urlretrieve(model_url, output_path)


def saxi_gradcam(args, out_model_path):
  print("Running Explainability....")
  with open(args.log_path,'w+') as log_f :
    log_f.write(f"{args.task},explainability,NaN,{args.num_classes}")

  NN = getattr(saxi_nets_lightning, args.nn)    
  model = NN.load_from_checkpoint(out_model_path, strict=False)

  model.eval()
  model.to(args.device)

  fname = os.path.basename(args.input_csv)
  predicted_csv = os.path.join(args.output_dir, fname.replace('.csv', "_prediction.csv"))
  df_test = pd.read_csv(predicted_csv)
    
  test_ds = SaxiDataset(df_test, transform=EvalTransform(), CN=True, 
                          surf_column=model.hparams.surf_column, mount_point = args.input_dir, 
                          class_column=None, scalar_column=None, **vars(args))
  test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=False)


  target_layer = getattr(model.convnet.module, '_blocks')
  mv_cam = LayerGradCam(model,target_layer[-1],device_ids=[0])

  out_dir = os.path.join(args.output_dir, "explainability", args.task)
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)
  targets = None
  
  for idx, (V, F, CN) in tqdm(enumerate(test_loader), total=len(test_loader)):
    # The generated CAM is processed and added to the input surface mesh (surf) as a point data array
    V = V.to(args.device)
    F = F.to(args.device)
    CN = CN.to(args.device)
    
    X_mesh = model.create_mesh(V, F, CN)
    X_pc = model.sample_points_from_meshes(X_mesh, model.hparams.sample_levels[0])  ## mhafb
    X_views, PF = model.render(X_mesh)

    surf = test_ds.getSurf(idx)
    surf_path = test_ds.getSurfPath(idx)

    args.target_class = None
    for class_idx in range(args.num_classes):
      if args.num_classes > 1:
        args.target_class = class_idx
      mv_att = mv_cam.attribute(inputs=(X_pc,X_views), target=class_idx,attr_dim_summation=False)
      # mv_att = multiview.attribute(inputs=(X_pc,X_views, x_v_fixed), target=class_idx,attr_dim_summation=False)

      mv_att = mv_att.sum(dim=1).cpu().detach() ## LayerIntegratedGradients

      mv_att_upscaled = scale_cam_image(mv_att.numpy(), target_size=(224,224))
      mv_att_upscaled = gradcam_process(args, mv_att_upscaled, F, PF, V,device=args.device)

      surf.GetPointData().AddArray(mv_att_upscaled)
      psp.MedianFilter(surf, mv_att_upscaled)

      out_surf_path = os.path.join(out_dir,os.path.basename(surf_path))
      utils.WriteSurf(surf, out_surf_path)
    
    with open(args.log_path,'w+') as log_f :
      log_f.write(f"{args.task},explainability,{idx},{args.num_classes}")


def saxi_predict(args,out_model_path):
    print("Running Prediction....")

    df = pd.read_csv(args.input_csv)
    with open(args.log_path,'w+') as log_f :
      log_f.write(f"{args.task},predict,NaN,{args.num_classes}")


    NN = getattr(saxi_nets_lightning, args.nn)
    model = NN.load_from_checkpoint(out_model_path, strict=False)
    model.eval()
    model.to(args.device)

    scale_factor = None
    if hasattr(model.hparams, 'scale_factor'):
        scale_factor = model.hparams.scale_factor
    
    test_ds = SaxiDataset(df, transform=EvalTransform(scale_factor), CN=True, 
                          surf_column=model.hparams.surf_column, mount_point = args.input_dir, 
                          class_column=None, scalar_column=None, **vars(args))
    
    test_loader = DataLoader(test_ds, batch_size=1, pin_memory=False)

    fname = os.path.basename(args.input_csv)

    with torch.no_grad():
      predictions = []
      softmax = nn.Softmax(dim=1)

      for idx, (V, F, CN) in tqdm(enumerate(test_loader), total=len(test_loader)):
        V = V.to(args.device)
        F = F.to(args.device)
        CN = CN.to(args.device)

        X_mesh = model.create_mesh(V, F, CN)
        X_pc = model.sample_points_from_meshes(X_mesh, model.hparams.sample_levels[0])
        X_views, X_PF = model.render(X_mesh)

        x = model(X_pc, X_views)
        
        if args.nn == 'SaxiMHAFBClassification': # no argmax for regression
          x = softmax(x).detach()
          x = torch.argmax(x, dim=1, keepdim=True)
        predictions.append(x)

        with open(args.log_path,'w+') as log_f :
          log_f.write(f"{args.task},predict,{idx+1},{args.num_classes}")


      predictions = torch.cat(predictions).cpu().numpy().squeeze()

      out_name = os.path.join(args.output_dir, fname.replace(".csv", "_prediction.csv"))
      if os.path.exists(out_name):
        df = pd.read_csv(out_name)

      df[f'{args.task}_prediction'] = predictions
      df.to_csv(out_name, index=False)

def linux2windows_path(filepath):
  if ':' in filepath:
    if '\\' in filepath:
      filepath = filepath.replace('\\', '/')
    drive, path_without_drive = filepath.split(':', 1)
    filepath = "/mnt/" + drive.lower() + path_without_drive
    return filepath
  else:
    return filepath

def create_csv(input_file):
  with open(input_file, "w") as f:
    f.write("surf\n")
   
def main(args):
  import torch

  args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # convert path if windows distribution
  args.input_csv = linux2windows_path(os.path.join(args.output_dir, f"files_{args.data_type}.csv"))
  args.input_dir = linux2windows_path(args.input_dir)
  args.output_dir = linux2windows_path(args.output_dir)
  args.log_path = linux2windows_path(args.log_path)

  with open(args.log_path,'w') as log_f:
    log_f.truncate(0)
  
  out_model_path = os.path.join(args.output_dir, args.model + '.ckpt')
  
  if os.path.exists(args.output_dir):
    if not os.path.exists(out_model_path):
      print("Downloading model...")
      download_model(args.model, out_model_path)

  if not os.path.exists(args.input_csv):
    create_csv(args.input_csv)
    csv_edit(args)

  saxi_predict(args, out_model_path)
  print("End prediction, starting explainability")

  saxi_gradcam(args, out_model_path)

  print("End explainability \nProcess Completed")


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir',type = str)
  parser.add_argument('output_dir',type=str)
  parser.add_argument('data_type',type = str)
  parser.add_argument('task', type=str)
  parser.add_argument('model',type=str)
  parser.add_argument('nn',type=str)
  parser.add_argument('num_classes',type=int)
  parser.add_argument('log_path',type=str)

  args = parser.parse_args()

  main(args)