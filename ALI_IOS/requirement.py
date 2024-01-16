import sys

def import_function(pip_path):
    import time
    import os
    import glob
    import sys
    import json


    import subprocess
    import platform
    import inspect
    import textwrap
    import urllib.request
    import shutil



    system = platform.system()
    try : 
        import numpy as np
    except : 
        command = [pip_path, 'install','numpy']
        result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
        import numpy as np

    try : 
        import vtk
    except : 
        command = [pip_path, 'install','vtk']
        result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
        import vtk

    try :
        from scipy import linalg
    except :
        command = [pip_path, 'install','scipy']
        result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
        from scipy import linalg

    try:
        import torch
    except ImportError:
        command = [pip_path, 'install','--no-cache-dir', 'torch==1.11.0+cu113', 'torchvision==0.12.0+cu113', 'torchaudio==0.11.0+cu113', '--extra-index-url' ,'https://download.pytorch.org/whl/cu113']
        result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
        import torch

    try :
        from monai.networks.nets import UNETR
    except ImportError:
        command = [pip_path, 'install', 'monai==0.7.0']
        subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
        from monai.networks.nets import UNETR

    from platform import system # to know which OS is used



    # Linux or Windows
    try:
        import pytorch3d
        if pytorch3d.__version__ != '0.6.2':
            raise ImportError
    except ImportError:
        try:
            pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
            version_str="".join([f"py3{sys.version_info.minor}_cu",torch.version.cuda.replace(".",""),f"_pyt{pyt_version_str}"])
            command = [pip_path, 'install','--upgrade' ,'pip']
            subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
            command = [pip_path, 'install','fvcore==0.1.5.post20220305']
            subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )     
            command = [pip_path, 'install','--no-index', '--no-cache-dir' ,'pytorch3d', '-f', f'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html']
            result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
        except: # install correct torch version
            command = [pip_path, 'install','--no-cache-dir', 'torch==1.11.0+cu113', 'torchvision==0.12.0+cu113', 'torchaudio==0.11.0+cu113', '--extra-index-url' ,'https://download.pytorch.org/whl/cu113']
            result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
            command = [pip_path, 'install','--no-index', '--no-cache-dir', 'pytorch3d', '-f', 'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html']
            result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
        import pytorch3d


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
    
    print("everything ok")
    

if __name__ == "__main__":
    

    print("Starting")
    print(sys.argv)


    path_pip = sys.argv[1]
    import_function(path_pip)