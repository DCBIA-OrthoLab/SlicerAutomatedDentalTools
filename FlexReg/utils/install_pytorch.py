#!/home/luciacev/Slicer-5.4.0-linux-amd64/bin/PythonSlicer
#/usr/bin/env pythonSlicer
# /usr/bin/env python-real

import sys
import subprocess
def import_function(pip_path):



    # system = platform.system()
    try:
        import torch
    except ImportError:
        command = [pip_path, 'install','--no-cache-dir', 'torch==1.11.0+cu113', 'torchvision==0.12.0+cu113', 'torchaudio==0.11.0+cu113', '--extra-index-url' ,'https://download.pytorch.org/whl/cu113']
        result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True )
        import torch
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
    from pytorch3d.structures import Meshes
    
    print("everything ok")
    

if __name__ == "__main__":
    

    print("Starting")
    print(sys.argv)


    path_pip = sys.argv[1]
    import_function(path_pip)