#!/usr/bin/env python-real


# def installPackages():
#     from slicer.util import pip_install, pip_uninstall
#     import sys
#     import os

#     try:
#         import pandas
#     except ImportError:
#         pip_install("pandas")

#     try:
#         import torch

#         pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
#         version_str = "".join(
#             [
#                 f"py3{sys.version_info.minor}_cu",
#                 torch.version.cuda.replace(".", ""),
#                 f"_pyt{pyt_version_str}",
#             ]
#         )
#         if version_str != "py39_cu113_pyt1120":
#             raise ImportError
#     except ImportError:
#         # pip_install('--no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113')
#         pip_install(
#             "--force-reinstall torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113"
#         )

#     try:
#         import monai
#     except ImportError:
#         pip_install("monai")

#     from platform import system  # to know which OS is used

#     if system() == "Darwin":  # MACOS
#         try:
#             import pytorch3d
#         except ImportError:
#             pip_install("pytorch3d")
#             import pytorch3d

#     else:  # Linux or Windows
#         try:
#             import pytorch3d

#             if pytorch3d.__version__ != "0.7.0":
#                 raise ImportError
#         except ImportError:
#             # try:
#             # #   import torch
#             #     pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
#             #     version_str="".join([f"py3{sys.version_info.minor}_cu",torch.version.cuda.replace(".",""),f"_pyt{pyt_version_str}"])
#             #     pip_install('--upgrade pip')
#             #     pip_install('fvcore==0.1.5.post20220305')
#             #     pip_install('--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
#             # except: # install correct torch version
#             #     pip_install('--no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113')
#             #     pip_install('--no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html')

#             try:
#                 code_path = os.sep.join(
#                     os.path.dirname(os.path.abspath(__file__)).split(os.sep)
#                 )
#                 # print(code_path)
#                 pip_install(
#                     os.path.join(
#                         code_path,
#                         "AREG_IOS_utils",
#                         "pytorch3d-0.7.0-cp39-cp39-linux_x86_64.whl",
#                     )
#                 )  # py39_cu113_pyt1120
#             except:
#                 pip_install(
#                     "--force-reinstall --no-deps --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1120/download.html"
#                 )

#     try:
#         import pytorch_lightning
#     except ImportError:
#         pip_install("pytorch_lightning==1.7.7")

#     import numpy

#     if float(".".join(numpy.__version__.split(".")[:2])) >= 1.23:
#         pip_install("numpy==1.21.1")


# installPackages()
import os
import sys
import argparse

import pandas
import torch
import monai
import pytorch3d
import pytorch_lightning
import numpy


# from tqdm import tqdm


fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from AREG_IOS_utils import (
    DatasetPatch,
    PredPatch,
    vtkMeshTeeth,
    vtkICP,
    ICP,
    WriteSurf,
    TransformSurf,
)


def main(args):

    if not os.path.exists(os.path.split(args.log_path)[0]):
        os.mkdir(os.path.split(args.log_path)[0])

    with open(args.log_path, "w") as log_f:
        log_f.truncate(0)

    dataset = DatasetPatch(args.T1, args.T2, "Universal_ID")
    Patched = PredPatch(args.model)

    Method = [vtkICP()]
    option = vtkMeshTeeth(list_teeth=[1], property="Butterfly")
    icp = ICP(Method, option=option)

    lower = False
    if dataset.isLower():
        lower = True

    # pbar = tqdm(total=len(dataset)*3,desc='Segment Palate')
    for idx in range(len(dataset)):

        name = os.path.basename(dataset.getUpperPath(idx, "T1"))
        # pbar.set_description(f'Patch {name}')
        surf_T1 = dataset.getUpperSurf(idx, "T1")
        surf_T1 = Patched(dataset[idx, "T1"], surf_T1)
        # pbar.update(1)

        name = os.path.basename(dataset.getUpperPath(idx, "T1"))
        WriteSurf(surf_T1, args.output, name, args.suffix)

        with open(args.log_path, "r+") as log_f:
            log_f.write(str(1))

        name = os.path.basename(dataset.getUpperPath(idx, "T2"))
        # pbar.set_description(f'Patch {name}')
        surf_T2 = dataset.getUpperSurf(idx, "T2")
        surf_T2 = Patched(dataset[idx, "T2"], surf_T2)
        # pbar.update(1)

        with open(args.log_path, "r+") as log_f:
            log_f.write(str(1))

        # pbar.set_description('ICP')

        output_icp = icp.run(surf_T2, surf_T1)

        name = os.path.basename(dataset.getUpperPath(idx, "T2"))
        WriteSurf(output_icp["source_Or"], args.output, name, args.suffix)


        if lower:
            surf_lower = dataset.getLowerSurf(idx, "T2")
            surf_lower = TransformSurf(surf_lower, output_icp["matrix"])
            name_lower = os.path.basename(dataset.getLowerPath(idx, "T2"))
            WriteSurf(surf_lower, args.output, name_lower, args.suffix)

            surf_lower = dataset.getLowerSurf(idx, "T1")
            if surf_lower!=None :
                name_lower = os.path.basename(dataset.getLowerPath(idx, "T1"))
                WriteSurf(surf_lower, args.output, name_lower, args.suffix)

        # pbar.update(1)

        with open(args.log_path, "r+") as log_f:
            log_f.write(str(1))


if __name__ == "__main__":

    print("Starting")
    print(sys.argv)

    parser = argparse.ArgumentParser()

    parser.add_argument("T1", type=str)
    parser.add_argument("T2", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("suffix", type=str)
    parser.add_argument("log_path", type=str)

    args = parser.parse_args()

    main(args)
