#!/usr/bin/env python-real
import glob
import os
import sys
import time
import argparse
import platform
import numpy as np

from tqdm import tqdm
from itertools import chain


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
    from ASO_IOS_utils.utils import UpperOrLower, search, ReadSurf, WriteSurf, WritefileError, saveMatrixAsTfm, PatientNumber
    from ASO_IOS_utils.icp import vtkICP, vtkMeanTeeth, InitIcp, ICP, ToothNoExist, NoSegmentationSurf
    from ASO_IOS_utils.data_file import Files_vtk_link, Jaw, Lower, Upper
    from ASO_IOS_utils.transformation import TransformSurf
    from ASO_IOS_utils.pre_icp import PrePreAso
    
else:
    from ASO_IOS_utils import (
        UpperOrLower, search, ReadSurf, WriteSurf, WritefileError, saveMatrixAsTfm, PatientNumber,
        vtkICP, vtkMeanTeeth, InitIcp, ICP, ToothNoExist, NoSegmentationSurf,
        Files_vtk_link, Jaw, Lower, Upper,
        TransformSurf,
        PrePreAso,
    )
    
# import ASO_IOS_utils


print("pre aso ios charge")


def main(args):
    print("icp meanteeth launch")

    list_extension = [".vtk", ".stl", ".off", ".obj", ".vtp"]

    lower = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    dic_teeth = {"Upper": [], "Lower": []}

    list_teeth = args.list_teeth[0].split(",")
    dic = {
        "UR8": 1,
        "UR7": 2,
        "UR6": 3,
        "UR5": 4,
        "UR4": 5,
        "UR3": 6,
        "UR2": 7,
        "UR1": 8,
        "UL1": 9,
        "UL2": 10,
        "UL3": 11,
        "UL4": 12,
        "UL5": 13,
        "UL6": 14,
        "UL7": 15,
        "UL8": 16,
        "LL8": 17,
        "LL7": 18,
        "LL6": 19,
        "LL5": 20,
        "LL4": 21,
        "LL3": 22,
        "LL2": 23,
        "LL1": 24,
        "LR1": 25,
        "LR2": 26,
        "LR3": 27,
        "LR4": 28,
        "LR5": 29,
        "LR6": 30,
        "LR7": 31,
        "LR8": 32,
    }

    for tooth in list_teeth:
        if dic[tooth] in lower:
            dic_teeth["Lower"].append(dic[tooth])
        else:
            dic_teeth["Upper"].append(dic[tooth])

    gold_files = list(
        chain.from_iterable(search(args.gold_folder[0], list_extension).values())
    )

    gold = {}

    gold[UpperOrLower(gold_files[0])] = ReadSurf(gold_files[0])
    gold[UpperOrLower(gold_files[1])] = ReadSurf(gold_files[1])

    if not os.path.exists(os.path.split(args.log_path[0])[0]):
        os.mkdir(os.path.split(args.log_path[0])[0])

    with open(args.log_path[0], "w") as log_f:
        log_f.truncate(0)

    if args.occlusion[0].lower() == "true":
        link = True
        if args.jaw[0] == "Upper":
            jaw = Jaw(Upper())

        elif args.jaw[0] == "Lower":
            jaw = Jaw(Lower())

    else:
        link = False

    if link:
        list_files = Files_vtk_link(args.input[0])

    else:
        list_files = list(
            chain.from_iterable(search(args.input[0], list_extension).values())
        )

    Method = [InitIcp(), vtkICP()]
    option_upper = vtkMeanTeeth(dic_teeth["Upper"])
    option_lower = vtkMeanTeeth(dic_teeth["Lower"])
    icp_upper = ICP(Method, option=option_upper)
    icp_lower = ICP(Method, option=option_lower)
    icp = {"Upper": icp_upper, "Lower": icp_lower}

    for index, file in tqdm(enumerate(list_files), total=len(list_files)):
        file_vtk = file
        if link:
            file_vtk = file[jaw()]
        if not link:
            jaw = Jaw(file_vtk)

        surf = ReadSurf(file_vtk)

        try:
            surf, matrix = PrePreAso(surf, gold[jaw()], dic_teeth[jaw()])
            output_icp = icp[jaw()].run(surf, gold[jaw()])
            
            final_matrix = np.matmul(output_icp["matrix"], matrix)
            patient_id = PatientNumber(file_vtk)
            tfm_path = os.path.join(args.output_folder[0], f"{patient_id}_SegOr.tfm")
            saveMatrixAsTfm(final_matrix, tfm_path)

        except ToothNoExist as tne:
            print(f"Error {tne}, for this file {file_vtk}")

            WritefileError(
                file_vtk,
                args.folder_error[0],
                f"Error {str(tne)}, for this file {file_vtk}",
            )

            with open(args.log_path[0], "r+") as log_f:
                log_f.write(str(index))
            continue

        except NoSegmentationSurf as nss:

            print(f"Error {nss}, for this file {file_vtk}")

            WritefileError(
                file_vtk,
                args.folder_error[0],
                f"Error {str(nss)}, for this file {file_vtk}",
            )

            with open(args.log_path[0], "r+") as log_f:
                log_f.write(str(index))
            continue

        print(f"DEBUG: About to write surface for {file_vtk}")
        print(f"DEBUG: output_icp source_Or type: {type(output_icp['source_Or'])}")
        try:
            WriteSurf(
                output_icp["source_Or"],
                args.output_folder[0],
                os.path.basename(file_vtk),
                args.add_inname[0],
            )
            print(f"DEBUG: WriteSurf completed successfully for {os.path.basename(file_vtk)}")
        except Exception as write_error:
            print(f"ERROR in WriteSurf: {str(write_error)}")
            print(f"  Output folder: {args.output_folder[0]}")
            print(f"  Filename: {os.path.basename(file_vtk)}")
            print(f"  Infix: {args.add_inname[0]}")
            raise

        if link:
            surf_lower = ReadSurf(file[jaw.inv()])
            output_lower = TransformSurf(surf_lower, matrix)
            output_lower = TransformSurf(output_lower, output_icp["matrix"])

            WriteSurf(
                output_lower,
                args.output_folder[0],
                os.path.basename(file[jaw.inv()]),
                args.add_inname[0],
            )

        with open(args.log_path[0], "w+") as log_f:
            log_f.write(str(index+1))


if __name__ == "__main__":

    print("Starting")
    print(sys.argv)

    parser = argparse.ArgumentParser()

    parser.add_argument("input", nargs=1)
    parser.add_argument("gold_folder", nargs=1)
    parser.add_argument("output_folder", nargs=1)
    parser.add_argument("add_inname", nargs=1)
    parser.add_argument("list_teeth", nargs=1)
    parser.add_argument("occlusion", nargs=1)
    parser.add_argument("jaw", nargs=1)
    parser.add_argument("folder_error", nargs=1)
    parser.add_argument("log_path", nargs=1)

    args = parser.parse_args()
    print(args)

    main(args)