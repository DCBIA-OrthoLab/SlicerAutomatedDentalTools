#!/usr/bin/env python-real

import os
import sys
import shutil
import argparse
import platform

# ===== DEPENDENCY CHECK =====
# Check and fix torch/torchvision compatibility before any imports
try:
    # Add parent for deps check
    areg_ios_path = os.path.dirname(__file__)
    if areg_ios_path not in sys.path:
        sys.path.insert(0, areg_ios_path)
    
    from AREG_IOS_utils.check_deps import ensure_compatible
    ensure_compatible()
except ImportError as e:
    print("[WARNING] Could not import dependency checker: {}".format(e))
# ===== END DEPENDENCY CHECK =====

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

def check_platform():
    if platform.system() == 'Windows':
        return "Windows"
    elif platform.system() == 'Linux':
        if 'Microsoft' in platform.release():
            return "WSL"
        else:
            return "Linux"
    else:
        return "Unknown"

if check_platform()=="WSL":
    from AREG_IOS_utils.dataset import DatasetPatch
    from AREG_IOS_utils.PredPatch import PredPatch
    from AREG_IOS_utils.vtkSegTeeth import vtkMeshTeeth
    from AREG_IOS_utils.ICP import vtkICP
    from AREG_IOS_utils.ICP import ICP
    from AREG_IOS_utils.utils import WriteSurf
    from AREG_IOS_utils.transformation import TransformSurf
    from AREG_IOS.AREG_IOS_utils.transformation import saveMatrixAsTfm

else : 
    from AREG_IOS_utils import (
        DatasetPatch,
        PredPatch,
        vtkMeshTeeth,
        vtkICP,
        ICP,
        WriteSurf,
        TransformSurf,
        saveMatrixAsTfm,
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
        
    for idx in range(len(dataset)):
        print("idx : ",idx)

        name = os.path.basename(dataset.getUpperPath(idx, "T1"))

        surf_T1 = dataset.getUpperSurf(idx, "T1")
        surf_T1 = Patched(dataset[idx, "T1"], surf_T1)

        name = os.path.basename(dataset.getUpperPath(idx, "T1"))

        WriteSurf(surf_T1, args.output, name, args.suffix)

        with open(args.log_path, "r+") as log_f:
            log_f.write(str(1))

        name = os.path.basename(dataset.getUpperPath(idx, "T2"))
        surf_T2 = dataset.getUpperSurf(idx, "T2")
        surf_T2 = Patched(dataset[idx, "T2"], surf_T2)

        with open(args.log_path, "r+") as log_f:
            log_f.write(str(1))

        output_icp = icp.run(surf_T2, surf_T1)

        name = os.path.basename(dataset.getUpperPath(idx, "T2"))
        WriteSurf(output_icp["source_Or"], args.output, name, args.suffix)
        
        patient_id = name.split("_T2")[0]
        
        if args.areg_mode == "Auto_IOS":
            # The .tfm files are saved in the input folders with pattern: PatientID_SegOr.tfm
            # Extract just the patient ID (e.g., "A2" from "A2_UpperT2_SegOr.vtk")
            patient_id_short = patient_id.split("_")[0] if "_" in patient_id else patient_id
            
            aso_tfm_path_T1 = os.path.join(args.T1, f"{patient_id_short}_SegOr.tfm")
            aso_tfm_path_T2 = os.path.join(args.T2, f"{patient_id_short}_SegOr.tfm")
            out_tfm_T1 = os.path.join(args.output, f"{patient_id_short}_T1_SegOr.tfm")
            
            print(f"DEBUG: Looking for T1 tfm at: {aso_tfm_path_T1}")
            print(f"DEBUG: Looking for T2 tfm at: {aso_tfm_path_T2}")
            
            try:
                if os.path.exists(aso_tfm_path_T1):
                    shutil.copy(aso_tfm_path_T1, out_tfm_T1)
                    print(f"Saved T1 matrix: {out_tfm_T1}")
                else:
                    print(f"Warning: T1 tfm file not found at {aso_tfm_path_T1}")
            except Exception as e:
                print(f"Error copying T1 matrix for {name}: {e}")

            saveMatrixAsTfm(output_icp["matrix"], aso_tfm_path_T2, args.output, patient_id_short, args.suffix, args.areg_mode)
        
        if lower:
            surf_lower = dataset.getLowerSurf(idx, "T2")
            surf_lower = TransformSurf(surf_lower, output_icp["matrix"])
            name_lower = os.path.basename(dataset.getLowerPath(idx, "T2"))
            WriteSurf(surf_lower, args.output, name_lower, args.suffix)

            surf_lower = dataset.getLowerSurf(idx, "T1")
            if surf_lower!=None :
                name_lower = os.path.basename(dataset.getLowerPath(idx, "T1"))
                WriteSurf(surf_lower, args.output, name_lower, args.suffix)

        with open(args.log_path, "w+") as log_f:
            log_f.write(str(idx + 1))


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
    parser.add_argument("areg_mode", type=str)

    args = parser.parse_args()
    main(args)
