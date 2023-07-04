#!/usr/bin/env python-real    
import glob
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch


fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)

from AREG_IOS_utils import TeethDatasetPatch, MonaiUNetHRes, PredPatch, vtkMeshTeeth, vtkICP,ICP, WriteSurf, TransformSurf




    
    
def main(args) :
        
        if not os.path.exists(os.path.split(args.log_path)[0]):
            os.mkdir(os.path.split(args.log_path)[0])

        with open(args.log_path,'w') as log_f :
            log_f.truncate(0)

        dataset = TeethDatasetPatch(args.T1,args.T2,'Universal_ID')
        Patched = PredPatch(args.model)

        methode = [vtkICP()]
        option = vtkMeshTeeth(list_teeth=[1],property='Butterfly')
        icp = ICP(methode, option=option)

        lower = False
        if dataset.isLower():
             lower = True


       
        pbar = tqdm(total=len(dataset)*3,desc='Segment Palate')
        for idx in range(len(dataset)) :
             
            name = os.path.basename(dataset.getUpperPath(idx,'T1'))
            pbar.set_description(f'Patch {name}')
            surf_T1 = dataset.getUpperSurf(idx,'T1')
            surf_T1 = Patched(dataset[idx,'T1'],surf_T1)
            pbar.update(1)

            with open(args.log_path,'r+') as log_f:
                log_f.write(str(1))  

            name = os.path.basename(dataset.getUpperPath(idx,'T2'))
            pbar.set_description(f'Patch {name}')
            surf_T2 = dataset.getUpperSurf(idx,'T2')
            surf_T2 = Patched(dataset[idx,'T2'],surf_T2)
            pbar.update(1)

            with open(args.log_path,'r+') as log_f:
                log_f.write(str(1))  

            pbar.set_description('ICP')

            output_icp = icp.run(surf_T2,surf_T1)

            name = os.path.basename(dataset.getUpperPath(idx,'T2'))
            WriteSurf(output_icp['source_Or'],args.output,name,args.suffix)


            if lower:
                 surf_lower = dataset.getLowerSurf(idx,'T2')
                 surf_lower = TransformSurf(surf_lower,output_icp['matrix'])
                 name_lower = os.path.basename(dataset.getLowerPath(idx,'T2'))
                 WriteSurf(surf_lower,args.output,name_lower,args.suffix)

            pbar.update(1)

            with open(args.log_path,'r+') as log_f:
                log_f.write(str(1))  

if __name__ == "__main__":
    

    print("Starting")
    print(sys.argv)

    parser = argparse.ArgumentParser()


    parser.add_argument('T1',type=str)
    parser.add_argument('T2',type=str)
    parser.add_argument('output',type=str)
    parser.add_argument('model',type=str)
    parser.add_argument('suffix',type=str)
    parser.add_argument('log_path',type=str)

    args = parser.parse_args()


    main(args)