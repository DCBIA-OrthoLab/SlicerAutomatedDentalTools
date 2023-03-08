#!/usr/bin/env python-real    
import glob
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain
# from pytorch3d.loss import chamfer_distance
# from torch import float32, tensor
# import torch


fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)

from ASO_IOS_utils import ( UpperOrLower, search, ReadSurf, WriteSurf,PatientNumber,ICP, InitIcp, vtkICP,
vtkMeanTeeth,TransformSurf,Files_vtk_link, Jaw ,Upper, Lower,ToothNoExist, vtkMeshTeeth, 
WritefileError, NoSegmentationSurf, PrePreAso)

# import ASO_IOS_utils





print('pre aso ios charge')
    
    
def main(args) :
    print('icp meanteeth launch')

    list_extension = ['.vtk','.stl','.off','.obj','.vtp']

    # device = torch.device('cuda')

    lower  = [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    dic_teeth={'Upper':[],'Lower':[]}

    list_teeth = args.list_teeth[0].split(',')
    dic = {'UR8': 1, 'UR7': 2, 'UR6': 3, 'UR5': 4, 'UR4': 5, 'UR3': 6, 'UR2': 7, 'UR1': 8, 'UL1': 9, 'UL2': 10, 'UL3': 11, 
        'UL4': 12, 'UL5': 13, 'UL6': 14, 'UL7': 15, 'UL8': 16, 'LL8': 17, 'LL7': 18, 'LL6': 19, 'LL5': 20, 'LL4': 21, 'LL3': 22, 
        'LL2': 23, 'LL1': 24, 'LR1': 25, 'LR2': 26, 'LR3': 27, 'LR4': 28, 'LR5': 29, 'LR6': 30, 'LR7': 31, 'LR8': 32}

    for tooth in list_teeth:
        if dic[tooth] in lower :
            dic_teeth['Lower'].append(dic[tooth])
        else :
            dic_teeth['Upper'].append(dic[tooth])






    gold_files = list(chain.from_iterable(search(args.gold_folder[0],list_extension).values()))

    gold ={}

    gold[UpperOrLower(gold_files[0])]= ReadSurf(gold_files[0])
    gold[UpperOrLower(gold_files[1])]= ReadSurf(gold_files[1])




    if not os.path.exists(os.path.split(args.log_path[0])[0]):
        os.mkdir(os.path.split(args.log_path[0])[0])

    with open(args.log_path[0],'w') as log_f :
        log_f.truncate(0)




    if args.occlusion[0].lower() == 'true':
        link = True
        if args.jaw[0] == 'Upper':
            jaw = Jaw(Upper())

        elif args.jaw[0] == 'Lower':
            jaw = Jaw(Lower())

    else:
        link = False

    if link:
        list_files=Files_vtk_link(args.input[0])

    
    else :
        list_files=list(chain.from_iterable(search(args.input[0],list_extension).values()))




    methode = [ InitIcp(),vtkICP()]
    option_upper = vtkMeanTeeth(dic_teeth['Upper'])
    option_lower = vtkMeanTeeth(dic_teeth['Lower'])
    icp_upper = ICP(methode, option=option_upper)
    icp_lower = ICP(methode, option=option_lower)
    icp = {'Upper':icp_upper,'Lower':icp_lower}



    for index , file in tqdm(enumerate(list_files),total=len(list_files)):
        file_vtk = file
        if link:
            file_vtk = file[jaw()]
        if not link :
            jaw = Jaw(file_vtk)
            
        surf = ReadSurf(file_vtk)
        

        try :
            surf, matrix = PrePreAso(surf,gold[jaw()],dic_teeth[jaw()])
            output_icp = icp[jaw()].run(surf,gold[jaw()])



        except ToothNoExist as tne:
            print(f'Error {tne}, for this file {file_vtk}')
          
            WritefileError(file_vtk,args.folder_error[0],f'Error {str(tne)}, for this file {file_vtk}')

            with open(args.log_path[0],'r+') as log_f:
                log_f.write(str(index))  
            continue



        except NoSegmentationSurf as nss :

            print(f'Error {nss}, for this file {file_vtk}')
          
            WritefileError(file_vtk,args.folder_error[0],f'Error {str(nss)}, for this file {file_vtk}')

            with open(args.log_path[0],'r+') as log_f:
                log_f.write(str(index))  
            continue

            

    

        WriteSurf(output_icp['source_Or'],args.output_folder[0],os.path.basename(file_vtk),args.add_inname[0])
        # matrix_final = np.matmul(output_icp['matrix'],matrix)
        # np.save(os.path.join(args.output_folder[0] ,f'matrix_{file["name"]}.npy'), matrix_final)

        if link:
            surf_lower = ReadSurf(file[jaw.inv()])
            output_lower = TransformSurf(surf_lower,matrix)
            output_lower = TransformSurf(output_lower,output_icp['matrix'])
            
            WriteSurf(output_lower,args.output_folder[0],os.path.basename(file[jaw.inv()]),args.add_inname[0])

        with open(args.log_path[0],'r+') as log_f:
            log_f.write(str(index))  
if __name__ == "__main__":
    

    print("Starting")
    print(sys.argv)

    parser = argparse.ArgumentParser()


    parser.add_argument('input',nargs=1)
    parser.add_argument('gold_folder',nargs=1)
    parser.add_argument('output_folder',nargs=1)
    parser.add_argument('add_inname',nargs=1)
    parser.add_argument('list_teeth',nargs=1)
    parser.add_argument('occlusion',nargs=1)
    parser.add_argument('jaw',nargs=1)
    parser.add_argument('folder_error',nargs=1)
    parser.add_argument('log_path',nargs=1)


    args = parser.parse_args()
    print(args)


    main(args)