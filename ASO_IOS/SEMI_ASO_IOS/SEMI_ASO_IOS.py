#!/usr/bin/env python-real    


import glob
import os

import sys
import argparse
from tqdm import tqdm
import numpy as np

fpath = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(fpath)

from ASO_IOS_utils import (vtkICP, InitIcp, SelectKey, ICP, TransformSurf, UpperOrLower, 
LoadJsonLandmarks, WriteSurf, WriteJsonLandmarks,
 listlandmark2diclandmark, ReadSurf, Files_vtk_json,Files_vtk_json_semilink, Jaw, Lower , Upper, ApplyTransform,WritefileError)



print('semi aso ios chargee')

def main(args):
    print('icp landmark launch')


    dic_landmark=listlandmark2diclandmark(args.list_landmark[0])


    dic_gold={}
    gold_json = glob.glob(args.gold_folder[0]+'/*json')

    dic_gold[UpperOrLower(gold_json[0])]= gold_json[0]
    dic_gold[UpperOrLower(gold_json[1])]= gold_json[1]




    if not os.path.exists(os.path.split(args.log_path[0])[0]):
        os.mkdir(os.path.split(args.log_path[0])[0])

    with open(args.log_path[0],'w') as log_f :
        log_f.truncate(0)

    

   





    methode = [InitIcp(),vtkICP()]
    option_upper = SelectKey(dic_landmark['Upper'])
    option_lower = SelectKey(dic_landmark['Lower'])
    print('dic landmark', dic_landmark)
    icp_upper = ICP(methode,option=option_upper)
    icp_lower = ICP(methode, option=option_lower)
    icp = {'Lower':icp_lower,'Upper':icp_upper}
    print('--------------'*10)


    if args.occlusion[0].lower() == 'true':
        link =True
        if args.jaw[0] == 'Upper':
            jaw = Jaw(Upper())
        elif args.jaw[0] == 'Lower':
            jaw = Jaw(Lower())

    else:
        link = False

    if link :
        list_file=Files_vtk_json_semilink(args.input[0])

    else :
        list_file = Files_vtk_json(args.input[0])

    

    for index ,file in tqdm(enumerate(list_file),total = len(list_file)) :
        file_jaw = file
        if link :
            file_jaw = file[jaw()]
        else :
            jaw = Jaw(file_jaw['json'])

        
        if file_jaw['json'] is None:
            continue

        try :
            output_icp = icp[jaw()].run(file_jaw['json'],dic_gold[jaw()])
        except KeyError as k:
            print('error  KeyError',k,file_jaw)
            WritefileError(file_jaw['json'],args.folder_error[0],f'Please verify this file {file_jaw["json"]} or {dic_gold[jaw()]}, we dont find this landmark {k} ')
            with open(args.log_path[0],'r+') as log_f:
                log_f.write(str(index))   
            continue

        surf_input = ReadSurf(file_jaw['vtk'])
        surf_output = TransformSurf(surf_input,output_icp['matrix'])

        WriteJsonLandmarks(output_icp['source_Or'], file_jaw['json'],file_jaw['json'],args.add_inname[0],args.output_folder[0])
        WriteSurf(surf_output,args.output_folder[0],file_jaw['vtk'],args.add_inname[0])
        np.save(os.path.join(args.output_folder[0] ,f'matrix_{file["name"]}.npy'), output_icp['matrix'])


        if link :
            surf_input = ReadSurf(file[jaw.inv()]['vtk'])
            surf_output = TransformSurf(surf_input,output_icp['matrix'])
            WriteSurf(surf_output,args.output_folder[0],file[jaw.inv()]['vtk'],args.add_inname[0])
            
            if not file[jaw.inv()]['json'] is None:

                json_input = LoadJsonLandmarks(file[jaw.inv()]['json'])
                json_output = ApplyTransform(json_input,output_icp['matrix'])
                WriteJsonLandmarks(json_output,file[jaw.inv()]['json'],file[jaw.inv()]['json'],args.add_inname[0],args.output_folder[0])



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
    parser.add_argument('list_landmark',nargs=1)
    parser.add_argument('occlusion',nargs=1)
    parser.add_argument('jaw',nargs=1)
    parser.add_argument('folder_error',nargs=1)
    parser.add_argument('log_path',nargs=1)

    args = parser.parse_args()


    main(args)