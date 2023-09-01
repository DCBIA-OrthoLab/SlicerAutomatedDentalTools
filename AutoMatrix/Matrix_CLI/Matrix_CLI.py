#!/usr/bin/env python-real

import argparse
import os
import sys
from pathlib import Path
import multiprocessing as mp
import numpy as np
import Apply_matrix_utils as amu


def main(args):
    
    path_patient_input = Path(args.path_patient_intput)

    # clear log file
    with open(args.logPath,'w') as log_f:
        log_f.truncate(0)

    ## Apply matrix on files != .nii.gz 
    ## Call the function in VTK_tools to Apply the matrixs and to save the new files
    ## The update of the file log for the progress bare is in the function CheckSharedListVTK

    idx = 0
    # patients,nb_files = amu.GetPatientsVTK(args.path_patient_intput,args.path_matrix_intput)
    # nb_worker = 6
    # if nb_files!=0:
    #     nb_scan_done = mp.Manager().list([0 for i in range(nb_worker)])
    #     idxProcess = mp.Value('i',idx)
    #     check = mp.Process(target=amu.CheckSharedListVTK,args=(nb_scan_done,nb_files,args.logPath,idxProcess)) 
    #     check.start()

    #     splits = np.array_split(list(patients.keys()),nb_worker)
        
    #     if path_patient_input.is_dir() : 
    #         processess = [mp.Process(target=amu.ApplyMatrixVTK,args=(patients,keys,args.path_patient_intput,args.path_patient_output,i,nb_scan_done,args.logPath,idx,args.suffix)) for i,keys in enumerate(splits)]
        
    #     elif path_patient_input.is_file() : 
    #         processess = [mp.Process(target=amu.ApplyMatrixVTK,args=(patients,keys,os.path.dirname(args.path_patient_intput),args.path_patient_output,i,nb_scan_done,args.logPath,idx,args.suffix)) for i,keys in enumerate(splits)]    

    #     for proc in processess: proc.start()
    #     for proc in processess: proc.join()
    #     check.join()

    


    ## Apply matrix on files == .nii.gz 
    ## Call the function in GZ_tools to Apply the matrixs and to save the new files
    ## The update of the file log for the progress bare is in the function CheckSharedList
    # patients,nb_files = amu.GetPatients(args.path_patient_intput,args.path_matrix_intput)
    # nb_worker = 6
    # if nb_files!=0:
    #     nb_scan_done = mp.Manager().list([0 for i in range(nb_worker)])
    #     idxProcess = mp.Value('i',idx)
    #     check = mp.Process(target=amu.CheckSharedList,args=(nb_scan_done,nb_files,args.logPath,idxProcess)) 
    #     check.start()

    #     splits = np.array_split(list(patients.keys()),nb_worker)
        
    #     if path_patient_input.is_dir() : 
    #         processess = [mp.Process(target=amu.ApplyMatrixGZ,args=(patients,keys,args.path_patient_intput,args.path_patient_output,i,nb_scan_done,args.logPath,idx,args.suffix)) for i,keys in enumerate(splits)]
        
    #     elif path_patient_input.is_file() : 
    #         processess = [mp.Process(target=amu.ApplyMatrixGZ,args=(patients,keys,os.path.dirname(args.path_patient_intput),args.path_patient_output,i,nb_scan_done,args.logPath,idx,args.suffix)) for i,keys in enumerate(splits)]    

    #     for proc in processess: proc.start()
    #     for proc in processess: proc.join()
    #     check.join()

    print("Applied matrix with success")








if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    
    parser.add_argument('path_patient_intput',type=str,help="choose a patient")
    parser.add_argument('path_matrix_intput',type=str,help="choose a matrix")
    parser.add_argument('path_patient_output',type=str,help="choose an output")
    parser.add_argument("suffix",type=str,help="choose a suffix")
    parser.add_argument("logPath",type=str, help="logpath")
    


    args = parser.parse_args()


    main(args)