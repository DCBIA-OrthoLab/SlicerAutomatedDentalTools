#!/usr/bin/env python-real

"""
AUTOMATIC LANDMARK IDENTIFICATION IN CBCT SCANS (ALI_CBCT)

Authors :
- Maxime Gillot (UoM)
- Baptiste Baquero (UoM)

"""

#### ##     ## ########   #######  ########  ########
 ##  ###   ### ##     ## ##     ## ##     ##    ##
 ##  #### #### ##     ## ##     ## ##     ##    ##
 ##  ## ### ## ########  ##     ## ########     ##
 ##  ##     ## ##        ##     ## ##   ##      ##
 ##  ##     ## ##        ##     ## ##    ##     ##
#### ##     ## ##         #######  ##     ##    ##


#region IMPORTS

import glob
import sys
import os
import time
import shutil
import argparse
import numpy as np
import ast

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from ALI_CBCT_utils import (
    Agent, GetAgentLst, Brain, DNet, Environment, GenEnvironmentLst,
    GetBrain, CorrectHisto, SetSpacing, convertdicom2nifti,
    MOVEMENTS, DEVICE,
)
    

#endregion

##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

#region Main

def main(input):
    print("Reading : ",args.input)
    if not os.path.exists(os.path.split(args.log_path)[0]):
        os.mkdir(os.path.split(args.log_path)[0])

    with open(args.log_path, "w") as log_f:
        log_f.truncate(0)
        

    scale_spacing = ast.literal_eval(args.spacing)
    speed_per_scale = ast.literal_eval(args.speed_per_scale)
    agent_FOV = ast.literal_eval(args.agent_FOV)
    lm_type = ast.literal_eval(f"{args.lm_type}")
    spawn_radius = int(args.spawn_radius)
    print("Selected spacings : ", scale_spacing)

    # If input in DICOM Format --> CONVERT THEM INTO NIFTI
    if args.DCMInput.lower() == "true":
        convertdicom2nifti(args.input)

    patients = {}
    if os.path.isfile(args.input):
        basename = os.path.basename(args.input)
        patients[basename] = {"scan": args.input, "scans":{}}

    else:
        normpath = os.path.normpath("/".join([args.input, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            basename = os.path.basename(img_fn)

            if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:

                if basename not in patients.keys():
                    patients[basename] = {"scan": img_fn, "scans":{}}




    temp_fold = args.temp_fold

    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)


    print(f"""<filter-progress>{0}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)
    print(f"""<filter-progress>{2}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)
    print(f"""<filter-progress>{0}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)


    for patient,data in patients.items():

        scan = data["scan"]

        scan_name = patient.split(".")

        tempPath = os.path.join(temp_fold, patient)

        if not os.path.exists(tempPath):
            CorrectHisto(scan, tempPath,0.01, 0.99)


        for sp in scale_spacing:
            new_name = ""
            spac = str(sp).replace(".","-")
            for i,element in enumerate(scan_name):
                if i == 0:
                    new_name = scan_name[0] + "_scan_sp" + spac
                else:
                    new_name += "." + element

            outpath = os.path.join(temp_fold,new_name)
            if not os.path.exists(outpath):
                SetSpacing(tempPath,[sp,sp,sp],outpath)
            patients[patient]["scans"][spac] = outpath

        print(f"""<filter-progress>{1}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.5)


    # #####################################
    #  Init_param
    # #####################################


    image_dim = len(agent_FOV) # Dimention of the images 2D or 3D

    SCALE_KEYS = [str(scale).replace('.','-') for scale in scale_spacing]




    environment_lst = GenEnvironmentLst(patient_dic = patients,env_type = Environment, padding =  np.array(agent_FOV)/2 + 1, device = DEVICE, scale_keys=SCALE_KEYS)

    agents_param = {
        "type" : Agent,
        "FOV" : agent_FOV,
        "movements" : MOVEMENTS,
        "scale_keys" : SCALE_KEYS,
        "spawn_rad" : spawn_radius,
        "speed_per_scale" : speed_per_scale,
        "verbose" : False,
        "landmarks": lm_type,
    }

    agent_lst = GetAgentLst(agents_param)
    brain_lst = GetBrain(args.dir_models)
    trainsitionLayerSize = 1024

    print(f"""<filter-progress>{0}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)
    print(f"""<filter-progress>{2}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)
    print(f"""<filter-progress>{0}</filter-progress>""")
    sys.stdout.flush()
    time.sleep(0.5)


    start_time = time.time()

    tot_step = 0
    fails = {}
    outPath = args.output_dir
    idx=0
    for environment in environment_lst:
        with open(args.log_path, "r+") as log_f:
            log_f.write(str(idx+1))

        print(environment.patient_id)
        print(len(agent_lst))
        for agent in agent_lst:
            brain = Brain(
                network_type = DNet,
                network_scales = SCALE_KEYS,
                device = DEVICE,
                in_channels = trainsitionLayerSize,
                out_channels = len(MOVEMENTS["id"]),
                batch_size= 1,
                generate_tensorboard=False,
                verbose=True
                )
            print(agent)
            brain.LoadModels(brain_lst[agent.target])
            agent.SetBrain(brain)
            agent.SetEnvironment(environment)
            search_result = agent.Search()
            agent.SetBrain(None)
            if search_result == -1:
                fails[agent.target] = fails.get(agent.target,0) + 1
            else:
                tot_step += search_result
            print(f"""<filter-progress>{1}</filter-progress>""")
            sys.stdout.flush()
            time.sleep(0.5)
            print(f"""<filter-progress>{0}</filter-progress>""")
            sys.stdout.flush()

        outputdir = outPath
        environment.SavePredictedLandmarks(SCALE_KEYS[-1],outputdir)

    print("Total steps:",tot_step)
    end_time = time.time()
    print('prediction time :' , end_time-start_time)


    for lm, nbr in fails.items():
        print(f"Fails for {lm} : {nbr}/{len(environment_lst)}")


    try:
        shutil.rmtree(temp_fold)
    except OSError as e:
        print("Error: %s : %s" % (temp_fold, e.strerror))


if __name__ == "__main__":

    print("Starting ALI-CBCT")
    print(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input folder or file")
    parser.add_argument("dir_models", type=str, help="Directory of the models")
    parser.add_argument("lm_type", type=str, help="Type of the landmarks to search")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("temp_fold", type=str, help="Temporary folder")
    parser.add_argument("DCMInput", type=str, help="Is the input in DICOM format")
    parser.add_argument("spacing", type=str, default="[1,0.3]", help="Spacing of the images")
    parser.add_argument("speed_per_scale", type=str, default="[1,1]", help="Speed of the agent per scale")
    parser.add_argument("agent_FOV", type=str, default="[64,64,64]", help="Field of view of the agent")
    parser.add_argument("spawn_radius", type=str, default="10", help="Spawn radius of the agent")
    parser.add_argument("log_path", type=str)

    args = parser.parse_args()
    
    main(args)

#endregion