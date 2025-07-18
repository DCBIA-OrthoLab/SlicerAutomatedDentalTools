import os
import json
import numpy as np
import torch
import SimpleITK as sitk
from monai.transforms import Compose, EnsureChannelFirst, BorderPad, ScaleIntensity, SpatialCrop

from ALI_CBCT_utils.constants import LABELS, LABEL_GROUPS, SCALE_KEYS, DEVICE, bcolors
from ALI_CBCT_utils.io import WriteJson, GenControlPoint

class Environment :
    def __init__(
        self,
        patient_id,
        padding,
        device,
        scale_keys = None,
        correct_contrast = False,
        verbose = False,

    ) -> None:
        """
        Args:
            images_path : path of the image with all the different scale,
            landmark_fiducial : path of the fiducial list linked with the image,
        """
        self.patient_id = patient_id
        self.padding = padding.astype(np.int16)
        self.device = device
        self.scale_keys = scale_keys if scale_keys is not None else SCALE_KEYS
        self.verbose = verbose
        self.transform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            BorderPad(spatial_border=self.padding.tolist())
        ])
        # self.transform = Compose([EnsureChannelFirst(),BorderPad(spatial_border=self.padding.tolist()),ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)])

        self.scale_nbr = 0

        # self.transform = Compose([EnsureChannelFirst(),BorderPad(spatial_border=self.padding.tolist())])
        self.available_lm = []

        self.data = {}

        self.predicted_landmarks = {}


    def LoadImages(self,images_path):

        scales = []

        for scale_id,path in images_path.items():
            data = {"path":path}
            img = sitk.ReadImage(path)
            img_ar = sitk.GetArrayFromImage(img)
            data["image"] = self.transform(img_ar).to(dtype=torch.int16)

            data["spacing"] = np.array(img.GetSpacing())
            origin = img.GetOrigin()
            data["origin"] = np.array([origin[2],origin[1],origin[0]])
            data["size"] = np.array(np.shape(img_ar))

            data["landmarks"] = {}

            self.data[scale_id] = data
            self.scale_nbr += 1



    def LoadJsonLandmarks(self,fiducial_path):
        # print(fiducial_path)
        # test = []

        with open(fiducial_path) as f:
            data = json.load(f)

        markups = data["markups"][0]["controlPoints"]
        for markup in markups:
            if markup["label"] not in LABELS:
                print(fiducial_path)
                print(f"{bcolors.WARNING}WARNING : {markup['label']} is an unusual landmark{bcolors.ENDC}")
            # test.append(markup["label"])
            mark_pos = markup["position"]
            lm_ph_coord = np.array([mark_pos[2],mark_pos[1],mark_pos[0]])
            self.available_lm.append(markup["label"])
            for scale,scale_data in self.data.items():
                lm_coord = ((lm_ph_coord+ abs(scale_data["origin"]))/scale_data["spacing"]).astype(np.int16)
                scale_data["landmarks"][markup["label"]] = lm_coord

        # print(test)


    def SavePredictedLandmarks(self,scale_key,out_path=None):
        img_path = self.data[scale_key]["path"]
        print(f"Saving predicted landmarks for patient{self.patient_id} at scale {scale_key}")

        ref_origin = self.data[scale_key]["origin"]
        ref_spacing = self.data[scale_key]["spacing"]
        physical_origin = abs(ref_origin/ref_spacing)

        # print(ref_origin,ref_spacing,physical_origin)

        landmark_dic = {}
        for landmark,pos in self.predicted_landmarks.items():

            real_label_pos = (pos-physical_origin)*ref_spacing
            real_label_pos = [real_label_pos[2],real_label_pos[1],real_label_pos[0]]
            # print(real_label_pos)
            if LABEL_GROUPS[landmark] in landmark_dic.keys():
                landmark_dic[LABEL_GROUPS[landmark]].append({"label": landmark, "coord":real_label_pos})
            else:landmark_dic[LABEL_GROUPS[landmark]] = [{"label": landmark, "coord":real_label_pos}]


        # print(landmark_dic)

        for group,list in landmark_dic.items():

            id = self.patient_id.split(".")[0]
            json_name = f"{id}_lm_Pred_{group}.mrk.json"

            if out_path is not None:
                file_path = os.path.join(out_path,json_name)
            else:
                file_path = os.path.join(os.path.dirname(img_path),json_name)
            groupe_data = {}
            for lm in list:
                groupe_data[lm["label"]] = {"x":lm["coord"][0],"y":lm["coord"][1],"z":lm["coord"][2]}

            lm_lst = GenControlPoint(groupe_data)
            WriteJson(lm_lst,file_path)

    def ResetLandmarks(self):
        for scale in self.data.keys():
            self.data[scale]["landmarks"] = {}

        self.available_lm = []

    def LandmarkIsPresent(self,landmark):
        return landmark in self.available_lm

    def GetLandmarkPos(self,scale,landmark):
        return self.data[scale]["landmarks"][landmark]

    def GetL2DistFromLandmark(self, scale, position, target):
        label_pos = self.GetLandmarkPos(scale,target)
        return np.linalg.norm(position-label_pos)**2

    def GetZone(self,scale,center,crop_size):
        cropTransform = SpatialCrop(center.tolist() + self.padding,crop_size)
        rescale = ScaleIntensity(minv = -1.0, maxv = 1.0, factor = None)
        crop = cropTransform(self.data[scale]["image"])
        # print(tor ch.max(crop))
        crop = rescale(crop).type(torch.float32)
        return crop

    def GetRewardLst(self,scale,position,target,mvt_matrix):
        agent_dist = self.GetL2DistFromLandmark(scale,position,target)
        get_reward = lambda move : agent_dist - self.GetL2DistFromLandmark(scale,position + move,target)
        reward_lst = list(map(get_reward,mvt_matrix))
        return reward_lst

    def GetRandomPoses(self,scale,target,radius,pos_nbr):
        if scale == self.scale_keys[0]:
            porcentage = 0.2 #porcentage of data around landmark
            centered_pos_nbr = int(porcentage*pos_nbr)
            rand_coord_lst = self.GetRandomPosesInAllScan(scale,pos_nbr-centered_pos_nbr)
            rand_coord_lst += self.GetRandomPosesArounfLabel(scale,target,radius,centered_pos_nbr)
        else:
            # print("RANDOOOOOOM AROUND LABEL")
            rand_coord_lst = self.GetRandomPosesArounfLabel(scale,target,radius,pos_nbr)

        return rand_coord_lst

    def GetRandomPosesInAllScan(self,scale,pos_nbr):
        max_coord = self.data[scale]["size"]
        get_rand_coord = lambda x: np.random.randint(1, max_coord, dtype=np.int16)
        rand_coord_lst = list(map(get_rand_coord,range(pos_nbr)))
        return rand_coord_lst

    def GetRandomPosesArounfLabel(self,scale,target,radius,pos_nbr):
        min_coord = [0,0,0]
        max_coord = self.data[scale]["size"]
        landmark_pos = self.GetLandmarkPos(scale,target)

        get_random_coord = lambda x: landmark_pos + np.random.randint([1,1,1], radius*2) - radius

        rand_coords = map(get_random_coord,range(pos_nbr))

        correct_coord = lambda coord: np.array([min(max(coord[0],min_coord[0]),max_coord[0]),min(max(coord[1],min_coord[1]),max_coord[1]),min(max(coord[2],min_coord[2]),max_coord[2])])
        rand_coords = list(map(correct_coord,rand_coords))

        return rand_coords

    def GetSampleFromPoses(self,scale,target,pos_lst,crop_size,mvt_matrix):

        get_sample = lambda coord : {
            "state":self.GetZone(scale,coord,crop_size),
            "target": np.argmax(self.GetRewardLst(scale,coord,target,mvt_matrix))
            }
        sample_lst = list(map(get_sample,pos_lst))

        return sample_lst

    def GetSpacing(self,scale):
        return self.data[scale]["spacing"]

    def GetSize(self,scale):
        return self.data[scale]["size"]

    def AddPredictedLandmark(self,lm_id,lm_pos):
        # print(f"Add landmark {lm_id} at {lm_pos}")
        self.predicted_landmarks[lm_id] = lm_pos

    def __str__(self):
        print(self.patient_id)
        for scale in self.data.keys():
            print(f"{scale}")
            print(self.data[scale]["spacing"])
            print(self.data[scale]["origin"])
            print(self.data[scale]["size"])
            print(self.data[scale]["landmarks"])
        return ""
    
def GenEnvironmentLst(patient_dic ,env_type, padding = 1, device = DEVICE, scale_keys = None):
    environement_lst = []
    for patient,data in patient_dic.items():
        print(f"{bcolors.OKCYAN}Generating Environement for the patient: {bcolors.OKBLUE}{patient}{bcolors.ENDC}")
        env = env_type(
            patient_id = patient,
            device = device,
            padding = padding,
            scale_keys = scale_keys,
            verbose = False,
        )
        env.LoadImages(data["scans"])
        environement_lst.append(env)
    return environement_lst
