import numpy as np
import time
from collections import deque

from ALI_CBCT_utils.constants import bcolors

def GetAgentLst(agents_param):
    print("-- Generating agents --")

    agent_lst = []
    for label in agents_param["landmarks"]:
        print(f"{bcolors.OKCYAN}Generating Agent for the lamdmark: {bcolors.OKBLUE}{label}{bcolors.ENDC}")
        agt = agents_param["type"](
            targeted_landmark=label,
            movements = agents_param["movements"],
            scale_keys = agents_param["scale_keys"],
            FOV=agents_param["FOV"],
            start_pos_radius = agents_param["spawn_rad"],
            speed_per_scale = agents_param["speed_per_scale"],
            verbose = agents_param["verbose"]
        )
        agent_lst.append(agt)

    print(f"{bcolors.OKGREEN}{len(agent_lst)} agent successfully generated. {bcolors.ENDC}")

    return agent_lst
    
def OUT_WARNING():
    print(f"{bcolors.WARNING}WARNING : Agent trying to go in a none existing space {bcolors.ENDC}")
    
class Agent :
    def __init__(
        self,
        targeted_landmark,
        movements,
        scale_keys,
        brain = None,
        environement = None,
        FOV = [32,32,32],
        start_pos_radius = 20,
        shortmem_size = 10,
        speed_per_scale = [2,1],
        verbose = False
    ) -> None:

        self.target = targeted_landmark
        self.scale_keys = scale_keys
        self.environement = environement
        self.scale_state = 0
        self.start_pos_radius = start_pos_radius
        self.start_position = np.array([0,0,0], dtype=np.int16)
        self.position = np.array([0,0,0], dtype=np.int16)
        self.FOV = np.array(FOV, dtype=np.int16)

        self.movement_matrix = movements["mat"]
        self.movement_id = movements["id"]

        self.brain = brain
        self.shortmem_size = shortmem_size

        self.verbose = verbose


        self.search_atempt = 0
        self.speed_per_scale = speed_per_scale
        self.speed = self.speed_per_scale[0]


    def SetEnvironment(self, environement):
        self.environement = environement
        position_mem = []
        position_shortmem = []
        for i in range(environement.scale_nbr):
            position_mem.append([])
            position_shortmem.append(deque(maxlen=self.shortmem_size))
        self.position_mem = position_mem
        self.position_shortmem = position_shortmem

    def SetBrain(self,brain): self.brain = brain

    def ClearShortMem(self):
        for mem in self.position_shortmem:
            mem.clear()

    def GoToScale(self,scale=0):
        self.position = (self.position*(self.environement.GetSpacing(self.scale_keys[self.scale_state])/self.environement.GetSpacing(self.scale_keys[scale]))).astype(np.int16)
        self.scale_state = scale
        self.search_atempt = 0
        self.speed = self.speed_per_scale[scale]

    def SetPosAtCenter(self):
        self.position = self.environement.GetSize(self.scale_keys[self.scale_state])/2

    def SetRandomPos(self):
        if self.scale_state == 0:
            rand_coord = np.random.randint(1, self.environement.GetSize(self.scale_keys[self.scale_state]), dtype=np.int16)
            self.start_position = rand_coord
            # rand_coord = self.environement.GetLandmarkPos(self.scale_keys[self.scale_state],self.target)
        else:
            rand_coord = np.random.randint([1,1,1], self.start_pos_radius*2) - self.start_pos_radius
            rand_coord = self.start_position + rand_coord
            rand_coord = np.where(rand_coord<0, 0, rand_coord)
            rand_coord = rand_coord.astype(np.int16)

        self.position = rand_coord


    def GetState(self):
        state = self.environement.GetZone(self.scale_keys[self.scale_state] ,self.position,self.FOV)
        return state

    def UpScale(self):
        scale_changed = False
        if self.scale_state < self.environement.scale_nbr-1:
            self.GoToScale(self.scale_state + 1)
            scale_changed = True
            self.start_position = self.position
        # else:
        #     OUT_WARNING()
        return scale_changed

    def PredictAction(self):
        return self.brain.Predict(self.scale_state,self.GetState())

    def Move(self, movement_idx):
        new_pos = self.position + self.movement_matrix[movement_idx]*self.speed
        if new_pos.all() > 0 and (new_pos < self.environement.GetSize(self.scale_keys[self.scale_state])).all():
            self.position = new_pos
            # if self.verbose:
            #     print("Moving ", self.movement_id[movement_idx])
        else:
            OUT_WARNING()
            self.ClearShortMem()
            self.SetRandomPos()
            self.search_atempt +=1

    def Train(self, data, dim):
        if self.verbose:
            print(f"{bcolors.OKCYAN}Training agent :{bcolors.OKBLUE}{self.target}{bcolors.ENDC}")
        self.brain.Train(data,dim)

    def Validate(self, data,dim):
        if self.verbose:
            print(f"{bcolors.OKCYAN}Validating agent :{bcolors.OKBLUE}{self.target}{bcolors.ENDC}")
        return self.brain.Validate(data,dim)

    def SavePos(self):
        self.position_mem[self.scale_state].append(self.position)
        self.position_shortmem[self.scale_state].append(self.position)

    def Focus(self,start_pos):
        explore_pos = np.array(
            [
                [1,0,0],
                [-1,0,0],
                [0,1,0],
                [0,-1,0],
                [0,0,1],
                [0,0,-1]
            ],
            dtype=np.int16
        )
        radius = 4
        final_pos = np.array([0,0,0], dtype=np.float64)
        for pos in explore_pos:
            found = False
            self.position_shortmem[self.scale_state].clear()
            self.position = start_pos + radius*pos
            while  not found:
                action = self.PredictAction()
                self.Move(action)
                if self.Visited():
                    found = True
                self.SavePos()
            final_pos += self.position
        return final_pos/len(explore_pos)

    def Search(self):
        # if self.verbose:
        tic = time.time()
        print("Searching landmark :",self.target)
        self.GoToScale()
        self.SetPosAtCenter()
        # self.SetRandomPos()
        self.SavePos()
        found = False
        tot_step = 0
        while not found and time.time()-tic < 15:
            # action = self.environement.GetBestMove(self.scale_state,self.position,self.target)
            tot_step+=1
            action = self.PredictAction()
            self.Move(action)
            if self.Visited():
                found = True
            self.SavePos()
            if found:
                if self.verbose:
                    print("Landmark found at scale :",self.scale_state)
                    print("Agent pos = ", self.position)
                    if self.environement.LandmarkIsPresent(self.target):
                        print("Landmark pos = ", self.environement.GetLandmarkPos(self.scale_keys[self.scale_state],self.target))
                scale_changed = self.UpScale()
                found = not scale_changed
            if self.search_atempt > 2:
                print(self.target, "landmark not found")
                self.search_atempt = 0
                return -1

        if not found: # Took too much time
            print(self.target, "landmark not found")
            self.search_atempt = 0
            return -1

        final_pos = self.Focus(self.position)
        print("Result :", final_pos)
        self.environement.AddPredictedLandmark(self.target,final_pos)
        return tot_step

    def Visited(self):
        visited = False
        # print(self.position, self.position_shortmem[self.scale_state],)
        for previous_pos in self.position_shortmem[self.scale_state]:
            if np.array_equal(self.position,previous_pos):
                visited = True
        return visited