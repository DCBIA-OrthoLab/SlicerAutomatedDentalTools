import numpy as np
import time
import logging
from collections import deque
import sys

from ALI_CBCT_utils.constants import bcolors

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("ALI_CBCT_Agent")
logger.setLevel(logging.INFO)

logger.propagate = False

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def GetAgentLst(agents_param):
    """Generate a list of agents with error handling."""
    logger.info("-- Generating agents --")
    
    if not agents_param:
        logger.error("agents_param cannot be empty")
        raise ValueError("agents_param is empty")
    
    if "landmarks" not in agents_param:
        logger.error("Missing 'landmarks' key in agents_param")
        raise KeyError("agents_param missing required key: 'landmarks'")

    agent_lst = []
    failed_landmarks = []
    
    for label in agents_param["landmarks"]:
        try:
            logger.debug(f"Generating Agent for the landmark: {label}")
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
        except Exception as e:
            logger.error(f"Failed to generate agent for landmark '{label}': {e}")
            failed_landmarks.append(label)

    if failed_landmarks:
        logger.warning(f"Failed to generate agents for landmarks: {failed_landmarks}")
    
    logger.info(f"{len(agent_lst)} agent(s) successfully generated.")
    
    if not agent_lst:
        logger.error("No agents were successfully created")
        raise RuntimeError("Agent list is empty after generation attempt")

    return agent_lst
    
def OUT_WARNING():
    logger.warning("WARNING: Agent trying to move to a non-existing space")
    
class Agent :
    """Agent class for landmark search with error handling."""
    
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
        try:
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
            
            logger.debug(f"Agent initialized for landmark: {targeted_landmark}")
        except Exception as e:
            logger.error(f"Error initializing Agent for landmark '{targeted_landmark}': {e}")
            raise


    def SetEnvironment(self, environement):
        """Set environment with error handling."""
        try:
            if environement is None:
                logger.error("Environment cannot be None")
                raise ValueError("Environment is None")
            
            self.environement = environement
            position_mem = []
            position_shortmem = []
            for i in range(environement.scale_nbr):
                position_mem.append([])
                position_shortmem.append(deque(maxlen=self.shortmem_size))
            self.position_mem = position_mem
            self.position_shortmem = position_shortmem
            logger.debug(f"Environment set for agent {self.target}")
        except Exception as e:
            logger.error(f"Error setting environment for agent {self.target}: {e}")
            raise

    def SetBrain(self, brain):
        """Set brain with error handling."""
        try:
            self.brain = brain
            if brain is not None:
                logger.debug(f"Brain set for agent {self.target}")
        except Exception as e:
            logger.error(f"Error setting brain for agent {self.target}: {e}")
            raise

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
        else:
            OUT_WARNING()
            self.ClearShortMem()
            self.SetRandomPos()
            self.search_atempt +=1

    def Train(self, data, dim):
        if self.verbose:
            logger.info(f"{bcolors.OKCYAN}Training agent :{bcolors.OKBLUE}{self.target}{bcolors.ENDC}")
        self.brain.Train(data,dim)

    def Validate(self, data,dim):
        if self.verbose:
            logger.info(f"{bcolors.OKCYAN}Validating agent :{bcolors.OKBLUE}{self.target}{bcolors.ENDC}")
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
        """Search for landmark with comprehensive error handling."""
        tic = time.time()
        logger.info(f"Starting search for landmark: {self.target}")
        
        try:
            if self.brain is None:
                logger.error(f"Brain not set for agent {self.target}")
                raise RuntimeError("Brain is not initialized")
            
            if self.environement is None:
                logger.error(f"Environment not set for agent {self.target}")
                raise RuntimeError("Environment is not initialized")
            
            self.GoToScale()
            self.SetPosAtCenter()
            self.SavePos()
            
            found = False
            tot_step = 0
            max_time = 15  # seconds
            
            while not found and time.time() - tic < max_time:
                tot_step += 1
                
                try:
                    action = self.PredictAction()
                    self.Move(action)
                    
                    if self.Visited():
                        found = True
                    
                    self.SavePos()
                    
                    if found:
                        logger.debug(f"Landmark {self.target} found at scale: {self.scale_state}")
                        logger.debug(f"Agent position: {self.position}")
                        
                        scale_changed = self.UpScale()
                        found = not scale_changed
                    
                    if self.search_atempt > 2:
                        logger.warning(f"Landmark {self.target} not found after {self.search_atempt} attempts")
                        self.search_atempt = 0
                        return -1
                        
                except Exception as e:
                    logger.error(f"Error during search step for {self.target}: {e}")
                    continue

            if not found:  # Took too much time
                logger.warning(f"Landmark {self.target} search timed out after {max_time} seconds")
                self.search_atempt = 0
                return -1

            try:
                final_pos = self.Focus(self.position)
                logger.info(f"Final position for {self.target}: {final_pos}")
                self.environement.AddPredictedLandmark(self.target, final_pos)
                return tot_step
            except Exception as e:
                logger.error(f"Error in focus phase for {self.target}: {e}")
                return -1
                
        except Exception as e:
            logger.error(f"Fatal error during search for {self.target}: {e}")
            return -1

    def Visited(self):
        visited = False
        for previous_pos in self.position_shortmem[self.scale_state]:
            if np.array_equal(self.position,previous_pos):
                visited = True
        return visited