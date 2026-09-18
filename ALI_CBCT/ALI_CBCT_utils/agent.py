import numpy as np
import time
from collections import deque

import os

from ALI_CBCT_utils.constants import bcolors

# --- LOGGING CONFIGURATION ---
from ADTLib.logging_setup import get_logger

logger = get_logger("ALI_CBCT_Agent")

# How much work one search may spend, in STEPS -- one network forward pass
# each. It used to be a number of SECONDS (15 on a GPU, 60 otherwise), which
# made the result depend on the machine: the same scan converged or not
# depending on whether the GPU was busy, the disk slow, or another module
# running. A step count is the same everywhere, so two runs of the same scan
# land on the same landmark.
#
# 1000 is about six times the longest search measured over ninety searches on
# three scans with the shipped models (32 to 171 steps, median around 100),
# and it is only ever reached by a search that is not converging.
#
# Override with ALI_SEARCH_MAX_STEPS. Raise it for an unusually large volume
# or a fine spacing, where crossing the scan takes more steps.
DEFAULT_MAX_STEPS = 1000

# A guard rail, not a budget. Nothing below decides on it in the normal case;
# it exists so that a machine slow enough to turn the step budget into hours
# -- CPU-only inference, mainly -- still gives the operator its scan back.
# Override with ALI_SEARCH_TIME_GUARD, in seconds.
DEFAULT_TIME_GUARD = 900.0

# How many steps in a row an agent may take without reaching a single
# position it has not already stood on. Past that it is going round a ring
# rather than searching, and no amount of budget will get it out. See
# Agent.Cycling. Sixty-four is one field of view worth of steps, which is
# far above anything a converging search does: over ninety searches measured
# on three scans with the shipped models, the count never left zero -- every
# single step of every one of them landed on ground the agent had not stood
# on before.
STALL_STEPS = 64


def _env_number(name, default, cast):
    """An environment override, or the default if it is not a number."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not a number, keeping {default}")
        return default


def SearchStepBudget():
    """How many steps one search may take. See DEFAULT_MAX_STEPS."""
    return _env_number("ALI_SEARCH_MAX_STEPS", DEFAULT_MAX_STEPS, int)


def SearchTimeGuard():
    """Seconds after which a search is cut short. See DEFAULT_TIME_GUARD."""
    return _env_number("ALI_SEARCH_TIME_GUARD", DEFAULT_TIME_GUARD, float)

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


def InsideSamplableZone(position, low, high):
    """May the agent stand here -- can GetZone read a whole field of view?

    `low` and `high` come from Environment.GetSamplableBounds, which derives
    them from the padding GetZone crops inside. One rule, both ends: the
    agent may stand wherever the state it is about to read is the real one.

    This replaces `new_pos.all() > 0 and (new_pos < GetSize(scale)).all()`,
    wrong on both sides, and the two sides have to be repaired together.

    The floor. `new_pos.all()` reduces the array to ONE boolean before the
    comparison: `True > 0` is True and `False > 0` is False, so the test read
    "no coordinate is exactly zero", never "every coordinate is positive".
    Two opposite faults at once. Coordinates far out on the negative side
    went through, and they are the expensive ones -- SpatialCrop clamps a
    negative crop start at zero, so they all read the SAME zone, the network
    keeps answering the same move, and the agent walks out of the volume for
    as long as its budget lasts. Meanwhile a step onto a coordinate of
    exactly ZERO -- a real voxel, which GetZone reads perfectly -- was
    refused, and cost one of the three attempts a search is allowed. That
    second fault is what kept `Me` out of reach on a scan whose chin sits on
    the bottom slice: measured on MG_test_scan, the search was refused the
    step from voxel 1 to voxel 0 three times and gave up.

    The ceiling. It was `GetSize`, the size of the volume BEFORE padding,
    while GetZone crops the padded tensor. The agent forbade itself a zone it
    can read, and paid an attempt for each step into it.

    Repairing the floor without widening the ceiling would make the agent
    give up EARLIER near a face, which is why neither half travels alone.
    """
    position = np.asarray(position)
    return bool((position >= low).all() and (position <= high).all())


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

            # Every position stood on since the current attempt began, and
            # how many steps in a row landed on one of them. See Cycling.
            self.ground = set()
            self.steps_on_known_ground = 0

            # Why the last Search returned -1, in words. See Search.
            self.failure_reason = None

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
        # A respawn starts an attempt over: what the previous one had walked
        # says nothing about whether this one is going in circles.
        self.ground = set()
        self.steps_on_known_ground = 0
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
        low, high = self.environement.GetSamplableBounds(
            self.scale_keys[self.scale_state], self.FOV)
        if InsideSamplableZone(new_pos, low, high):
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
        # `while not found` with nothing else to stop it: a probe that keeps
        # moving without ever landing twice inside the short memory hangs the
        # whole run, with no budget above it to cut it short -- Focus is
        # called after the search loop has ended. The same step budget bounds
        # it. A probe normally settles in a handful of steps, so reaching the
        # budget means this one is not converging; the others still vote.
        max_steps = SearchStepBudget()
        for pos in explore_pos:
            found = False
            step = 0
            self.position_shortmem[self.scale_state].clear()
            self.position = start_pos + radius*pos
            while not found and step < max_steps:
                step += 1
                action = self.PredictAction()
                self.Move(action)
                if self.Visited():
                    found = True
                self.SavePos()
            if not found:
                logger.warning(
                    f"Focus probe {pos} for {self.target} did not settle "
                    f"within {max_steps} steps; its last position is used")
            final_pos += self.position
        return final_pos/len(explore_pos)

    def Search(self):
        """Search for landmark with comprehensive error handling.

        Returns the number of steps it took, or -1 when the landmark was not
        placed. On -1 `self.failure_reason` says which of the ways it was --
        the caller puts that in front of the operator, since a landmark that
        is simply absent from the output file is a result nobody can read.
        """
        tic = time.time()
        self.failure_reason = None
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
            self.ground = set()
            self.steps_on_known_ground = 0

            found = False
            tot_step = 0
            max_steps = SearchStepBudget()
            time_guard = SearchTimeGuard()

            while not found and tot_step < max_steps:
                tot_step += 1

                if time.time() - tic > time_guard:
                    logger.error(
                        f"Landmark {self.target} abandoned after {tot_step} "
                        f"steps: the {time_guard}s guard rail fired before the "
                        f"{max_steps} step budget. This machine is too slow "
                        "for the budget it was given -- see ALI_SEARCH_MAX_STEPS "
                        "and ALI_SEARCH_TIME_GUARD.")
                    self.search_atempt = 0
                    self.failure_reason = (
                        f"the {time_guard}s guard rail fired after "
                        f"{tot_step} steps")
                    return -1

                try:
                    action = self.PredictAction()
                    self.Move(action)
                    
                    if self.Visited():
                        found = True

                    self.SavePos()

                    if not found and self.Cycling():
                        logger.warning(
                            f"Landmark {self.target} is going in circles at "
                            f"scale {self.scale_state}, step {tot_step}: "
                            f"{STALL_STEPS} steps without reaching one "
                            f"position it had not already stood on, out of "
                            f"{len(self.ground)} of them. Respawning "
                            f"(attempt {self.search_atempt + 1}).")
                        self.ClearShortMem()
                        self.SetRandomPos()
                        self.search_atempt += 1

                    if found:
                        logger.debug(f"Landmark {self.target} found at scale: {self.scale_state}")
                        logger.debug(f"Agent position: {self.position}")
                        
                        scale_changed = self.UpScale()
                        found = not scale_changed
                    
                    if self.search_atempt > 2:
                        logger.warning(f"Landmark {self.target} not found after {self.search_atempt} attempts")
                        self.search_atempt = 0
                        self.failure_reason = (
                            f"gave up after {tot_step} steps and three "
                            "respawns: the agent kept leaving the zone it "
                            "can read, or going round in circles")
                        return -1
                        
                except Exception as e:
                    # `continue` here: a real fault -- an index out of range,
                    # a tensor of the wrong shape -- was logged once per step
                    # and the loop started over on the SAME state, so it
                    # failed the same way until the budget ran out. What the
                    # operator was then told was "not found", never the
                    # cause. Nothing about a step changes when it fails, so
                    # there is nothing to retry: the error goes up, where the
                    # handler below names the landmark and the caller counts
                    # the scan as failed.
                    logger.error(
                        f"Search for {self.target} failed at step {tot_step}, "
                        f"scale {self.scale_state}, position {self.position}: {e}")
                    self.failure_reason = f"failed at step {tot_step}: {e}"
                    raise

            if not found:  # Spent its whole budget without settling
                logger.warning(
                    f"Landmark {self.target} not found within its budget of "
                    f"{max_steps} steps")
                self.search_atempt = 0
                self.failure_reason = (
                    f"never settled within its budget of {max_steps} steps")
                return -1

            try:
                final_pos = self.Focus(self.position)
                logger.info(f"Final position for {self.target}: {final_pos}")
                self.environement.AddPredictedLandmark(self.target, final_pos)
                return tot_step
            except Exception as e:
                logger.error(f"Error in focus phase for {self.target}: {e}")
                self.failure_reason = f"found, but the focus phase failed: {e}"
                return -1

        except Exception as e:
            logger.error(f"Fatal error during search for {self.target}: {e}")
            if self.failure_reason is None:
                self.failure_reason = str(e)
            return -1

    def Cycling(self):
        """Is the agent walking a ring it has already been round?

        `Visited()` only compares against the last `shortmem_size` positions
        -- ten. A cycle longer than that never satisfies it, so an agent
        caught in one used to keep going until its budget ran out, and the
        operator was told the landmark could not be found, not that the
        search had been turning on the spot.

        Here every position stood on since the attempt began is kept, and
        the answer is yes once STALL_STEPS go by without the agent reaching
        one it had not already stood on. Both counters restart on a respawn,
        in SetRandomPos.
        """
        key = (self.scale_state,) + tuple(int(c) for c in self.position)
        if key in self.ground:
            self.steps_on_known_ground += 1
        else:
            self.ground.add(key)
            self.steps_on_known_ground = 0
        return self.steps_on_known_ground >= STALL_STEPS

    def Visited(self):
        visited = False
        for previous_pos in self.position_shortmem[self.scale_state]:
            if np.array_equal(self.position,previous_pos):
                visited = True
        return visited