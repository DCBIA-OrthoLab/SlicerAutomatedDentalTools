# What stops an ALI CBCT search: work, not seconds.
#
# The loop stopped on `time.time() - tic < max_time`, with 15 seconds on a GPU
# and 60 otherwise. The same scan therefore converged or not depending on
# whether the GPU was busy, the disk slow, or another module running: for a
# clinical tool, a result nobody can reproduce.
#
# The bound is now on the number of STEPS -- one forward pass of the network
# each. That is the same amount of work everywhere, so the same result. These
# cases show it by slowing one step down on purpose: the step count does not
# move, where a budget in seconds would have returned fewer.
#
# No model and no GPU here: the brain and the environment are stand-ins, and
# only the loop in Agent.Search is under test.
import os
import sys
import time
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
for _path in (os.path.join(_ROOT, "ADT"), os.path.join(_ROOT, "ALI_CBCT")):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# dicom2nifti does not import outside Slicer, so the module is stubbed.
sys.modules.setdefault("dicom2nifti", types.ModuleType("dicom2nifti"))

import numpy as np  # noqa: E402

from ALI_CBCT_utils import agent as agent_module  # noqa: E402
from ALI_CBCT_utils.agent import (  # noqa: E402
    Agent, DEFAULT_MAX_STEPS, SearchStepBudget, SearchTimeGuard,
)
from ALI_CBCT_utils.constants import MOVEMENTS  # noqa: E402

FOV = [64, 64, 64]
SCALE = "1"
# Large enough that the stand-ins' walk never reaches an edge.
SIZE = np.array([10000, 10000, 10000])


class FakeEnvironment:
    """A volume holding nothing: only its bounds are ever read."""

    scale_nbr = 1

    def __init__(self):
        self.predicted = {}

    def GetSpacing(self, scale):
        return np.array([1.0, 1.0, 1.0])

    def GetSize(self, scale):
        return SIZE

    def GetSamplableBounds(self, scale, crop_size):
        return np.array([-1, -1, -1]), SIZE + 1

    def GetZone(self, scale, center, crop_size):
        return None

    def AddPredictedLandmark(self, lm_id, lm_pos):
        self.predicted[lm_id] = lm_pos


class FakeBrain:
    """A brain that counts its passes and can make one of them slow.

    The moves alternate on two axes: the agent never revisits a position, so
    `Visited()` is never true and the search runs to the end of its budget.
    That is the case worth measuring.
    """

    def __init__(self, delay=0.0):
        self.calls = 0
        self.delay = delay

    def Predict(self, dim, state):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return 0 if self.calls % 2 else 2


def an_agent(brain):
    agent = Agent(targeted_landmark="Me", movements=MOVEMENTS,
                  scale_keys=[SCALE], FOV=FOV, speed_per_scale=[1, 1])
    agent.SetBrain(brain)
    agent.SetEnvironment(FakeEnvironment())
    return agent


def steps_for(budget, delay=0.0):
    """The forward passes actually made, under this budget."""
    os.environ["ALI_SEARCH_MAX_STEPS"] = str(budget)
    try:
        brain = FakeBrain(delay=delay)
        result = an_agent(brain).Search()
    finally:
        del os.environ["ALI_SEARCH_MAX_STEPS"]
    return brain.calls, result


class StepBudgetTest(unittest.TestCase):

    def test_a_search_that_never_settles_spends_exactly_its_budget(self):
        calls, result = steps_for(40)
        self.assertEqual(calls, 40)
        self.assertEqual(result, -1)

    def test_the_same_budget_gives_the_same_work_twice(self):
        self.assertEqual(steps_for(40), steps_for(40))

    def test_a_slow_machine_does_not_shorten_the_search(self):
        """The heart of the matter.

        At 2 ms a step, forty steps take more than forty times what an
        instant step takes. Under a budget in seconds the slow run would have
        stopped well before the fast one; under a budget in steps, the two do
        exactly the same work.
        """
        fast_calls, fast_result = steps_for(40)
        tic = time.time()
        slow_calls, slow_result = steps_for(40, delay=0.002)
        slow_seconds = time.time() - tic

        self.assertEqual(slow_calls, fast_calls)
        self.assertEqual(slow_result, fast_result)
        self.assertGreater(slow_seconds, 0.04,
                           "the slow run really has to be slower")

    def test_the_budget_is_what_changes_the_work(self):
        self.assertEqual(steps_for(7)[0], 7)
        self.assertEqual(steps_for(23)[0], 23)


class BrokenBrain:
    """A brain that fails, the way a malformed tensor would."""

    def __init__(self):
        self.calls = 0

    def Predict(self, dim, state):
        self.calls += 1
        raise IndexError("index 64 is out of bounds for axis 0 with size 64")


class FailedStepTest(unittest.TestCase):
    """A failed step was swallowed and turned into a "timeout".

    The `except ... continue` replayed the same step on the same state until
    the budget ran out: the operator read "not found", never the cause.
    Nothing changes between two attempts, so there is nothing to retry.
    """

    def setUp(self):
        os.environ["ALI_SEARCH_MAX_STEPS"] = "500"

    def tearDown(self):
        os.environ.pop("ALI_SEARCH_MAX_STEPS", None)

    def test_a_failing_step_is_not_replayed_until_the_budget_runs_out(self):
        brain = BrokenBrain()
        result = an_agent(brain).Search()
        self.assertEqual(brain.calls, 1, "one attempt, not five hundred")
        self.assertEqual(result, -1)

    def test_the_cause_reaches_the_log(self):
        brain = BrokenBrain()
        with self.assertLogs("ADT.ALI_CBCT_Agent", level="ERROR") as logged:
            an_agent(brain).Search()
        self.assertTrue(
            any("out of bounds" in line for line in logged.output),
            f"the original message has to surface: {logged.output}")


class RingWalkingBrain:
    """A brain that walks the agent round a ring of forty steps.

    `Visited()` compares against the last TEN positions only: on a longer
    ring no position comes back soon enough, the search never settles, and it
    used to spend its whole budget going round.
    """

    RING = 40

    def __init__(self):
        self.calls = 0

    def Predict(self, dim, state):
        phase = self.calls % self.RING
        self.calls += 1
        if phase < 10:
            return 0   # +x
        if phase < 20:
            return 2   # +y
        if phase < 30:
            return 1   # -x
        return 3       # -y


class CyclingTest(unittest.TestCase):
    """An agent going in circles is stopped well before the budget ends."""

    BUDGET = 2000

    def setUp(self):
        os.environ["ALI_SEARCH_MAX_STEPS"] = str(self.BUDGET)

    def tearDown(self):
        os.environ.pop("ALI_SEARCH_MAX_STEPS", None)

    def test_the_ring_is_longer_than_the_short_memory(self):
        """Otherwise the case would prove nothing: `Visited()` would catch it."""
        self.assertGreater(RingWalkingBrain.RING, Agent(
            targeted_landmark="Me", movements=MOVEMENTS, scale_keys=[SCALE],
            FOV=FOV).shortmem_size)

    def test_without_the_detection_the_whole_budget_goes_to_the_ring(self):
        """The state before, measured: the whole budget goes round the ring."""
        saved = agent_module.STALL_STEPS
        agent_module.STALL_STEPS = 10 ** 9
        try:
            brain = RingWalkingBrain()
            result = an_agent(brain).Search()
        finally:
            agent_module.STALL_STEPS = saved
        self.assertEqual(brain.calls, self.BUDGET)
        self.assertEqual(result, -1)

    def test_with_it_the_search_stops_far_earlier(self):
        brain = RingWalkingBrain()
        result = an_agent(brain).Search()
        self.assertEqual(result, -1)
        self.assertLess(brain.calls, self.BUDGET // 2,
                        "stopped well before the budget ends")

    def test_the_diagnosis_says_what_happened(self):
        with self.assertLogs("ADT.ALI_CBCT_Agent", level="WARNING") as logged:
            an_agent(RingWalkingBrain()).Search()
        self.assertTrue(
            any("going in circles" in line for line in logged.output),
            f"the diagnosis has to be readable: {logged.output}")

    def test_a_converging_search_is_never_called_cycling(self):
        """FakeBrain never revisits a position, so it is never flagged."""
        os.environ["ALI_SEARCH_MAX_STEPS"] = "300"
        agent = an_agent(FakeBrain())
        agent.Search()
        self.assertEqual(agent.steps_on_known_ground, 0)


class BudgetSettingTest(unittest.TestCase):

    def setUp(self):
        self.saved = {name: os.environ.pop(name, None)
                      for name in ("ALI_SEARCH_MAX_STEPS",
                                   "ALI_SEARCH_TIME_GUARD")}

    def tearDown(self):
        for name, value in self.saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

    def test_the_default_is_used_when_nothing_is_set(self):
        self.assertEqual(SearchStepBudget(), DEFAULT_MAX_STEPS)

    def test_the_environment_variable_wins(self):
        os.environ["ALI_SEARCH_MAX_STEPS"] = "250"
        self.assertEqual(SearchStepBudget(), 250)
        os.environ["ALI_SEARCH_TIME_GUARD"] = "12.5"
        self.assertEqual(SearchTimeGuard(), 12.5)

    def test_a_value_that_is_not_a_number_falls_back_to_the_default(self):
        os.environ["ALI_SEARCH_MAX_STEPS"] = "beaucoup"
        self.assertEqual(SearchStepBudget(), DEFAULT_MAX_STEPS)

    def test_the_time_guard_stays_far_above_the_step_budget(self):
        """It is no longer what decides in the ordinary case."""
        self.assertGreater(SearchTimeGuard(), 60.0)


if __name__ == "__main__":
    unittest.main()
