import time
import numpy as np
# Return a path on map where rectangle reaches from start to end point
from ..simulator.standard_config import StandardConfig
from ..manual_models import BaseManualModel
from ..RL.training_utils import create_multi_env
from ..samplers import NDIMSampler
from ..storages import GNAT
from ..nodes import GoalNode
from ..storage_wrappers import RectEndWrapper
from ..planners import VecEnvPlanner, PlannerResponse
from ..utils.seed_manager import manager


class _LinearModel(BaseManualModel):
    def predict(self, obs, **kwargs):
        return obs, None


class RectPath:
    def __init__(self, maker, node_manager, cfg: StandardConfig, distance_fnc, max_steps=100_000):
        self.maker = maker
        self.cfg = cfg
        self.node_manager = node_manager
        self.distance_fnc = distance_fnc
        self.model = _LinearModel()
        self._time_taken = 0
        self.max_steps = max_steps
        self.rng = np.random.default_rng(
            manager().get_seed(self.__class__.__name__))

    def get_path(self, annealing_iterations=0):
        env = create_multi_env(self.maker, 1, normalize=False)
        sampler = NDIMSampler((0, 0), (self.cfg["width"], self.cfg["height"]))
        storage = GNAT(self.distance_fnc)
        overall_goal = self._overall_goal()
        s_wrapper = RectEndWrapper(
            storage, self.distance_fnc, overall_goal, self.cfg)
        self.planner = VecEnvPlanner(env, self.model, self.cfg)
        start_node = env.env_method("export_state")
        response = PlannerResponse(start_node, {})
        s_wrapper.save_to_storage(response)

        start_time = time.time()
        iter_cnt = 0
        while self.planner.step_cnt < self.max_steps:
            if not s_wrapper.want_next_iter:
                reachead_goal = True
                print("Heuristics Goal reached")
                break
            qrand_raw = sampler.sample()
            self.node_manager.wanted_position = qrand_raw
            qrand = self.node_manager.create_goal()
            nearest = s_wrapper.get_nearest(qrand)
            response = self.planner.check_path(nearest, qrand)

            s_wrapper.save_to_storage(response)
            iter_cnt += 1
            if iter_cnt % 100 == 0:
                print("Heur Iterations: ", iter_cnt)
                print("Heur steps sum: ", self.planner.step_cnt)
                self._time_taken = time.time() - start_time
        self._time_taken = time.time() - start_time
        path = s_wrapper.get_path()
        if annealing_iterations != 0:
            path = self._path_annealing(path, annealing_iterations)
        return path

    def _path_annealing(self, path, annealing_iterations):
        if len(path.nodes) == 0:
            return path
        for _ in range(annealing_iterations):
            p1 = self.rng.integers(0, len(path.nodes))
            p2 = self.rng.integers(0, len(path.nodes))
            if p1 == p2:
                continue
            startI = min(p1, p2)
            endI = max(p1, p2)
            start = path.nodes[startI]
            self.node_manager.wanted_position = path.nodes[endI].agent_pos
            end = self.node_manager.create_goal()
            response = self.planner.check_path(start, end)
            if response.data['reached'] and not response.data['fail']:
                rest = endI + 1 if endI != len(path.nodes) - 1 else endI
                path.nodes = path.nodes[:startI + 1] + \
                    response.path + path.nodes[rest:]
        return path

    def _overall_goal(self):
        return GoalNode(
            (self.cfg['width'] - 200, self.cfg['height'] // 2), threshold=250)

    @property
    def time_taken(self):
        return self._time_taken
