import numpy as np
import pygame
import pickle
import json
from pathlib import Path
from ..envs.rrt_env import BaseEnv
from ..rendering.debug_renderer import DebugRenderer
from ..makers.makers import *
from ..nodes import VelTreeNode
import warnings


class SavePathReplayer:
    def __init__(self, fpath: Path):
        self.fpath = fpath
        self.pickled_path = None
        self.fdata = None
        self._get_filedata()

    def _get_filedata(self):
        with open(self.fpath, 'rb') as f:
            try:
                self.pickled_path = pickle.load(f)
            except pickle.UnpicklingError:
                warnings.warn(
                    "No pickled path found, will try to load json path instead")

            self.fdata = json.loads(f.read().decode('utf-8'))

    def _replay_from_pickle(self, env):
        path = self.pickled_path
        for i in range(0, len(path.nodes)):
            node = path.nodes[i]
            assert isinstance(node, VelTreeNode), "Node is not VelTreeNode"
            # print(f"node {i} buffer:{len(node.velocity_buffer)}")
            for vel in node.velocity_buffer:
                env.map.sim.step()
                env.map.agent.velocity = vel
            print(f"node {i} pos:{env.map.agent.position}")
            env.map.sim.import_from(node.state)
            env.map.sim.step()

        env.map.agent.velocity = np.zeros_like(env.map.agent.velocity)

    def _replay_from_json(self, env):
        for i in range(0, len(self.fdata['nodes'])):
            node = self.fdata['nodes'][i]
            for vel in node['velocity']:
                env.map.sim.step()
                env.map.agent.velocity = np.array(vel)
            env.map.agent.position = np.array(node['agent_pos'])
            # print(node['agent_rot'])
            env.map.agent.orientation = node['agent_rot']
            print(f"node {i} pos:{env.map.agent.position}")

            env.map.sim.step()

    def replay(self):
        renderer = DebugRenderer(self.fdata['cfg'])
        maker_name = self.fdata['maker_name']
        my_cls = maker_name.split('=')[0]
        method_name = maker_name.split('=')[1]
        map_name = self.fdata['map_name']
        cfg = self.fdata['cfg']
        try:
            maker, _, _ = globals()[my_cls].__dict__[
                method_name](map_name, cfg, render_mode='human')
        except KeyError:
            raise KeyError(
                f"Maker {maker_name} not found. Might take a look into {self.fdata['script_name']}")

        env = maker()
        if not isinstance(env, BaseEnv):
            env = env.unwrapped

        env.map.sim.attach_renderer(renderer)

        if self.pickled_path is not None:
            self._replay_from_pickle(env)
        else:
            warnings.warn(
                "No pickled path found, will play the json path instead")
            self._replay_from_json(env)
