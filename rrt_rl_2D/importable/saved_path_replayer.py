import numpy as np
import pygame
import pickle
import json
from pathlib import Path
from ..envs.rrt_env import BaseEnv
from ..rendering.debug_renderer import DebugRenderer
from ..CLIENT.makers.makers import *
from ..nodes import VelTreeNode
import warnings


class SavePathReplayer:
    """
    Replay path save by VelPathSaver
    """

    def __init__(self, fpath: Path, renderer=None):
        self.fpath = fpath
        self.pickled_path = None
        self.fdata = None
        self.renderer = renderer
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

        if len(path.nodes) == 0:
            warnings.warn("No nodes found in path pickle")
            return False

        for i in range(0, len(path.nodes)):
            node = path.nodes[i]
            assert isinstance(node, VelTreeNode), "Node is not VelTreeNode"
            # print(f"node {i} buffer:{len(node.velocity_buffer)}")
            for vel in node.velocity_buffer:
                env.map.sim.step()
                env.map.agent.velocity = vel
            # print(f"node {i} pos:{env.map.agent.position}")
            env.map.sim.import_from(node.state)
            env.map.sim.step()

        env.map.agent.velocity = np.zeros_like(env.map.agent.velocity)
        return True

    def _replay_from_json(self, env):
        path_length = len(self.fdata['nodes'])
        if path_length == 0:
            warnings.warn("No nodes found in path JSON")
            return False

        for i in range(0, path_length):
            node = self.fdata['nodes'][i]
            for vel in node['velocity']:
                env.map.sim.step()
                env.map.agent.velocity = np.array(vel)
            env.map.agent.position = np.array(node['agent_pos'])
            # print(node['agent_rot'])
            env.map.agent.orientation = node['agent_rot']
            # print(f"node {i} pos:{env.map.agent.position}")

            env.map.sim.step()

    def _process_data(self):

        try:
            self.fdata['data']['nodes']
        except KeyError:
            warnings.warn("No RRT nodes found in data")
            return

        print("PROCESS DATA")
        print(len(self.fdata['data']['nodes']))

        def clb(screen, font):
            for node in self.fdata['data']['nodes']:
                color = np.random.randint(0, 255, 3)
                if type(node[0]) == list:
                    for point in node:
                        pygame.draw.circle(screen, color, point, 2)
                    for i in range(1, len(node)):
                        pygame.draw.line(screen, color, node[i - 1], node[i])
                else:
                    pygame.draw.circle(screen, color, node, 2)
        self.renderer.one_time_draw(clb)

    def _show_path_nodes(self):
        def clb(screen, font):
            try:
                nodes = self.fdata['nodes']
                for i in range(len(nodes)):
                    node = nodes[i]
                    pos = np.array(node['agent_pos'])
                    if len(pos.shape) == 1:
                        pygame.draw.circle(screen, (0, 0, 255), pos, 2)
                        if i != 0:
                            pygame.draw.line(screen, (0, 0, 255),
                                             nodes[i - 1]['agent_pos'], pos)
                    else:
                        for point in pos:
                            pygame.draw.circle(screen, (0, 0, 255), point, 2)

            except KeyError:
                warnings.warn("No nodes found in JSON path")
                return
        self.renderer.one_time_draw(clb)

    def _show_rrt_nodes(self, env):
        while True:
            self.renderer.render(env.map.sim)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

    def replay(self):
        if self.renderer is None:
            self.renderer = DebugRenderer(self.fdata['cfg'])
        self._process_data()
        self._show_path_nodes()
        maker_name = self.fdata['maker_name']
        my_cls = maker_name.split('=')[0]
        method_name = maker_name.split('=')[1]
        map_name = self.fdata['map_name']
        cfg = self.fdata['cfg']
        try:
            maker_factory = globals()[my_cls](
                render_mode='human', map_name=map_name, cfg=cfg, resetable=True)
            maker, maker_name, objects = getattr(maker_factory, method_name)()
        except KeyError:
            raise KeyError(
                f"Maker {maker_name} not found. Might take a look into {self.fdata['script_name']}")

        env = maker()
        if not isinstance(env, BaseEnv):
            env = env.unwrapped

        env.map.sim.attach_renderer(self.renderer)

        if self.pickled_path is not None:
            ret = self._replay_from_pickle(env)
        else:
            warnings.warn(
                "No pickled path found, will play the json path instead")
            ret = self._replay_from_json(env)

        if not ret:
            print("No path found, showing nodes only")
            self._show_rrt_nodes(env)
