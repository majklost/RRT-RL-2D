from pathlib import Path
import json
import warnings
import numpy as np
import pygame
import datetime
import pickle

from ..nodes import VelTreeNode


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


class VelPathSaver:
    def __init__(self, maker_name: str, path, cfg: dict, map_name: str, data: dict, script_name: Path):

        self.maker_name = maker_name
        self.path = path
        self.data = data
        self.cfg = cfg
        self.map_name = map_name
        self.creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.script_name = script_name
        self.nodes = self._extract_nodes()

    def save(self, filepath: Path, fname: str):
        filepath = Path(filepath)
        if len(self.nodes) == 0:
            warnings.warn("No nodes to save")
            return
        with open(filepath / f"{fname}.rpath", 'wb') as f:
            pickle.dump(self.path, f)
            s = json.dumps(self._create_structure(),
                           default=default).encode('utf-8')
            f.write(b'\n' + s)

    def _create_structure(self):
        structure = {}
        structure['nodes'] = self.nodes
        structure['creation_date'] = self.creation_date
        structure['script_name'] = self.script_name
        structure['map_name'] = self.map_name
        structure['maker_name'] = self.maker_name
        structure['cfg'] = self.cfg
        structure['data'] = self.data
        return structure

    def _extract_nodes(self):
        nodes = []
        for i in range(0, len(self.path.nodes)):
            node = self.path.nodes[i]
            assert isinstance(node, VelTreeNode), "Node is not VelTreeNode"
            cur_node = {}
            cur_node['agent_pos'] = node.agent_pos
            cur_node['agent_rot'] = node.agent_rot
            cur_node['velocity'] = node.velocity_buffer
            nodes.append(cur_node.copy())

        return nodes
