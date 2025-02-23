from .node import Node


class TreeNode(Node):
    """
    Used to represent a node in the tree. 
    """

    def __init__(self):
        super().__init__()
        self._state = None
        self.agent_pos = None
        self.parent = None  # Parent node : TreeNode | None


    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state


class VelTreeNode(TreeNode):
    def __init__(self):
        super().__init__()
        self.velocity_buffer = []
