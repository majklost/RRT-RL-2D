import numpy as np
from .base_model import BaseManualModel


class BlendManualModel(BaseManualModel):
    def __init__(self, segnum):
        self.segnum = segnum
        super().__init__()

    def predict(self, obs):
        SAFETY_THRESHOLD = 100
        obs = obs[0]
        target_vecs = obs[:self.segnum * 2].reshape(self.segnum, 2)
        obstacle_vecs = obs[self.segnum * 2:].reshape(self.segnum, 2)
        obstacle_norms = np.linalg.norm(obstacle_vecs, axis=1)

        obstacle_weights = 1 / (1 + np.exp(obstacle_norms - SAFETY_THRESHOLD))
        target_weights = 2 * (1 - obstacle_weights) - 1
        return [target_weights], None
