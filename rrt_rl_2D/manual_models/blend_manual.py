import numpy as np
from .base_model import BaseManualModel


class BlendManualModel(BaseManualModel):
    def __init__(self, segnum):
        self.segnum = segnum
        self.d = 0.15
        self.cnt = 0
        self.cnt2 = 0
        self.last_targets = np.inf * np.ones(2 * self.segnum)
        super().__init__()

    def predict(self, obs, **kwargs):

        obs = obs[0]
        targets = obs[:self.segnum * 2]
        n = np.linalg.norm(targets - self.last_targets)
        if n > 100:
            # print("RESET")
            self.d = 0.15
            self.cnt = 0
            self.cnt2 = 0
        elif n < 5:
            # print("UPDATE ", self.cnt)
            self.cnt += 1
            self.d += 0.01
        elif n > 10:
            # print("Downstep ", self.cnt2)
            self.cnt2 += 1
            self.d -= 0.01
        self.d = np.clip(self.d, 0, 0.5)

        self.last_targets = targets

        obstacle_vecs = obs[self.segnum * 2:].reshape(self.segnum, 2)

        obstacle_norms = np.linalg.norm(obstacle_vecs, axis=1)

        target_weights = 1 / obstacle_norms + self.d
        target_weights = np.clip(target_weights, 0, 1)
        # shift to [-1,1] to fit with BlendEnv setup
        return [2 * target_weights - 1], None
