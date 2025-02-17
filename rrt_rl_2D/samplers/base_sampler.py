import numpy as np


class BaseSampler:
    def __init__(self):
        pass

    def sample(self, *args) -> np.ndarray:
        raise NotImplementedError

    def analytics(self):
        return None
