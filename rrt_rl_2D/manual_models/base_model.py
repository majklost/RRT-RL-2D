import numpy as np

from typing import Any, Tuple


class BaseManualModel:
    def __init__(self):
        pass

    def predict(self, obs) -> 'Tuple[np.ndarray, Any]':
        raise NotImplementedError("Predict method must be implemented")
