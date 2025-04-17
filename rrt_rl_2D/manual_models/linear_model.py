from .base_model import BaseManualModel


class LinearModel(BaseManualModel):
    def __init__(self, slow_down_coeff=1):
        super().__init__()
        self.slow_down_coeff = slow_down_coeff

    def predict(self, obs, **kwargs):
        return obs / self.slow_down_coeff, None
