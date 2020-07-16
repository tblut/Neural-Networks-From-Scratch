import numpy as np


class L1:
    def __init__(self, l=0.01):
        self.l = l

    def __call__(self, params):
        return self.l * np.sum(np.abs(params).reshape(-1))

    def get_grad(self, params):
        return self.l * np.sign(params)


class L2:
    def __init__(self, l=0.01):
        self.l = l

    def __call__(self, params):
        return self.l * np.sum(np.square(params).reshape(-1))

    def get_grad(self, params):
        return 2.0 * self.l * params
