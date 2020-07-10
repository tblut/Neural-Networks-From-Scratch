import numpy as np


_EPS = 1.0e-7


class MSELoss:
    def __call__(self, preds, targets):
        return np.mean(np.square(preds - targets))

    def get_grad_in(self, preds, targets):
        return 2.0 * (preds - targets) / preds.shape[0]


class BinaryCrossEntropyLoss:
    def __call__(self, preds, targets):
        return -np.mean(targets * np.log(preds + _EPS) + (1.0 - targets) * np.log(1.0 - preds + _EPS))

    def get_grad_in(self, preds, targets):
        return ((1.0 - targets) / (1.0 - preds + _EPS) - targets / (preds + _EPS)) / preds.shape[0]
