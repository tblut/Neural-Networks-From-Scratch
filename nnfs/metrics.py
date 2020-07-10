import numpy as np


def binary_accuracy(y_pred, y_target, threshold=0.5):
    y_pred = (y_pred > threshold).astype(np.int32)
    return np.mean(y_pred.reshape(-1) == y_target.reshape(-1))
