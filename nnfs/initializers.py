import numpy as np


def zeros(n_inputs, n_outputs):
    return np.zeros((n_inputs, n_outputs))


def ones(n_inputs, n_outputs):
    return np.ones((n_inputs, n_outputs))


def glorot_uniform(n_inputs, n_outputs):
    x = np.sqrt(6.0 / (n_inputs + n_outputs))
    return np.random.uniform(-x, x, (n_inputs, n_outputs))


def glorot_normal(n_inputs, n_outputs):
    std = np.sqrt(2.0 / (n_inputs + n_outputs))
    return np.random.normal(0.0, std, (n_inputs, n_outputs))


def he_uniform(n_inputs, n_outputs):
    x = np.sqrt(6.0 / n_inputs)
    return np.random.uniform(-x, x, (n_inputs, n_outputs))


def he_normal(n_inputs, n_outputs):
    std = np.sqrt(2.0 / n_inputs)
    return np.random.normal(0.0, std, (n_inputs, n_outputs))
