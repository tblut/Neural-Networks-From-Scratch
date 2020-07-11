import numpy as np


def generate_data(samples):
    X = np.random.uniform(-1.0, 1.0, (samples, 2)).astype(np.float32)
    y = [int(x[0] > 0.0) ^ int(x[1] > 0.0) for x in X]
    y = np.array(y, dtype=np.int32).reshape((len(y), 1))
    return X, y
