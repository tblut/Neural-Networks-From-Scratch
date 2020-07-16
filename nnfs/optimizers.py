import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def apply_gradients(self, parameters):
        if self.momentum > 0.0:
            if not self.v:
                self.v = [np.zeros(p.shape) for p in parameters]
            for index, param in enumerate(parameters):
                self.v[index] = self.momentum * self.v[index] - self.lr * param.grad
                param.value += self.v[index]
        else:
            for param in parameters:
                param.value -= self.lr * param.grad
