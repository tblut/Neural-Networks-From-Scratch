import numpy as np
from nnfs.initializers import zeros, glorot_normal


class Linear:
    def __init__(self, n_inputs, n_neurons,
                 weights_inititalizer=glorot_normal,
                 bias_initializer=zeros):
        self.weights = weights_inititalizer(n_inputs, n_neurons)
        self.biases = zeros(1, n_neurons)

    def forward(self, inputs):
        self._inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, grad_out):
        return np.dot(grad_out, self.weights.T)

    def get_grad_param(self, grad_out):
        grad_weights = np.dot(self._inputs.T, grad_out)
        grad_biases = np.sum(grad_out, axis=0)
        return [grad_weights, grad_biases]

    def update_parameters(self, optimizer, grad_param):
        self.weights = optimizer(self.weights, grad_param[0])
        self.biases = optimizer(self.biases, grad_param[1])


class Sigmoid:
    def forward(self, inputs):
        self._inputs = inputs
        return 1.0 / (1.0 + np.exp(-inputs))

    def backward(self, grad_out):
        sig = 1.0 / (1.0 + np.exp(-self._inputs))
        return grad_out * sig * (1.0 - sig)


class ReLU:
    def forward(self, inputs):
        self._inputs = inputs
        return np.maximum(0.0, inputs)

    def backward(self, grad_out):
        grad_in = np.where(self._inputs > 0, 1.0, 0.0)
        return grad_out * grad_in
