import numpy as np
from nnfs.layers import Linear
from nnfs.utils import accuracy_score


class Model:
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.grad_table = {}

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def train(self, X, y, epochs=20, batch_size=32, validation_data=None):
        n_batches = (len(X) + batch_size - 1) // batch_size
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_index in range(n_batches):
                batch_start = batch_index * batch_size
                batch_end = max((batch_index + 1) * batch_size, X.shape[0])
                X_batch = X[batch_start:batch_end, ...]
                y_batch = y[batch_start:batch_end, ...]

                y_pred = self.predict(X_batch)
                batch_loss = self.loss(y_pred, y_batch)
                total_loss += batch_loss

                grad_in = self.loss.get_grad_in(y_pred, y_batch)
                for layer in reversed(self.layers):
                    if isinstance(layer, Linear):
                        self.grad_table[layer] = grad_in
                    grad_in = layer.backward(grad_in)

                for layer in self.layers:
                    if not isinstance(layer, Linear):
                        continue
                    grad_out = self.grad_table[layer]
                    grad_param = layer.get_grad_param(grad_out)
                    layer.update_parameters(self.optimizer, grad_param)

            log_str = f"epoch: {epoch+1}/{epochs} - loss: {total_loss/n_batches}"
            if validation_data:
                valid_accuracy = 0.0
                n_valid_batches = (len(validation_data[0]) + batch_size - 1) // batch_size
                for batch_index in range(n_valid_batches):
                    batch_start = batch_index * batch_size
                    batch_end = max((batch_index + 1) * batch_size, validation_data[0].shape[0])
                    X_batch = validation_data[0][batch_start:batch_end, ...]
                    y_batch = validation_data[1][batch_start:batch_end, ...]
                    y_pred = (self.predict(X_batch) > 0.5).astype(np.int32)
                    valid_accuracy += accuracy_score(y_pred, y_batch)
                log_str += f" - valid_acc: {valid_accuracy/n_valid_batches}"
            print(log_str)
