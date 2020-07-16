import numpy as np
from nnfs.layers import Linear
from nnfs.optimizers import SGD


class Model:
    def __init__(self, layers, loss, optimizer=SGD(lr=0.01)):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def save_weights(self, filename):
        weights = []
        for layer in self.layers:
            for param in layer.get_parameters():
                weights.append(param.value)
        np.savez(filename, *weights)

    def load_weights(self, filename):
        weights = np.load(filename)
        param_index = 0
        for layer in self.layers:
            for param in layer.get_parameters():
                param.value = weights[f'arr_{param_index}']
                param_index += 1

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def train(self, X, y, epochs=20, batch_size=32, validation_data=None, metrics=None, verbose=1):
        history = {'train_loss': [0.0] * epochs}
        if validation_data:
            history['valid_loss'] = [0.0] * epochs
        if metrics:
            for name, _ in metrics.items():
                history[f'train_{name}'] = [0.0] * epochs
                if validation_data:
                    history[f'valid_{name}'] = [0.0] * epochs

        n_batches = (len(X) + batch_size - 1) // batch_size
        for epoch in range(epochs):
            train_loss = 0.0
            for batch_index in range(n_batches):
                batch_start = batch_index * batch_size
                batch_end = min((batch_index + 1) * batch_size, X.shape[0])
                X_batch = X[batch_start:batch_end, ...]
                y_batch = y[batch_start:batch_end, ...]

                y_pred = self.predict(X_batch)
                batch_loss = self.loss(y_pred, y_batch)
                batch_loss += np.sum([layer.get_loss() for layer in self.layers])
                train_loss += batch_loss / n_batches

                parameters = []
                grad_in = self.loss.get_grad_in(y_pred, y_batch)
                for layer in reversed(self.layers):
                    grad_in = layer.backward(grad_in)
                    for param in layer.get_parameters():
                        parameters.append(param)

                self.optimizer.apply_gradients(parameters)

                if metrics:
                    for name, metric in metrics.items():
                        history[f'train_{name}'][epoch] += metric(y_pred, y_batch) / n_batches

            history['train_loss'][epoch] = train_loss

            if validation_data:
                valid_loss = 0.0
                n_valid_batches = (len(validation_data[0]) + batch_size - 1) // batch_size
                for batch_index in range(n_valid_batches):
                    batch_start = batch_index * batch_size
                    batch_end = min((batch_index + 1) * batch_size, validation_data[0].shape[0])
                    X_batch = validation_data[0][batch_start:batch_end, ...]
                    y_batch = validation_data[1][batch_start:batch_end, ...]
                    y_pred = self.predict(X_batch)
                    batch_loss = self.loss(y_pred, y_batch)
                    batch_loss += np.sum([layer.get_loss() for layer in self.layers])
                    valid_loss += batch_loss / n_valid_batches
                    if metrics:
                        for name, metric in metrics.items():
                            history[f'valid_{name}'][epoch] += metric(y_pred, y_batch) / n_valid_batches
                history['valid_loss'][epoch] = valid_loss

            if not verbose:
                continue
            log_str = f"epoch: {epoch+1}/{epochs} - train_loss: {train_loss:.8f}"
            if metrics:
                for name, metric in metrics.items():
                    value = history[f'train_{name}'][epoch]
                    log_str += f" - train_{name}: {value:.8f}"
            if validation_data:
                log_str += f" - valid_loss: {valid_loss:.8f}"
                if metrics:
                    for name, metric in metrics.items():
                        value = history[f'valid_{name}'][epoch]
                        log_str += f" - valid_{name}: {value:.8f}"
            print(log_str)
        return history
