# Neural Networks from Scratch (NNFS)
This is a learning project for understanding how neural networks are trained and backpropagation works. The framework is implemented in Python and NumPy and is inspired by Keras.

## Features
- Backpropagation with mini-batches
- Gradient checking
- Layers: Linear, ReLU, Sigmoid, Softmax
- Glorot and He weight initialization
- L1 and L2 regularization
- Stochastic gradient descent (SGD) optimizer with momentum
- MSE and cross-entropy losses
- Full training loop with validation set and custom metrics
- Built-in MNIST dataset with automatic download and caching
- Saving and loading model weights

## Example
```python
import nnfs

(X_train_full, y_train_full), (X_test, y_test) = nnfs.datasets.mnist.load_data()
y_train_full = nnfs.utils.to_one_hot(y_train_full)
y_test = nnfs.utils.to_one_hot(y_test)
X_train, X_valid, y_train, y_valid = nnfs.utils.split_train_test(
    X_train_full, y_train_full, test_size=0.2)

layers = [
    nnfs.layers.Linear(X_train.shape[1], 16),
    nnfs.layers.ReLU(),
    nnfs.layers.Linear(16, 16),
    nnfs.layers.ReLU(),
    nnfs.layers.Linear(16, y_train.shape[1]),
    nnfs.layers.Softmax()
]
model = nnfs.model.Model(layers=layers,
                         loss=nnfs.losses.CrossEntropyLoss(),
                         optimizer=nnfs.optimizers.SGD(lr=0.01))
model.train(X_train, y_train,
            epochs=100, batch_size=480,
            validation_data=(X_valid, y_valid),
            metrics={'acc': nnfs.metrics.multi_class_accuracy})

y_pred = model.predict(X_test)
print("test_accuracy:", nnfs.metrics.multi_class_accuracy(y_pred, y_test))
```
