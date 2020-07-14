import numpy as np
from matplotlib import pyplot as plt
from nnfs.model import Model
from nnfs.layers import Linear, Sigmoid, ReLU, Softmax
from nnfs.losses import MSELoss, BinaryCrossEntropyLoss, CrossEntropyLoss
from nnfs.optimizers import SGD
from nnfs.utils import to_one_hot, split_train_test, check_gradients, download_file
from nnfs.initializers import *
from nnfs.metrics import binary_accuracy, multi_class_accuracy
from nnfs.datasets import mnist, xor

import time

if __name__ == "__main__":
    np.random.seed(42)
    
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    y_train_full = to_one_hot(y_train_full)
    y_test = to_one_hot(y_test)

    #n_samples = 4096
    #X, y = xor.generate_data(n_samples)
    #X = np.concatenate([X, (X[:,0] * X[:,1]).reshape((n_samples,1))], axis=1)
    #X = (X[:,0] * X[:,1]).reshape((n_samples,1))
    #X_train_full, X_test, y_train_full, y_test = split_train_test(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = split_train_test(X_train_full, y_train_full, test_size=0.2)
    print(len(X_train), len(y_train), len(X_test), len(y_test))

    layers = [
        Linear(X_train.shape[1], 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, y_train.shape[1]),
        Softmax()
    ]
    model = Model(layers=layers,
                  loss=CrossEntropyLoss(),
                  optimizer=SGD(lr=0.01))
    history = model.train(X_train, y_train,
                          epochs=100, batch_size=480,
                          validation_data=(X_valid, y_valid),
                          metrics={'acc': multi_class_accuracy})
    
    #check_gradients(model, batch_size=7, eps=0.0001)

    print("test_accuracy:", multi_class_accuracy(model.predict(X_test), y_test))

    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['valid_loss'], label='valid_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
