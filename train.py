import numpy as np
from matplotlib import pyplot as plt
from nnfs.model import Model
from nnfs.layers import Linear, Sigmoid, ReLU
from nnfs.losses import MSELoss, BinaryCrossEntropyLoss
from nnfs.optimizers import SGD
from nnfs.utils import gen_xor_data, split_train_test, check_gradients
from nnfs.initializers import he_normal, ones, zeros
from nnfs.metrics import binary_accuracy


if __name__ == "__main__":
    np.random.seed(42)

    n_samples = 4096
    X, y = gen_xor_data(n_samples)
    #X = np.concatenate([X, (X[:,0] * X[:,1]).reshape((n_samples,1))], axis=1)
    #X = (X[:,0] * X[:,1]).reshape((n_samples,1))
    X_train_full, X_test, y_train_full, y_test = split_train_test(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = split_train_test(X_train_full, y_train_full, test_size=0.2)
    print(len(X_train), len(y_train), len(X_test), len(y_test))

    layers = [
        Linear(X.shape[1], 4, weights_inititalizer=he_normal),
        ReLU(),
        Linear(4, 4, weights_inititalizer=he_normal),
        ReLU(),
        Linear(4, 1, weights_inititalizer=he_normal),
        Sigmoid()
    ]
    model = Model(layers=layers,
                  loss=BinaryCrossEntropyLoss(),
                  optimizer=SGD(lr=0.01))
    history = model.train(X_train, y_train,
                          epochs=200, batch_size=32,
                          validation_data=(X_valid, y_valid),
                          metrics={'acc': binary_accuracy},
                          verbose=True)
    
    #check_gradients(model, batch_size=8, eps=0.0001)

    y_pred = model.predict(X_test)
    print("test_acc:", binary_accuracy(y_pred, y_test.astype(np.int32)))

    #plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['valid_loss'], label='valid_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
