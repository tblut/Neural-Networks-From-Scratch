import numpy as np
import math
import tarfile
import gzip
import shutil
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen
from urllib.parse import urlparse
from nnfs.layers import Linear


def to_one_hot(labels, n_classes=None):
    return np.identity(n_classes or labels.max() + 1)[labels.reshape(-1)]


def download_file(url, target_dir, extract=None, verbose=1):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    target_path = Path(target_dir, filename)
    if not target_path.exists():
        print(f"Downloading {url} ...")
        with urlopen(url) as response:
            with open(target_path, 'wb') as file:
                file.write(response.read())

    if extract:
        print(f"Extracting {target_path} ...")
        target_path = Path(target_path)
        extract_path = Path(target_dir, target_path.stem)
        if target_path.suffix == '.zip':
            with ZipFile(target_path, 'r') as zip_file:
                zip_file.extractall(extract_path)
        elif '.tar' in target_path.suffixes:
            with tarfile.open(target_path) as tar:
                tar.extractall(extract_path)
        elif target_path.suffix == '.gz':
            with gzip.open(target_path, 'rb') as file_in:
                with open(extract_path, 'wb') as file_out:
                    shutil.copyfileobj(file_in, file_out)
        
    return str(extract_path) if extract else target_path


def split_train_test(*arrays, test_size=None, train_size=None, shuffle=True, random_seed=None):
    if not test_size and not train_size:
        test_size = 0.2
    if not test_size:
        test_size = 1.0 - train_size if isinstance(train_size, float) else len(arrays[0]) - train_size
    if not train_size:
        train_size = 1.0 - test_size if isinstance(test_size, float) else len(arrays[0]) - test_size
    if isinstance(test_size, float):
        test_size = int(test_size * len(arrays[0]))
    if isinstance(train_size, float):
        train_size = int(train_size * len(arrays[0]))
    if test_size < 0 or train_size < 0 or test_size + train_size > len(arrays[0]):
        raise ValueError("test_size + train_size must be less or equal than 1.0 (or the length of the arrays) and can't be negative")
    
    if shuffle:
        shuffle_indices = np.random.permutation(len(arrays[0]))
        if random_seed:
            np.random.seed(random_seed)
    splits = []
    for array in arrays:
        if len(array) != len(arrays[0]):
            raise ValueError("All arrays must be the same length")
        if shuffle:
            array = array[shuffle_indices]
        splits.append(array[:train_size])
        splits.append(array[train_size:train_size+test_size])
    return splits


def compute_finite_grad(X, y, model, layer, eps=0.001):
    if not isinstance(layer, Linear):
        raise TypeError("layer must be of type Linear")

    grad_weights = np.zeros((layer.weights.size,))
    param_index = 0
    weights = layer.weights
    for i in range(weights.size):
        layer.weights = layer.weights.flatten()
        layer.weights[i] += eps
        layer.weights = layer.weights.reshape(weights.shape)
        output_pos = model.loss(model.predict(X), y)
        layer.weights = weights

        layer.weights = layer.weights.flatten()
        layer.weights[i] -= eps
        layer.weights = layer.weights.reshape(weights.shape)
        output_neg = model.loss(model.predict(X), y)
        layer.weights = weights

        grad_weights[param_index] = (output_pos - output_neg) / (2.0 * eps)
        param_index += 1
    grad_weights = grad_weights.reshape(layer.weights.shape)

    grad_biases = np.zeros((layer.biases.size,))
    param_index = 0
    biases = layer.biases
    for i in range(biases.size):
        layer.biases = layer.biases.flatten()
        layer.biases[i] += eps
        layer.biases = layer.biases.reshape(biases.shape)
        output_pos = model.loss(model.predict(X), y)
        layer.biases = biases

        layer.biases = layer.biases.flatten()
        layer.biases[i] -= eps
        layer.biases = layer.biases.reshape(biases.shape)
        output_neg = model.loss(model.predict(X), y)
        layer.biases = biases

        grad_biases[param_index] = (output_pos - output_neg) / (2.0 * eps)
        param_index += 1
    grad_biases = grad_biases.reshape(layer.biases.shape)
    return grad_weights, grad_biases


def check_gradients(model, batch_size=4, eps=0.0001):
    n_inputs = model.layers[0].weights.shape[0]
    n_outputs = model.layers[-2].weights.shape[1]
    X = np.random.rand(batch_size, n_inputs)
    y = np.identity(n_outputs)[np.random.random_integers(0, n_outputs-1, batch_size)]

    n_params = 0
    for layer in model.layers:
        if isinstance(layer, Linear):
            n_params += layer.weights.size + layer.biases.size
    grad_finite = np.zeros((n_params,))

    param_index = 0
    for layer in model.layers:
        if not isinstance(layer, Linear):
            continue
        grad_weights, grad_biases = compute_finite_grad(X, y, model, layer, eps=eps)
        for grad in grad_weights.flatten():
            grad_finite[param_index] = grad
            param_index += 1
        for grad in grad_biases.flatten():
            grad_finite[param_index] = grad
            param_index += 1

    param_index = 0
    grad_bprop = np.zeros((n_params,))
    grad_in = model.loss.get_grad_in(model.predict(X), y)
    for layer in reversed(model.layers):
        if isinstance(layer, Linear):
            model.grad_table[layer] = grad_in
        grad_in = layer.backward(grad_in)

    for layer in model.layers:
        if not isinstance(layer, Linear):
            continue
        grad_out = model.grad_table[layer]
        grad_param = layer.get_grad_param(grad_out)
        for grad in grad_param:
            for param in grad.flatten():
                grad_bprop[param_index] = param
                param_index += 1

    print("grad_finite:", grad_finite)
    print("grad_bprop:", grad_bprop)
    print("grad_error:", np.linalg.norm(grad_finite - grad_bprop) / (np.linalg.norm(grad_finite) + np.linalg.norm(grad_bprop)))
