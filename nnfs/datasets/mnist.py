import numpy as np
from pathlib import Path
from nnfs.utils import download_file


def _read_images_file(path):
    with open(path, mode='rb') as file:
        data = file.read()

    n_images = int.from_bytes(data[4:8], byteorder='big', signed=True)
    width = int.from_bytes(data[8:12], byteorder='big', signed=True)
    height = int.from_bytes(data[12:16], byteorder='big', signed=True)
    images = np.empty((n_images, width * height), dtype=np.float32)

    image_size = width * height
    for i in range(n_images):
        start = 16 + i * image_size
        end = start + image_size
        images[i, :] = np.frombuffer(data[start:end], dtype=np.uint8) / 255.0

    return images


def _read_labels_file(path):
    with open(path, mode='rb') as file:
        data = file.read()

    n_items = int.from_bytes(data[4:8], byteorder='big', signed=True)
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    labels = labels.astype(np.int32).reshape((n_items, 1))
    return labels

def load_data(cache_dir='.cache'):
    cache_file_path = Path(cache_dir, 'mnist.npz')
    if not cache_file_path.exists():
        url_train_images = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        url_train_labels = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        url_test_images = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
        url_test_labels = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

        train_images_path = download_file(url_train_images, cache_dir, extract=True)
        train_labels_path = download_file(url_train_labels, cache_dir, extract=True)
        test_images_path = download_file(url_test_images, cache_dir, extract=True)
        test_labels_path = download_file(url_test_labels, cache_dir, extract=True)

        train_images = _read_images_file(train_images_path)
        train_labels = _read_labels_file(train_labels_path)
        test_images = _read_images_file(test_images_path)
        test_labels = _read_labels_file(test_labels_path)

        np.savez(cache_file_path, train_images, train_labels, test_images, test_labels)
    
    data = np.load(cache_file_path)
    return (data['train_images'], data['train_labels']), (data['test_images'], data['test_labels'])
