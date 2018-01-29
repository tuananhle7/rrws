# Adapted from
# https://github.com/tensorflow/models/tree/master/research/rebar and
# https://github.com/duvenaud/relax/blob/master/datasets.py

import logging
import numpy as np
import os
import scipy.io
import urllib.request

BINARIZED_MNIST_URL = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist'
BINARIZED_MNIST_DATA_DIR = '/Users/tuananhle/Documents/research/datasets/binarized-mnist'
OMNIGLOT_URL = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'
OMNIGLOT_DATA_FILE = '/Users/tuananhle/Documents/research/datasets/omniglot/omniglot.mat'


def download_binarized_mnist(
    data_dir=BINARIZED_MNIST_DATA_DIR,
    binarized_mnist_url=BINARIZED_MNIST_URL,
    splits=['train', 'valid', 'test']
):
    """Downloads the binarized MNIST dataset and saves to .npy files."""
    for split in splits:
        filename = 'binarized_mnist_{}.amat'.format(split)
        url = '{}/binarized_mnist_{}.amat'.format(binarized_mnist_url, split)
        local_filename = os.path.join(data_dir, filename)
        if not os.path.exists(local_filename):
            urllib.request.urlretrieve(url, local_filename)
            logging.info('Downloaded {} to {}'.format(url, local_filename))

        npy_filename = 'binarized_mnist_{}.npy'.format(split)
        local_npy_filename = os.path.join(data_dir, npy_filename)
        if not os.path.exists(local_npy_filename):
            with open(local_filename, 'rb') as f:
                np.save(
                    local_npy_filename,
                    np.array([list(map(int, line.split())) for line in f.readlines()], dtype='uint8')
                )
                logging.info('Saved to {}'.format(local_npy_filename))


def load_binarized_mnist(
    data_dir=BINARIZED_MNIST_DATA_DIR,
    binarized_mnist_url=BINARIZED_MNIST_URL,
    splits=['train', 'valid', 'test']
):
    binarized_mnist = []
    for split in splits:
        npy_filename = 'binarized_mnist_{}.npy'.format(split)
        local_npy_filename = os.path.join(data_dir, npy_filename)
        if not os.path.exists(local_npy_filename):
            download_binarized_mnist(data_dir, binarized_mnist_url, [split])

        binarized_mnist.append(np.load(local_npy_filename))
        logging.info('Loaded {}'.format(local_npy_filename))

    return tuple(binarized_mnist)


def download_omniglot(
    data_file=OMNIGLOT_DATA_FILE,
    omniglot_url=OMNIGLOT_URL
):
    if not os.path.exists(data_file):
        urllib.request.urlretrieve(omniglot_url, data_file)
        logging.info('Downloaded {} to {}'.format(omniglot_url, data_file))


def load_binarized_omniglot(
    data_file=OMNIGLOT_DATA_FILE,
    omniglot_url=OMNIGLOT_URL
):
    download_omniglot(data_file, omniglot_url)
    n_validation = 1345

    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

    omni_raw = scipy.io.loadmat(data_file)
    logging.info('Loaded {}'.format(data_file))

    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # Binarize the data with a fixed seed
    np.random.seed(5)
    train_data = (np.random.rand(*train_data.shape) < train_data).astype(float)
    test_data = (np.random.rand(*test_data.shape) < test_data).astype(float)

    shuffle_seed = 123
    permutation = np.random.RandomState(seed=shuffle_seed).permutation(train_data.shape[0])
    train_data = train_data[permutation]

    x_train = train_data[:-n_validation]
    x_valid = train_data[-n_validation:]
    x_test = test_data

    return x_train, x_valid, x_test
