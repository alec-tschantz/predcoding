# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import torch
import torchvision


def get_mnist_train_set():
    return torchvision.datasets.MNIST("MNIST_train", download=True, train=True)


def get_mnist_test_set():
    return torchvision.datasets.MNIST("MNIST_test", download=True, train=False)


def onehot(label, n_classes=10):
    arr = np.zeros([10])
    arr[int(label)] = 1.0
    return arr


def img_to_np(img):
    return np.array(img).reshape([784]) / 255.0


def get_imgs(dataset):
    imgs = np.array([img_to_np(dataset[i][0]) for i in range(len(dataset))])
    return np.swapaxes(imgs, 0, 1)


def get_labels(dataset):
    labels = np.array([onehot(dataset[i][1]) for i in range(len(dataset))])
    return np.swapaxes(labels, 0, 1)


def scale_imgs(imgs, scale_factor):
    return imgs * scale_factor + 0.5 * (1 - scale_factor) * np.ones(imgs.shape)


def scale_labels(labels, scale_factor):
    return labels * scale_factor + 0.5 * (1 - scale_factor) * np.ones(labels.shape)
