# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import matplotlib.pyplot as plt
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


def mnist_accuracy(pred_labels, labels):
    correct = 0
    batch_size = pred_labels.size(1)
    for b in range(batch_size):
        if torch.argmax(pred_labels[:, b]) == torch.argmax(labels[:, b]):
            correct += 1
    return correct / batch_size


def get_batches(imgs, labels, batch_size):
    n_data = imgs.shape[1]
    n_batches = int(np.ceil(n_data / batch_size))

    img_batches = [[] for _ in range(n_batches)]
    label_batches = [[] for _ in range(n_batches)]

    for batch in range(n_batches):
        if batch == n_batches - 1:
            start = batch * batch_size
            img_batches[batch] = imgs[:, start:]
            label_batches[batch] = labels[:, start:]
        else:
            start = batch * batch_size
            end = (batch + 1) * batch_size
            img_batches[batch] = imgs[:, start:end]
            label_batches[batch] = labels[:, start:end]

    return img_batches, label_batches


def plot_imgs(img_batch, save_path):
    img_batch = img_batch.detach().cpu().numpy()
    batch_size = img_batch.shape[1]
    dim = nearest_square(batch_size)

    imgs = [np.reshape(img_batch[:, i], [28, 28]) for i in range(dim ** 2)]
    _, axes = plt.subplots(dim, dim)
    axes = axes.flatten()
    for i, img in enumerate(imgs):
        axes[i].imshow(img)
        axes[i].set_axis_off()
    plt.savefig(save_path)
    plt.close('all') 


def nearest_square(limit):
    answer = 0
    while (answer + 1) ** 2 < limit:
        answer += 1
    return answer

