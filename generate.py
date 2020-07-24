# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import torch

import mnist_utils
import functions as F
from network import PredictiveCodingNetwork


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def main(cf):
    print(f"device [{cf.device}]")
    print("loading MNIST data...")
    train_set = mnist_utils.get_mnist_train_set()
    test_set = mnist_utils.get_mnist_test_set()

    img_train = mnist_utils.get_imgs(train_set)
    img_test = mnist_utils.get_imgs(test_set)
    label_train = mnist_utils.get_labels(train_set)
    label_test = mnist_utils.get_labels(test_set)

    if cf.data_size is not None:
        test_size = cf.data_size // 5
        img_train = img_train[:, 0 : cf.data_size]
        label_train = label_train[:, 0 : cf.data_size]
        img_test = img_test[:, 0:test_size]
        label_test = label_test[:, 0:test_size]

    msg = "img_train {} img_test {} label_train {} label_test {}"
    print(msg.format(img_train.shape, img_test.shape, label_train.shape, label_test.shape))

    print("performing preprocessing...")
    if cf.apply_scaling:
        img_train = mnist_utils.scale_imgs(img_train, cf.img_scale)
        img_test = mnist_utils.scale_imgs(img_test, cf.img_scale)
        label_train = mnist_utils.scale_labels(label_train, cf.label_scale)
        label_test = mnist_utils.scale_labels(label_test, cf.label_scale)

    if cf.apply_inv:
        img_train = F.f_inv(img_train, cf.act_fn)
        img_test = F.f_inv(img_test, cf.act_fn)

    model = PredictiveCodingNetwork(cf)

    with torch.no_grad():
        for epoch in range(cf.n_epochs):
            print(f"\nepoch {epoch}")

            img_batches, label_batches = mnist_utils.get_batches(img_train, label_train, cf.batch_size)
            print(f"training on {len(img_batches)} batches of size {cf.batch_size}")
            model.train_epoch(label_batches, img_batches, epoch_num=epoch)

            img_batches, label_batches = mnist_utils.get_batches(img_test, label_test, cf.batch_size)
            print("generating images...")
            pred_imgs = model.generate_data(label_batches[0])
            mnist_utils.plot_imgs(pred_imgs, cf.img_path.format(epoch))

            perm = np.random.permutation(img_train.shape[1])
            img_train = img_train[:, perm]
            label_train = label_train[:, perm]


if __name__ == "__main__":
    cf = AttrDict()

    cf.img_path = "imgs/{}.png"

    cf.n_epochs = 100
    cf.data_size = None
    cf.batch_size = 128

    cf.apply_inv = True
    cf.apply_scaling = True
    cf.label_scale = 0.94
    cf.img_scale = 1.0

    cf.neurons = [10, 500, 500, 784]
    cf.n_layers = len(cf.neurons)
    cf.act_fn = F.TANH
    cf.var_out = 1
    cf.vars = torch.ones(cf.n_layers)

    cf.itr_max = 50
    cf.beta = 0.1
    cf.div = 2
    cf.condition = 1e-6
    cf.d_rate = 0

    # optim parameters
    cf.l_rate = 1e-3
    cf.optim = "ADAM"
    cf.eps = 1e-8
    cf.decay_r = 0.9
    cf.beta_1 = 0.9
    cf.beta_2 = 0.999

    cf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(cf)

