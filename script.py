# pylint: disable=unsubscriptable-object

import numpy as np

import mnist_utils
import utils
import functions as F
from network import PredictiveCodingNetwork


def main(cf):
    print("loading MNIST data...")
    train_set = mnist_utils.get_mnist_train_set()
    test_set = mnist_utils.get_mnist_test_set()

    img_train = mnist_utils.get_imgs(train_set)
    img_test = mnist_utils.get_imgs(test_set)

    label_train = mnist_utils.get_labels(train_set)
    label_test = mnist_utils.get_labels(test_set)

    if cf.datasize is not None:
        img_train = img_train[:, 0:cf.datasize]
        img_test = img_test[:, 0:cf.datasize]
        label_train = label_train[:, 0:cf.datasize]
        label_test = label_test[:, 0:cf.datasize]

    msg = "img_train {} img_test {} label_train {} label_test {}"
    print(msg.format(img_train.shape, img_test.shape, label_train.shape, label_test.shape))

    print("performing preprocessing...")
    img_train = mnist_utils.scale_imgs(img_train, cf.img_scale)
    img_test = mnist_utils.scale_imgs(img_test, cf.img_scale)

    label_train = mnist_utils.scale_labels(label_train, cf.label_scale)
    label_test = mnist_utils.scale_labels(label_test, cf.label_scale)

    img_train = F.f_inv(img_train, cf.act_fn)
    img_test = F.f_inv(img_test, cf.act_fn)

    cf.n_input = img_train.shape[0]
    model = PredictiveCodingNetwork(cf)

    print("starting training...")
    for epoch in range(cf.n_epochs):
        print(f"epoch: {epoch}")
        model.train_epoch(img_train, label_train)
        model.test(img_test, label_test)


if __name__ == "__main__":
    """ check elementwise squaring """

    cf = utils.AttrDict()

    cf.n_epochs = 100
    cf.batch_size = 20
    cf.datasize = 1000

    cf.neurons = [784, 500, 500, 10]
    cf.n_layers = len(cf.neurons)
    # cf.act_fn = "logsig"
    cf.act_fn = "tanh"

    cf.img_scale = 1.0
    cf.label_scale = 0.94

    cf.var_out = 1
    cf.vars = np.ones(cf.n_layers)

    cf.itr_max = 50
    cf.beta = 0.1
    cf.div = 2
    cf.d_rate = 0 * 0.001
    cf.l_rate = 1e-3
    cf.condition = 1e-6

    main(cf)

