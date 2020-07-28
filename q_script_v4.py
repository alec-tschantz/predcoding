# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import torch

import mnist_utils
import functions as F
from q_network_v3 import QCodingNetwork

"""
Precision term which weights influence - in theory a learnable parameter?
"""

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

    model = QCodingNetwork(cf)

    q_accs = []
    h_accs = []
    p_accs = [] 
    init_errs = []
    end_errs = []

    with torch.no_grad():
        for epoch in range(cf.n_epochs):
            print(f"\nepoch {epoch}")

            img_batches, label_batches = mnist_utils.get_batches(img_train, label_train, cf.batch_size)
            print(f"> training on {len(img_batches)} batches of size {cf.batch_size}")
            end_err, init_err, its = model.train_epoch(img_batches, label_batches, epoch_num=epoch)
            print("end_err {} / init_err {} / its {}".format(end_err, init_err, its))
            init_errs.append(init_err)
            end_errs.append(end_err)

            if epoch % cf.test_every == 0:
                img_batches, label_batches = mnist_utils.get_batches(img_test, label_test, cf.batch_size)
                print("> generating images...")
                pred_imgs = model.generate_data(label_batches[0])
                mnist_utils.plot_imgs(pred_imgs, cf.img_path.format(epoch))

                img_batches, label_batches = mnist_utils.get_batches(img_test, label_test, cf.batch_size)
                print(f"> testing hybrid acc on {len(img_batches)} batches of size {cf.batch_size}")
                accs, its = model.test_epoch(img_batches, label_batches, itr_max=cf.test_itr_max)
                mean_h_acc = np.mean(np.array(accs))
                h_accs.append(mean_h_acc)
                print(f"average hybrid accuracy {mean_h_acc} / its {its}")
                
                img_batches, label_batches = mnist_utils.get_batches(img_test, label_test, cf.batch_size)
                print(f"> testing amortised acc {len(img_batches)} batches of size {cf.batch_size}")
                accs = model.test_amortised_epoch(img_batches, label_batches)
                mean_q_acc = np.mean(np.array(accs))
                q_accs.append(mean_q_acc)
                print(f"average amortised accuracy {mean_q_acc}")

                img_batches, label_batches = mnist_utils.get_batches(img_test, label_test, cf.batch_size)
                print(f"> testing PC acc on {len(img_batches)} batches of size {cf.batch_size}")
                accs, its = model.test_pc_epoch(img_batches, label_batches, itr_max=cf.test_itr_max)
                mean_p_acc = np.mean(np.array(accs))
                p_accs.append(mean_p_acc)
                print(f"average PC accuracy {mean_p_acc} / its {its}")

                np.save(cf.hybird_path, h_accs)
                np.save(cf.amortised_path, q_accs)
                np.save(cf.pc_path, p_accs)
                np.save(cf.init_errs_path, init_errs)
                np.save(cf.end_errs_path, end_errs)

                perm = np.random.permutation(img_train.shape[1])
                img_train = img_train[:, perm]
                label_train = label_train[:, perm]




if __name__ == "__main__":
    cf = AttrDict() 

    cf.img_path = "imgs/epoch_{}.png"
    cf.hybird_path = "data/h_accs_6"
    cf.amortised_path = "data/q_accs_6"
    cf.pc_path = "data/pc_accs_6"
    cf.init_errs_path = "data/init_errs_6"
    cf.end_errs_path = "data/end_errs_6"
    cf.test_every = 1

    cf.n_epochs = 100
    cf.data_size = None
    cf.batch_size = 128

    cf.apply_inv = True
    cf.apply_scaling = True
    cf.label_scale = 0.94
    cf.img_scale = 1.0

    cf.td_neurons = [10, 500, 500, 784]
    cf.bu_neurons = [784, 500, 500, 10]
    cf.n_layers = len(cf.td_neurons)
    cf.act_fn = F.TANH
    cf.var_out = 1
    cf.vars = torch.ones(cf.n_layers)

    # TODO 
    cf.itr_max = 1000
    cf.test_itr_max = 1000
    # TODO change stuff here
    cf.amortised_prec = 0.1
    cf.generative_prec = 0.1
    cf.beta = 0.1
    cf.div = 2
    # TODO TODO
    cf.condition = 1e-6
    cf.d_rate = 0

    # optim parameters
    cf.l_rate = 1e-5
    # TODO q_l_rate low?
    cf.q_l_rate = 1e-5
    # TODO
    cf.optim = "ADAM"
    cf.eps = 1e-8
    cf.decay_r = 0.9
    cf.beta_1 = 0.9
    cf.beta_2 = 0.999

    cf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(cf)

