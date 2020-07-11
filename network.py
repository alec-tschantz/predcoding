import numpy as np

import utils
import functions as F


class PredictiveCodingNetwork(object):
    def __init__(self, cf):
        self.n_layers = cf.n_layers
        self.act_fn = cf.act_fn
        self.neurons = cf.neurons
        self.vars = cf.vars
        self.itr_max = cf.itr_max
        self.batch_size = cf.batch_size
        self.beta = cf.beta
        self.div = cf.div
        self.d_rate = cf.d_rate
        self.l_rate = cf.l_rate
        self.condition = cf.condition / (sum(cf.neurons) - cf.n_input)

        self.W = None
        self.b = None
        self.momentum = None
        self._init_params()

    def propagate(self, x, batch_size):
        for l in range(1, self.n_layers):
            x[l] = self.W[l - 1] @ F.f(x[l - 1], self.act_fn) + np.tile(self.b[l - 1], (1, batch_size))
        return x

    def infer(self, x, batch_size):
        errors = [[] for _ in range(self.n_layers)]
        f_n_arr = [[] for _ in range(self.n_layers)]
        f_p_arr = [[] for _ in range(self.n_layers)]
        f_0 = 0
        its = 0
        beta = self.beta

        for l in range(1, self.n_layers):
            b = np.tile(self.b[l - 1], (1, batch_size))
            f_n, f_p = F.f_b(x[l - 1], self.act_fn, l)
            errors[l] = (x[l] - self.W[l - 1] @ f_n - b) / self.vars[l]
            f_0 = f_0 - self.vars[l] * np.sum(np.multiply(errors[l], errors[l]), axis=0)
            f_n_arr[l - 1] = f_n
            f_p_arr[l - 1] = f_p

        for itr in range(self.itr_max):
            for l in range(1, self.n_layers - 1):
                g = self.W[l].T @ errors[l + 1]
                g = np.multiply(g, f_p[l])
                x[l] = x[l] + self.beta * (-errors[l] + g)

            f = 0
            for l in range(1, self.n_layers):
                f_n, f_p = F.f_b(x[l - 1], self.act_fn, l)
                errors[l] = (x[l] - self.W[l - 1] @ f_n - self.b[l - 1]) / self.vars[l]
                f = f - self.vars[l] * np.sum(np.multiply(errors[l], errors[l]), axis=0)
                f_n_arr[l - 1] = f_n
                f_p_arr[l - 1] = f_p

            diff = f - f_0
            threshold = self.condition * self.beta / self.vars[self.n_layers - 1]
            if np.any(diff < 0):
                beta = beta / self.div
            elif np.mean(diff) < threshold:
                break

            f_0 = f
            its = itr

        return x, errors, its

    def train_epoch(self, imgs, labels):
        img_batches, label_batches, batch_sizes = self._get_batches(imgs, labels, self.batch_size)
        n_batches = len(img_batches)
        print(f"training on {n_batches} batches of size {self.batch_size}")

        for batch in range(n_batches):
            batch_size = batch_sizes[batch]

            x = [[] for _ in range(self.n_layers)]
            x[0] = img_batches[batch]
            x = self.propagate(x, batch_size)
            x[-1] = label_batches[batch]

            x, errors, _ = self.infer(x, batch_size)

            grad_w = [[] for _ in range(self.n_layers - 1)]
            grad_b = [[] for _ in range(self.n_layers - 1)]

            for l in range(self.n_layers - 1):
                grad_w[l] = (
                    self.vars[-1] * (1 / batch_size) * errors[l + 1] @ F.f(x[l], self.act_fn).T
                    - self.d_rate * self.W[l]
                )
                grad_b[l] = self.vars[-1] * (1 / batch_size) * np.sum(errors[l + 1], axis=1)

            for l in range(self.n_layers - 1):
                self.W[l] = self.W[l] + self.l_rate * grad_w[l]
                self.b[l] = self.b[l] + self.l_rate * np.expand_dims(grad_b[l], axis=1)

            if batch % 50 == 0:
                avg_errs = [np.mean(error) for error in errors]
                print(f"batch {batch}/{n_batches}: avg errors {avg_errs}")
                

    def _init_params(self):
        momentum = utils.AttrDict()
        momentum.c_b = [[] for _ in range(self.n_layers)]
        momentum.c_w = [[] for _ in range(self.n_layers)]
        momentum.v_b = [[] for _ in range(self.n_layers)]
        momentum.v_w = [[] for _ in range(self.n_layers)]

        weights = [[] for _ in range(self.n_layers)]
        bias = [[] for _ in range(self.n_layers)]

        for l in range(self.n_layers - 1):
            norm_b = 0
            if self.act_fn == "logsig":
                norm_w = 4 * np.sqrt(6 / (self.neurons[l + 1] + self.neurons[l]))

            weights[l] = np.random.uniform(-1, 1, size=(self.neurons[l + 1], self.neurons[l])) * norm_w
            bias[l] = np.zeros((self.neurons[l + 1], 1)) + norm_b * np.ones((self.neurons[l + 1], 1))

            momentum.c_b[l] = np.zeros_like(bias[l])
            momentum.c_w[l] = np.zeros_like(weights[l])
            momentum.v_b[l] = np.zeros_like(bias[l])
            momentum.v_w[l] = np.zeros_like(weights[l])

        self.W = weights
        self.b = bias
        self.momentum = momentum

    def _get_batches(self, imgs, labels, batch_size):
        n_data = imgs.shape[1]
        n_batches = int(np.ceil(n_data / batch_size))

        img_batches = [[] for _ in range(n_batches)]
        label_batches = [[] for _ in range(n_batches)]
        batch_sizes = [[] for _ in range(n_batches)]

        for batch in range(n_batches):
            if batch == n_batches - 1:
                start = batch * batch_size
                img_batches[batch] = imgs[:, start:]
                label_batches[batch] = labels[:, start:]
                batch_sizes[batch] = int(n_data - batch_size * (n_batches - 1))
            else:
                start = batch * batch_size
                end = (batch + 1) * batch_size
                img_batches[batch] = imgs[:, start:end]
                label_batches[batch] = labels[:, start:end]
                batch_sizes[batch] = batch_size

        return img_batches, label_batches, batch_sizes

