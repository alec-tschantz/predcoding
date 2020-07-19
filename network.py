import numpy as np

import mnist_utils
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
        self.condition = cf.condition / (sum(cf.neurons) - cf.neurons[0])

        self.optim = cf.optim
        self.eps = cf.eps
        self.decay_r = cf.decay_r
        self.c_b = [[] for _ in range(self.n_layers)]
        self.c_w = [[] for _ in range(self.n_layers)]

        self.W = None
        self.b = None
        self._init_params()

    def train_epoch(self, img_batches, label_batches):
        for img_batch, label_batch in zip(img_batches, label_batches):
            batch_size = img_batch.shape[1]

            x = [[] for _ in range(self.n_layers)]
            x[0] = img_batch
            for l in range(1, self.n_layers):
                x[l] = self.W[l - 1] @ F.f(x[l - 1], self.act_fn) + np.tile(self.b[l - 1], (1, batch_size))
            x[self.n_layers - 1] = label_batch

            x, errors, _ = self.infer(x, batch_size)
            self.update_params(x, errors, batch_size)

    def test_epoch(self, img_batches, label_batches):
        accs = []
        for img_batch, label_batch in zip(img_batches, label_batches):
            batch_size = img_batch.shape[1]

            x = [[] for _ in range(self.n_layers)]
            x[0] = img_batch
            for l in range(1, self.n_layers):
                x[l] = self.W[l - 1] @ F.f(x[l - 1], self.act_fn) + np.tile(self.b[l - 1], (1, batch_size))
            pred_label = x[-1]

            acc = mnist_utils.mnist_accuracy(pred_label, label_batch)
            accs.append(acc)
        return accs

    def infer(self, x, batch_size):
        errors = [[] for _ in range(self.n_layers)]
        f_x_arr = [[] for _ in range(self.n_layers)]
        f_x_deriv_arr = [[] for _ in range(self.n_layers)]
        f_0 = 0
        its = 0
        beta = self.beta

        for l in range(1, self.n_layers):
            f_x = F.f(x[l - 1], self.act_fn)
            f_x_deriv = F.f_deriv(x[l - 1], self.act_fn)
            errors[l] = (x[l] - self.W[l - 1] @ f_x - np.tile(self.b[l - 1], (1, batch_size))) / self.vars[l]
            f_0 = f_0 - self.vars[l] * np.sum(np.multiply(errors[l], errors[l]), axis=0)
            f_x_arr[l - 1] = f_x
            f_x_deriv_arr[l - 1] = f_x_deriv

        for itr in range(self.itr_max):
            for l in range(1, self.n_layers - 1):
                g = np.multiply(self.W[l].T @ errors[l + 1], f_x_deriv_arr[l])
                x[l] = x[l] + beta * (-errors[l] + g)

            f = 0
            for l in range(1, self.n_layers):
                f_x = F.f(x[l - 1], self.act_fn)
                f_x_deriv = F.f_deriv(x[l - 1], self.act_fn)
                f_x_arr[l - 1] = f_x
                f_x_deriv_arr[l - 1] = f_x_deriv
                errors[l] = (x[l] - self.W[l - 1] @ f_x - self.b[l - 1]) / self.vars[l]
                f = f - self.vars[l] * np.sum(np.multiply(errors[l], errors[l]), axis=0)

            diff = f - f_0
            threshold = self.condition * self.beta / self.vars[self.n_layers - 1]
            if np.any(diff < 0):
                beta = beta / self.div
            elif np.mean(diff) < threshold:
                break

            f_0 = f
            its = itr

        return x, errors, its

    def update_params(self, x, errors, batch_size):
        grad_w = [[] for _ in range(self.n_layers - 1)]
        grad_b = [[] for _ in range(self.n_layers - 1)]

        for l in range(self.n_layers - 1):
            grad_w[l] = (
                self.vars[-1] * (1 / batch_size) * errors[l + 1] @ F.f(x[l], self.act_fn).T
                - self.d_rate * self.W[l]
            )
            grad_b[l] = self.vars[-1] * (1 / batch_size) * np.sum(errors[l + 1], axis=1)

        self._apply_gradients(grad_w, grad_b)

    def _apply_gradients(self, grad_w, grad_b):
        if self.optim is "RMSPROP":
            for l in range(self.n_layers - 1):
                grad_b[l] = np.expand_dims(grad_b[l], axis=1)
                self.c_w[l] = self.decay_r * self.c_w[l] + (1 - self.decay_r) * grad_w[l] ** 2
                self.c_b[l] = self.decay_r * self.c_b[l] + (1 - self.decay_r) * grad_b[l] ** 2

                self.W[l] = self.W[l] + self.l_rate * np.divide(grad_w[l], (np.sqrt(self.c_w[l]) + self.eps))
                self.b[l] = self.b[l] + self.l_rate * np.divide(grad_b[l], (np.sqrt(self.c_b[l]) + self.eps))

        elif self.optim is "SGD" or self.optim is None:
            for l in range(self.n_layers - 1):
                self.W[l] = self.W[l] + self.l_rate * grad_w[l]
                self.b[l] = self.b[l] + self.l_rate * np.expand_dims(grad_b[l], axis=1)

        else:
            raise ValueError(f"{self.optim} not supported")

    def _init_params(self):
        weights = [[] for _ in range(self.n_layers)]
        bias = [[] for _ in range(self.n_layers)]

        for l in range(self.n_layers - 1):
            norm_b = 0
            if self.act_fn is F.LINEAR:
                norm_w = np.sqrt(1 / (self.neurons[l + 1] + self.neurons[l]))
            elif self.act_fn is F.TANH:
                norm_w = np.sqrt(6 / (self.neurons[l + 1] + self.neurons[l]))
            elif self.act_fn is F.LOGSIG:
                norm_w = 4 * np.sqrt(6 / (self.neurons[l + 1] + self.neurons[l]))
            else:
                raise ValueError(f"{self.act_fn} not supported")

            weights[l] = np.random.uniform(-1, 1, size=(self.neurons[l + 1], self.neurons[l])) * norm_w
            bias[l] = np.zeros((self.neurons[l + 1], 1)) + norm_b * np.ones((self.neurons[l + 1], 1))

        self.W = weights
        self.b = bias

        for l in range(self.n_layers - 1):
            self.c_b[l] = np.zeros_like(self.b[l])
            self.c_w[l] = np.zeros_like(self.W[l])

