# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import torch

import mnist_utils
import functions as F


def set_tensor(arr, device):
    return torch.from_numpy(arr).float().to(device)


class PredictiveCodingNetwork(object):
    def __init__(self, cf):
        self.device = cf.device
        self.n_layers = cf.n_layers
        self.act_fn = cf.act_fn
        self.neurons = cf.neurons
        self.vars = cf.vars.float().to(self.device)
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
        for batch_id, (img_batch, label_batch) in enumerate(zip(img_batches, label_batches)):
            
            if batch_id % 500 == 0 and batch_id > 0:
                print(f"batch {batch_id}")

            img_batch = set_tensor(img_batch, self.device)
            label_batch = set_tensor(label_batch, self.device)
            batch_size = img_batch.size(1)

            x = [[] for _ in range(self.n_layers)]
            x[0] = img_batch
            for l in range(1, self.n_layers):
                b = self.b[l - 1].repeat(1, batch_size)
                x[l] = self.W[l - 1] @ F.f(x[l - 1], self.act_fn) + b
            x[self.n_layers - 1] = label_batch

            x, errors, _ = self.infer(x, batch_size)
            self.update_params(x, errors, batch_size)

    def test_epoch(self, img_batches, label_batches):
        accs = []
        for img_batch, label_batch in zip(img_batches, label_batches):
            img_batch = set_tensor(img_batch, self.device)
            label_batch = set_tensor(label_batch, self.device)
            batch_size = img_batch.size(1)

            x = [[] for _ in range(self.n_layers)]
            x[0] = img_batch
            for l in range(1, self.n_layers):
                b = self.b[l - 1].repeat(1, batch_size)
                x[l] = self.W[l - 1] @ F.f(x[l - 1], self.act_fn) + b
            pred_label = x[-1]

            acc = mnist_utils.mnist_accuracy(pred_label, label_batch)
            accs.append(acc)
        return accs

    def infer(self, x, batch_size):
        errors = [[] for _ in range(self.n_layers)]
        f_xs = [[] for _ in range(self.n_layers)]
        f_x_derivs = [[] for _ in range(self.n_layers)]
        f_0 = 0
        its = 0
        beta = self.beta

        for l in range(1, self.n_layers):
            f_x = F.f(x[l - 1], self.act_fn)
            f_x_deriv = F.f_deriv(x[l - 1], self.act_fn)

            b = self.b[l - 1].repeat(1, batch_size)
            errors[l] = (x[l] - self.W[l - 1] @ f_x - b) / self.vars[l]

            f_0 = f_0 - self.vars[l] * torch.sum(torch.mul(errors[l], errors[l]), dim=0)
            f_xs[l - 1] = f_x
            f_x_derivs[l - 1] = f_x_deriv

        for itr in range(self.itr_max):
            for l in range(1, self.n_layers - 1):
                g = torch.mul(self.W[l].T @ errors[l + 1], f_x_derivs[l])
                x[l] = x[l] + beta * (-errors[l] + g)

            f = 0
            for l in range(1, self.n_layers):
                f_x = F.f(x[l - 1], self.act_fn)
                f_x_deriv = F.f_deriv(x[l - 1], self.act_fn)
                f_xs[l - 1] = f_x
                f_x_derivs[l - 1] = f_x_deriv

                errors[l] = (x[l] - self.W[l - 1] @ f_x - self.b[l - 1]) / self.vars[l]
                f = f - self.vars[l] * torch.sum(torch.mul(errors[l], errors[l]), dim=0)

            diff = f - f_0
            threshold = self.condition * self.beta / self.vars[self.n_layers - 1]
            if torch.any(diff < 0):
                beta = beta / self.div
            elif torch.mean(diff) < threshold:
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
            grad_b[l] = self.vars[-1] * (1 / batch_size) * torch.sum(errors[l + 1], axis=1)

        self._apply_gradients(grad_w, grad_b)

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

            layer_w = np.random.uniform(-1, 1, size=(self.neurons[l + 1], self.neurons[l])) * norm_w
            layer_b = np.zeros((self.neurons[l + 1], 1)) + norm_b * np.ones((self.neurons[l + 1], 1))
            weights[l] = set_tensor(layer_w, self.device)
            bias[l] = set_tensor(layer_b, self.device)

        self.W = weights
        self.b = bias

        for l in range(self.n_layers - 1):
            self.c_b[l] = torch.zeros_like(self.b[l])
            self.c_w[l] = torch.zeros_like(self.W[l])

    def _apply_gradients(self, grad_w, grad_b):

        if self.optim is "RMSPROP":
            for l in range(self.n_layers - 1):
                grad_b[l] = grad_b[l].unsqueeze(dim=1)
                self.c_w[l] = self.decay_r * self.c_w[l] + (1 - self.decay_r) * grad_w[l] ** 2
                self.c_b[l] = self.decay_r * self.c_b[l] + (1 - self.decay_r) * grad_b[l] ** 2

                self.W[l] = self.W[l] + self.l_rate * (grad_w[l] / (torch.sqrt(self.c_w[l]) + self.eps))
                self.b[l] = self.b[l] + self.l_rate * (grad_b[l] / (torch.sqrt(self.c_b[l]) + self.eps))

        elif self.optim is "SGD" or self.optim is None:
            for l in range(self.n_layers - 1):
                self.W[l] = self.W[l] + self.l_rate * grad_w[l]
                self.b[l] = self.b[l] + self.l_rate * grad_b[l].unsqueeze(dim=1)

        else:
            raise ValueError(f"{self.optim} not supported")

