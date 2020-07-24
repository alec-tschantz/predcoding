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

        self.beta_1 = cf.beta_1
        self.beta_2 = cf.beta_2
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
        self.v_b = [[] for _ in range(self.n_layers)]
        self.v_w = [[] for _ in range(self.n_layers)]

        self.W = None
        self.b = None
        self._init_params()

    def train_epoch(self, x_batches, y_batches, epoch_num=None):
        n_batches = len(x_batches)
        for batch_id, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):

            if batch_id % 500 == 0 and batch_id > 0:
                print(f"batch {batch_id}")

            x_batch = set_tensor(x_batch, self.device)
            y_batch = set_tensor(y_batch, self.device)
            batch_size = x_batch.size(1)

            x = [[] for _ in range(self.n_layers)]
            x[0] = x_batch
            for l in range(1, self.n_layers):
                b = self.b[l - 1].repeat(1, batch_size)
                x[l] = self.W[l - 1] @ F.f(x[l - 1], self.act_fn) + b
            x[self.n_layers - 1] = y_batch

            x, errors, _ = self.infer(x, batch_size)
            self.update_params(x, errors, batch_size, epoch_num=epoch_num, n_batches=n_batches, curr_batch=batch_id)

    def test_epoch(self, x_batches, y_batches):
        accs = []
        for x_batch, y_batch in zip(x_batches, y_batches):
            x_batch = set_tensor(x_batch, self.device)
            y_batch = set_tensor(y_batch, self.device)
            batch_size = x_batch.size(1)

            x = [[] for _ in range(self.n_layers)]
            x[0] = x_batch
            for l in range(1, self.n_layers):
                b = self.b[l - 1].repeat(1, batch_size)
                x[l] = self.W[l - 1] @ F.f(x[l - 1], self.act_fn) + b
            pred_y = x[-1]

            acc = mnist_utils.mnist_accuracy(pred_y, y_batch)
            accs.append(acc)
        return accs

    def generate_data(self, x_batch):
        x_batch = set_tensor(x_batch, self.device)
        batch_size = x_batch.size(1)

        x = [[] for _ in range(self.n_layers)]
        x[0] = x_batch
        for l in range(1, self.n_layers):
            b = self.b[l - 1].repeat(1, batch_size)
            x[l] = self.W[l - 1] @ F.f(x[l - 1], self.act_fn) + b
        pred_y = x[-1]
        return pred_y

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
            f_x_arr[l - 1] = f_x
            f_x_deriv_arr[l - 1] = f_x_deriv

            # eq. 2.17
            b = self.b[l - 1].repeat(1, batch_size)
            errors[l] = (x[l] - self.W[l - 1] @ f_x - b) / self.vars[l]
            f_0 = f_0 - self.vars[l] * torch.sum(torch.mul(errors[l], errors[l]), dim=0)

        for itr in range(self.itr_max):
            # update node activity
            for l in range(1, self.n_layers - 1):
                # eq. 2.18
                g = torch.mul(self.W[l].T @ errors[l + 1], f_x_deriv_arr[l])
                x[l] = x[l] + beta * (-errors[l] + g)

            # update errors
            f = 0
            for l in range(1, self.n_layers):
                f_x = F.f(x[l - 1], self.act_fn)
                f_x_deriv = F.f_deriv(x[l - 1], self.act_fn)
                f_x_arr[l - 1] = f_x
                f_x_deriv_arr[l - 1] = f_x_deriv

                # eq. 2.17
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

    def update_params(self, x, errors, batch_size, epoch_num=None, n_batches=None, curr_batch=None):
        grad_w = [[] for _ in range(self.n_layers - 1)]
        grad_b = [[] for _ in range(self.n_layers - 1)]

        for l in range(self.n_layers - 1):
            # eq. 2.19 (with weight decay)
            grad_w[l] = (
                self.vars[-1] * (1 / batch_size) * errors[l + 1] @ F.f(x[l], self.act_fn).T
                - self.d_rate * self.W[l]
            )
            grad_b[l] = self.vars[-1] * (1 / batch_size) * torch.sum(errors[l + 1], axis=1)

        self._apply_gradients(grad_w, grad_b, epoch_num=epoch_num, n_batches=n_batches, curr_batch=curr_batch)

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
            self.v_b[l] = torch.zeros_like(self.b[l])
            self.v_w[l] = torch.zeros_like(self.W[l])

    def _apply_gradients(self, grad_w, grad_b, epoch_num=None, n_batches=None, curr_batch=None):

        if self.optim is "RMSPROP":
            for l in range(self.n_layers - 1):
                grad_b[l] = grad_b[l].unsqueeze(dim=1)
                self.c_w[l] = self.decay_r * self.c_w[l] + (1 - self.decay_r) * grad_w[l] ** 2
                self.c_b[l] = self.decay_r * self.c_b[l] + (1 - self.decay_r) * grad_b[l] ** 2

                self.W[l] = self.W[l] + self.l_rate * (grad_w[l] / (torch.sqrt(self.c_w[l]) + self.eps))
                self.b[l] = self.b[l] + self.l_rate * (grad_b[l] / (torch.sqrt(self.c_b[l]) + self.eps))

        elif self.optim is "ADAM":
            for l in range(self.n_layers - 1):
                grad_b[l] = grad_b[l].unsqueeze(dim=1)
                self.c_b[l] = self.beta_1 * self.c_b[l] + (1 - self.beta_1) * grad_b[l] 
                self.c_w[l] = self.beta_1 * self.c_w[l] + (1 - self.beta_1) * grad_w[l] 

                self.v_b[l] = self.beta_2 * self.v_b[l] + (1 - self.beta_2) * grad_b[l] ** 2
                self.v_w[l] = self.beta_2 * self.v_w[l] + (1 - self.beta_2) * grad_w[l] ** 2

                t = (epoch_num) * n_batches + curr_batch
                self.W[l] = self.W[l] + self.l_rate * np.sqrt(1 - self.beta_2**t) * self.c_w[l] / (torch.sqrt(self.v_w[l]) + self.eps)
                self.b[l] = self.b[l] + self.l_rate * np.sqrt(1 - self.beta_2**t) * self.c_b[l] / (torch.sqrt(self.v_b[l]) + self.eps)

        elif self.optim is "SGD" or self.optim is None:
            for l in range(self.n_layers - 1):
                self.W[l] = self.W[l] + self.l_rate * grad_w[l]
                self.b[l] = self.b[l] + self.l_rate * grad_b[l].unsqueeze(dim=1)

        else:
            raise ValueError(f"{self.optim} not supported")

