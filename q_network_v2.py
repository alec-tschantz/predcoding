# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import torch

import mnist_utils
import functions as F


def set_tensor(arr, device):
    return torch.from_numpy(arr).float().to(device)


class QCodingNetwork(object):
    def __init__(self, cf):
        self.device = cf.device
        self.n_layers = cf.n_layers
        self.act_fn = cf.act_fn
        self.td_neurons = cf.td_neurons
        self.bu_neurons = cf.bu_neurons
        self.vars = cf.vars.float().to(self.device)
        self.itr_max = cf.itr_max
        self.batch_size = cf.batch_size

        self.beta_1 = cf.beta_1
        self.beta_2 = cf.beta_2
        self.beta = cf.beta
        self.div = cf.div
        self.d_rate = cf.d_rate
        self.l_rate = cf.l_rate
        self.q_l_rate = cf.q_l_rate
        self.condition = cf.condition / (sum(cf.td_neurons) - cf.td_neurons[0])

        self.optim = cf.optim
        self.eps = cf.eps
        self.decay_r = cf.decay_r
        self.c_b = [[] for _ in range(self.n_layers)]
        self.c_w = [[] for _ in range(self.n_layers)]
        self.v_b = [[] for _ in range(self.n_layers)]
        self.v_w = [[] for _ in range(self.n_layers)]
        self.c_b_q = [[] for _ in range(self.n_layers)]
        self.c_w_q = [[] for _ in range(self.n_layers)]
        self.v_b_q = [[] for _ in range(self.n_layers)]
        self.v_w_q = [[] for _ in range(self.n_layers)]

        self.W = None
        self.b = None
        self.Wq = None
        self.bq = None
        self._init_params()

    def train_epoch(self, img_batches, label_batches, epoch_num=None):
        init_err = 0
        end_err = 0
        avg_itr = 0
        n_batches = len(img_batches)

        for batch_id, (img_batch, label_batch) in enumerate(zip(img_batches, label_batches)):
            img_batch = set_tensor(img_batch, self.device)
            label_batch = set_tensor(label_batch, self.device)
            batch_size = img_batch.size(1)

            x = [[] for _ in range(self.n_layers)]
            q = [[] for _ in range(self.n_layers)]

            # bottom-up inference
            # self.W_q.reverse()
            # self.b_q.reverse()
            q[0] = img_batch
            for l in range(1, self.n_layers):
                b_q = self.b_q[l - 1].repeat(1, batch_size)
                q[l] = self.W_q[l - 1] @ F.f(q[l - 1], self.act_fn) + b_q


            x = q[::-1]
            x[0] = label_batch
            x[self.n_layers - 1] = img_batch
            init_err += self.get_errors(x, batch_size)

            # top-down inference
            x, errors, its = self.infer(x, batch_size)
            end_err += self.get_errors(x, batch_size)
            avg_itr += its

            # bottom-up inference
            # self.W_q.reverse()
            # self.b_q.reverse() 
            q = x[::-1]
            q[0] = img_batch
            q[self.n_layers-1] = label_batch
            q, q_errors, its = self.amortised_infer(q, batch_size)
            # self.W_q.reverse()
            # self.b_q.reverse()

            # update top-down parameters
            self.update_params(
                x,
                q,
                errors,
                q_errors,
                batch_size,
                img_batch,
                label_batch,
                epoch_num=epoch_num,
                n_batches=n_batches,
                curr_batch=batch_id,
            )

        return end_err / n_batches, init_err / n_batches, avg_itr / n_batches

    def test_epoch(self, x_batches, y_batches, itr_max=None):
        accs = []
        n_batches = len(x_batches)
        avg_itr = 0
        for x_batch, y_batch in zip(x_batches, y_batches):
            x_batch = set_tensor(x_batch, self.device)
            y_batch = set_tensor(y_batch, self.device)
            batch_size = x_batch.size(1)

            x = [[] for _ in range(self.n_layers)]
            q = [[] for _ in range(self.n_layers)]

            q[0] = x_batch
            for l in range(1, self.n_layers):
                b_q = self.b_q[l - 1].repeat(1, batch_size)
                q[l] = self.W_q[l - 1] @ F.f(q[l - 1], self.act_fn) + b_q
            x = q[::-1]
            x[self.n_layers - 1] = x_batch

            x, errors, its = self.infer_and_classify(x, batch_size, x_batch, itr_max=itr_max)
            pred_y = x[0]
            acc = mnist_utils.mnist_accuracy(pred_y, y_batch)
            accs.append(acc)
            avg_itr += its
        return accs, avg_itr / n_batches

    def test_pc_epoch(self, x_batches, y_batches, itr_max=None):
        accs = []
        n_batches = len(x_batches)
        avg_itr = 0
        for x_batch, y_batch in zip(x_batches, y_batches):
            x_batch = set_tensor(x_batch, self.device)
            y_batch = set_tensor(y_batch, self.device)
            batch_size = x_batch.size(1)

            x = [[] for _ in range(self.n_layers)]
            q = [[] for _ in range(self.n_layers)]

            x[0] = torch.empty_like(y_batch).normal_(mean=0.0, std=0.1)
            for l in range(1, self.n_layers):
                b = self.b[l - 1].repeat(1, batch_size)
                x[l] = self.W[l - 1] @ F.f(x[l - 1], self.act_fn) + b
            x[self.n_layers - 1] = x_batch

            x, errors, its = self.infer_and_classify(x, batch_size, x_batch, itr_max=itr_max)
            pred_y = x[0]
            acc = mnist_utils.mnist_accuracy(pred_y, y_batch)
            accs.append(acc)
            avg_itr += its
        return accs, avg_itr / n_batches

    def test_amortised_epoch(self, x_batches, y_batches):            
        # self.W_q.reverse()
        # self.b_q.reverse()

        accs = []
        for x_batch, y_batch in zip(x_batches, y_batches):
            x_batch = set_tensor(x_batch, self.device)
            y_batch = set_tensor(y_batch, self.device)
            batch_size = x_batch.size(1)

            q = [[] for _ in range(self.n_layers)]
            q[0] = x_batch
            for l in range(1, self.n_layers):
                b_q = self.b_q[l - 1].repeat(1, batch_size)
                q[l] = self.W_q[l - 1] @ F.f(q[l - 1], self.act_fn) + b_q
            pred_y = q[-1]
            acc = mnist_utils.mnist_accuracy(pred_y, y_batch)
            accs.append(acc)

        # self.W_q.reverse()
        # self.b_q.reverse()
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

    def amortised_infer(self, x, batch_size, itr_max=None):
        itr_max = self.itr_max if itr_max is None else itr_max
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
            b = self.b_q[l - 1].repeat(1, batch_size)
            errors[l] = (x[l] - self.W_q[l - 1] @ f_x - b) / self.vars[l]
            f_0 = f_0 - self.vars[l] * torch.sum(torch.mul(errors[l], errors[l]), dim=0)

        for itr in range(itr_max):
            # update node activity
            for l in range(1, self.n_layers - 1):
                # eq. 2.18
                g = torch.mul(self.W_q[l].T @ errors[l + 1], f_x_deriv_arr[l])
                x[l] = x[l] + beta * (-errors[l] + g)

            # update errors
            f = 0
            for l in range(1, self.n_layers):
                f_x = F.f(x[l - 1], self.act_fn)
                f_x_deriv = F.f_deriv(x[l - 1], self.act_fn)
                f_x_arr[l - 1] = f_x
                f_x_deriv_arr[l - 1] = f_x_deriv

                # eq. 2.17
                errors[l] = (x[l] - self.W_q[l - 1] @ f_x - self.b_q[l - 1]) / self.vars[l]
                f = f - self.vars[l] * torch.sum(torch.mul(errors[l], errors[l]), dim=0)

            diff = f - f_0
            threshold = self.condition * self.beta / self.vars[self.n_layers - 1]
            if torch.any(diff < 0):
                beta = beta / self.div
            elif torch.mean(diff) < threshold:
                # print(f"broke @ {its} its")
                break

            f_0 = f
            its = itr

        return x, errors, its

    def infer(self, x, batch_size, itr_max=None):
        itr_max = self.itr_max if itr_max is None else itr_max
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

        for itr in range(itr_max):
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
                # print(f"broke @ {its} its")
                break

            f_0 = f
            its = itr

        return x, errors, its

    def infer_and_classify(self, x, batch_size, x_batch, itr_max=None):
        """ this version infers top layer, rather than keeping it fixed """
        itr_max = self.itr_max if itr_max is None else itr_max
        errors = [[] for _ in range(self.n_layers)]
        f_x_arr = [[] for _ in range(self.n_layers)]
        f_x_deriv_arr = [[] for _ in range(self.n_layers)]
        f_0 = 0
        its = 0
        beta = self.beta

        x[self.n_layers - 1] = x_batch

        for l in range(1, self.n_layers):
            f_x = F.f(x[l - 1], self.act_fn)
            f_x_deriv = F.f_deriv(x[l - 1], self.act_fn)
            f_x_arr[l - 1] = f_x
            f_x_deriv_arr[l - 1] = f_x_deriv

            # eq. 2.17
            b = self.b[l - 1].repeat(1, batch_size)
            errors[l] = (x[l] - self.W[l - 1] @ f_x - b) / self.vars[l]
            f_0 = f_0 - self.vars[l] * torch.sum(torch.mul(errors[l], errors[l]), dim=0)

        for itr in range(itr_max):
            # TODO (updating top layer)
            g = torch.mul(self.W[0].T @ errors[1], f_x_deriv_arr[0])
            x[0] = x[0] + beta * g 

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

    def get_errors(self, x, batch_size):
        total_err = 0
        for l in range(1, self.n_layers - 1):
            b = self.b[l - 1].repeat(1, batch_size)
            err = (x[l] - self.W[l - 1] @ F.f(x[l - 1], self.act_fn) - b) / self.vars[l]
            total_err += torch.sum(torch.mul(err, err), dim=0)
        return torch.sum(total_err)

    def update_params(
        self, x, q, errors, q_errors, batch_size, x_batch, y_batch, epoch_num=None, n_batches=None, curr_batch=None
    ):

        grad_w = [[] for _ in range(self.n_layers - 1)]
        grad_b = [[] for _ in range(self.n_layers - 1)]
        grad_w_q = [[] for _ in range(self.n_layers - 1)]
        grad_b_q = [[] for _ in range(self.n_layers - 1)]

        for l in range(self.n_layers - 1):
            grad_w[l] = (
                self.vars[-1] * (1 / batch_size) * errors[l + 1] @ F.f(x[l], self.act_fn).T
                - self.d_rate * self.W[l]
            )
            grad_b[l] = self.vars[-1] * (1 / batch_size) * torch.sum(errors[l + 1], axis=1)

        for l in range(self.n_layers - 1):
            grad_w_q[l] = (
                self.vars[-1] * (1 / batch_size) * q_errors[l + 1] @ F.f(q[l], self.act_fn).T
                - self.d_rate * self.W_q[l]
            )
            grad_b_q[l] = self.vars[-1] * (1 / batch_size) * torch.sum(q_errors[l + 1], axis=1)

        self._apply_gradients(
            grad_w,
            grad_b,
            grad_w_q,
            grad_b_q,
            epoch_num=epoch_num,
            n_batches=n_batches,
            curr_batch=curr_batch,
        )

    def _apply_gradients(
        self, grad_w, grad_b, grad_w_q, grad_b_q, epoch_num=None, n_batches=None, curr_batch=None
    ):

        if self.optim is "RMSPROP":
            for l in range(self.n_layers - 1):
                grad_b[l] = grad_b[l].unsqueeze(dim=1)

                self.c_w[l] = self.decay_r * self.c_w[l] + (1 - self.decay_r) * grad_w[l] ** 2
                self.c_b[l] = self.decay_r * self.c_b[l] + (1 - self.decay_r) * grad_b[l] ** 2

                self.W[l] = self.W[l] + self.l_rate * (grad_w[l] / (torch.sqrt(self.c_w[l]) + self.eps))
                self.b[l] = self.b[l] + self.l_rate * (grad_b[l] / (torch.sqrt(self.c_b[l]) + self.eps))

                self.c_w_q[l] = self.decay_r * self.c_w[l] + (1 - self.decay_r) * grad_w[l] ** 2
                self.c_b_q[l] = self.decay_r * self.c_b[l] + (1 - self.decay_r) * grad_b[l] ** 2

                self.W_q[l] = self.W_q[l] + self.q_l_rate * (grad_w_1[l] / (torch.sqrt(self.c_w_q[l]) + self.eps))
                self.b_q[l] = self.b_q[l] + self.q_l_rate * (grad_b_q[l] / (torch.sqrt(self.c_b_q[l]) + self.eps))


        elif self.optim is "ADAM":
            for l in range(self.n_layers - 1):
                grad_b[l] = grad_b[l].unsqueeze(dim=1)
                self.c_b[l] = self.beta_1 * self.c_b[l] + (1 - self.beta_1) * grad_b[l]
                self.c_w[l] = self.beta_1 * self.c_w[l] + (1 - self.beta_1) * grad_w[l]

                self.v_b[l] = self.beta_2 * self.v_b[l] + (1 - self.beta_2) * grad_b[l] ** 2
                self.v_w[l] = self.beta_2 * self.v_w[l] + (1 - self.beta_2) * grad_w[l] ** 2

                t = (epoch_num) * n_batches + curr_batch
                self.W[l] = self.W[l] + self.l_rate * np.sqrt(1 - self.beta_2 ** t) * self.c_w[l] / (
                    torch.sqrt(self.v_w[l]) + self.eps
                )
                self.b[l] = self.b[l] + self.l_rate * np.sqrt(1 - self.beta_2 ** t) * self.c_b[l] / (
                    torch.sqrt(self.v_b[l]) + self.eps
                )

                grad_b_q[l] = grad_b_q[l].unsqueeze(dim=1)
                self.c_b_q[l] = self.beta_1 * self.c_b_q[l] + (1 - self.beta_1) * grad_b_q[l]
                self.c_w_q[l] = self.beta_1 * self.c_w_q[l] + (1 - self.beta_1) * grad_w_q[l]

                self.v_b_q[l] = self.beta_2 * self.v_b_q[l] + (1 - self.beta_2) * grad_b_q[l] ** 2
                self.v_w_q[l] = self.beta_2 * self.v_w_q[l] + (1 - self.beta_2) * grad_w_q[l] ** 2

                t = (epoch_num) * n_batches + curr_batch
                self.W_q[l] = self.W_q[l] + self.q_l_rate * np.sqrt(1 - self.beta_2 ** t) * self.c_w_q[l] / (
                    torch.sqrt(self.v_w_q[l]) + self.eps
                )
                self.b_q[l] = self.b_q[l] + self.q_l_rate * np.sqrt(1 - self.beta_2 ** t) * self.c_b_q[l] / (
                    torch.sqrt(self.v_b_q[l]) + self.eps
                )

        elif self.optim is "SGD" or self.optim is None:
            for l in range(self.n_layers - 1):
                self.W[l] = self.W[l] + self.l_rate * grad_w[l]
                self.b[l] = self.b[l] + self.l_rate * grad_b[l].unsqueeze(dim=1)

                self.W_q[l] = self.W_q[l] + self.q_l_rate * grad_w_q[l]
                self.b_q[l] = self.b_q[l] + self.q_l_rate * grad_b_q[l].unsqueeze(dim=1)

        else:
            raise ValueError(f"{self.optim} not supported")


    def _init_params(self):
        """
        Top down
        """

        weights = [[] for _ in range(self.n_layers - 1)]
        bias = [[] for _ in range(self.n_layers - 1)]

        for l in range(self.n_layers - 1):
            norm_b = 0
            if self.act_fn is F.LINEAR:
                norm_w = np.sqrt(1 / (self.td_neurons[l + 1] + self.td_neurons[l]))
            elif self.act_fn is F.TANH:
                norm_w = np.sqrt(6 / (self.td_neurons[l + 1] + self.td_neurons[l]))
            elif self.act_fn is F.LOGSIG:
                norm_w = 4 * np.sqrt(6 / (self.td_neurons[l + 1] + self.td_neurons[l]))
            else:
                raise ValueError(f"{self.act_fn} not supported")

            layer_w = np.random.uniform(-1, 1, size=(self.td_neurons[l + 1], self.td_neurons[l])) * norm_w
            layer_b = np.zeros((self.td_neurons[l + 1], 1)) + norm_b * np.ones((self.td_neurons[l + 1], 1))
            weights[l] = set_tensor(layer_w, self.device)
            bias[l] = set_tensor(layer_b, self.device)

        self.W = weights
        self.b = bias

        for l in range(self.n_layers - 1):
            self.c_b[l] = torch.zeros_like(self.b[l])
            self.c_w[l] = torch.zeros_like(self.W[l])
            self.v_b[l] = torch.zeros_like(self.b[l])
            self.v_w[l] = torch.zeros_like(self.W[l])

        """
        Bottom up
        """

        q_weights = [[] for _ in range(self.n_layers - 1)]
        q_bias = [[] for _ in range(self.n_layers - 1)]

        for l in range(self.n_layers - 1):
            norm_b = 0
            if self.act_fn is F.LINEAR:
                norm_w = np.sqrt(1 / (self.bu_neurons[l + 1] + self.bu_neurons[l]))
            elif self.act_fn is F.TANH:
                norm_w = np.sqrt(6 / (self.bu_neurons[l + 1] + self.bu_neurons[l]))
            elif self.act_fn is F.LOGSIG:
                norm_w = 4 * np.sqrt(6 / (self.bu_neurons[l + 1] + self.bu_neurons[l]))
            else:
                raise ValueError(f"{self.act_fn} not supported")

            layer_w = np.random.uniform(-1, 1, size=(self.bu_neurons[l + 1], self.bu_neurons[l])) * norm_w
            layer_b = np.zeros((self.bu_neurons[l + 1], 1)) + norm_b * np.ones((self.bu_neurons[l + 1], 1))
            q_weights[l] = set_tensor(layer_w, self.device)
            q_bias[l] = set_tensor(layer_b, self.device)

        self.W_q = q_weights
        self.b_q = q_bias

        for l in range(self.n_layers - 1):
            self.c_b_q[l] = torch.zeros_like(self.b_q[l])
            self.c_w_q[l] = torch.zeros_like(self.W_q[l])
            self.v_b_q[l] = torch.zeros_like(self.b_q[l])
            self.v_w_q[l] = torch.zeros_like(self.W_q[l])
