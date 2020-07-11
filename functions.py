import numpy as np

""" TODO fix numerical issues """


def f(x, act_fn):
    if act_fn == "logsig":
        div = np.ones(x.shape) + np.exp(-x)
        y = np.divide(1, div + 1e-7)
    elif act_fn == "tanh":
        y = np.tanh(x)
    else:
        raise ValueError(f"{act_fn} not supported")
    return y


def f_inv(x, act_fn):
    if act_fn == "logsig":
        div = (np.ones(x.shape) - x) + 1e-7
        y = np.log(np.divide(x, div) + 1e-7)
    elif act_fn == "tanh":
        num = np.ones(x.shape) + x
        div = (np.ones(x.shape) - x) + 1e-7
        y = 0.5 * np.log(np.divide(num, div))
    else:
        raise ValueError(f"{act_fn} not supported")
    return y


def f_b(x, act_fn, layer):
    if act_fn == "logsig":
        div = np.ones(x.shape) + np.exp(-x)
        f_n = np.log(np.divide(1, div + 1e-7) + 1e-7)
        f_p = np.multiply(f_n, (np.ones(x.shape) - f_n))
    elif act_fn == "tanh":
        f_n = np.tanh(x)
        # TODO is this element wise
        f_p = np.ones(x.shape) - f_n ** 2
    return f_n, f_p
