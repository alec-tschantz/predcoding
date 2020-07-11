import numpy as np


def f(x, act_fn):
    if act_fn == "logsig":
        div = np.ones(x.shape) + np.exp(-x)
        y = np.divide(1, div + 1e-7)
    else:
        raise ValueError(f"{act_fn} not supported")
    return y


def f_inv(x, act_fn):
    if act_fn == "logsig":
        div = (np.ones(x.shape) - x) + 1e-7
        y = np.log(np.divide(x, div) + 1e-7)
    else:
        raise ValueError(f"{act_fn} not supported")
    return y


def f_b(x, act_fn, layer):
    if act_fn == "logsig":
        # TODO
        # print(f"layer {layer} max {x.max()}")
        div = np.ones(x.shape) + np.exp(-x)
        f_n = np.log(np.divide(1, div + 1e-7) + 1e-7)
        f_p = np.multiply(f_n, (np.ones(x.shape) - f_n))
    return f_n, f_p
