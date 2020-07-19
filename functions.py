import numpy as np

LINEAR = "linear"
TANH = "tanh"
LOGSIG = "logsig"


def f(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is LINEAR:
        m = x
    elif act_fn is TANH:
        m = np.tanh(x)
    elif act_fn is LOGSIG:
        div = np.ones(x.shape) + np.exp(-x)
        m = np.divide(1, div + 1e-7)
    else:
        raise ValueError(f"{act_fn} not supported")
    return m


def f_inv(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is LINEAR:
        m = x
    elif act_fn is TANH:
        num = np.ones(x.shape) + x
        div = (np.ones(x.shape) - x) + 1e-7
        m = 0.5 * np.log(np.divide(num, div))
    elif act_fn == LOGSIG:
        div = (np.ones(x.shape) - x) + 1e-7
        m = np.log(np.divide(x, div) + 1e-7)
    else:
        raise ValueError(f"{act_fn} not supported")
    return m


def f_b(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is LINEAR:
        f = x
        f_p = np.ones(x.shape)
    elif act_fn is TANH:
        f = np.tanh(x)
        f_p = np.ones(x.shape) - f ** 2
    elif act_fn is LOGSIG:
        div = np.ones(x.shape) + np.exp(-x)
        f = np.divide(1, div + 1e-7)
        f_p = np.multiply(f, (np.ones(x.shape) - f))
    else:
        raise ValueError(f"{act_fn} not supported")
    return f, f_p
