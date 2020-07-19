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
        m = np.divide(1, (np.ones(x.shape) + np.exp(-x)) + 1e-7)
    else:
        raise ValueError(f"{act_fn} not supported")
    return m


def f_deriv(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is LINEAR:
        deriv = np.ones(x.shape)
    elif act_fn is TANH:
        deriv = np.ones(x.shape) - np.tanh(x) ** 2
    elif act_fn is LOGSIG:
        f = np.divide(1, (np.ones(x.shape) + np.exp(-x)) + 1e-7)
        deriv = np.multiply(f, (np.ones(x.shape) - f))
    else:
        raise ValueError(f"{act_fn} not supported")
    return deriv


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

