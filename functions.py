import numpy as np


def f(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is "lin":
        m = x
    elif act_fn is "tanh":
        m = np.tanh(x)
    elif act_fn is "logsig":
        div = np.ones(x.shape) + np.exp(-x)
        m = np.divide(1, div + 1e-7)
    elif act_fn is "exp":
        m = np.exp(x)
    else:
        raise ValueError(f"{act_fn} not supported")
    return m


def f_inv(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is "lin":
        m = x
    elif act_fn is "tanh":
        # prone to numerical issues
        num = np.ones(x.shape) + x
        div = (np.ones(x.shape) - x) + 1e-7
        m = 0.5 * np.log(np.divide(num, div))
    elif act_fn == "logsig":
        div = (np.ones(x.shape) - x) + 1e-7
        m = np.log(np.divide(x, div) + 1e-7)
    elif act_fn == "exp":
        m = np.log(x)
    else:
        raise ValueError(f"{act_fn} not supported")
    return m


def f_b(x, act_fn):
    """ (activation_size, batch_size) """
    if act_fn is "linear":
        f = x
        f_p = np.ones(x.shape)
    elif act_fn is "tanh":
        f = np.tanh(x)
        f_p = np.ones(x.shape) - f ** 2
    elif act_fn is "logsig":
        div = np.ones(x.shape) + np.exp(-x)
        f = np.divide(1, div + 1e-7) 
        f_p = np.multiply(f, (np.ones(x.shape) - f))
    elif act_fn is "exp":
        f = np.exp(x)
        f_p = np.exp(x)
    else:
        raise ValueError(f"{act_fn} not supported")
    return f, f_p


if __name__ == "__main__":
    size = 784
    batch_size = 20
    act_fn = "logsig"

    # predefined to compare with MATLAB
    x = np.zeros((4, 2))
    x[:, 0] = [0.1, 0.4, 0.2, 0.7]
    x[:, 1] = [0.2, 0.5, 0.05, 0.02]

    pred_x_1, pred_x_2 = f_b(x, act_fn)
    print(pred_x_1.round(3))
    print(pred_x_2.round(3))
