from functools import partial
import numpy as np


def step_func(x: np.ndarray):
    return (x >= 0).astype(float) * 2 - 1

def lineal_func(x: np.ndarray):
    return x

def get_sigmoid_tanh(b: float):
    return lambda x:np.tanh(b * x), lambda x:1/np.cosh(b * x) * b





def sigmoid_exp(b: float, x: np.ndarray):
    return 1/(1 + np.exp(-2 * b * x))

def sigmoid_exp_der(b: float, x: np.ndarray):
    t = np.exp(-2 * b * x)
    return 2 * b * t / (1 + t) ** 2

def get_sigmoid_exp(b: float):
    return partial(sigmoid_exp, b), partial(sigmoid_exp_der, b)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.linspace(-5, 5, 200)
    f, d = get_sigmoid_tanh(1)
    plt.plot(X, f(X), label="tanh")
    plt.plot(X, d(X), label="dtanh")

    f, d = get_sigmoid_exp(1)
    plt.plot(X, f(X), label="exp")
    plt.plot(X, d(X), label="dexp")

    plt.legend()

    plt.show()
