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
