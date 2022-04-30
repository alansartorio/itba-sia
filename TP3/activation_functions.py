import numpy as np


def step_func(x: np.ndarray):
    return (x >= 0).astype(float) * 2 - 1
