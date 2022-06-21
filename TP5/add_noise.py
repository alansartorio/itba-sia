import numpy as np
import numpy.typing as npt

def add_noise(array: npt.NDArray, probability: float = 1, amp: float = 0.1):
    array = array.copy()
    rand = np.random.uniform(0, 1, array.shape)
    to_add = np.random.uniform(-amp, amp, array.shape)
    where = rand < probability
    array[where] += to_add[where]
    array = np.clip(array, 0, 1)

    return array


if __name__ == '__main__':
    print(add_noise(np.full((10, 10), 0.5), 1))
