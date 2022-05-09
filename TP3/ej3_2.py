from typing import Generator, TypeVar
from single_data import SingleData
from activation_functions import get_sigmoid_tanh
from network import Network
import numpy as np
from more_itertools import take


def read_digits():
    with open('digits.txt') as digits_file:
        # print(digits_file.readlines())
        # print(list(split(digits_file.readlines(), '\n')))
        for digit in digits_file.read().split('\n\n'):
            if digit.strip():
                yield np.array([list(map(bool, map(int, line.strip().split(' ')))) for line in digit.split('\n')])

digits_data = []
empty_out = np.full((10,), -1, dtype=np.float64)
for i, digit in take(8, enumerate(read_digits())):
    out = empty_out.copy()
    out[i] = 1
    digits_data.append(SingleData(digit.flatten().astype(np.float64) * 2 - 1, out))
    

print(digits_data)



model = Network.with_zeroed_weights(5 * 7, (10, 10), *get_sigmoid_tanh(10))

# error = model.error(digits_data)
# while error > 0:
    # model.train(0.1, digits_data)
    # error = model.error(digits_data)
    # print(error)

print(model.evaluate(np.array([digit.inputs for digit in digits_data])))
