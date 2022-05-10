import numpy as np
from single_data import SingleData


def read_digits():
    with open('digits.txt') as digits_file:
        # print(digits_file.readlines())
        # print(list(split(digits_file.readlines(), '\n')))
        for digit in digits_file.read().split('\n\n'):
            if digit.strip():
                yield np.array([list(map(bool, map(int, line.strip().split(' ')))) for line in digit.split('\n')])

def get_parity_data():
    parity_data = []
    empty_out = np.full((2,), -1, dtype=np.float64)
    for i, digit in enumerate(read_digits()):
        out = empty_out.copy()
        out[i % 2] = 1
        parity_data.append(SingleData(digit.flatten().astype(np.float64) * 2 - 1, out))
        # digits_data.append(SingleData(digit.flatten().astype(np.float64), out))
        
    return parity_data

def get_digits_data():
    digits_data = []
    empty_out = np.full((10,), -1, dtype=np.float64)
    for i, digit in enumerate(read_digits()):
        out = empty_out.copy()
        out[i] = 1
        digits_data.append(SingleData(digit.flatten().astype(np.float64) * 2 - 1, out))
        # digits_data.append(SingleData(digit.flatten().astype(np.float64), out))
        
    return digits_data
