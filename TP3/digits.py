import numpy as np
from single_data import SingleData
from typing import Sequence


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


def rand_bit(value): 
    if np.random.random() < 0.002:
        return -value
    return value
            
def random_digits_data():
    digits_data = []
    empty_out = np.full((10,), -1, dtype=np.float64)
    for i, digit in enumerate(read_digits()):
        out = empty_out.copy()
        out[i] = 1
        #digits_data.append(SingleData(digit.flatten().astype(np.float64) * 2 - 1, out))
        digits_data.append(SingleData(list(map(rand_bit, digit.flatten().astype(np.float64) * 2 - 1)), out))
        # digits_data.append(SingleData(digit.flatten().astype(np.float64), out))
        print(digit.flatten().astype(np.float64) * 2 - 1)
    return digits_data

"""
def random_digits_data(prob: float):
    digits_data = get_digits_data()
    inputs = [d.inputs for d in digits_data]

    new_inputs = list()
    for digit in inputs:
        d = []
        for bit in digit:
            value = np.random.random()
            if  value < 0.02:
                if bit == 1:
                    bit = -1
                else: 
                    bit = 1
            d.append(bit)
        new_inputs.append(d)
    print(inputs)
    print(new_inputs)
    return digits_data
"""
