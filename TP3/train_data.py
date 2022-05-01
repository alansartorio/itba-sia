from single_data import SingleData
import numpy as np


def load_values(filename: str):
    with open(filename, 'r') as file:
        for line in file:
            yield np.array(tuple(map(float, line.split())))


def load_single_data(inputs_filename: str, outputs_filename: str):
    for input, output in zip(load_values(inputs_filename), load_values(outputs_filename)):
        yield SingleData(input, output)


ej1_and_data = (SingleData(np.array([-1, 1]), np.array([-1])),
                SingleData(np.array([-1, -1]), np.array([-1])),
                SingleData(np.array([1, -1]), np.array([-1])),
                SingleData(np.array([1, 1]), np.array([1])))

ej1_xor_data = (SingleData(np.array([-1, 1]), np.array([1])),
                SingleData(np.array([-1, -1]), np.array([-1])),
                SingleData(np.array([1, -1]), np.array([1])),
                SingleData(np.array([1, 1]), np.array([-1])))

ej2_data = list(load_single_data('ej2_inputs.txt', 'ej2_outputs.txt'))

