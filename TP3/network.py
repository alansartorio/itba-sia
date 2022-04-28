
from functools import cached_property
import numpy as np
from typing import Callable, NamedTuple

from activation_functions import step_func


class Layer:
    def __init__(self, weights: np.ndarray, activation_function: Callable[[np.ndarray], np.ndarray]) -> None:
        self.weights = weights
        self.activation_function = activation_function

    @cached_property
    def perceptron_count(self):
        return self.weights.shape[1]

    @cached_property
    def previous_layer_perceptron_count(self):
        return self.weights.shape[0]

    def propagate(self, inputs):
        return self.activation_function(np.dot(inputs, self.weights))

    def backpropagate(self, output_delta):
        pass


class SingleData:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs

class Network:
    def __init__(self, weights: tuple[np.ndarray, ...], activation_function: Callable[[np.ndarray], np.ndarray]) -> None:
        self.layers = tuple(Layer(layer_weights, activation_function)
                            for layer_weights in weights)

    def layer_size(self, layer_index: int):
        return self.layers[layer_index].perceptron_count

    @cached_property
    def input_size(self):
        return self.layers[0].previous_layer_perceptron_count

    def evaluate(self, inputs: np.ndarray):
        values = inputs
        for layer in self.layers:
            values = layer.propagate(values)

        return values

    def train_single(self, single_data: SingleData):
        pass

    def train(self, train_data: list[SingleData]):
        for single_data in train_data:
            self.train_single(single_data)

    @classmethod
    def with_random_weights(cls, input_size: int, layers: tuple[int, ...], activation_function: Callable[[np.ndarray], np.ndarray]):
        return cls(tuple(np.random.rand(previous, current) * 2 - 1 for previous, current in zip((input_size, ) + layers, layers)), activation_function)



net = Network.with_random_weights(1, (2, 3), step_func)

print(net.evaluate(np.array([1])))
