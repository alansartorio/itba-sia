
from abc import ABC
from functools import cached_property
import random
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import numpy.typing as npt

from activation_functions import step_func
from single_data import SingleData
from train_data import ej1_data

float_array = npt.NDArray[np.float64]


class Layer:
    def __init__(self, weights: np.ndarray, activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray]) -> None:
        self.weights = weights
        self.activation_function = activation_function
        self.derivated_activation_function = derivated_activation_function

    @cached_property
    def perceptron_count(self):
        return self.weights.shape[0]

    @cached_property
    def previous_layer_perceptron_count(self):
        return self.weights.shape[1]

    def propagate_without_activation(self, inputs):
        return np.dot(self.weights, inputs)

    def calculate_previous_delta(self, previous_layer_h: float_array, output_delta: float_array):
        prod = sum(self.weights[i, :] * output_delta[i]
                   for i in range(self.perceptron_count))
        return self.derivated_activation_function(previous_layer_h) * prod


class EvaluationData:
    def __init__(self, h: Optional[float_array], v: float_array) -> None:
        self.h = h
        self.v = v


class Network(ABC):
    def __init__(self, weights: tuple[np.ndarray, ...], activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.full(x.shape, 1)) -> None:
        self.layers = tuple(Layer(layer_weights, activation_function, derivated_activation_function)
                            for layer_weights in weights)

    def layer_size(self, layer_index: int):
        return self.layers[layer_index].perceptron_count

    @cached_property
    def input_size(self):
        return self.layers[0].previous_layer_perceptron_count

    def calculate_vs_and_hs(self, inputs: float_array):
        values = [EvaluationData(None, inputs)]
        for layer in self.layers:
            current_h = layer.propagate_without_activation(values[-1].v)
            values.append(EvaluationData(
                current_h, layer.activation_function(current_h)))

        return values

    def evaluate(self, inputs: float_array):
        return self.calculate_vs_and_hs(inputs)[-1].v

    def error(self, evaluation_data: Sequence[SingleData]):
        return 0.5 * sum((data.outputs - self.evaluate(data.inputs)) ** 2 for data in evaluation_data)

    def calculate_deltas(self, single_data: SingleData, evaluation_data: list[EvaluationData]):
        last_hs = evaluation_data[-1].h
        assert last_hs is not None, "Last layer Hs should not be None."
        last_vs = evaluation_data[-1].v
        last_layer_delta = self.layers[-1].derivated_activation_function(
            last_hs) * (single_data.outputs - last_vs)
        deltas = [last_layer_delta]

        for layer in reversed(self.layers[1:]):
            hs = evaluation_data[-1].h
            deltas.insert(0, layer.calculate_previous_delta(hs, deltas[0]))

        return deltas

    def train_single(self, learning_rate: float, single_data: SingleData):
        evaluation = self.calculate_vs_and_hs(single_data.inputs)
        deltas = self.calculate_deltas(single_data, evaluation)
        for m, layer in enumerate(self.layers):
            # delta_w = learning_rate *\
            #     np.dot(np.expand_dims(deltas[m], 1),
            #            np.expand_dims(evaluation[m].v, 0))
            delta_w = [[learning_rate * d * v for v in evaluation[m].v]
                       for d in deltas[m]]
            layer.weights += delta_w

    def train(self, learning_rate: float, train_data: Sequence[SingleData]):
        train_data = list(train_data)
        random.shuffle(train_data)
        # for single_data in train_data:
            # self.train_single(learning_rate, single_data)
        self.train_single(learning_rate, random.sample(train_data, 1)[0])

    @classmethod
    def with_random_weights(cls, input_size: int, layers: tuple[int, ...], activation_function: Callable[[np.ndarray], np.ndarray], derivated_activation_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.full(x.shape, 1)):
        # return cls(tuple(np.array([[0.1, 0.2]]) for previous, current in zip((input_size, ) + layers, layers)), activation_function, derivated_activation_function)
        return cls(tuple((np.random.rand(current, previous) * 2 - 1) * 0.1 for previous, current in zip((input_size, ) + layers, layers)), activation_function, derivated_activation_function)
        # return cls(tuple(np.zeros((previous, current)) for previous, current in zip((input_size, ) + layers, layers)), activation_function, derivated_activation_function)


model = Network.with_random_weights(2, (1, ), step_func)

print(model.layers[0].weights.flatten())
# print(model.error(ej1_data))
for i in range(10000):
    model.train(0.1, ej1_data)
    # print(model.error(ej1_data))
    print(model.layers[0].weights.flatten(), model.error(ej1_data))
    # for data in ej1_data:
    # print(data.inputs, data.outputs, model.evaluate(data.inputs), end=' | ')
    # print()
# single_neuron = Network.with_random_weights(1, (2, 3), step_func)


# print(single_neuron.evaluate(np.array([1])))
