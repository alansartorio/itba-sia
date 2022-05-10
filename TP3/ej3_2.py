from typing import Generator, TypeVar
from digits import get_digits_data, get_parity_data
from simple_image_plot import ImagePlot
from single_data import SingleData
from activation_functions import get_sigmoid_exp, get_sigmoid_tanh, softmax
from network import Network
import numpy as np
from more_itertools import take

digits_data = get_parity_data()

model = Network.with_zeroed_weights(5 * 7, (10, 2,), *get_sigmoid_tanh(10))
model.randomize_weights(0.01)

error = model.error(digits_data)
plot = ImagePlot(2, 2, (0, 2), (2, 0), -1, 1)
plot.ax.set_xlabel("Input")
plot.ax.set_ylabel("Prediction")

# for _ in range(1000):
while error > 0:
    model.train(0.00001, digits_data)
    error = model.error(digits_data)
    evaluation = model.evaluate(np.array([digit.inputs for digit in digits_data]))
    # evaluation = softmax(evaluation)
    plot.draw(evaluation)
    print(error)

import matplotlib.pyplot as plt
plt.show()
