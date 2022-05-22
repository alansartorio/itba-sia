from typing import Generator, TypeVar
from digits import get_digits_data, random_digits_data
from simple_image_plot import ImagePlot
from single_data import SingleData
from activation_functions import get_sigmoid_exp, get_sigmoid_tanh, softmax
from network import Network
import numpy as np
from more_itertools import take

digits_data = get_digits_data()
# random_digits_data = random_digits_data()

model = Network.with_zeroed_weights(5 * 7, (10, 10,), *get_sigmoid_tanh(10))
model.randomize_weights(0.01)

error = model.error(digits_data)
plot = ImagePlot(10, 10, (0, 10), (10, 0), -1, 1)
plot.ax.set_xlabel("Input")
plot.ax.set_ylabel("Prediction")

try:
    # while error > 0:
    for i in range(200):
        model.train(0.001, digits_data)
        error = model.error(digits_data)
        evaluation = model.evaluate(np.array([digit.inputs for digit in digits_data])).T
        # evaluation = model.evaluate(np.array([digit.inputs for digit in random_digits_data])).T
        # evaluation = softmax(evaluation)
        plot.draw(evaluation)

        print(i)
except KeyboardInterrupt:
    pass

for (i, j), z in np.ndenumerate(evaluation):
    plot.ax.text(j + 0.5, i + 0.5, '{:0.1f}'.format(z), ha='center', va='center')
plot.draw(evaluation)

import matplotlib.pyplot as plt
plt.show()
