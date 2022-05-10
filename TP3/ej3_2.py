import matplotlib.pyplot as plt
from typing import Generator, TypeVar

from matplotlib.animation import PillowWriter
from digits import get_digits_data, get_parity_data
from simple_image_plot import ImagePlot
from single_data import SingleData
from activation_functions import get_sigmoid_exp, get_sigmoid_tanh, softmax
from network import Network
import numpy as np
from more_itertools import take

digits_data = get_parity_data()

model = Network.with_zeroed_weights(5 * 7, (2,), *get_sigmoid_tanh(10))
model.randomize_weights(0.01)

digits = list(range(10))
predictions = ['Odd', 'Even']
error = model.error(digits_data)
plot = ImagePlot(10, 2, (0, 10), (0, 2), -1, 1)
plot.ax.set_xlabel("Digit")
plot.ax.set_ylabel("Prediction")

plot.ax.set_xticks(np.arange(len(digits)) + 0.5, labels=digits)
plot.ax.set_yticks(np.arange(len(predictions)) + 0.5, labels=predictions)

writer = PillowWriter(fps=10)
writer.setup(plot.fig, 'plots/ej3_2.gif', dpi=200)

try:
    # for _ in range(1000):
    while error > 0:
        model.train(0.0001, digits_data[:8])
        error = model.error(digits_data)
        evaluation = model.evaluate(
            np.array([digit.inputs for digit in digits_data]))
        # evaluation = softmax(evaluation)
        plot.draw(evaluation.T)
        print(error)
        writer.grab_frame()
except KeyboardInterrupt:
    pass

for _ in range(30):
    writer.grab_frame()
writer.finish()

plt.show()
