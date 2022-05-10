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
# plot = ImagePlot(10, 2, (0, 10), (0, 2), -1, 1)
# plot.ax.set_xlabel("Digit")
# plot.ax.set_ylabel("Prediction")

# plot.ax.set_xticks(np.arange(len(digits)) + 0.5, labels=digits)
# plot.ax.set_yticks(np.arange(len(predictions)) + 0.5, labels=predictions)

# writer = PillowWriter(fps=10)
# writer.setup(plot.fig, 'plots/ej3_2.gif', dpi=200)

try:
    # while error > 0:
    for _ in range(1000):
        model.train(0.0001, digits_data[:8] + digits_data[10:])
        error = model.error(digits_data)
        evaluation = model.evaluate(
            np.array([digit.inputs for digit in digits_data]))
        # evaluation = softmax(evaluation)
        # plot.draw(evaluation.T)
        print(error)
        # writer.grab_frame()
except KeyboardInterrupt:
    pass

# for _ in range(30):
    # writer.grab_frame()
# writer.finish()


evaluation = model.evaluate(np.array([digit.inputs for digit in digits_data]))
evaluation = evaluation[8:10, :]
even = evaluation[::2, :]
real_even = np.count_nonzero(even[:, 0] > even[:, 1])
false_even = np.count_nonzero(even[:, 0] <= even[:, 1])
odd = evaluation[1::2, :]
false_odd = np.count_nonzero(odd[:, 0] > odd[:, 1])
real_odd = np.count_nonzero(odd[:, 0] <= odd[:, 1])

confusion = [[real_even, false_even], [false_odd, real_odd]]

fig = plt.gcf()
fig.set_size_inches(3, 3)
fig.set_dpi(200)
ax = plt.subplot()
ax.matshow(confusion, cmap="Blues")
for (i, j), z in np.ndenumerate(confusion):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

ax.xaxis.set_label_position('top')
ax.set_xticks([1, 0], labels=predictions)
ax.set_yticks([1, 0], labels=predictions)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()

plt.show()
