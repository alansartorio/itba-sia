from matplotlib.animation import PillowWriter
from image_plot import ImageEvaluationPlot
from activation_functions import get_sigmoid_tanh, lineal_func, step_func
from network import Network
from train_data import ej1_and_data, ej1_xor_data
import matplotlib.pyplot as plt
import numpy as np
from plot_line import Plot


# model = Network.with_random_weights(2, (3, 1, ), step_func)
# model = Network.with_random_weights(2, (4, 4, 4, 4, 1, ), step_func)
# model = Network.with_random_weights(2, (3, 3, 1, ), lineal_func)
model = Network.with_random_weights(2, (100, 100, 1, ), *get_sigmoid_tanh(100))

# data = ej1_and_data
data = ej1_xor_data

res = 120
plot = ImageEvaluationPlot(model, res, res, (-2, 2), (-2, 2))
plot.ax.scatter([-1, -1, 1, 1], [-1, 1, -1, 1], color='black')
plot.draw()

writer = PillowWriter(fps = 10)
writer.setup(plot.fig, 'plots/ej3_1_tanh.gif', dpi=200)

# print(model.layers[0].weights.flatten())
# print(model.error(ej1_data))
try:
    while model.error(data) > 0:
        # model.train(0.001, data)
        model.train(0.000001, data)
        print(model.error(data))
        # print(model.layers[0].weights.flatten(), model.error(ej1_data))
        # for single_data in data:
            # print(single_data.inputs, single_data.outputs, model.evaluate(single_data.inputs), end=' | ')
        # print()

        plot.draw()
        writer.grab_frame()
except KeyboardInterrupt:
    pass

for _ in range(10):
    writer.grab_frame()
writer.finish()

plt.show()

# single_neuron = Network.with_random_weights(1, (2, 3), step_func)

# print(single_neuron.evaluate(np.array([1])))
