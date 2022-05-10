from activation_functions import step_func
from network import Network
from train_data import ej1_and_data, ej1_xor_data
import matplotlib.pyplot as plt
import numpy as np
from plot_line import Plot
from matplotlib.animation import PillowWriter

# model = Network.with_zeroed_weights(2, (1, ), step_func)
model = Network.with_random_weights(2, (1, ), step_func)

# data = ej1_and_data
data = ej1_xor_data
plot = Plot([d.inputs for d in data], [d.outputs for d in data], model)


writer = PillowWriter(fps = 10)
writer.setup(plot.fig, 'plots/ej1.gif', dpi=200)

# print(model.layers[0].weights.flatten())
# print(model.error(ej1_data))
try:
    while model.error(data) > 0:
        model.train(0.0001, data)
        print(model.layers[0].weights.flatten(), model.error(data))
        # print(model.layers[0].weights.flatten(), model.error(ej1_data))
        # for single_data in data:
            # print(single_data.inputs, single_data.outputs, model.evaluate(single_data.inputs), end=' | ')
        # print()
        plot.update()
        writer.grab_frame()
except KeyboardInterrupt:
    pass
for _ in range(10):
    writer.grab_frame()
# single_neuron = Network.with_random_weights(1, (2, 3), step_func)
writer.finish()

plt.show()

# print(single_neuron.evaluate(np.array([1])))
