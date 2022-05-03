from activation_functions import step_func
from network import Network
from train_data import ej1_and_data, ej1_xor_data
import matplotlib.pyplot as plt
import numpy as np
from plot_line import Plot

# model = Network.with_zeroed_weights(2, (1, ), step_func)
model = Network.with_random_weights(2, (1, ), step_func)

data = ej1_and_data
# data = ej1_xor_data
plot = Plot([d.inputs for d in data], [d.outputs for d in data], model)



# print(model.layers[0].weights.flatten())
# print(model.error(ej1_data))
while model.error(data) > 0:
    model.train(0.0001, data)
    print(model.layers[0].weights.flatten(), model.error(data))
    # print(model.layers[0].weights.flatten(), model.error(ej1_data))
    # for single_data in data:
        # print(single_data.inputs, single_data.outputs, model.evaluate(single_data.inputs), end=' | ')
    # print()
    plot.update()
# single_neuron = Network.with_random_weights(1, (2, 3), step_func)

plt.show()

# print(single_neuron.evaluate(np.array([1])))
