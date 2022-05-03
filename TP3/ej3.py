from activation_functions import get_sigmoid_tanh, lineal_func, step_func
from network import Network
from train_data import ej1_and_data, ej1_xor_data
import matplotlib.pyplot as plt
import numpy as np
from plot_line import Plot


model = Network.with_random_weights(2, (3, 1, ), step_func)
# model = Network.with_random_weights(2, (4, 4, 4, 4, 1, ), step_func)
# model = Network.with_random_weights(2, (3, 3, 1, ), lineal_func)
# model = Network.with_random_weights(2, (3, 1, ), *get_sigmoid_tanh(1))

# data = ej1_and_data
data = ej1_xor_data

res = 0.025
coordinates = np.mgrid[-2:2:res, 2:-2:-res]
# print(coordinates.shape)
# coordinates = np.array(np.meshgrid([1, -1], [-1, 1]))

minX = np.min(coordinates[0])
maxX = np.max(coordinates[0])
minY = np.min(coordinates[1])
maxY = np.max(coordinates[1])

fig, ax = plt.subplots()

# plt.ion()
image = ax.imshow(np.zeros(coordinates.shape[1:]), extent=(minX, maxX, minY, maxY), vmin=-1, vmax=1)#, cmap='Wistia')
ax.scatter([-1, -1, 1, 1], [-1, 1, -1, 1], color='black')
def draw():
    out = model.evaluate(coordinates)
    out = np.swapaxes(out, 0, -1)
    # out = np.ones(coordinates.shape[1:])
    image.set_array(out)

    
    # plt.draw()
    fig.canvas.draw()
    fig.canvas.start_event_loop(0.0000001)

draw()
plt.show(block=False)


# print(model.layers[0].weights.flatten())
# print(model.error(ej1_data))
while model.error(data) > 0:
    model.train(0.1, data)
    print(model.layers[0].weights.flatten(), model.error(data))
    # print(model.layers[0].weights.flatten(), model.error(ej1_data))
    # for single_data in data:
        # print(single_data.inputs, single_data.outputs, model.evaluate(single_data.inputs), end=' | ')
    # print()

    draw()
plt.show()

# single_neuron = Network.with_random_weights(1, (2, 3), step_func)

# print(single_neuron.evaluate(np.array([1])))
