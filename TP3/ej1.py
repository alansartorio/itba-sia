from activation_functions import step_func
from network import Network
from train_data import ej1_and_data, ej1_xor_data
import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, inputs, outputs, model) -> None:
        self.model = model
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        self.scatter = self.ax.scatter(*zip(*inputs), c=outputs)
        self.line, = self.ax.plot([0, 0], [0, 0])
        plt.draw()
        self.update()

    def update(self):
        C, A, B = self.model.layers[0].weights[0]
        X = [-2, 2]
        Y = [-(A/B)*x+C/B for x in X]
        self.line.set_xdata(X)
        self.line.set_ydata(Y)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


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
