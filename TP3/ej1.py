from activation_functions import step_func
from network import Network
from train_data import ej1_and_data, ej1_xor_data
import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, inputs, outputs, model) -> None:
        self.model = model
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        scatter = ax.scatter(*zip(*inputs), c=outputs)
        fig.canvas.draw()

        self.axbackground = fig.canvas.copy_from_bbox(ax.bbox)

        self.line, = ax.plot([0, 0], [0, 0])
        self.fig, self.ax = fig, ax

        self.update()
        plt.show(block=False)


    def update(self):
        self.fig.canvas.restore_region(self.axbackground)

        C, A, B = self.model.layers[0].weights[0]
        X = [-2, 2]
        Y = [-(A/B)*x+C/B for x in X]
        self.line.set_xdata(X)
        self.line.set_ydata(Y)
        self.ax.draw_artist(self.line)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()


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
