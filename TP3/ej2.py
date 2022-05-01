from activation_functions import lineal_func
from network import Network
from train_data import ej2_data
import matplotlib.pyplot as plt


class MovingPlot:
    def __init__(self, points, evaluate) -> None:
        self.points = points
        self.evaluate = evaluate
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.scatter = self.ax.scatter(*zip(*self.points))

    def update(self):
        self.scatter.set_array([self.evaluate(inputs) for inputs in self.points])
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

model = Network.with_random_weights(3, (1,), lineal_func)
inputs = [d.inputs for d in ej2_data]
outputs = [d.outputs for d in ej2_data]
plot = MovingPlot(inputs, lambda x:model.evaluate(x)[0])


def truth_plot():
    ax = plot.fig.add_subplot(1, 2, 2, projection='3d')
    scatter = ax.scatter(*zip(*inputs), c=outputs)
    plt.draw()

truth_plot()


print(model.layers[0].weights.flatten(), model.error(ej2_data))
for i in range(1000):
    model.train(0.0001, ej2_data)
    
    plot.update()
    print(model.layers[0].weights.flatten(), model.error(ej2_data))

