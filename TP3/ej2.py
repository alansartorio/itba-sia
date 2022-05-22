from matplotlib.animation import PillowWriter
from activation_functions import get_sigmoid_tanh, lineal_func, get_sigmoid_exp, sigmoid_exp, sigmoid_exp_der
from network import Network
from train_data import ej2_data
import matplotlib.pyplot as plt
from metrics import Metrics
import numpy as np

class MovingPlot:
    def __init__(self, points, evaluate) -> None:
        self.points = points
        self.evaluate = evaluate
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        self.scatter = self.ax.scatter(*zip(*self.points))
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        plt.colorbar(self.scatter)
        plt.show(block=False)

    def update(self):
        self.scatter.set_array(self.evaluate(np.array(self.points)))
        self.scatter.set_clim(0, 100)
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.0000001)

def get_sigmoid_big_exp(b: float):
    return lambda x:100 * sigmoid_exp(b, x), lambda x:100 * sigmoid_exp_der(b, x)

# def train_with(model: Network):

    # # for data in ej2_data:
        # # data.outputs /= 100

    # model.randomize_weights()
    # (training, test) = Metrics.split(ej2_data, 0.2)
    # errors = []
    # errors.append(model.mean_squared_error(test))
    # for i in range(4000):
        # model.train(0.00001, training)
        # errors.append(model.mean_squared_error(test)) 

    # return errors


model_lin = Network.with_random_weights(3, (1,), lineal_func)
# # model = Network.with_random_weights(3, (1,), *get_sigmoid_tanh(1))
model_exp = Network.with_random_weights(3, (1,), *get_sigmoid_big_exp(0.3))

# e1 = train_with(model_lin)
# plt.plot(e1)
# e2 = train_with(model_exp)
# plt.plot(e2)

# plt.yticks(list(plt.yticks()[0]) + [min(e1), min(e2)])
# # plt.yscale('log')
# plt.xlabel("Epochs")
# plt.ylabel("Mean Squared Error")
# plt.grid(which='both')

# plt.show()

# model = model_lin
model = model_exp

inputs = [d.inputs for d in ej2_data]
outputs = [d.outputs for d in ej2_data]

# def truth_plot():
    # ax = plot.fig.add_subplot(1, 2, 2, projection='3d')
    # scatter = ax.scatter(*zip(*inputs), c=outputs, vmin=0, vmax=100)
    # plt.colorbar(scatter)
    # plt.draw()

plot = MovingPlot(inputs, lambda x:model.evaluate(x)[:, 0])
# truth_plot()

writer = PillowWriter(fps=10)
writer.setup(plot.fig, 'plots/ej2_anim_lin.gif', dpi=200)

plot.update()
#print(model.layers[0].weights.flatten(), model.error(ej2_data))
for i in range(100):
    model.train(0.0001, ej2_data)
    
    plot.update()
    print(i, model.layers[0].weights.flatten(), model.error(ej2_data))

    writer.grab_frame()

writer.finish()
plt.show()
