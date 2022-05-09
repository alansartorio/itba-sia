from network import Network
import numpy as np
import matplotlib.pyplot as plt

def image_evaluate(model: Network, width: int, height: int, xlim: tuple[float, float], ylim: tuple[float, float]):
    coordinates = np.array(np.meshgrid(np.linspace(*xlim, width), np.linspace(*ylim, height)))
    coordinates = np.swapaxes(coordinates, 0, -1)
    print(coordinates.shape)

    out = model.evaluate(coordinates)

    return out

class ImageEvaluationPlot:

    def __init__(self, model: Network, width: int, height: int, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        # coordinates = np.mgrid[-2:2:res, 2:-2:-res]
        # print(coordinates.shape)
        coordinates = np.array(np.meshgrid(-np.linspace(*xlim, width), np.linspace(*ylim, height)))
        coordinates = np.swapaxes(coordinates, 0, -1)

        minX = np.min(coordinates[:, 0])
        maxX = np.max(coordinates[:, 0])
        minY = np.min(coordinates[:, 1])
        maxY = np.max(coordinates[:, 1])

        fig, ax = plt.subplots()

        image = ax.imshow(np.zeros(coordinates.shape[1:]), extent=(minX, maxX, minY, maxY), vmin=-1, vmax=1)#, cmap='Wistia')

        self.image, self.coordinates, self.fig = image, coordinates, fig
        self.model = model
        self.ax = ax

        plt.show(block=False)

    def draw(self):
        out = self.model.evaluate(self.coordinates)
        # out = np.ones(coordinates.shape[1:])
        if out.shape[2] == 3:
            out = (out + 1) / 2
        self.image.set_array(out)

        
        # plt.draw()
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.0000001)
