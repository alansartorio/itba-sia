from simple_image_plot import ImagePlot
from network import Network
import numpy as np
import matplotlib.pyplot as plt


def image_evaluate(model: Network, width: int, height: int, xlim: tuple[float, float], ylim: tuple[float, float]):
    coordinates = np.array(np.meshgrid(np.linspace(
        *xlim, width), np.linspace(*ylim, height)))
    coordinates = np.swapaxes(coordinates, 0, -1)
    print(coordinates.shape)

    out = model.evaluate(coordinates)

    return out


class ImageEvaluationPlot(ImagePlot):
    def __init__(self, model: Network, width: int, height: int, xlim: tuple[float, float], ylim: tuple[float, float], vmin=-1, vmax=1) -> None:
        super().__init__(width, height, xlim, ylim, vmin=vmin, vmax=vmax)

        coordinates = np.array(
            np.meshgrid(-np.linspace(*xlim, width), np.linspace(*ylim, height)))
        coordinates = np.swapaxes(coordinates, 0, -1)

        self.coordinates = coordinates
        self.model = model

    def draw(self):
        out = self.model.evaluate(self.coordinates)
        # out = np.ones(coordinates.shape[1:])
        if out.shape[2] == 3:
            out = (out + 1) / 2

        super().draw(out)
