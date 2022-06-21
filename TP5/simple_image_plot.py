from typing import Sequence
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math



class ImagePlot:
    def __init__(self, count: int, width: int, height: int, xlim: tuple[float, float], ylim: tuple[float, float], vmin=-1, vmax=1) -> None:
        size = math.ceil(math.sqrt(count))
        fig, axs = plt.subplots(size, size, squeeze=False)
        axs = axs.flatten()
        for ax in map(axs.__getitem__, range(count, size*size)):
            fig.delaxes(ax)

        self.width = width
        self.height = height
        self.images = []
        for i, ax in zip(range(count), axs):
            image = ax.matshow(np.zeros((height, width)), extent=(*xlim, *ylim), vmin=vmin, vmax=vmax, cmap='Wistia')
            ax.xaxis.set_label_position('top') 

            self.images.append(image)
        plt.colorbar(image)
        self.axs = axs
        self.fig = fig

        plt.show(block=False)

    def draw(self, images: Sequence[npt.NDArray[np.float64]]):
        for p, image in zip(self.images, images):
            p.set_array(image.reshape((self.height, self.width)))

        # self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.0000001)

class LetterPlot(ImagePlot):
    def __init__(self, count: int) -> None:
        super().__init__(count, 5, 7, (0, 5), (0, 7), 0, 1)
