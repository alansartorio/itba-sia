import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt



class ImagePlot:
    def __init__(self, width: int, height: int, xlim: tuple[float, float], ylim: tuple[float, float], vmin=-1, vmax=1) -> None:
        fig, ax = plt.subplots()

        image = ax.imshow(np.zeros((height, width)), extent=(*xlim, *ylim), vmin=vmin, vmax=vmax)#, cmap='Wistia')
        plt.colorbar(image)

        self.image, self.fig = image, fig
        self.ax = ax

        plt.show(block=False)

    def draw(self, image: npt.NDArray[np.float64]):
        self.image.set_array(image)

        # self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.0000001)
