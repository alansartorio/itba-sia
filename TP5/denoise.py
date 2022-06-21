import random
from typing import Sequence
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math
from activation_functions import get_sigmoid_exp
from network import AutoEncoder
from single_data import SingleData
from fonts import to_bin_array, font_2, font_2_char
from add_noise import add_noise


class ImagePlot:
    def __init__(self, count: int, width: int, height: int, xlim: tuple[float, float], ylim: tuple[float, float], vmin=-1, vmax=1) -> None:
        w = 3
        h = math.ceil(count / w)
        fig, axs = plt.subplots(h, w, squeeze=False)
        axs = axs.flatten()
        for ax in map(axs.__getitem__, range(count, w*h)):
            fig.delaxes(ax)

        self.width = width
        self.height = height
        self.images = []
        for i, ax in zip(range(count), axs):
            image = ax.matshow(np.zeros((height, width)), extent=(*xlim, *ylim), vmin=vmin, vmax=vmax, cmap='Wistia')
            ax.xaxis.set_label_position('top') 
            # For X-axis
            ax.xaxis.set_ticklabels([])
            # For Y-axis
            ax.yaxis.set_ticklabels([])

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

letter_count = 32

offset = 0
letters = font_2_char[offset:letter_count + offset]
images = [to_bin_array(font_2[font_2_char.index(letter)]) for letter in letters]

data = [SingleData(add_noise(image, probability=0.0, amp=1), image) for image in images]

with open('denoise.txt') as file:
    net = AutoEncoder.load_from_file(file, 0.001, *get_sigmoid_exp(1))

denoised = net.evaluate(np.array([d.inputs for d in data]))

plot = LetterPlot(letter_count * 3)
to_draw = [(d.outputs, d.inputs, denoise) for d, denoise in zip(data, denoised)]
plot.draw([x for xs in to_draw for x in xs])
plt.show()

