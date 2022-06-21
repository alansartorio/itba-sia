
from activation_functions import get_sigmoid_exp
from simple_image_plot import ImagePlot, LetterPlot
from network import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np
from fonts import font_2, font_2_char, to_bin_array


with open('model.txt') as file:
    net = AutoEncoder.load_from_file(file, 0, *get_sigmoid_exp(1))

def get_letter(ch: str):
    return to_bin_array(font_2[font_2_char.index(ch)])


# plot = LetterPlot(1)
# i = net.decode(np.array([0, 0, 0, 0, 0], dtype=np.float64))
# plot.draw([i])
l = font_2_char
encodings = net.encode(np.array(list(map(get_letter, l))))

print(encodings)

fig, ax = plt.subplots()
x, y = zip(*encodings[:, :2])
ax.scatter(x, y)

for _x, _y, letter in zip(x, y, l):
    ax.annotate(letter, (_x, _y))

plt.show()

