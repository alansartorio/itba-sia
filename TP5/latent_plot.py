
from activation_functions import get_sigmoid_exp
from simple_image_plot import ImagePlot, LetterPlot
from network import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np
from fonts import font_2, font_2_char, to_bin_array
import seaborn as sns
sns.set_theme()

import sys

model = sys.argv[1]
letter_count = int(sys.argv[2])

with open(model) as file:
    net = AutoEncoder.load_from_file(file, 0, *get_sigmoid_exp(1))

def get_letter(ch: str):
    return to_bin_array(font_2[font_2_char.index(ch)])


# plot = LetterPlot(1)
# i = net.decode(np.array([0, 0, 0, 0, 0], dtype=np.float64))
# plot.draw([i])
l = font_2_char[:letter_count]

encodings = net.encode(np.array(list(map(get_letter, l))))
print(encodings)

fig, ax = plt.subplots()
x, y = zip(*encodings[:, :2])
ax.set_xlim((-0.2, 1.2))
ax.set_ylim((-0.2, 1.2))
ax.scatter(x, y)

fig.set_size_inches(4, 4)

for _x, _y, letter in zip(x, y, l):
    ax.annotate(letter, (_x, _y))

plt.tight_layout()
fig.savefig(f'plots/{model}.png', dpi=200)

# plot = LetterPlot(letter_count)
# plot.draw(net.decode(encodings))
# plt.tight_layout()
# plot.fig.savefig(f'plots/{model}-letters.png', dpi=200)

# plt.show()

