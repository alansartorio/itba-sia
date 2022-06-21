from activation_functions import get_sigmoid_exp
from simple_image_plot import ImagePlot, LetterPlot
from network import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np
from fonts import font_2, font_2_char, to_bin_array
import seaborn as sns
sns.set_theme()

import sys

with open('15_letters.txt') as file:
    net = AutoEncoder.load_from_file(file, 0, *get_sigmoid_exp(1))

def get_letter(ch: str):
    return to_bin_array(font_2[font_2_char.index(ch)])


# plot = LetterPlot(1)
# i = net.decode(np.array([0, 0, 0, 0, 0], dtype=np.float64))
# plot.draw([i])
l = 'DM'
f = get_letter(l[0])
k = get_letter(l[1])

f_code, k_code = map(lambda x:x.numpy(), net.encode(np.array([f, k])))

x = np.linspace(f_code[0], k_code[0], 8)
y = np.interp(x, (f_code[0], k_code[0]), (f_code[1], k_code[1]))
# print(interp)

fig, ax = plt.subplots()
ax.set_xlim((-0.2, 1.2))
ax.set_ylim((-0.2, 1.2))
ax.scatter(x, y)

fig.set_size_inches(4, 4)

for (_x, _y), letter in zip((f_code, k_code), l):
    ax.annotate(letter, (_x, _y))

plt.tight_layout()
fig.savefig(f'plots/interp-latent.png', dpi=200)


plot = LetterPlot(len(x))
plot.draw(net.decode(np.array(list(zip(x, y)))))
plt.tight_layout()
plot.fig.savefig(f'plots/interp.png', dpi=200)

# plt.show()

