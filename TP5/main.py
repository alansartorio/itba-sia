
import matplotlib.pyplot as plt
from itertools import count
from activation_functions import get_relu, lineal_func
from simple_image_plot import ImagePlot, LetterPlot
from fonts import print_char, to_bin_array, font_2, font_2_char
from single_data import SingleData
from network import AutoEncoder
import numpy as np
from activation_functions import get_sigmoid_exp, get_sigmoid_tanh
from add_noise import add_noise
import random

import sys

# letter_count = int(sys.argv[1])
letter_count = 5

# letters = ('H', 'I', 'A', 'O', 'T')
# letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:letter_count]
# letters = font_2_char[:letter_count]
offset = 1
letters = font_2_char[offset:letter_count + offset]
# letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# data = [to_bin_array(letter) for letter in font_2]
images = [to_bin_array(font_2[font_2_char.index(letter)]) for letter in letters]
for l in images:
    print_char(l)
# l = images[1]
# print_char(l)
# for i in range(10):
    # print_char(add_noise(l, probability=0.2, amp=1))
# exit(0)
input_size = images[0].size
# layers = (30, 27, 25, 22, 20, 17, 15, 12, 10)
latent_size = 10
# layers = tuple(range(35, latent_size, -1))
# layers = tuple(range(35, latent_size, -10))
# layers = (100, 40, 10)
# layers = (30, 30)
layers = (30, 20)

net = AutoEncoder.with_random_weights(
    0.001, input_size, (*layers, latent_size, *layers[::-1], input_size), len(layers) + 1, *get_sigmoid_exp(1))
# net = AutoEncoder.with_random_weights(
    # 0.01, input_size, (*layers, latent_size, *layers[::-1], input_size), len(layers) + 1, lineal_func, lambda x: 1)


# data = [SingleData(input, input) for input in data]
plot = LetterPlot(len(images))

try:
    for i in count():
        data = [SingleData(add_noise(image, probability=random.uniform(0, 0.2), amp=1), image) for image in images]
        for _ in range(10):
            net.train(data)
            error = net.mean_squared_error(data)
            print(error)

        if error < 0.5:
            break

        # print(code, decoded)
        # if i % 10 == 0:
            # # for plot, data2 in zip(plots, data):
            # # code = net.encode(data2.inputs)
            # # decoded = net.decode(code)
        plot.draw(net.evaluate(np.array(list(d.inputs for d in data))))
        # print_char(decoded)
except KeyboardInterrupt:
    pass
finally:
    with open(f'denoise_2.txt', 'w') as file:
        net.save_weights(file)

# plt.show()
