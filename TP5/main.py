
from itertools import count
from simple_image_plot import ImagePlot
from fonts import print_char, to_bin_array, font_2
from single_data import SingleData
from network import AutoEncoder
import numpy as np
from activation_functions import get_sigmoid_exp, get_sigmoid_tanh

# letters = ('H', 'I', 'A', 'O', 'T') 
letter_count = 6
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:letter_count]
# letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# data = [to_bin_array(letter) for letter in font_2]
data = [to_bin_array(font_2[ord(letter) - ord('@')]) for letter in letters]
for l in data:
    print_char(l)
input_size = data[0].size
# layers = (30, 27, 25, 22, 20, 17, 15, 12, 10)
latent_size = 35
# layers = tuple(range(35, latent_size, -1))
# layers = tuple(range(35, latent_size, -10))
# layers = (30,)
# layers = (30, 30)
layers = ()

net = AutoEncoder.with_random_weights(input_size, (*layers, latent_size, *layers[::-1], input_size), len(layers) + 1, *get_sigmoid_exp(1))


import matplotlib.pyplot as plt

data = [SingleData(input, input) for input in data]
plot = ImagePlot(len(data), 5, 7, (0, 5), (0, 7), 0, 1)

for i in count():
    net.train(data)
    print(net.mean_squared_error(data))

    # print(code, decoded)
    # if i % 500 == 0:
        # # for plot, data2 in zip(plots, data):
            # # code = net.encode(data2.inputs)
            # # decoded = net.decode(code)
    plot.draw(net.evaluate(np.array(list(d.inputs for d in data))))
    # print_char(decoded)

plt.show()
