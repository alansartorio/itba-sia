from itertools import product
import sys
from single_data import SingleData
from image_plot import ImageEvaluationPlot
from network import Network
from activation_functions import get_sigmoid_exp, get_sigmoid_tanh
from cv2 import cv2
import numpy as np



image_filename = sys.argv[1]
image = cv2.imread(image_filename)

image = cv2.resize(image, (16, 16))
image = image[:,:,::-1].astype(np.float64) / 256 * 2 - 1

data = []
for y, x in product(range(image.shape[0]), range(image.shape[1])):
    # print(x, y, c)
    data.append(SingleData(np.array([x, y], dtype=np.float64) / image.shape[0:2] * 2 - 1, image[y, x, :]))
    # print(np.indices(image.shape).shape)
print(data)
# data = 

model = Network.with_zeroed_weights(2, (3, 3), *get_sigmoid_tanh(10))
model.randomize_weights(0.001)
plot = ImageEvaluationPlot(model, *image.shape[:-1], (-1, 1), (-1, 1))


error = model.error(data)
while error > 0:
    lr = min(error / 100, 0.1)
    # lr = 0.0001
    model.train(lr, data)
    plot.draw()
    error = model.error(data)
    print(error)
