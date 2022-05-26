import numpy as np
import pandas as pd
import csv
import statistics


class Oja:
    def __init__(self, data, registers, learning_rate: float, size: int) -> None:
        self.data = np.array(data)
        self.registers = registers
        self.size = size
        self.learning_rate = learning_rate
        self.weights = np.zeros([self.size])

    def randomize_weights(self, amplitude: float = 0.01):
        self.weights[:] = (np.random.rand(*self.weights.shape) * 2 - 1) * amplitude
   
    def train(self, data):
        for i in range(self.size):
            s = np.sum(data[i] * self.weights)
            self.weights += self.learning_rate * s * (data[i] - s * self.weights)

def standarize(data):
    cols = np.empty([7,28])
    for i in range(7):
        col = np.array([row[i] for row in data])
        mean = statistics.mean(col)
        cols[i] = (col - mean)/ statistics.stdev(col)
    return cols.transpose()

data = []
with open('europe.csv', newline='\n') as File:
    reader = csv.reader(File)
    header = next(reader)
    for country, *row in reader:
        data.append([country] + list(map(float, row))) 

countries = [row[0] for row in data]
data = [row[1:] for row in data]
original_df = pd.DataFrame(data, columns=header[1:], index=countries)
original_df

data = standarize(data)
net = Oja(data, countries, 0.001, 7)
net.randomize_weights()
for i in range(5000):
    net.train(data)
    print(net.weights)
print(net.weights)
