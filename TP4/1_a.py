import math
import pandas as pd
import numpy as np

class Kohonen:
    #sample_data_input es una entrada cualquiera de datos, se usa para inicializar pesos
    def __init__(self, sample_data_input: list[float], q_neurons: int, learning_rate: float, initial_R: int):
        if math.sqrt(q_neurons) ** 2 != q_neurons:
            raise Exception('Wrong number of neurons')
        self.matrix = [[[] for j in range(int(math.sqrt(q_neurons)))] for i in range(int(math.sqrt(q_neurons)))]
        self.weights = [[sample_data_input.to_numpy() for j in range(int(math.sqrt(q_neurons)))] for i in range(int(math.sqrt(q_neurons)))]
        self.learning_rate = learning_rate
        self.R = initial_R

    def get_min_distance(self, entry: pd.DataFrame) -> tuple[int,int]:
        country = entry.iloc[0]
        entry = entry.iloc[1:].to_numpy()
        min_distance = float('inf')
        min_i = float('inf')
        min_j = float('inf')

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                dist = np.linalg.norm(entry - self.weights[i][j])
                if dist < min_distance:
                    min_distance = dist
                    min_i = i
                    min_j = j
        return (min_i, min_j)

    def single_train(self, entry: pd.DataFrame):
        (row,col) = self.get_min_distance(entry)
        #TODO en base al radio actual actualizar pesos del vecindario
        #TODO definir si usar vecindario que toma diagonales o no
        #TODO achicar radio considerando que al final tiene que llegar a 1
                

    def train(self, data: pd.DataFrame):
        for i in range(len(data)):
            self.single_train(data.iloc[i])
        

data = pd.read_csv('europe.csv')
net = Kohonen(data.iloc[0].iloc[1:], 36, 0.001, 6)
net.train(data)