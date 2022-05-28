import math
import pandas as pd
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class Kohonen:
    #sample_data_input es una entrada cualquiera de datos, se usa para inicializar pesos
    def __init__(self, sample_data_input: list[float], q_neurons: int, learning_rate: float, initial_R: float):
        if math.sqrt(q_neurons) ** 2 != q_neurons:
            raise Exception('Wrong number of neurons')
        self.weights = [[sample_data_input.to_numpy() for j in range(int(math.sqrt(q_neurons)))] for i in range(int(math.sqrt(q_neurons)))]
        self.learning_rate = learning_rate
        self.R = initial_R

    def get_min_distance(self, entry: pd.DataFrame) -> tuple[int,int]:
        country = entry.iloc[0]
        entry = entry.iloc[1:].to_numpy()
        min_distance = float('inf')
        min_i = float('inf')
        min_j = float('inf')

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                dist = np.linalg.norm(entry - self.weights[i][j])
                if dist < min_distance:
                    min_distance = dist
                    min_i = i
                    min_j = j
        return (min_i, min_j)

    def getNeighbors(self, row: int, col: int) -> list[tuple[int,int]]:
        flooredR = math.floor(self.R)
        neighbors = []
        #TODO incluir o no a la propia neurona?

        #Direcciones normales
        for i in range(row-flooredR, row+flooredR+1):
            for j in range(col-flooredR, col+flooredR+1):
                if i >= 0 and j >= 0 and i < len(self.weights) and j < len(self.weights[i]):
                    neighbors.append((i,j))
        #Diagonales
        for i in range(1,flooredR):
            if row-i >= 0 and col-i >= 0 and row-i < len(self.weights) and col-i < len(self.weights[row-i]):
                neighbors.append((row-i,col-i))
            if row-i >= 0 and col+i >= 0 and row-i < len(self.weights) and col+i < len(self.weights[row-i]):
                neighbors.append((row-i,col+i))
            if row+i >= 0 and col-i >= 0 and row+i < len(self.weights) and col-i < len(self.weights[row+i]):
                neighbors.append((row+i,col-i))
            if row+i >= 0 and col+i >= 0 and row+i < len(self.weights) and col+i < len(self.weights[row+i]):
                neighbors.append((row+i,col+i))


        return neighbors

    def single_train(self, entry: pd.DataFrame):
        (row,col) = self.get_min_distance(entry)
        neighbors = self.getNeighbors(row,col)
        for neighbor in neighbors:
            self.weights[neighbor[0]][neighbor[1]] = self.weights[neighbor[0]][neighbor[1]] + self.learning_rate * (entry.iloc[1:] - self.weights[neighbor[0]][neighbor[1]])
        
        
        
                

    def train(self, data: pd.DataFrame, epochs: int):
        vars = len(data.iloc[0].iloc[1:])
        rDecreaseRate = (self.R-1) / (epochs*vars)
        nDecreaseRate = (self.learning_rate-0.0001) / (epochs*vars)
        for epoch in range(epochs):
            for i in range(len(data)):
                self.single_train(data.iloc[i])
                self.R -= rDecreaseRate
                self.learning_rate -= nDecreaseRate

    def evaluate(self, entry: pd.DataFrame) -> tuple[int,int]:
        (row,col) = self.get_min_distance(entry)
        return (row,col)

    def avgNeighDist(self, i: int, j: int) -> float:
        dist = 0
        cant = 0
        if i > 1:
            dist += np.linalg.norm(self.weights[i][j] - self.weights[i-1][j])
            cant += 1
        if i < len(self.weights)-1:
            dist += np.linalg.norm(self.weights[i][j] - self.weights[i+1][j])
            cant += 1
        if j > 1:
            dist += np.linalg.norm(self.weights[i][j] - self.weights[i][j-1])
            cant += 1
        if j < len(self.weights[i])-1:
            dist += np.linalg.norm(self.weights[i][j] - self.weights[i][j+1])
            cant += 1
            
        return dist/cant
        
data = pd.read_csv('europe.csv')
scaler = StandardScaler()
train_data = scaler.fit_transform(data.iloc[1:,1:])
train_data = pd.DataFrame(train_data)


q_neurons = 25
net = Kohonen(data.iloc[0].iloc[1:], q_neurons, 0.01, 5.0)
vars = len(data.iloc[0].iloc[1:])
net.train(data, 500 * vars)



plt.grid()
plt.xticks(np.arange(0, int(math.sqrt(q_neurons))+1, 1))
plt.yticks(np.arange(0, int(math.sqrt(q_neurons))+1, 1))
frame = plt.gca()
frame.axes.xaxis.set_ticklabels([])
frame.axes.yaxis.set_ticklabels([])
map = [[[] for j in range(int(math.sqrt(q_neurons)))] for i in range(int(math.sqrt(q_neurons)))]
for i in range(len(data)):
    (row,col) = net.evaluate(data.iloc[i])
    map[row][col].append(data.iloc[i].iloc[0])

for row in range(len(map)):
    for col in range(len(map[row])):
        ydiff = 0.15
        for country in map[row][col]:
            plt.text(col+0.15, 5 - row - ydiff,country, fontsize=8)
            ydiff += 0.2

plt.title('Agrupaciones de paises resultantes')
plt.savefig('1_a_agrupaciones.png')
plt.cla()

cant_map = [[len(map[i][j]) for j in range(len(map[i]))] for i in range(len(map))]
ax = sns.heatmap( np.array(cant_map) , linewidth = 0.5 , cmap = 'coolwarm' )  
plt.title( "Mapa de calor: Cant. paises agrupados por neurona" )
frame = plt.gca()
frame.axes.xaxis.set_ticklabels([])
frame.axes.yaxis.set_ticklabels([])

plt.savefig('1_a_mapa_cantidades.png')
plt.cla()
plt.clf()


dist_map = [[net.avgNeighDist(i,j) for j in range(int(math.sqrt(q_neurons)))] for i in range(int(math.sqrt(q_neurons)))]
dist_map = list(reversed(dist_map))
ax = sns.heatmap( np.array(dist_map) , linewidth = 0.5 , cmap = 'coolwarm' )  
plt.title( "Mapa de calor: Distancia promedio" )
frame = plt.gca()
frame.axes.xaxis.set_ticklabels([])
frame.axes.yaxis.set_ticklabels([])
plt.savefig('1_a_mapa_distancias.png')