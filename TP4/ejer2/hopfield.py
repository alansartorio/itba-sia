from multiprocessing.dummy import Array
import numpy as np
import pandas as pd
import csv
from alphabet import Alphabet
from letter import Letter

class Hopfield:
    def __init__(self, n: int) -> None:
        self.size = n * n
        self.states = np.empty([self.size])
        self.weights = np.empty([self.size,self.size])
        self.patterns = []

    def set_weights(self):
        for i in range(self.size):
            for j in range(self.size):
                if j == i:
                    self.weights[i][j] = 0
                else:
                    suma = 0
                    for pattern in self.patterns:
                        suma +=  np.sum(pattern[i]*pattern[j])
                        self.weights[i][j] = (1/self.size) * suma
   
    def save_pattern(self, pattern: Letter) -> int:
        self.patterns.append(np.array(pattern.get_array()).flatten())
        return len(self.patterns)

    def calculate_hs(self) -> np.array:
        hs = np.zeros([self.size])
        for i in range(self.size):
            suma = 0
            for j in range(self.size):
                if i != j:
                    suma += self.weights[i][j] * self.states[j]
                    #suma += self.weights[i][j] * self.states[j]
            hs[i] = np.sum(suma)
        return hs

    def update_states(self, hs: np.array):
        changed = False
        for i in range(len(hs)):
            aux = self.states[i]
            self.states[i] = np.sign(hs[i])
            changed = aux != self.states[i]
        return changed
            
    def train(self, letter: Letter):
        self.states = np.array(letter.get_array()).flatten()
        changed = True
        i = 0
        while i < 50 and changed:
            print( changed)
            i += 1
            hs = self.calculate_hs()
            changed = self.update_states(hs)
            print(Letter.get_letter(self.states))


def change_matrix(matrix: np.array, prob: 0.2):
    mat = matrix.flatten()
    for n in range(len(mat)):
        if np.random.rand() < prob:
            mat[n] = -mat[n]

    return mat.reshape([5,5])
        
    
letters = Alphabet()
"""
for letter in letters.letters:
    print(letter)
    print("\n")
"""

hop = Hopfield(5)
print("Saved patterns: \n")
print(letters.letters[4])
print(letters.letters[8])
print(letters.letters[15])
print(letters.letters[16])

hop.save_pattern(letters.letters[4])
hop.save_pattern(letters.letters[8])
hop.save_pattern(letters.letters[15])
hop.save_pattern(letters.letters[16])

hop.set_weights()
print("My letter: \n")

letter = Letter(5)
modified_matrix = change_matrix(letters.letters[16].get_array(), 0.1)
letter.add(modified_matrix)
print(letter)
print("-----------------")
hop.train(letter)



    



