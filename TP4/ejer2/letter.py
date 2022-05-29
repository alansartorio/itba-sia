import numpy as np
from enum import Enum


class Colors(Enum):
    RED = 'r'
    WHITE = 'w'

    def __str__(self):
        return '\033[1;' + {
            Colors.RED: '31m',
            Colors.WHITE: '37m',
        }[self] + '█\033[0m'




class Letter:
    def __init__(self, size: int) -> None:
        self.letter = np.empty([size,size])
        self.size = size

    def add(self, matrix):
        self.letter = matrix


    def get_array(self) -> np.array:
        return np.array(self.letter)
        
    def __str__(self):
        printable = ""
        for row in self.letter:
            for num in row:
                if num == 0:
                    printable += '\033[1;37m█\033[0m'
                else:
                    printable += '\033[1;31m█\033[0m'
            printable += '\n'

        return printable

