import numpy as np
import csv 
from letter import Letter

class Alphabet:
    def __init__(self) -> None:
        self.letters = self.get_letters()

    def get_letters(self):
        letters = []
        with open('patterns.tsv') as File:
            read_tsv = csv.reader(File, delimiter="\t")
            l =[]
            letter = Letter(5)
            for row in read_tsv:
                if len(row) == 0:
                    letter.add(l)
                    l=[]
                    letters.append(letter)
                    letter = Letter(5)
                    continue
                int_row = list(map(int, row))
                int_row = list(map(lambda x:  -1 if x == 0 else 1, int_row))
                for n in int_row:
                    if n == 0: n = -1
                l.append(int_row)
            return letters
