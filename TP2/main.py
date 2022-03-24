

from chromosome import BinaryChromosome
from crossover import Crossover, OnePointCrossover, NPointCrossover, UniformCrossover
from selection import *


crossover = UniformCrossover(BinaryChromosome)
a = BinaryChromosome([True] * 10)
b = BinaryChromosome([False] * 10)
for _ in range(10):
    print(crossover.apply(a, b))
