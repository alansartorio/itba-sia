

from mutation import BinaryMutation
from chromosome import BinaryChromosome
from crossover import Crossover, OnePointCrossover, NPointCrossover, UniformCrossover
from selection import *


crossover: UniformCrossover[bool, BinaryChromosome] = UniformCrossover(lambda l:BinaryChromosome(l))
a = BinaryChromosome([True] * 10)
b = BinaryChromosome([False] * 10)
# for _ in range(10):
    # print(crossover.apply(a, b))

for _ in range(10):
    print(BinaryMutation(0.1).apply(b))
