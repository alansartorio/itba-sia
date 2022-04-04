from functools import cached_property
import random
from typing import TextIO
from bag import BagChromosome, generate_valid_bags
from population import Population
from algorythm import GeneticAlgorythm
from mutation import BinaryMutation
from chromosome import BinaryChromosome
from crossover import Crossover, OnePointCrossover, NPointCrossover, UniformCrossover
from selection import *
from more_itertools import take


crossover: UniformCrossover[bool, BagChromosome] = UniformCrossover(
    lambda l: BagChromosome(l))
population_count = 10
selection = BoltzmannSelection(population_count, False)
mutation = BinaryMutation(lambda l: BagChromosome(l), 0.01)
initial_population = Population(take(population_count, generate_valid_bags()))

algorythm = GeneticAlgorythm(mutation, crossover, selection)

print("hi")
# TODO: Add more stop criteria parameters.
def stop_criteria(generations: int, previous_fitnesses: list[float], time_since_start: float):
    return generations > 1000


for g, population in enumerate(algorythm.run(initial_population, stop_criteria)):
    best = max(population, key=lambda c: c.fitness)
    print(best.fitness)
