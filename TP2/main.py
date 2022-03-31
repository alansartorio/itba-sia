from functools import cached_property
from typing import TextIO
from algorythm import GeneticAlgorythm
from mutation import BinaryMutation
from chromosome import BinaryChromosome
from crossover import Crossover, OnePointCrossover, NPointCrossover, UniformCrossover
from selection import *
from more_itertools import take

data = []


def read_ints(file: TextIO):
    for line in file:
        yield map(int, line.split())


with open('mochila.txt') as file:
    parsed = read_ints(file)
    count, max_weight = next(parsed)
    for value, weight in parsed:
        data.append((weight, value))


class BagChromosome(BinaryChromosome):
    @cached_property
    def fitness(self):
        return sum(value for present, (_, value) in zip(self, data) if present)

    @cached_property
    def is_valid(self):
        return sum(weight for present, (weight, _) in zip(self, data) if present) < max_weight

    @classmethod
    def random(cls, length: int, p: float):
        return cls(random.random() < p for _ in range(length))


def generate_valid_bags(length: int):
    while True:
        bag = BagChromosome.random(length, 0.01)
        if bag.is_valid:
            yield bag


crossover: UniformCrossover[bool, BagChromosome] = UniformCrossover(
    lambda l: BagChromosome(l))
population_count = 10
selection = TournamentSelection(population_count, False, 0.8)
mutation = BinaryMutation(lambda l: BagChromosome(l), 0.01)
initial_population = Population(
    take(population_count, generate_valid_bags(len(data))))

algorythm = GeneticAlgorythm(mutation, crossover, selection)


# TODO: Add more stop criteria parameters (time, iterations since last fitness improvement).
# Llega un punto en que deja de mejorar el best fitness, e incluso empieza a bajar. Podria ser un
# criterio de corte. El tema es que habria que hacerlo generico y que considere el mejor fitness actual
# y los mejores fitness anteriores.
def stop_criteria(generations: int, previous_fitnesses: list[float], time_since_start: float):
    print(generations, previous_fitnesses, time_since_start)
    return generations > 1000


for g, population in enumerate(algorythm.run(initial_population, stop_criteria)):
    best = max(population, key=lambda c: c.fitness)
    print(best.fitness)
