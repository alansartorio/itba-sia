from enum import Enum, auto
from typing import Generic, TypeVar
from utils import Generator
from population import Population
from executor import CombinatorBuilder, Executor, entuple
from selection import *
from mutation import *
from crossover import *
from chromosome import *
from bag import *
from algorythm import GeneticAlgorythm

from more_itertools import take

def create_chromosome(l): return BagChromosome(l)
C = BagChromosome
class Parameters(NamedTuple):
    population_size: int
    population: Population[C]
    mutation: Mutation[C]
    crossover: Crossover[bool, C]
    selection: Selection[C]


combinations = CombinatorBuilder.initialize([10, 50, 100]) \
                 .append_map(lambda p: (Population(take(p, generate_valid_bags())), ))\
                .add_product(entuple([BinaryMutation(create_chromosome, p) for p in [0.01, 0.02, 0.05]]))\
                 .add_product(entuple([OnePointCrossover(create_chromosome), NPointCrossover(create_chromosome, 2), UniformCrossover(create_chromosome)]))\
                 .append_map(lambda p_c, p, m, c: (EliteSelection(p_c), ))\
                 .map(lambda p_c, p, m, c, s: Parameters(p_c, p, m, c, s))

class StopReason(Enum):
    MaxGenerationCount = auto()
    MaxTimeExceeded = auto()
    NotEnoughVariation = auto()
    NotEnoughImprovement = auto()

def stop_criteria(generations: int, previous_fitnesses: list[float], time_since_start: float):
    if generations > 1000:
        return StopReason.MaxGenerationCount
    # if time_since_start > 1:
        # return StopReason.MaxTimeExceeded
    last_fitnesses = previous_fitnesses[-10:]
    if len(previous_fitnesses) > 10 and max(last_fitnesses) - min(last_fitnesses) < 5:
        return StopReason.NotEnoughImprovement
    # if len(previous_fitnesses) > 10 and abs(max(previous_fitnesses) - min(previous_fitnesses)) < 5:
        # return StopReason.NotEnoughVariation
    return False


def evaluate(p: Parameters):
    algorythm=GeneticAlgorythm(p.mutation, p.crossover, p.selection)
    generations = Generator(algorythm.run(p.population, stop_criteria))
    best_fitness = max((p.best_chromosome.fitness for p in generations), default=0)
    return best_fitness, generations.value


for i, o in Executor(combinations).run(evaluate):
    p = Parameters(*i)
    print(p.population_size, o)
