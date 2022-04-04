from enum import Enum, auto
from functools import cache
from typing import Generic, TypeVar
from utils import Generator, StopReason
from population import Population
from executor import CombinatorBuilder, Executor, entuple
from selection import *
from mutation import *
from crossover import *
from chromosome import *
from bag import *
from algorythm import GeneticAlgorythm
import json
from tqdm import tqdm

from more_itertools import take


def create_chromosome(l): return BagChromosome(l)


C = BagChromosome


class Parameters(NamedTuple):
    population_size: int
    population: Population[C]
    mutation: BinaryMutation[C]
    crossover: Crossover[bool, C]
    selection: Selection[C]
    algorythm: GeneticAlgorythm[bool, C]


@cache
def get_random_population(size: int):
    return Population(take(size, generate_valid_bags()))


combinations = CombinatorBuilder.initialize([10, 50, 100]) \
    .append_map(lambda p_c: (get_random_population(p_c), ))\
    .add_product(entuple([BinaryMutation(create_chromosome, p) for p in [0.01, 0.02, 0.05]]))\
    .add_product(entuple([OnePointCrossover(create_chromosome), NPointCrossover(create_chromosome, 2), UniformCrossover(create_chromosome)]))\
    .append_map(lambda p_c, p, m, c: (EliteSelection(p_c), ))\
    .append_map(lambda p_c, p, m, c, s: (GeneticAlgorythm(p_c, m, c, s),))\
    .map(lambda p_c, p, m, c, s, a: Parameters(p_c, p, m, c, s, a))


def stop_criteria(generations: int, previous_fitnesses: list[float], time_since_start: float):
    if generations > 1000:
        return StopReason.MaxGenerationCount
    # if time_since_start > 1:
        # return StopReason.MaxTimeExceeded
    last_fitnesses = previous_fitnesses[-10:]
    if len(previous_fitnesses) > 10 and max(last_fitnesses) - min(last_fitnesses) < 100:
        return StopReason.NotEnoughImprovement
    # if len(previous_fitnesses) > 10 and abs(max(previous_fitnesses) - min(previous_fitnesses)) < 5:
        # return StopReason.NotEnoughVariation
    return False


def evaluate(p: Parameters):
    generations = Generator(p.algorythm.run(p.population, stop_criteria))
    best_fitness = max(
        (p.best_chromosome.fitness for p in generations), default=0)
    return best_fitness, generations.stop_reason


def run():
    for i, (best_fitness, stop_reason) in Executor(combinations).run(evaluate):
        p = Parameters(*i)
        yield {'algorythm': p.algorythm.to_dict(), 'result': {'best_fitness': best_fitness, 'stop_reason': stop_reason.value}}


# data = dict(((p.population_size, type(p.mutation), type(p.crossover), type(p.selection)), o)
        # for p, o in ((Parameters(*i), o) for i, o in Executor(combinations).run(evaluate)))
# print(json.dumps(data, indent=2))
results = []

for v in tqdm(run(), total=len(combinations)):
    results.append(v)

print(json.dumps(results, indent=2))
