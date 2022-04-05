from enum import Enum, auto
from functools import cache, partial
from typing import Callable, Generic, TypeVar
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


def generate_combinations(population_counts: list[int], mutation_probability: list[float], crossovers: list[Crossover], selections: list[Callable[[int], Selection]]):
    return CombinatorBuilder.initialize(population_counts)\
        .append_map(lambda p_c: (get_random_population(p_c), ))\
        .add_product(entuple([BinaryMutation(create_chromosome, p) for p in mutation_probability]))\
        .add_product(entuple(crossovers))\
        .product_map(lambda p_c, p, m, c: entuple([selection(p_c) for selection in selections]), len(selections))\
        .append_map(lambda p_c, p, m, c, s: (GeneticAlgorythm(p_c, m, c, s),))\
        .map(lambda p_c, p, m, c, s, a: Parameters(p_c, p, m, c, s, a))\
        .repeat(20)


def stop_criteria(generations: int, previous_fitnesses: list[float], time_since_start: float):
    if generations > 500:
        return StopReason.MaxGenerationCount
    # if time_since_start > 1:
        # return StopReason.MaxTimeExceeded
    prev_fitnesses = previous_fitnesses[:-25]
    new_fitnesses = previous_fitnesses[-25:]
    if len(previous_fitnesses) > 25 and max(prev_fitnesses) >= max(new_fitnesses):
        return StopReason.NotEnoughImprovement
    # if len(previous_fitnesses) > 10 and abs(max(previous_fitnesses) - min(previous_fitnesses)) < 5:
        # return StopReason.NotEnoughVariation
    return False


def evaluate(p: Parameters):
    generations = Generator(p.algorythm.run(p.population, stop_criteria))
    fitnesses = []
    for pop in generations:
        fitnesses.append(pop.best_chromosome.fitness)
        # print(pop.best_chromosome.fitness)
        # best_fitness = max(
        # (p.best_chromosome.fitness for p in generations), default=0)
    return fitnesses, generations.stop_reason


def generate_data(plot_name: str, combinations, show_in_legend: tuple[str]):

    def run():
        for i, (fitnesses, stop_reason) in Executor(combinations).run(evaluate):
            p = Parameters(*i)
            if stop_reason == StopReason.NotEnoughImprovement:
                fitnesses = fitnesses[:]
            yield {'algorythm': p.algorythm.to_dict(), 'result': {'fitnesses': fitnesses, 'stop_reason': stop_reason.value}}

    results = []

    for v in tqdm(run(), total=len(combinations), leave=False):
        results.append(v)

    with open(f'data/{plot_name}.json', 'w') as file:
        json.dump({
            'show_in_legend': show_in_legend,
            'results': results,
        }, file, indent=2)


best_population_size = 500
best_mutation = 0.005
best_crossover = OnePointCrossover(create_chromosome)
population_sizes = [62, 125, 250, 500, 1000]
def best_selection(p): return EliteSelection(p)


files = {
    'population_variable': (('Population Size', ), population_sizes, [best_mutation], [best_crossover], [best_selection]),
    'mutation_probability': (('Mutation Probability', ), [best_population_size], [0.0025, 0.005, 0.01, 0.02, 0.04], [best_crossover], [best_selection]),
    'crossover_variable': (('Crossover', ), [best_population_size], [best_mutation], [OnePointCrossover(create_chromosome), NPointCrossover(create_chromosome, 2), NPointCrossover(create_chromosome, 3), UniformCrossover(create_chromosome)], [best_selection]),
    'selection_variable': (('Selection', ), [best_population_size], [best_mutation], [best_crossover], [
        lambda p:EliteSelection(p),
        lambda p:RankSelection(p, False),
        lambda p:RouletteSelection(p, False),
        lambda p:TournamentSelection(p, False, 0.8),
        lambda p:BoltzmannSelection(p, False, 0.001),
        lambda p:TruncatedSelection(p, 100)
    ]),
    'boltzmann_k': (('Selection', ), [best_population_size], [best_mutation], [best_crossover], [
        partial(BoltzmannSelection, replace=False, k=k, T0=10000) for k in [0.001, 0.01, 0.1, 1, 10]
    ]),
    'tournament': (('Selection', ), [best_population_size], [best_mutation], [best_crossover], [
        lambda p:TournamentSelection(p, False, 0.9),
        lambda p:TournamentSelection(p, True, 0.9),
        lambda p:TournamentSelection(p, False, 0.5),
        lambda p:TournamentSelection(p, True, 0.5),
        lambda p:TournamentSelection(p, False, 0.8),
        lambda p:TournamentSelection(p, True, 0.8),
    ]),
    'truncated': (('Selection', ), [best_population_size], [best_mutation], [best_crossover], [
        partial(TruncatedSelection, truncate_count=count) for count in [10, 20, 100, 200]
    ]),
}

for file, (show_in_legend, *params) in tqdm(files.items()):
    generate_data(file, generate_combinations(*params), show_in_legend)

# .add_product(entuple([OnePointCrossover(create_chromosome), NPointCrossover(create_chromosome, 2), UniformCrossover(create_chromosome)]))\
