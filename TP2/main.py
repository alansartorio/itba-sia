from functools import cached_property
import random
from typing import Optional, TextIO, TypedDict
from utils import StopReason
from bag import BagChromosome, BagPopulation, generate_valid_bags
from population import Population
from algorythm import GeneticAlgorythm, GeneticAlgorythmDict
from mutation import BinaryMutation
from chromosome import BinaryChromosome
from crossover import Crossover, OnePointCrossover, NPointCrossover, UniformCrossover
from selection import *
from more_itertools import take
import json


class ConfigDict(TypedDict):
    max_generations: Optional[int]
    max_time: Optional[float]
    algorythm: GeneticAlgorythmDict


with open('input.json', 'r') as file:
    data: ConfigDict = json.load(file)

alg = GeneticAlgorythm.parse_binary(
    lambda c: BagChromosome(c),
    data['algorythm']
)
initial_population = BagPopulation.random(alg.population_count)

def stop_criteria(generations: int, previous_fitnesses: list[float], time_since_start: float):
    if data['max_generations'] is not None and generations > data['max_generations']:
        return StopReason.MaxGenerationCount
    if data['max_time'] is not None and time_since_start > data['max_time']:
        return StopReason.MaxTimeExceeded
    return False

for p in alg.run(initial_population, stop_criteria):
    print(p.best_chromosome.fitness)
