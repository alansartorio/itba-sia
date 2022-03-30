
from abc import ABC, abstractmethod
import random
from typing import Generic, TypeVar
from population import Population
from chromosome import Chromosome
import numpy as np

C = TypeVar('C', bound=Chromosome)


class Selection(ABC, Generic[C]):
    def __init__(self, population_count: int) -> None:
        self.population_count = population_count

    @abstractmethod
    def apply(self, population: Population[C]) -> Population[C]: ...


def sorted_population(population: Population[C]) -> list[C]:
    return sorted(population, key=lambda c: c.fitness, reverse=True)


class EliteSelection(Selection[C], Generic[C]):
    def apply(self, population: Population[C]) -> Population[C]:
        return Population(sorted_population(population)[:self.population_count])


def roulette_probabilities(population: Population[C]) -> list[float]:
    fitness_sum = sum(c.fitness for c in population)
    return [c.fitness / fitness_sum for c in population]

class RouletteSelection(Selection[C], Generic[C]):
    def apply(self, population: Population[C]) -> Population[C]:
        population_1d = np.empty(len(population),dtype=object)
        population_1d[:] = population
        return Population(np.random.choice(population_1d, self.population_count, replace=False, p=roulette_probabilities(population)))


# TODO: Implement
class RankSelection(Selection[C], Generic[C]):
    def apply(self, population: Population[C]) -> Population[C]:
        ...


# TODO: Implement
class TournamentSelection(Selection[C], Generic[C]):
    def apply(self, population: Population[C]) -> Population[C]:
        ...


# TODO: Implement
class BoltzmannSelection(Selection[C], Generic[C]):
    def apply(self, population: Population[C]) -> Population[C]:
        ...


class TruncatedSelection(Selection[C], Generic[C]):
    def __init__(self, population_count: int, truncate_count: int) -> None:
        self.truncate_count = truncate_count
        super().__init__(population_count)

    def apply(self, population: Population[C]) -> Population[C]:
        truncated = sorted_population(population)[:-self.truncate_count]
        return Population(random.sample(truncated, self.population_count))
