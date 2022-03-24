
from abc import ABC, abstractmethod
import random
from typing import Generic, TypeVar
from chromosome import Chromosome
from more_itertools import take

C = TypeVar('C', bound=Chromosome)


class Selection(ABC, Generic[C]):
    def __init__(self, population_count: int) -> None:
        self.population_count = population_count

    @abstractmethod
    def apply(self, population: list[C]) -> list[C]: ...


def sorted_population(population: list[C]) -> list[C]:
    return sorted(population, key=lambda c: c.fitness)


class EliteSelection(Selection[C], Generic[C]):
    def apply(self, population: list[C]) -> list[C]:
        return list(sorted_population(population)[:self.population_count])


class StocasticSelection(Selection[C], Generic[C]):
    def apply(self, population: list[C]) -> list[C]:
        ...


class RouletteSelection(Selection[C], Generic[C]):
    def apply(self, population: list[C]) -> list[C]:
        ...


class RankSelection(Selection[C], Generic[C]):
    def apply(self, population: list[C]) -> list[C]:
        ...


class TournamentSelection(Selection[C], Generic[C]):
    def apply(self, population: list[C]) -> list[C]:
        ...


class BoltzmannSelection(Selection[C], Generic[C]):
    def apply(self, population: list[C]) -> list[C]:
        ...


class TruncatedSelection(Selection[C], Generic[C]):
    def __init__(self, population_count: int, truncate_count: int) -> None:
        self.truncate_count = truncate_count
        super().__init__(population_count)

    def apply(self, population: list[C]) -> list[C]:
        truncated = sorted_population(population)[:-self.truncate_count]
        return random.sample(truncated, self.population_count)
