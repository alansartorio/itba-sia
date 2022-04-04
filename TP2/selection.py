
from abc import ABC, abstractmethod
from asyncio import selector_events
import random
from typing import Generic, TypeVar
from population import Population
from chromosome import Chromosome
import numpy as np

__all__ = [
    'Selection',
    'EliteSelection',
    'RankSelection',
    'RouletteSelection',
    'TournamentSelection',
    'BoltzmannSelection',
    'TruncatedSelection',
]


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


class SelectionWithReplacement(Generic[C], Selection[C]):
    def __init__(self, population_count: int, replace: bool) -> None:
        self.replace = replace
        super().__init__(population_count)


class RouletteSelection(SelectionWithReplacement[C], Generic[C]):
    def apply(self, population: Population[C]) -> Population[C]:
        population_1d = np.empty(len(population), dtype=object)
        population_1d[:] = population
        return Population(np.random.choice(population_1d, self.population_count, replace=self.replace, p=np.array(roulette_probabilities(population))))


def rank_probabilities(population: Population[C]) -> list[float]:
    sorted_list = sorted_population(population)
    total_population_count = len(population)

    def rank(i):
        return sorted_list.index(i) + 1

    def f1(i):
        return (total_population_count - rank(i)) / total_population_count
    sum_f1 = sum(f1(i) for i in population)
    return [f1(i) / sum_f1 for i in population]


class RankSelection(SelectionWithReplacement[C], Generic[C]):
    def apply(self, population: Population[C]) -> Population[C]:
        population_1d = np.empty(len(population), dtype=object)
        population_1d[:] = population
        return Population(np.random.choice(population_1d, self.population_count, replace=self.replace, p=np.array(rank_probabilities(population))))


class TournamentSelection(SelectionWithReplacement[C], Generic[C]):
    def __init__(self, population_count: int, replace: bool, threshold: float) -> None:
        self.threshold = threshold
        super().__init__(population_count, replace)

    def battle(self, competitors: list[C]) -> C:
        def compare_fitness(c1: C, c2: C, get_best: bool) -> C:
            if(c1.fitness > c2.fitness):
                return c1 if get_best else c2
            else:
                return c2 if get_best else c1
        rand = random.random()
        couple1_winner = compare_fitness(
            competitors[0], competitors[1], rand < self.threshold)
        rand = random.random()
        couple2_winner = compare_fitness(
            competitors[2], competitors[3], rand < self.threshold)
        rand = random.random()
        return compare_fitness(couple1_winner, couple2_winner, rand < self.threshold)

    def apply(self, population: Population[C]) -> Population[C]:
        original_population = list(population)
        new_population = []
        for i in range(self.population_count):
            winner = self.battle(random.sample(original_population, 4))
            if not self.replace:
                original_population.remove(winner)
            new_population.append(winner)
        return Population(new_population)


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
