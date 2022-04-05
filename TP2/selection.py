
from abc import ABC, abstractmethod
from asyncio import selector_events
from functools import partial
import random
from typing import Any, Generic, TypeVar, TypedDict
from population import Population
from chromosome import Chromosome
import numpy as np
from utils import weighted_multisample
from math import exp
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

SelectionParams = dict[str, Any]


class SelectionDict(TypedDict):
    type: str
    params: SelectionParams


class Selection(ABC, Generic[C]):
    def __init__(self, population_count: int) -> None:
        self.population_count = population_count

    @abstractmethod
    def apply(self, population: Population[C],
              generation_number: int) -> Population[C]: ...

    @classmethod
    def parse(cls, population_count: int, data: SelectionDict):
        # {"type": "RouletteSelection", "params": null}
        # {"type": "TournamentSelection", "params": {"replace": true, "threshold": 0.5}}
        selection_type = data['type']
        params = data['params']
        selection_class = {
            'EliteSelection': EliteSelection,
            'RankSelection': RankSelection,
            'RouletteSelection': RouletteSelection,
            'TournamentSelection': TournamentSelection,
            'BoltzmannSelection': BoltzmannSelection,
            'TruncatedSelection': TruncatedSelection,
        }[selection_type]
        return selection_class(population_count, **params)

    @abstractmethod
    def params_dict(self) -> SelectionParams: ...

    def to_dict(self):
        return SelectionDict(
            type={
                EliteSelection: 'EliteSelection',
                RankSelection: 'RankSelection',
                RouletteSelection: 'RouletteSelection',
                TournamentSelection: 'TournamentSelection',
                BoltzmannSelection: 'BoltzmannSelection',
                TruncatedSelection: 'TruncatedSelection',
            }[type(self)],
            params=self.params_dict()
        )


def sorted_population(population: Population[C]) -> list[C]:
    return sorted(population, key=lambda c: c.fitness, reverse=True)


class EliteSelection(Selection[C], Generic[C]):
    def apply(self, population: Population[C], generation_number: int) -> Population[C]:
        return Population(sorted_population(population)[:self.population_count])

    def params_dict(self) -> SelectionParams:
        return {}


def roulette_probabilities(population: Population[C]) -> list[float]:
    fitness_sum = sum(c.fitness for c in population)
    return [c.fitness / fitness_sum for c in population]


def temperature(generation: int, k: float, Tc: float = 15, T0: float = 100) -> float:
    answer = Tc + (T0 - Tc) * exp(-k * generation)
        
    return answer


def boltzmann_probabilities(population: Population[C], T) -> list[float]:
    fitness_sum = sum(exp((c.fitness/T)) for c in population)
    ans = [exp(c.fitness/T) / fitness_sum for c in population]
    return ans


class SelectionWithReplacement(Generic[C], Selection[C]):
    def __init__(self, population_count: int, replace: bool) -> None:
        self.replace = replace
        super().__init__(population_count)

    def params_dict(self) -> SelectionParams:
        return {'replace': self.replace}


class RouletteSelection(SelectionWithReplacement[C], Generic[C]):
    def apply(self, population: Population[C], generation_number: int) -> Population[C]:
        p = dict(zip(population, roulette_probabilities(population)))
        return Population(weighted_multisample(population, self.population_count, p.__getitem__, self.replace))


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
    def apply(self, population: Population[C], generation_number: int) -> Population[C]:
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

    def apply(self, population: Population[C], generation_number: int) -> Population[C]:
        original_population = list(population)
        new_population = []
        for i in range(self.population_count):
            winner = self.battle(random.sample(original_population, 4))
            if not self.replace:
                original_population.remove(winner)
            new_population.append(winner)
        return Population(new_population)

    def params_dict(self) -> SelectionParams:
        return {'replace': self.replace, 'threshold': self.threshold}


class BoltzmannSelection(SelectionWithReplacement[C], Generic[C]):
    def __init__(self, population_count: int, replace: bool, k: float, T0: float = 10000, Tc: float = 15) -> None:
        self.k = k
        self.T0 = T0
        self.Tc = Tc
        super().__init__(population_count, replace)

    def apply(self, population: Population[C], generation_number: int) -> Population[C]:
        temp = temperature(generation_number, k=self.k)
        # print('\t', temp)
        p = dict(zip(population, boltzmann_probabilities(population, temp)))
        # p = np.array(boltzmann_probabilities(population, temperature(generation_number)))

        return Population(weighted_multisample(population, self.population_count, p.__getitem__, self.replace))

    def params_dict(self) -> SelectionParams:
        return {'replace': self.replace, 'k': self.k, 'T0': self.T0, 'Tc': self.Tc}


class TruncatedSelection(Selection[C], Generic[C]):
    def __init__(self, population_count: int, truncate_count: int) -> None:
        self.truncate_count = truncate_count
        super().__init__(population_count)

    def apply(self, population: Population[C], generation_number: int) -> Population[C]:
        truncated = sorted_population(population)[:-self.truncate_count]
        return Population(random.sample(truncated, self.population_count))

    def params_dict(self) -> SelectionParams:
        return {'truncate_count': self.truncate_count}
