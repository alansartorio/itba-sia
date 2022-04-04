import random
from typing import Callable, Generic, TypeVar

import time

from selection import Selection
from crossover import Crossover
from mutation import Mutation
from population import Population
from chromosome import Chromosome

C = TypeVar('C', bound=Chromosome)
T = TypeVar('T')


def weighted_sample(c: list[T], key: Callable[[T], float]) -> T:
    mapped = tuple((v, key(v)) for v in c)
    total = sum(weight for _, weight in mapped)
    mapped = tuple((v, (weight / total) if total > 0 else (1 / len(c)))
                   for v, weight in mapped)
    r = random.random()

    for v, p in mapped:
        r -= p
        if r <= 0:
            return v

    # print(r)
    raise RuntimeError()


def weighted_multisample(c: list[T], count: int, key: Callable[[T], float]) -> tuple[T, ...]:
    if len(c) < count:
        raise AttributeError()

    picked = []

    for _ in range(count):
        p = weighted_sample(c, key)
        picked.append(p)
        c.remove(p)

    return tuple(picked)


StopCriteria = Callable[[int, list[float], float], bool]


class GeneticAlgorythm(Generic[T, C]):
    def __init__(self, mutation_operator: Mutation, crossover_operator: Crossover, selection_operator: Selection[C]) -> None:
        self.mutation_operator = mutation_operator
        self.crossover_operator = crossover_operator
        self.selection_operator = selection_operator

    def run(self, initial_population: Population[C], stop_criteria: StopCriteria):
        population = initial_population
        generations = 1
        previous_fitnesses = []
        start_time = time.process_time()
        stop_value = False
        while not stop_value:
            children: list[C] = []
            while len(children) < len(population):
                p1, p2 = weighted_multisample(
                    list(population), 2, lambda c: c.fitness)
                c1, c2 = self.crossover_operator.apply(p1, p2)
                c1 = self.mutation_operator.apply(c1)
                c2 = self.mutation_operator.apply(c2)
                if c1.is_valid:
                    children.append(c1)
                if c2.is_valid:
                    children.append(c2)

            population = self.selection_operator.apply(
                Population(list(population) + children))
            yield population
            generations += 1

            best_fitness = max(population, key=lambda c: c.fitness)
            previous_fitnesses.append(best_fitness.fitness)
            # if len(previous_fitnesses) > 10:
            # previous_fitnesses.pop(0)
            stop_value = stop_criteria(
                generations, previous_fitnesses, time.process_time() - start_time)
        return stop_value
