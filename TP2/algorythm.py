import random
from typing import Any, Callable, Generic, TypeVar, TypedDict

import time
from utils import weighted_multisample

from selection import Selection, SelectionDict
from crossover import Crossover, CrossoverDict
from mutation import BinaryMutation, BinaryMutationDict, Mutation
from population import Population
from chromosome import BinaryChromosome, Chromosome, CreateChromosome

C = TypeVar('C', bound=Chromosome)
T = TypeVar('T')

StopCriteria = Callable[[int, list[float], float], Any]


class GeneticAlgorythmDict(TypedDict):
    population_count: int
    mutation: BinaryMutationDict
    crossover: CrossoverDict
    selection: SelectionDict


class GeneticAlgorythm(Generic[T, C]):
    def __init__(self, population_count: int, mutation_operator: Mutation, crossover_operator: Crossover, selection_operator: Selection[C]) -> None:
        self.population_count = population_count
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
                children.append(c1)
                children.append(c2)
            new_population = Population(list(population) + children)
            population = self.selection_operator.apply(new_population, generations)
            yield population
            generations += 1

            best_fitness = max(population, key=lambda c: c.fitness)
            previous_fitnesses.append(best_fitness.fitness)
            # if len(previous_fitnesses) > 10:
            # previous_fitnesses.pop(0)
            stop_value = stop_criteria(
                generations, previous_fitnesses, time.process_time() - start_time)
        return stop_value

    @classmethod
    def parse_binary(cls, create_chromosome: CreateChromosome[bool, BinaryChromosome], data: GeneticAlgorythmDict):
        population_count = data['population_count']
        return cls(population_count, BinaryMutation.parse(create_chromosome, data['mutation']), Crossover.parse(
            create_chromosome, data['crossover']), Selection.parse(population_count, data['selection']))

    def to_dict(self):
        return GeneticAlgorythmDict(
            population_count=self.population_count,
            mutation=self.mutation_operator.to_dict(),
            crossover=self.crossover_operator.to_dict(),
            selection=self.selection_operator.to_dict()
        )
