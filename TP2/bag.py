

from abc import abstractproperty
from collections import namedtuple
from functools import cached_property
from more_itertools import take
import random
from typing import Iterable, NamedTuple, TextIO
from population import Population
from chromosome import BinaryChromosome
from dataclasses import dataclass

BagItem = namedtuple('BagItem', ['weight', 'value'])


@dataclass
class ProblemData:
    max_weight: int
    items: tuple[BagItem, ...]


data: list[BagItem] = []


def read_ints(file: TextIO):
    for line in file:
        yield map(int, line.split())


with open('mochila.txt') as file:
    parsed = read_ints(file)
    count, max_weight = next(parsed)
    for value, weight in parsed:
        data.append(BagItem(weight, value))


class BagChromosome(BinaryChromosome):
    problem_data: ProblemData = ProblemData(max_weight, tuple(data))

    @cached_property
    def fitness(self):
        return sum(item.value for present, item in zip(self, self.problem_data.items) if present)

    @cached_property
    def is_valid(self):
        return sum(item.weight for present, item in zip(self, self.problem_data.items) if present) < self.problem_data.max_weight

    @classmethod
    def random(cls, p: float):
        return cls(random.random() < p for _ in range(len(cls.problem_data.items)))


def generate_valid_bags():
    while True:
        bag = BagChromosome.random(0.01)
        if bag.is_valid:
            yield bag

class BagPopulation(Population):
    @classmethod
    def random(cls, population_count: int):
        return Population(take(population_count, generate_valid_bags()))
