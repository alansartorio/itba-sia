from functools import cached_property
from typing import TypeVar
from chromosome import Chromosome

__all__ = ['Population']

C = TypeVar('C', bound=Chromosome)


class Population(tuple[C]):
    @cached_property
    def best_chromosome(self) -> C:
        return max(self, key=lambda c: c.fitness)
