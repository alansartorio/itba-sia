
from abc import abstractmethod
from typing import Any, Callable, Generic, TypeVar
from chromosome import T, Chromosome
from random import randint, randrange


T = TypeVar('T')
C = TypeVar('C', bound=Chromosome)


class Crossover(Generic[T, C]):
    def __init__(self, create_chromosome: Callable[[list[T]], C]) -> None:
        self.create_chromosome = create_chromosome

    @abstractmethod
    def apply(self, a: C, b: C) -> tuple[C, C]: ...

class OnePointCrossover(Generic[T, C], Crossover[T, C]):
    def apply(self, a: C, b: C) -> tuple[C, C]:
        r = randint(0, len(a))
        c1 = a[:r] + b[r:]
        c2 = b[:r] + a[r:]
        return self.create_chromosome(c1), self.create_chromosome(c2)

class TwoPointCrossover(Generic[T, C], Crossover[T, C]):
    def apply(self, a: C, b: C) -> tuple[C, C]:
        one = OnePointCrossover(self.create_chromosome)
        c1, c2 = one.apply(a, b)
        c1, c2 = one.apply(c1, c2)
        return c1, c2

class UniformCrossover(Generic[T, C], Crossover[T, C]):
    def apply(self, a: C, b: C) -> tuple[C, C]:
        c1, c2 = [], []
        for ia, ib in zip(a, b):
            if randrange(2) == 0:
                ia, ib = ib, ia
            c1.append(ia)
            c2.append(ib)
        c1, c2 = self.create_chromosome(c1), self.create_chromosome(c2)
        return c1, c2
            
