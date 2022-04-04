
from abc import abstractmethod
from typing import Any, Callable, Generic, Iterable, TypeVar, TypedDict
from chromosome import T, Chromosome, CreateChromosome
from random import randint, randrange

__all__ = [
    'Crossover',
    'OnePointCrossover',
    'NPointCrossover',
    'UniformCrossover'
]


T = TypeVar('T')
C = TypeVar('C', bound=Chromosome)


class CrossoverDict(TypedDict):
    type: str
    params: dict[str, Any]


class Crossover(Generic[T, C]):
    def __init__(self, create_chromosome: CreateChromosome[T, C]) -> None:
        self.create_chromosome = create_chromosome

    @abstractmethod
    def apply(self, a: C, b: C) -> tuple[C, C]: ...

    @classmethod
    def parse(cls, create_chromosome: CreateChromosome[T, C], data: CrossoverDict):
        crossover_type = data['type']
        params = data['params']
        crossover_class = {
            'OnePointCrossover': OnePointCrossover,
            'NPointCrossover': NPointCrossover,
            'UniformCrossover': UniformCrossover,
        }[crossover_type]
        return crossover_class(create_chromosome, **params)

    @abstractmethod
    def params_dict(self) -> dict[str, Any]: ...

    def to_dict(self):
        return CrossoverDict(type={
            OnePointCrossover: 'OnePointCrossover',
            NPointCrossover: 'NPointCrossover',
            UniformCrossover: 'UniformCrossover',
        }[type(self)], params=self.params_dict())


class OnePointCrossover(Generic[T, C], Crossover[T, C]):
    def apply(self, a: C, b: C) -> tuple[C, C]:
        r = randint(0, len(a))
        c1 = a[:r] + b[r:]
        c2 = b[:r] + a[r:]
        return self.create_chromosome(c1), self.create_chromosome(c2)

    def params_dict(self) -> dict[str, Any]:
        return {}


class NPointCrossover(Generic[T, C], Crossover[T, C]):
    def __init__(self, create_chromosome: CreateChromosome[T, C], points: int) -> None:
        self.points = points
        super().__init__(create_chromosome)

    def apply(self, a: C, b: C) -> tuple[C, C]:
        one = OnePointCrossover(self.create_chromosome)
        c1, c2 = a, b
        for _ in range(self.points):
            c1, c2 = one.apply(c1, c2)
        return c1, c2

    def params_dict(self) -> dict[str, Any]:
        return {'points': self.points}


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

    def params_dict(self) -> dict[str, Any]:
        return {}
