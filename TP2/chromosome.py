

from abc import ABC, abstractclassmethod, abstractproperty
import random
from typing import Generic, Hashable, TypeVar, Iterable
from typing_extensions import Self


__all__ = [
    'Chromosome',
    'BinaryChromosome',
    'FloatChromosome',
]

T = TypeVar('T', bound=Hashable)


class Chromosome(ABC, Generic[T], tuple[T, ...]):
    @abstractproperty
    def fitness(self) -> float: ...

    @abstractproperty
    def is_valid(self) -> bool: ...

    def __new__(cls, information: Iterable[T]):
        return super().__new__(cls, information)

    @abstractclassmethod
    def random(cls, length: int) -> Self: ...


class BinaryChromosome(Chromosome[bool]):
    @classmethod
    def random(cls, length: int) -> Self:
        return cls(random.randrange(2) == 1 for _ in range(length))


class FloatChromosome(Chromosome[float]):
    pass
