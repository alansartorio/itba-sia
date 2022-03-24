

from abc import ABC, abstractproperty
from typing import Generic, Hashable, TypeVar, Iterable
from typing_extensions import Self

T = TypeVar('T', bound=Hashable)


class Chromosome(ABC, Generic[T], tuple[T, ...]):
    @abstractproperty
    def fitness(self) -> float: ...

    def __new__(cls, information: Iterable[T]):
        return super().__new__(cls, information)


class BinaryChromosome(Chromosome[bool]):
    pass


class FloatChromosome(Chromosome[float]):
    pass
