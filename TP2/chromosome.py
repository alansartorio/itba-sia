

from typing import Generic, TypeVar, Iterable

T = TypeVar('T')

class Chromosome(Generic[T], list[T]):
    def __init__(self, information: Iterable[T]):
        super().__init__(information)

class BinaryChromosome(Chromosome[bool]):
    pass

class FloatChromosome(Chromosome[float]):
    pass
