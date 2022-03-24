from abc import ABC, abstractmethod
import random
from typing import Callable, Generic, Iterable, TypeVar
from chromosome import BinaryChromosome, Chromosome, FloatChromosome

C = TypeVar('C', bound=Chromosome)


class Mutation(ABC, Generic[C]):
    def __init__(self, create_chromosome: Callable[[Iterable[bool]], C]) -> None:
        self.create_chromosome = create_chromosome

    @abstractmethod
    def apply(self, chromosome: C) -> C: ...


B = TypeVar('B', bound=BinaryChromosome)


class BinaryMutation(Mutation[B]):
    def __init__(self, create_chromosome: Callable[[Iterable[bool]], B], probability: float) -> None:
        self.probability = probability
        super().__init__(create_chromosome)

    # def apply(self, chromosome: BinaryChromosome) -> BinaryChromosome:
        # index = random.randrange(len(chromosome))
        # new = list(chromosome)
        # new[index] = not new[index]

        # return BinaryChromosome(new)

    def apply(self, chromosome: B) -> B:
        copy = list(chromosome)
        for i in range(len(copy)):
            if random.random() < self.probability:
                copy[i] = not copy[i]
        return self.create_chromosome(copy)

# TODO: Implement mutation for chromosomes with real values.
class RealMutation(Mutation[FloatChromosome]):
    pass
