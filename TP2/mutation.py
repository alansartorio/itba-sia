from abc import ABC, abstractmethod
import random
from typing import Generic, TypeVar
from chromosome import BinaryChromosome, Chromosome, FloatChromosome

C = TypeVar('C', bound=Chromosome)


class Mutation(ABC, Generic[C]):
    @abstractmethod
    def apply(self, chromosome: C) -> C: ...


class BinaryMutation(Mutation[BinaryChromosome]):
    def __init__(self, probability: float) -> None:
        self.probability = probability

    # def apply(self, chromosome: BinaryChromosome) -> BinaryChromosome:
        # index = random.randrange(len(chromosome))
        # new = list(chromosome)
        # new[index] = not new[index]

        # return BinaryChromosome(new)

    def apply(self, chromosome: BinaryChromosome) -> BinaryChromosome:
        copy = list(chromosome)
        for i in range(len(copy)):
            if random.random() < self.probability:
                copy[i] = not copy[i]
        return BinaryChromosome(copy)

class RealMutation(Mutation[FloatChromosome]):
    pass
