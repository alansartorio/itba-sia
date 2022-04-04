from abc import ABC, abstractclassmethod, abstractmethod
import random
from typing import Any, Callable, Generic, Iterable, Literal, TypeVar, TypedDict
from typing_extensions import Self
from chromosome import BinaryChromosome, Chromosome, CreateChromosome, FloatChromosome

__all__ = [
    'Mutation',
    'BinaryMutation',
    'RealMutation'
]


C = TypeVar('C', bound=Chromosome)
D = TypeVar('D', bound=TypedDict)


class Mutation(ABC, Generic[C, D]):
    def __init__(self, create_chromosome: CreateChromosome[bool, C]) -> None:
        self.create_chromosome = create_chromosome

    @abstractmethod
    def apply(self, chromosome: C) -> C: ...

    @abstractclassmethod
    def parse(cls, data: D) -> Self: ...

    @abstractmethod
    def to_dict(self) -> D: ...


B = TypeVar('B', bound=BinaryChromosome)


class BinaryMutationDict(TypedDict):
    probability: float

class BinaryMutation(Mutation[B, BinaryMutationDict]):
    def __init__(self, create_chromosome: CreateChromosome[bool, B], probability: float) -> None:
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

    @classmethod
    def parse(cls, create_chromosome: CreateChromosome[bool, B], data: BinaryMutationDict) -> Self:
        probability = data['probability']
        return cls(create_chromosome, probability)

    def to_dict(self) -> BinaryMutationDict:
        return BinaryMutationDict(probability=self.probability)



class RealMutationDict(TypedDict):
    type: Literal['uniform', 'normal']
    value: float

# TODO: Implement mutation for chromosomes with real values.
class RealMutation(Mutation[FloatChromosome, RealMutationDict]):
    pass
