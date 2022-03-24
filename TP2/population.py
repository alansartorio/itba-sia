from typing import TypeVar
from chromosome import Chromosome

C = TypeVar('C', bound=Chromosome)


class Population(tuple[C]):
    pass
