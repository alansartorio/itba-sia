from functools import partial
from itertools import cycle, product, zip_longest, tee
from typing import Any, Callable, Collection, Generator, Generic, Iterable, Iterator, Optional, Sized, TypeVar, get_args
from typing_extensions import Self, TypeVarTuple, Unpack
from multiprocessing.pool import Pool

__all__ = ['CombinatorBuilder', 'Executor', 'entuple']

AnyTuple = tuple[Any, ...]

T = TypeVar('T')


class SizedGenerator(Collection[T], Generic[T]):
    def __init__(self, generator: Iterator[T], size: int) -> None:
        self.generator = generator
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[T]:
        return iter(self.generator)

    def __contains__(self, __x: object) -> bool:
        raise NotImplementedError


T_tu = TypeVarTuple('T_tu')
O_tu = TypeVarTuple('O_tu')


class CombinatorBuilder(Collection[tuple[Unpack[T_tu]]], Generic[Unpack[T_tu]]):
    def __init__(self, initial_inputs: Collection[tuple[Unpack[T_tu]]]) -> None:
        self.inputs = initial_inputs
        self.length = len(initial_inputs)

    def append_end(self, inputs: list[tuple[Unpack[O_tu]]]) -> 'CombinatorBuilder[Unpack[T_tu], Unpack[O_tu]]':
        assert len(self) == len(inputs)
        return CombinatorBuilder(SizedGenerator((prev + new for prev, new in zip(self, inputs)), len(self)))

    def add_product(self, inputs: list[tuple[Unpack[O_tu]]]) -> 'CombinatorBuilder[Unpack[T_tu], Unpack[O_tu]]':
        return CombinatorBuilder(SizedGenerator((prev + new for prev, new in product(self, inputs)), len(inputs) * len(self)))

    def append_map(self, func: Callable[[Unpack[T_tu]], tuple[Unpack[O_tu]]]) -> 'CombinatorBuilder[Unpack[T_tu], Unpack[O_tu]]':
        return CombinatorBuilder(SizedGenerator((prev + func(*prev) for prev in self), len(self)))

    def product_map(self, func: Callable[[Unpack[T_tu]], list[tuple[Unpack[O_tu]]]], length: int) -> 'CombinatorBuilder[Unpack[T_tu], Unpack[O_tu]]':
        return CombinatorBuilder(SizedGenerator((prev + new for prev in self for new in func(*prev)), length * len(self)))

    def map(self, func: Callable[[Unpack[T_tu]], tuple[Unpack[O_tu]]]) -> 'CombinatorBuilder[Unpack[O_tu]]':
        return CombinatorBuilder(SizedGenerator((func(*t) for t in self), len(self)))

    def __iter__(self):
        assert self.inputs is not None
        return iter(self.inputs)

    def __len__(self):
        return self.length

    def __contains__(self, __x: object) -> bool:
        raise NotImplementedError

    @classmethod
    def initialize(cls, initial_values: Collection[T]) -> 'CombinatorBuilder[T]':
        return CombinatorBuilder(SizedGenerator(((v,) for v in initial_values), len(initial_values)))


def entuple(l: list[T]) -> list[tuple[T]]:
    return [(v, ) for v in l]


O = TypeVar('O')


class Executor(Generic[Unpack[T_tu]]):
    def __init__(self, combinations: Collection[tuple[Unpack[T_tu]]]) -> None:
        self.combinations = combinations

    def run(self, func: Callable[[tuple[Unpack[T_tu]]], O]) -> SizedGenerator[tuple[tuple[Unpack[T_tu]], O]]:
        def inner():
            with Pool(12) as pool:
                comb1, comb2 = tee(self.combinations)
                for v, o in zip(comb2, pool.imap(func, comb1)):
                    yield v, o
        return SizedGenerator(inner(), len(self.combinations))


if __name__ == '__main__':
    combinations = CombinatorBuilder.initialize([0, 1, 2])              \
                                    .add_product(entuple([3, 4, 5]))    \
                                    .append_end(entuple([6, 7, 8]*3))

    executor = Executor(combinations)
    outs = executor.run(sum)
    print(len(outs))
    for i, o in outs:
        print(i, o)
