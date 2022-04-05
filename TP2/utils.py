from enum import Enum, auto
import random
from typing import Callable, TypeVar


class Generator:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.stop_reason = yield from self.gen

class StopReason(Enum):
    MaxGenerationCount = 'MaxGenerationCount'
    MaxTimeExceeded = 'MaxTimeExceeded'
    NotEnoughVariation = 'NotEnoughVariation'
    NotEnoughImprovement = 'NotEnoughImprovement'


T = TypeVar('T')

def weighted_sample(c: list[T], key: Callable[[T], float]) -> T:
    mapped = tuple((v, key(v)) for v in c)
    total = sum(weight for _, weight in mapped)
    mapped = tuple((v, (weight / total) if total > 0 else (1 / len(c)))
                   for v, weight in mapped)
    r = random.random()

    for v, p in mapped:
        r -= p
        if r <= 0:
            return v

    raise RuntimeError()


def weighted_multisample(c: list[T], count: int, key: Callable[[T], float]) -> tuple[T, ...]:
    if len(c) < count:
        raise AttributeError()

    picked = []

    for _ in range(count):
        p = weighted_sample(c, key)
        picked.append(p)
        c.remove(p)

    return tuple(picked)
