from enum import Enum, auto


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

