import random
import unittest
from functools import cached_property
from population import Population
from selection import EliteSelection, TruncatedSelection

from chromosome import FloatChromosome


class SimpleChromosome(FloatChromosome):
    @cached_property
    def fitness(self):
        return self[0]

    def __new__(cls, value: float):
        return super().__new__(cls, (value, ))


class TestSelections(unittest.TestCase):
    def setUp(self) -> None:
        self.population = [SimpleChromosome(
            float(i)) for i in range(9, -1, -1)]

        self.mixed = self.population.copy()
        random.shuffle(self.mixed)

    def test_elite(self):
        selection = EliteSelection(5)
        self.assertEquals(set(selection.apply(Population(self.mixed))),
                          set(self.population[:5]))

    def test_truncated(self):
        selection = TruncatedSelection(5, 5)
        self.assertEquals(set(selection.apply(Population(self.mixed))),
                          set(self.population[:5]))

    def test_truncated2(self):
        selection = TruncatedSelection(2, 5)
        for _ in range(10):
            self.assertTrue(set(self.population[:5]).issuperset(
                set(selection.apply(Population(self.mixed)))))


if __name__ == '__main__':
    unittest.main()
