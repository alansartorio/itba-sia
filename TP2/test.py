import random
import unittest
from functools import cached_property
from algorythm import GeneticAlgorythm
from population import Population
from selection import EliteSelection, Selection, TournamentSelection, TruncatedSelection
from crossover import *
from mutation import *

from chromosome import BinaryChromosome, FloatChromosome


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
        self.assertEqual(set(selection.apply(Population(self.mixed))),
                         set(self.population[:5]))

    def test_truncated(self):
        selection = TruncatedSelection(5, 5)
        self.assertEqual(set(selection.apply(Population(self.mixed))),
                         set(self.population[:5]))

    def test_truncated2(self):
        selection = TruncatedSelection(2, 5)
        for _ in range(10):
            self.assertTrue(set(self.population[:5]).issuperset(
                set(selection.apply(Population(self.mixed)))))


class TestSelectionParsing(unittest.TestCase):
    def test_parse_elite(self):
        parsed = Selection.parse(10, {"type": "EliteSelection", "params": {}})
        self.assertIsInstance(parsed, EliteSelection)

    def test_parse_tournament(self):
        parsed = Selection.parse(10, {"type": "TournamentSelection", "params": {
                                 "replace": False, "threshold": 0.4}})
        self.assertIsInstance(parsed, TournamentSelection)
        self.assertEqual(parsed.threshold, 0.4)
        self.assertEqual(parsed.replace, False)


class TestCrossoverParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.create_chromosome = lambda l: BinaryChromosome(l)

    def test_parse_one_point(self):
        parsed = Crossover.parse(self.create_chromosome, {
                                 "type": "OnePointCrossover", "params": {}})
        self.assertIsInstance(parsed, OnePointCrossover)

    def test_parse_n_point(self):
        parsed = Crossover.parse(self.create_chromosome, {
                                 "type": "NPointCrossover", "params": {"points": 3}})
        self.assertIsInstance(parsed, NPointCrossover)
        self.assertEqual(parsed.points, 3)

    def test_parse_uniform(self):
        parsed = Crossover.parse(self.create_chromosome, {
                                 "type": "UniformCrossover", "params": {}})
        self.assertIsInstance(parsed, UniformCrossover)


class TestMutationParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.create_chromosome = lambda l: BinaryChromosome(l)

    def test_parse_binary(self):
        parsed = BinaryMutation.parse(
            self.create_chromosome, {"probability": 0.03})
        self.assertIsInstance(parsed, BinaryMutation)
        self.assertEqual(parsed.probability, 0.03)


class TestAlgorythmParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.create_chromosome = lambda l: BinaryChromosome(l)

    def test_parse(self):
        parsed = GeneticAlgorythm.parse_binary(self.create_chromosome, {
            'population_count': 10,
            'mutation': {
                'probability': 0.1
            },
            'crossover': {
                'type': 'NPointCrossover',
                'params': {'points': 5}
            },
            'selection': {
                'type': 'EliteSelection',
                'params': {}
            },
        })

        self.assertIsInstance(parsed, GeneticAlgorythm)
        self.assertIsInstance(parsed.mutation_operator, BinaryMutation)
        self.assertIsInstance(parsed.crossover_operator, NPointCrossover)
        self.assertEqual(parsed.crossover_operator.points, 5)
        self.assertIsInstance(parsed.selection_operator, EliteSelection)


if __name__ == '__main__':
    unittest.main()
