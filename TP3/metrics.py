import math
from typing import Sequence
from network import Network
from single_data import SingleData
import numpy as np
import random

class Metrics:
    def __init__(self) -> None:
        self.TP = 0.0
        self.TN = 0.0
        self.FP = 0.0
        self.FN = 0.0
        pass

    def accuracy(self) -> float:
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
    def precision(self) -> float:
        return (self.TP) / (self.TP + self.FP)
    def recall(self) -> float:
        return (self.TP) / (self.TP + self.FN)
    def f1(self) -> float:
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall())
        

    #Para 20% test_size deberia ser 0.2
    @classmethod
    def split(cls, seq: Sequence[SingleData], test_size: float, seed: int = None):
        data = list(seq)
        training_size = math.floor(len(data) * (1 - test_size))
        random.seed(seed)
        random.shuffle(data)
        training, test = data[:training_size], data[training_size:]
        return (training,test)

    @classmethod
    def analyze(cls, expected_output: np.ndarray, predicted_output: np.ndarray):
        if(len(expected_output) != len(predicted_output)):
            raise ValueError("expected_output and predicted_output must have the same length")
        metrics = cls()
        for i in range(len(expected_output)):
            if(expected_output[i] == predicted_output[i]):
                if(expected_output[i] == 1):
                    metrics.TP += 1 #TODO unos y ceros o true false?
                else:
                    metrics.TN += 1
            else:
                if(expected_output[i] == 1):
                    metrics.FN += 1
                else:
                    metrics.FP += 1
        return metrics

