import numpy as np


class Normalizer:
    def __init__(self, x: np.ndarray):
        self.means = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1e-5

    def normalize(self, x):
        return (x - self.means) / self.std
