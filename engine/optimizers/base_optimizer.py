import abc
import numpy as np


class BaseOptimizer(abc.ABC):
    @abc.abstractmethod
    def optimize(self, x: np.ndarray, y: np.ndarray):
        pass
