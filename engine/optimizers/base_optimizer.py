import abc
import numpy as np


class BaseOptimizer(abc.ABC):
    @abc.abstractmethod
    def optimize(self, x: np.ndarray, y: np.ndarray, epochs: int=5, save_hist: bool=False):
        # NOTE when save_hist is True, must return histories where histories[0] is the history of w
        pass
