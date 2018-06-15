import abc
from typing import Callable

import numpy as np


class BaseEstimator(abc.ABC):
    def __init__(self, projection: Callable[[np.ndarray], np.ndarray]):
        self.projection = projection

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, save_hist: bool=False):
        pass

    @abc.abstractmethod
    def predict(self, x: np.ndarray):
        pass

    def score_accuracy(self, x: np.ndarray, y_true: np.ndarray):
        x = self.projection(x)
        y_pred = self.predict(x)
        errors = y_true != y_pred
        error_rate = np.sum(errors)/len(errors)
        return 1 - error_rate
