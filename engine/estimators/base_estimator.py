import abc
import numpy as np


class BaseEstimator(abc.ABC):
    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, x: np.ndarray):
        pass

