import abc
import numpy as np


class BaseEstimator(abc.ABC):
    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, save_hist: bool=False):
        pass

    @abc.abstractmethod
    def predict(self, x: np.ndarray):
        pass

    def score_accuracy(self, x: np.ndarray, y_true: np.ndarray):
        y_pred = self.predict(x)
        errors = y_true != y_pred
        error_rate = np.sum(errors)/len(errors)
        return 1 - error_rate

