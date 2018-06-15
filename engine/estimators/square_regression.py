from typing import Callable

import numpy as np
from engine.estimators.base_estimator import BaseEstimator
from engine.optimizers.base_optimizer import BaseOptimizer
from engine.optimizers.sgd_square import SquareSGD
from engine.utils import projections


class LogisticRegression(BaseEstimator):
    def __init__(self, optimizer: BaseOptimizer=SquareSGD(2, 1e-4),
                 projection: Callable[[np.ndarray], np.ndarray]=projections.identity_projection):
        super().__init__(projection)
        self.optimizer = optimizer
        self.w = np.zeros(1)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int=5, save_hist: bool=False):
        x = self.projection(x)
        if not save_hist:
            self.w = self.optimizer.optimize(x, y, epochs=epochs)
        else:
            histories = self.optimizer.optimize(x, y, epochs=epochs, save_hist=True)
            self.w = histories[0][-1]
            return histories

    def predict(self, x: np.ndarray, threshold: int=0.5):
        x = self.projection(x)
        y_pred = np.dot(x, self.w)
        y_pred[y_pred < threshold] = -1
        y_pred[y_pred >= threshold] = 1
        return y_pred
