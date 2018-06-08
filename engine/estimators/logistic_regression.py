import numpy as np
from engine.estimators.base_estimator import BaseEstimator
from engine.optimizers.base_optimizer import BaseOptimizer
from engine.optimizers.sgd_logistic import LogisticSGD


class LogisticRegression(BaseEstimator):
    def __init__(self, optimizer: BaseOptimizer=LogisticSGD(2, 1e-4)):
        super().__init__()
        self.optimizer = optimizer
        self.w = np.zeros(1)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int=5, save_hist: bool=False):
        if not save_hist:
            self.w = self.optimizer.optimize(x, y, epochs=epochs)
        else:
            histories = self.optimizer.optimize(x, y, epochs=epochs, save_hist=True)
            self.w = histories[0][-1]
            return histories

    def predict(self, x: np.ndarray, threshold: int=0.5):
        y_pred = 1 / (1 + np.exp(- np.dot(x, self.w)))
        y_pred[y_pred < threshold] = -1
        y_pred[y_pred >= threshold] = 1
        return y_pred
