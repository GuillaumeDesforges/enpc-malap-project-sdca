import numpy as np
from engine.optimizers.base_sgd import BaseSGD


def square_loss(x, y, w, c):
    return c * np.sum([(np.dot(w.T, x[i]) - y[i]) ** 2 for i in range(x.shape[0])]) + np.dot(w, w) / 2


def square_increment(x_i, y_i, w, c, eps):
    return - eps * (c * 2 * x_i * (np.dot(w.T, x_i) - y_i) + w)


class SquareSGD(BaseSGD):
    def __init__(self, c, eps):
        self.c = c
        self.eps = eps

        def loss(x, y, w):\
            return square_loss(x, y, w, self.c)

        def increment(x_i, y_i, w):
            return square_increment(x_i, y_i, w, self.c, self.eps)

        super().__init__(loss, increment)