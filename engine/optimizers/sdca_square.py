import numpy as np
from engine.optimizers.base_sdca import BaseSDCA


def square_loss(x, y, w, c):
    return c * np.sum([(np.dot(w.T, x[i]) - y[i]) ** 2 for i in range(x.shape[0])]) + np.dot(w, w) / 2


def square_increment(x_i, y_i, w, alpha_i, c, n):
    return (y_i - np.dot(x_i.T, w) - alpha_i / 2) / (0.5 + np.linalg.norm(x_i) ** 2 / (c * n))


class SquareSDCA(BaseSDCA):
    def __init__(self, c):
        def loss(x, y, w):\
            return square_loss(x, y, w, c)

        def increment(x_i, y_i, w, alpha_i, n):
            return square_increment(x_i, y_i, w, alpha_i, c, n)

        super().__init__(loss, increment, c)