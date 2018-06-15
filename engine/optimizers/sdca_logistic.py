import numpy as np
from engine.optimizers.base_sdca import BaseSDCA


def logistic_loss(x, y, w, c):
    a = - y * np.dot(x, w)
    # floating point arithmetic seems to make log(1 + exp(x)) = x if x > 40
    b = np.where(a < 40, np.log(1 + np.exp(a)), a)
    z = c * np.sum(b) + np.dot(w, w)/2
    return z


def logistic_increment(x_i, y_i, w, alpha_i, c, n):
    return (y_i / (1 + np.exp(np.dot(x_i.T, w) * y_i)) - alpha_i) / max(1, 0.25 + np.linalg.norm(x_i) ** 2 / (c * n))


class LogisticSDCA(BaseSDCA):
    def __init__(self, c):
        def loss(x, y, w):\
            return logistic_loss(x, y, w, c)

        def increment(x_i, y_i, w, alpha_i, n):
            return logistic_increment(x_i, y_i, w, alpha_i, c, n)

        super().__init__(loss, increment, c)