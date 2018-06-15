import numpy as np
from engine.optimizers.base_sgd import BaseSGD


def logistic_loss(x, y, w, c):
    a = - y * np.dot(x, w)
    # floating point arithmetic seems to make log(1 + exp(x)) = x if x > 40
    b = np.where(a < 40, np.log(1 + np.exp(a)), a)
    z = c * np.sum(b) + np.dot(w, w)/2
    return z


def logistic_increment(x_i, y_i, w, c, eps):
    return eps * (c * y_i * x_i / (1 + np.exp(y_i * np.dot(x_i, w))) - w)


class LogisticSGD(BaseSGD):
    def __init__(self, c: float, eps: float):
        self.c = c
        self.eps = eps

        def loss(x, y, w):\
            return logistic_loss(x, y, w, self.c)

        def increment(x_i, y_i, w):
            return logistic_increment(x_i, y_i, w, self.c, self.eps)

        super().__init__(loss, increment)