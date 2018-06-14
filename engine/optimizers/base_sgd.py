import numpy as np
from engine.optimizers.base_optimizer import BaseOptimizer


class BaseSGD(BaseOptimizer):
    def __init__(self, loss, increment):
        super().__init__()
        self.loss = loss
        self.increment = increment

    def optimize(self, x: np.ndarray, y: np.ndarray, epochs=5, save_hist: bool=False):
        if len(x.shape) != 2:
            raise Exception("x must have ndim==2")

        if len(y.shape) != 1:
            raise Exception("y must have ndim==1")

        n, d = x.shape

        if len(y) != n:
            raise Exception("x 1st dim and y dim must be equal")

        # weight vector
        w = np.zeros(d)

        hist_w = []
        hist_loss = []
        if save_hist:
            hist_w.append(np.copy(w))
            loss = self.loss(x, y, w)
            hist_loss.append(loss)

        iter_max = int(n*epochs)
        for iter_k in range(iter_max):
            i = np.random.randint(n)
            w += self.increment(x[i], y[i], w)
            if save_hist:
                hist_w.append(np.copy(w))
                loss = self.loss(x, y, w)
                hist_loss.append(loss)

        if save_hist:
            return hist_w, hist_loss

        return w
