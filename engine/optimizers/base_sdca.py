import numpy as np
from engine.optimizers.base_optimizer import BaseOptimizer


class BaseSDCA(BaseOptimizer):
    def __init__(self, loss, increment, c):
        super().__init__()
        self.loss = loss
        self.increment = increment
        self.c = c

    def optimize(self, x: np.ndarray, y: np.ndarray, epochs=5, save_hist: bool=False):
        if len(x.shape) != 2:
            raise Exception("x must have ndim==2")

        if len(y.shape) != 1:
            raise Exception("y must have ndim==1")

        n, d = x.shape

        if len(y) != n:
            raise Exception("x 1st dim and y dim must be equal")

        # dual variable vector
        alpha = np.zeros(n)

        # weight vector
        w = np.zeros(d)

        hist_alpha = []
        hist_w = []
        hist_loss = []

        print("Nombre d'itérations :", int(n*epochs))

        for k in range(int(n*epochs)):
            #print("Itération n°", k)
            i = np.random.randint(n)
            delta = self.increment(x[i], y[i], w, alpha[i], n)
            alpha[i] += delta
            w += delta*x[i] / (self.c * n)

            if save_hist:
                hist_alpha.append(np.copy(alpha))
                hist_w.append(np.copy(w))
                loss = self.loss(x, y, w)
                hist_loss.append(loss)

        # do epochs
        '''for epoch in range(epochs):
            # at each epoch we fit to each data in a random order
            index_order = np.arange(n)
            np.random.shuffle(index_order)

            for i in index_order:
                delta = self.increment(x[i], y[i], w, alpha[i], n)
                alpha[i] += delta
                w += delta*x[i] / (self.c * n)

                if save_hist:
                    hist_alpha.append(np.copy(alpha))
                    hist_w.append(np.copy(w))
                    loss = self.loss(x, y, w)
                    hist_loss.append(loss)'''



        if save_hist:
            return hist_w, hist_loss, hist_alpha

        return w