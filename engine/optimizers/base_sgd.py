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
        
        best_w = np.copy(w)
        best_loss = self.loss(x, y, w)

        print("Nombre d'itérations :", int(n * epochs))

        for k in range(int(n*epochs)):
            #print("Itération n°", k)
            i = np.random.randint(n)
            w += self.increment(x[i], y[i], w)
            loss = self.loss(x, y, w)
            if loss < best_loss:
                best_loss = loss
                best_w = np.copy(w)
            if save_hist:
                hist_w.append(np.copy(w))
                hist_loss.append(loss)

        # do epochs
        '''for epoch in range(epochs):
            # at each epoch we fit to each data in a random order
            index_order = np.arange(n)
            np.random.shuffle(index_order)
            for i in index_order:
                w += self.increment(x[i], y[i], w)
                if save_hist:
                    hist_w.append(np.copy(w))
                    loss = self.loss(x, y, w)
                    hist_loss.append(loss)'''

        if save_hist:
            return hist_w, hist_loss

        return best_w