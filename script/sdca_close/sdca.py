#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from .losses import build_increment


def sdca(X, Y, loss, n_iter, lamb):
    # problem shape
    n, d = X.shape

    # alpha
    alpha = np.ones(n)

    # alpha history
    hist_alpha = np.zeros((n_iter, n))
    hist_alpha[0] = np.copy(alpha)
    
    # weights
    w = np.sum(alpha[:, np.newaxis]*X, axis=0)/(lamb*n)

    # weights history
    hist_w = np.zeros((n_iter, d))
    hist_w[0] = np.copy(w)

    # get increment function depending on the loss
    compute_increment = build_increment(loss, lamb, n)
    
    # iterations
    for t in range(1, n_iter):
        # select i in [0, n[
        i = np.random.randint(n)

        # get concerned data
        x = X[i]
        y = Y[i]
        alpha_i = alpha[i]

        # compute increment $\delta\alpha_i$
        delta = compute_increment(x, y, w, alpha_i)
        
        # update alpha
        alpha[i] += delta

        # update weights
        w += delta*x/(lamb*n)

        # save history
        hist_alpha[t] = np.copy(alpha)
        hist_w[t] = np.copy(w)

    return hist_alpha, hist_w

