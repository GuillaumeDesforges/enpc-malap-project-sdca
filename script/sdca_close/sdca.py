#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from .losses import build_increment


def fit(X, Y, loss, n_iter, lamb, history=False):
    # problem shape
    n, d = X.shape

    # alpha
    alpha = np.ones(n)

    # weights
    w = np.sum(alpha[:, np.newaxis]*X, axis=0)/(lamb*n)

    # get increment function depending on the loss
    compute_increment = build_increment(loss, lamb, n)
    
    # computation histories
    if history:
        # alpha history
        hist_alpha = np.zeros((n_iter, n))
        hist_alpha[0] = np.copy(alpha)
        
        # weights history
        hist_w = np.zeros((n_iter, d))
        hist_w[0] = np.copy(w)

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
        
        if history:
            # save history
            hist_alpha[t] = np.copy(alpha)
            hist_w[t] = np.copy(w)

    if history:
        hist_alpha = []
        return hist_alpha, hist_w

    return alpha, w


def predict(X, w):
    Y_pred = np.dot(X, w)
    Y_pred[Y_pred <  0] = -1
    Y_pred[Y_pred >= 0] = 1
    return Y_pred


def score(X, Y, w):
    n, _ = X.shape
    Y_pred = predict(X, w)
    errors = np.not_equal(Y_pred, Y)
    error_rate = np.sum(errors)/n
    return 1-error_rate

