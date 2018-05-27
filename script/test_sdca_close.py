#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from util.malaptools import gen_arti, plot_data, plot_frontiere

from sdca_close.sdca import fit, predict, score


def main():
    # Data
    data_args = {'data_type':0, 'epsilon': 0.1, 'sigma':0.5}
    X, Y = gen_arti(nbex=1000, **data_args)
    plot_data(X, Y)
    plt.show()
    
    # Compute SDCA
    n_iter = 100
    hist_alpha, hist_w = fit(X, Y, loss='square_loss', n_iter=n_iter, lamb=1, history=True)
    w = hist_w[-1]

    # Plot result
    plot_frontiere(X, lambda X: predict(X, w))
    plot_data(X, Y)
    plt.show()

    # Plot some stats
    n, _ = X.shape
    plt.title("Evolution of alpha")
    for i in range(n):
        plt.plot(hist_alpha[:, i])
    plt.show()
    
    X_test, Y_test = gen_arti(nbex=100, **data_args)
    hist_error_rate = [score(X_test, Y_test, hist_w[t]) for t in range(n_iter)]

    plt.title("Evolution of the error rate")
    plt.plot(hist_error_rate)
    plt.show()
    

if __name__ == '__main__':
    main()
