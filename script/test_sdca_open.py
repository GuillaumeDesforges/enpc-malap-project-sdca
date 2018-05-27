#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from util.malaptools import gen_arti, plot_data, plot_frontiere

from sdca_open.sdca import sdca

def main():
    X, Y = gen_arti(nbex=100)
    
    w = sdca(X, Y, verbose=True, plot_alphas=True)
    
    def predict(X):
        Y_pred = np.dot(X, w)
        Y_pred[Y_pred > 0] = 1
        Y_pred[Y_pred <= 0] = -1
        return Y_pred
    
    plot_frontiere(X, predict)
    plot_data(X, Y)
    plt.show()

if __name__ == '__main__':
    main()
