#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from util.malaptools import gen_arti, plot_data, plot_frontiere


def main():
    # Data
    X_train, y_train = gen_arti(data_type = 0, sigma = 0.5, nbex = n_train, epsilon = 0.1)
    X_test, y_test = gen_arti(data_type = 0, sigma = 0.5, nbex = n_test, epsilon = 0.1)
    plot_data(X_train, y_train)
    plt.show()




if __name__ == '__main__':
    main()
