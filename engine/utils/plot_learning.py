from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import engine.utils.malaptools as malaptools
from engine.estimators.logistic_regression import LogisticRegression
from engine.optimizers.sdca_logistic import LogisticSDCA
from engine.optimizers.sgd_logistic import LogisticSGD
from engine.utils import projections

DEFAULT_SGD = LogisticSGD(c=1, eps=1e-3)
DEFAULT_SDCA = LogisticSDCA(c=1)


def plot_learning(x, y, chosen_sgd=DEFAULT_SGD, chosen_sdca=DEFAULT_SDCA, nb_epochs=1,
                  comp_sgd=True, comp_sdca=True, is_malaptool=False, verbose_all=False,
                  projection: Callable[[np.ndarray], np.ndarray]=projections.identity_projection):
    # make estimator
    if comp_sgd:
        sgd = chosen_sgd
        sgd_clf = LogisticRegression(optimizer=sgd, projection=projection)

        # train estimator with history
        sgd_hist_w, sgd_hist_loss = sgd_clf.fit(x, y, epochs=nb_epochs, save_hist=True)
        sgd_hist_w = np.array(sgd_hist_w)

        if verbose_all:
            # plot histories
            plt.figure()
            plt.title("Evolution of the weights")
            for d in range(sgd_hist_w.shape[1]):
                plt.plot(sgd_hist_w[:, d])
            plt.show()

            plt.figure()
            plt.title("Evolution of the loss")
            plt.plot(sgd_hist_loss)
            plt.show()

            # verify result
            if is_malaptool:
                plt.figure()
                plt.title("Estimator regions")
                malaptools.plot_frontiere(x, sgd_clf.predict)
                malaptools.plot_data(x, y)
                plt.show()

    # do it again with SDCA !
    if comp_sdca:
        sdca = chosen_sdca
        sdca_clf = LogisticRegression(optimizer=sdca, projection=projection)

        sdca_hist_w, sdca_hist_loss = sdca_clf.fit(x, y, epochs=nb_epochs, save_hist=True)
        sdca_hist_w = np.array(sdca_hist_w)

        if verbose_all:
            plt.figure()
            plt.title("Evolution of the weights")
            for d in range(sdca_hist_w.shape[1]):
                plt.plot(sdca_hist_w[:, d])
            plt.show()

            plt.figure()
            plt.title("Evolution of the loss")
            plt.plot(sdca_hist_loss)
            plt.show()

            if is_malaptool:
                plt.figure()
                plt.title("Estimator regions")
                malaptools.plot_frontiere(x, sdca_clf.predict)
                malaptools.plot_data(x, y)
                plt.show()

    # comparison
    if comp_sgd and comp_sdca:
        plt.figure()
        plt.title("Comparison of the evolution of the loss")
        plt.plot(sgd_hist_loss, label="SGD")
        plt.plot(sdca_hist_loss, label="SDCA")
        plt.legend()
        plt.show()
