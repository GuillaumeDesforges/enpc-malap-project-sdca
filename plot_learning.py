import numpy as np
import matplotlib.pyplot as plt
import engine.utils.data_gen as data_gen
import engine.utils.data_sets as data_sets
import engine.utils.malaptools as malaptools


from engine.estimators.logistic_regression import LogisticRegression
from engine.optimizers.sdca_logistic import LogisticSDCA
from engine.optimizers.sgd_logistic import LogisticSGD
from engine.optimizers.sdca_square import SquareSDCA
from engine.optimizers.sgd_square import SquareSGD


def plot_learning(x, y, chosen_sgd=LogisticSGD(c=1, eps=1e-3), chosen_sdca=LogisticSDCA(c=1), nb_epochs=1,
                  comp_sgd=True, comp_sdca=True, is_malaptool=False, verbose_all=False):
    # make estimator
    if comp_sgd:
        sgd = chosen_sgd
        sgd_clf = LogisticRegression(optimizer=sgd)

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

            # verify resultd783b05f493dbfb35511a7e543e67cd8e02eb2aa
            if is_malaptool:
                plt.figure()
                plt.title("Estimator regions")
                malaptools.plot_frontiere(x, sgd_clf.predict)
                malaptools.plot_data(x, y)
                plt.show()

    # do it again with SDCA !
    if comp_sdca:
        sdca = chosen_sdca
        sdca_clf = LogisticRegression(optimizer=sdca)

        sdca_hist_w, sdca_hist_loss, sdca_hist_alpha = sdca_clf.fit(x, y, epochs=nb_epochs, save_hist=True)
        sdca_hist_w = np.array(sdca_hist_w)
        sdca_hist_alpha = np.array(sdca_hist_alpha)

        if verbose_all:
            plt.figure()
            plt.title("Evolution of the weights")
            for d in range(sdca_hist_w.shape[1]):
                plt.plot(sdca_hist_w[:, d])
            plt.show()

            plt.figure()
            plt.title("Evolution of the dual variables")
            for n in range(sdca_hist_alpha.shape[1]):
                plt.plot(sdca_hist_alpha[:, n])
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


def main():
    # x, y = data_gen.gen_circle_data(n1=100, n2=100, r1=3, r2=1)
    # x, y = data_gen.gen_gaussian_data(n1=100, n2=100)
    x, y = data_sets.real_data_set(data_set_name="lfw", n=1000)

    plot_learning(x, y, chosen_sgd=LogisticSGD(c=0.1, eps=1e-3), chosen_sdca=LogisticSDCA(c=0.1), nb_epochs=1,
                  comp_sgd=True, comp_sdca=True, is_malaptool=False)
    '''plot_learning(x, y, chosen_sgd=LogisticSGD(c=1, eps=1e-3), chosen_sdca=LogisticSDCA(c=1), nb_epochs=1,
                  comp_sgd=True, comp_sdca=True, is_malaptool=False)
    plot_learning(x, y, chosen_sgd=LogisticSGD(c=10, eps=1e-3), chosen_sdca=LogisticSDCA(c=10), nb_epochs=1,
                  comp_sgd=True, comp_sdca=True, is_malaptool=False)'''

    '''plot_learning(x, y, chosen_sgd=SquareSGD(c=0.1, eps=1e-3), chosen_sdca=SquareSDCA(c=0.1), nb_epochs=1,
                  comp_sgd=True, comp_sdca=True, is_malaptool=False)
    plot_learning(x, y, chosen_sgd=SquareSGD(c=1, eps=1e-3), chosen_sdca=SquareSDCA(c=1), nb_epochs=1,
                  comp_sgd=True, comp_sdca=True, is_malaptool=False)
    plot_learning(x, y, chosen_sgd=SquareSGD(c=10, eps=1e-3), chosen_sdca=SquareSDCA(c=10), nb_epochs=1,
                  comp_sgd=True, comp_sdca=True, is_malaptool=False)'''


if __name__ == '__main__':
    main()
