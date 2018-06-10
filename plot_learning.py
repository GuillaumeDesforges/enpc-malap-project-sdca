import numpy as np
import matplotlib.pyplot as plt
import engine.utils.malaptools as malaptools

from engine.estimators.logistic_regression import LogisticRegression
from engine.optimizers.sdca_logistic import LogisticSDCA
from engine.optimizers.sgd_logistic import LogisticSGD


def plot_learning(x, y):
    # make estimator
    sgd = LogisticSGD(c=1, eps=1e-3)
    sgd_clf = LogisticRegression(optimizer=sgd)

    # train estimator with history
    sgd_hist_w, sgd_hist_loss = sgd_clf.fit(x, y, epochs=20, save_hist=True)
    sgd_hist_w = np.array(sgd_hist_w)

    # verify result
    plt.figure()
    plt.title("Estimator regions")
    malaptools.plot_frontiere(x, sgd_clf.predict)
    malaptools.plot_data(x, y)
    plt.show()

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

    # do it again with SDCA !
    sdca = LogisticSDCA(c=1)
    sdca_clf = LogisticRegression(optimizer=sdca)

    sdca_hist_w, sdca_hist_loss, sdca_hist_alpha = sdca_clf.fit(x, y, epochs=20, save_hist=True)
    sdca_hist_w = np.array(sdca_hist_w)
    sdca_hist_alpha = np.array(sdca_hist_alpha)
    
    plt.figure()
    plt.title("Estimator regions")
    malaptools.plot_frontiere(x, sdca_clf.predict)
    malaptools.plot_data(x, y)
    plt.show()
    
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

    # comparison
    plt.figure()
    plt.title("Comparison of the evolution of the loss")
    plt.plot(sgd_hist_loss, label="SGD")
    plt.plot(sdca_hist_loss, label="SDCA")
    plt.legend()
    plt.show()


def main():
    # make data
    np.random.seed(0)

    n1, n2 = 100, 100
    x1 = np.random.normal(loc=(-1, -1), scale=(1, 1), size=(n1, 2))
    x2 = np.random.normal(loc=(1, 1), scale=(1, 1), size=(n2, 2))
    x = np.concatenate([x1, x2])

    y1 = -np.ones(shape=n1)
    y2 = np.ones(shape=n2)
    y = np.concatenate([y1, y2])

    plot_learning(x, y)


if __name__ == '__main__':
    main()
