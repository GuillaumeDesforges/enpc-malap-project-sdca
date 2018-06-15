import pickle
from typing import Union, Callable

import matplotlib.pyplot as plt
import numpy as np

import engine.utils.projections as projections
from engine.estimators.logistic_regression import LogisticRegression
from engine.optimizers.sgd_logistic import LogisticSGD
from engine.utils.normalize import Normalizer


def compute_search(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                   param_name: str, param_values: Union[list, np.ndarray],
                   optimizer_type, optimizer_kwargs: dict=None,
                   projection: Callable[[np.ndarray], np.ndarray]=projections.identity_projection):
    scores_train = list()
    scores_test = list()

    plt.ion()
    plt.show()

    for param_value in param_values:
        np.random.seed(50307)

        # gather parameters
        param_kwarg = {param_name: param_value}
        if optimizer_kwargs is None:
            optimizer_kwargs = param_kwarg
        else:
            optimizer_kwargs.update(param_kwarg)

        # init optimizer and estimator
        optimizer = optimizer_type(**optimizer_kwargs)
        estimator = LogisticRegression(optimizer=optimizer, projection=projection)

        # fit estimator
        hist_w, hist_loss = estimator.fit(x_train, y_train, epochs=15, save_hist=True)

        # plot learning
        plt.clf()
        plt.title(", ".join(map(lambda x: "{}={}".format(*x), optimizer_kwargs.items())))
        plt.plot(hist_loss)
        plt.draw()
        plt.pause(0.001)

        # evaluate
        score_train = estimator.score_accuracy(x_train, y_train)
        score_test = estimator.score_accuracy(x_test, y_test)
        scores_train.append(score_train)
        scores_test.append(score_test)

    plt.close()
    plt.ioff()

    return scores_train, scores_test


def plot_search(param_name: str, param_values: Union[list, np.ndarray], scores_train, scores_test, logarithmic: bool=False):
    plot_method = plt.semilogx if logarithmic else plt.plot
    plot_method(param_values, scores_train, label='Train set')
    plot_method(param_values, scores_test, label='Test set')

    plt.title("Accuracy vs hyper parameter")
    plt.xlabel(param_name)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


def main():
    # importation of data
    with open("datasets/Heart_disease/heart_disease_data.pkl", 'rb') as file:
        x = pickle.load(file)
        y = pickle.load(file)

    # dividing data in 2 classes
    y = np.where(y == 1, 1, -1)

    # normalisation
    normalizer = Normalizer(x)
    x = normalizer.normalize(x)

    # split
    n, d = x.shape
    n_train = int(n*0.9)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    # test setup
    param_name = 'eps'
    param_values = np.logspace(-7, 0, num=10)
    optimizer_type = LogisticSGD
    optimizer_kwargs = {'c': 1}

    # compute training
    projection = projections.identity_projection
    scores_train, scores_test = compute_search(x_train, y_train, x_test, y_test,
                                               param_name, param_values,
                                               optimizer_type, optimizer_kwargs,
                                               projection)

    # plot result
    plot_search(param_name, param_values, scores_train, scores_test, logarithmic=True)


if __name__ == '__main__':
    main()
