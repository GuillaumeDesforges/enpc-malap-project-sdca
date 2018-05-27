#!/usr/bin/python

import numpy as np
import sklearn.datasets
from scipy.stats import describe
import matplotlib.pyplot as plt

from sdca_close.sdca import fit, predict, score


def main():
    # No randomness for the true warriors of data science
    np.random.seed(0)

    # Load datasets
    boston = sklearn.datasets.load_boston()
    iris = sklearn.datasets.load_iris()
    diabetes = sklearn.datasets.load_diabetes()
    digits = sklearn.datasets.load_digits()
    datasets = [boston, iris, diabetes, digits]

    # Extract data
    get_name = lambda dataset: dataset['DESCR'].split('\n')[0]
    datasets_name = [get_name(dataset) for dataset in datasets]
    get_data = lambda dataset: (dataset['data'], dataset['target'])
    datasets_data = [get_data(dataset) for dataset in datasets]

    # Turn datasets into logistic regression
    datasets_labels = []
    datasets_labels_sample = []
    for i, (X, Y) in enumerate(datasets_data):
        labels = np.unique(Y)
        labels_n = len(labels)
        labels_sample_n = len(labels) // 2
        labels_sample = labels[:labels_sample_n]
        Y_nonlogical = np.copy(Y)
        Y[np.isin(Y_nonlogical, labels_sample)] = -1
        Y[np.isin(Y_nonlogical, labels_sample, invert=True)] = 1
        datasets_labels.append(labels)
        datasets_labels_sample.append(labels_sample)

    # Resume datasets
    for dataset_name, (X, Y), dataset_labels, dataset_labels_sample in zip(datasets_name, datasets_data, datasets_labels, datasets_labels_sample):
        print(dataset_name)
        # print('X', X)
        print('X shape', X.shape)
        # print('Y', Y)
        print('Y shape', Y.shape)
        print('Y labels', dataset_labels)
        print('Y -1 labels', dataset_labels_sample)
        # print('------------------------------------')
        print()
    
    # Fit each
    datasets_n_iter = [10000, 500, 50, 10000]
    datasets_hist_w = [fit(X, Y, loss='square_loss', n_iter=n_iter, lamb=1, history=True)[1] for (X, Y), n_iter in zip(datasets_data, datasets_n_iter)]
    datasets_w = [hist_w[-1] for hist_w in datasets_hist_w]

    # Score each
    datasets_score = [score(X, Y, w) for (X, Y), w in zip(datasets_data, datasets_w)]

    # Scores :
    for name, dataset_score in zip(datasets_name, datasets_score):
        print(name, dataset_score)

    # Plot score evolutions
    for name, (X, Y), n_iter, hist_w in zip(datasets_name, datasets_data, datasets_n_iter, datasets_hist_w):
        hist_score = [score(X, Y, hist_w[t]) for t in range(n_iter)]
        plt.title("Evolution of score for dataset {}".format(name))
        plt.plot(hist_score)
        plt.xlabel("Iterations")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.show()


if __name__ == '__main__':
    main()
