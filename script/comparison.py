import numpy as np
import matplotlib.pyplot as plt
from sgd.classifier_sgd import classifier_SGD

from util.malaptools import gen_arti, plot_data, plot_frontiere

from sdca_close.sdca import fit, predict, score
from sdca_close.losses import LOSSES

import sklearn.datasets

# No randomness for the true warriors of data science
np.random.seed(0)

#number of iterations
T = 10

# Calcul of the score for logistic regression
def score_logistic(X, Y, w):
    z = X.dot(w)
    y_proba = 1/(1 + np.exp(-z))
    y_predict = np.where(y_proba > 0.5, 1, -1)
    score = len(np.where(y_predict.ravel() == Y.ravel())[0])/len(y_predict)
    return score

# Load datasets
boston = sklearn.datasets.load_boston()
iris = sklearn.datasets.load_iris()
diabetes = sklearn.datasets.load_diabetes()
digits = sklearn.datasets.load_digits()
# datasets = [boston, iris, diabetes, digits]
datasets = [digits]

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
    # split into two even classes : labels_sample_n = len(labels) // 2
    # 1vAll
    labels_sample_n = 1
    labels_sample = labels[:labels_sample_n]
    Y_nonlogical = np.copy(Y)
    Y[np.isin(Y_nonlogical, labels_sample)] = -1
    Y[np.isin(Y_nonlogical, labels_sample, invert=True)] = 1
    datasets_labels.append(labels)
    datasets_labels_sample.append(labels_sample)

# Fit each : SDCA
datasets_n_iter = [T for dataset in datasets]
datasets_hist_w = [fit(X, Y, loss='logistic_loss', n_iter=n_iter, lamb=1, history=True)[1] for (X, Y), n_iter in zip(datasets_data, datasets_n_iter)]
datasets_w = [hist_w[-1] for hist_w in datasets_hist_w]

# Fit each : SGD
sdca_clf = []
for (X, Y), n_iter in zip(datasets_data, datasets_n_iter):
    # TODO change that, as it SUCKS
    clf = classifier_SGD(maxIter=n_iter, eps=10**-8)
    clf.fit(X, Y)
    sdca_clf.append(clf)

# Plot performances
for i in range(len(datasets)):
    (X, Y) = datasets_data[i]
    plt.figure()
    vect_score_sdca = []
    vect_score_sgd = []
    for t in range(T):
        vect_score_sdca.append(score_logistic(X, Y, datasets_hist_w[i][t]))
        vect_score_sgd.append(score_logistic(X, Y, sdca_clf[i].vect_w[t]))
    plt.plot(vect_score_sgd, c='b', label="SGD")
    plt.plot(vect_score_sdca, c='g', label="SDCA")
    plt.title("Evolution of the score with the number of iterations")
    plt.xlabel("t")
    plt.ylabel("score")
    plt.legend()
    plt.show()

