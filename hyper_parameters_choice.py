import pickle
from typing import Union, Callable

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

import engine.utils.projections as projections
from engine.estimators.logistic_regression import LogisticRegression
from engine.optimizers.sgd_logistic import LogisticSGD
from engine.optimizers.sdca_logistic import LogisticSDCA
from engine.utils.normalize import Normalizer

from engine.utils.data_sets import load_sklearn_dataset, load_adults_dataset


def compute_search(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                   param_name: str, param_values: Union[list, np.ndarray],
                   optimizer_type, optimizer_kwargs: dict=None,
                   projection: Callable[[np.ndarray], np.ndarray]=projections.identity_projection):
    scores_train = list()
    scores_test = list()

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
        estimator.fit(x_train, y_train, epochs=15, save_hist=True)

        # evaluate
        score_train = estimator.score_accuracy(x_train, y_train)
        score_test = estimator.score_accuracy(x_test, y_test)
        scores_train.append(score_train)
        scores_test.append(score_test)

    return scores_train, scores_test


def plot_search(param_name: str, param_values: Union[list, np.ndarray], scores_train, scores_test, logarithmic: bool=False):
    plt.figure()

    plot_method = plt.semilogx if logarithmic else plt.plot
    plot_method(param_values, scores_train, label='Train set')
    plot_method(param_values, scores_test, label='Test set')

    plt.title("Accuracy vs hyper parameter")
    plt.xlabel(param_name)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


def eval_c(data, labels, vect_param, nb_epoch, data_name, eps_base=10 ** -6):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    
    vect_train_accuracy_sgd = []
    vect_train_accuracy_sdca = []
    
    vect_test_accuracy_sgd = []
    vect_test_accuracy_sdca = []
    
    for param in vect_param:
        # make estimator
        sgd = LogisticSGD(c=param, eps=eps_base)
        sgd_clf = LogisticRegression(optimizer=sgd)
        
        sdca = LogisticSDCA(c=param)
        sdca_clf = LogisticRegression(optimizer=sdca)
        
        # train estimators without history
        sgd_clf.fit(X_train, y_train, epochs=nb_epoch, save_hist=False)
        sdca_clf.fit(X_train, y_train, epochs=nb_epoch, save_hist=False)
        
        vect_train_accuracy_sgd.append(sgd_clf.score_accuracy(X_train, y_train))
        vect_train_accuracy_sdca.append(sdca_clf.score_accuracy(X_train, y_train))
        
        vect_test_accuracy_sgd.append(sgd_clf.score_accuracy(X_test, y_test))
        vect_test_accuracy_sdca.append(sdca_clf.score_accuracy(X_test, y_test))
    
    plt.figure()
    plt.semilogx(vect_param, vect_train_accuracy_sgd, 'b', label="train")
    plt.semilogx(vect_param, vect_test_accuracy_sgd, 'r', label="test")
    plt.title("SGD accuracy vs. hyperparameter C\n on data set " + data_name)
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.figure()
    plt.semilogx(vect_param, vect_train_accuracy_sdca, 'b', label="train")
    plt.semilogx(vect_param, vect_test_accuracy_sdca, 'r', label="test")
    plt.title("SDCA accuracy vs. hyperparameter C\n on data set " + data_name)
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.legend()


def eval_eps(data, labels, vect_param, nb_epoch, data_name, param_c=10**1):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    
    vect_train_accuracy_sgd = []
    
    vect_test_accuracy_sgd = []
    
    for param in vect_param:
        # make estimator
        sgd = LogisticSGD(c=param_c, eps=param)
        sgd_clf = LogisticRegression(optimizer=sgd)
        
        # train estimators without history
        sgd_clf.fit(X_train, y_train, epochs=nb_epoch, save_hist=False)
        
        vect_train_accuracy_sgd.append(sgd_clf.score_accuracy(X_train, y_train))
        
        vect_test_accuracy_sgd.append(sgd_clf.score_accuracy(X_test, y_test))
    
    plt.figure()
    plt.semilogx(vect_param, vect_train_accuracy_sgd, 'b', label="train")
    plt.semilogx(vect_param, vect_test_accuracy_sgd, 'r', label="test")
    plt.title("Accuracy of SGD vs. hyperparameter epsilon \non data set " + data_name)
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()


# get historic of accuracy
def get_hist_accuracy(x, y, hist_w, estimator):
    best_w = np.copy(estimator.w)
    hist_accuracy = []
    for w in hist_w:
        estimator.w = w
        accuracy = estimator.score_accuracy(x, y)
        hist_accuracy.append(accuracy)
    estimator.w = np.copy(best_w)
    return hist_accuracy


def plot_training(data, labels, nb_epoch, data_name, c_sgd, c_sdca, eps_sgd):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    
    # make estimator
    sgd = LogisticSGD(c=c_sgd, eps=eps_sgd)
    sgd_clf = LogisticRegression(optimizer=sgd)
    
    sdca = LogisticSDCA(c=c_sdca)
    sdca_clf = LogisticRegression(optimizer=sdca)
    
    # train estimator with history
    sgd_hist_w, sgd_hist_loss = sgd_clf.fit(X_train, y_train, epochs=nb_epoch, save_hist=True)
    sgd_hist_w = np.array(sgd_hist_w)
    
    # plot histories
    '''plt.figure()
    plt.title("Evolution of the weights : SGD")
    for d in range(sgd_hist_w.shape[1]):
        plt.plot(sgd_hist_w[:, d])'''
    
    plt.figure()
    plt.plot(sgd_hist_loss)
    plt.title("SGD learning loss vs. iteration\non data set " + data_name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    # final accuracy
    print("final accuracy SGD :", sgd_clf.score_accuracy(X_test, y_test))

    # do it again with SDCA !
    
    sdca_hist_w, sdca_hist_loss = sdca_clf.fit(X_train, y_train, epochs=nb_epoch, save_hist=True)
    sdca_hist_w = np.array(sdca_hist_w)
    
    '''plt.figure()
    plt.title("Evolution of the weights : SDCA")
    for d in range(sdca_hist_w.shape[1]):
        plt.plot(sdca_hist_w[:, d])'''
    
    plt.figure()
    plt.title("SDCA learning loss vs. iteration\non data set " + data_name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(sdca_hist_loss)
    
    # final accuracy
    print("final accuracy SDCA :", sdca_clf.score_accuracy(X_test, y_test))
    
    sgd_hist_accuracy = get_hist_accuracy(X_test, y_test, sgd_hist_w, sgd_clf)
    sdca_hist_accuracy = get_hist_accuracy(X_test, y_test, sdca_hist_w, sdca_clf)
    plt.figure()
    plt.plot(sgd_hist_accuracy, c='b', label="SGD")
    plt.plot(sdca_hist_accuracy, c='g', label="SDCA")
    plt.title("Test accuracy vs. iteration\non data set " + data_name)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()


def import_data_arrhythmia():
    # importation of data
    with open("datasets/Heart_disease/heart_disease_data.pkl", 'rb') as file:
        x = pickle.load(file)
        y = pickle.load(file)

    # dividing data in 2 classes
    y = np.where(y == 1, 1, -1)
    
    return x, y


def gaussian_kernel(x, Base_proj, h):
    a = ((x - Base_proj)/h)**2
    b = np.sum(a, axis=1)
    return np.exp(-b)


def gaussian_proj(data, Base_proj, h):
    N, dim = data.shape
    n, _ = Base_proj.shape
    data_proj = np.zeros((N, n))
    for i in range(N):
        data_proj[i,:] = gaussian_kernel(data[i,:], Base_proj, h)
    return data_proj


def create_base(data, prop=0.1):
    N, dim = data.shape
    n = int(prop*N)
    index = np.arange(N)
    np.random.shuffle(index)
    return data[index[:n],:]


def plot_gausian_arr():
    data_name = "Arrhythmia"
    nb_epoch = 40

    x, y = import_data_arrhythmia()

    h = 10
    prop = 0.1
    # base of projection
    Base_proj = create_base(x, prop=prop)
    x = gaussian_proj(x, Base_proj, h)

    # normalization
    normalizer = Normalizer(x)
    x = normalizer.normalize(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    # make estimator
    sgd = LogisticSGD(c=10**3, eps=10**-6)
    sgd_clf = LogisticRegression(optimizer=sgd)

    sdca = LogisticSDCA(c=10**-1)
    sdca_clf = LogisticRegression(optimizer=sdca)

    # train estimator with history
    sgd_hist_w, sgd_hist_loss = sgd_clf.fit(X_train, y_train, epochs=nb_epoch, save_hist=True)
    sgd_hist_w = np.array(sgd_hist_w)

    sdca_hist_w, sdca_hist_loss = sdca_clf.fit(X_train, y_train, epochs=nb_epoch, save_hist=True)
    sdca_hist_w = np.array(sdca_hist_w)

    plt.figure()
    plt.plot(sgd_hist_loss)
    plt.title("SGD learning loss vs. iteration\non data set " + data_name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.figure()
    plt.title("SDCA learning loss vs. iteration\non data set " + data_name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(sdca_hist_loss)

    sgd_hist_accuracy = get_hist_accuracy(X_test, y_test, sgd_hist_w, sgd_clf)
    sdca_hist_accuracy = get_hist_accuracy(X_test, y_test, sdca_hist_w, sdca_clf)
    plt.figure()
    plt.plot(sgd_hist_accuracy, c='b', label="SGD")
    plt.plot(sdca_hist_accuracy, c='g', label="SDCA")
    plt.title("Test accuracy vs. iteration\non data set " + data_name)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()


def eval_h(data, labels, vect_param, nb_epoch, data_name, prop_base, c_sgd, c_sdca, eps_sgd):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)

    Base_proj = create_base(x_train, prop=prop_base)
    dim, _ = Base_proj.shape
    print("dim :", dim)

    vect_train_accuracy_sgd = []
    vect_train_accuracy_sdca = []

    vect_test_accuracy_sgd = []
    vect_test_accuracy_sdca = []

    for param in vect_param:
        X_train = gaussian_proj(x_train, Base_proj, param)
        X_test = gaussian_proj(x_test, Base_proj, param)

        # normalisation
        normalizer = Normalizer(X_train)
        X_train = normalizer.normalize(X_train)
        X_test = normalizer.normalize(X_test)

        # make estimator
        sgd = LogisticSGD(c=c_sgd, eps=eps_sgd)
        sgd_clf = LogisticRegression(optimizer=sgd)

        sdca = LogisticSDCA(c=c_sdca)
        sdca_clf = LogisticRegression(optimizer=sdca)

        # train estimators without history
        sgd_clf.fit(X_train, y_train, epochs=nb_epoch, save_hist=False)
        sdca_clf.fit(X_train, y_train, epochs=nb_epoch, save_hist=False)

        vect_train_accuracy_sgd.append(sgd_clf.score_accuracy(X_train, y_train))
        vect_train_accuracy_sdca.append(sdca_clf.score_accuracy(X_train, y_train))

        vect_test_accuracy_sgd.append(sgd_clf.score_accuracy(X_test, y_test))
        vect_test_accuracy_sdca.append(sdca_clf.score_accuracy(X_test, y_test))

    plt.figure()
    plt.semilogx(vect_param, vect_train_accuracy_sgd, 'b', label="train")
    plt.semilogx(vect_param, vect_test_accuracy_sgd, 'r', label="test")
    plt.title("SGD accuracy vs. hyperparameter h\nfor gaussian projection (dim = {})\n on data set ".format(dim) + data_name)
    plt.xlabel("h")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.figure()
    plt.semilogx(vect_param, vect_train_accuracy_sdca, 'b', label="train")
    plt.semilogx(vect_param, vect_test_accuracy_sdca, 'r', label="test")
    plt.title("SDCA accuracy vs. hyperparameter h\nfor gaussian projection (dim = {})\n on data set ".format(dim) + data_name)
    plt.xlabel("h")
    plt.ylabel("Accuracy")
    plt.legend()


def main1():
    x, y = load_adults_dataset()

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
    
    plt.show()


def main2():
    # Gaussian projection

    # Arrhythmia
    vect_h = 10**np.linspace(-2, 7, 70)
    data_name = "Arrhythmia"
    x, y = import_data_arrhythmia()
    nb_epoch = 10
    eval_h(x, y, vect_h, nb_epoch, data_name, 0.2, 10**3, 10**-1, 10**-5)

    # Adults
    vect_h = 10**np.linspace(-2, 7, 70)
    data_name = "Adults"
    x, y = load_adults_dataset()
    N, dim = x.shape
    n_sample = 1200
    index = np.arange(N)
    np.random.shuffle(index)
    x = x[index[:n_sample],:]
    y = y[index[:n_sample]]
    nb_epoch = 10
    eval_h(x, y, vect_h, nb_epoch, data_name, 0.2, 10**4, 5*10**-2, 5*10**-6)

    plt.show()


if __name__ == '__main__':
    main2()
