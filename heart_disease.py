import numpy as np
import pickle

import matplotlib.pyplot as plt
import engine.utils.data_gen as data_gen
import engine.utils.data_sets as data_sets
import engine.utils.malaptools as malaptools


from engine.estimators.logistic_regression import LogisticRegression
from engine.optimizers.sdca_logistic import LogisticSDCA
from engine.optimizers.sgd_logistic import LogisticSGD
from engine.optimizers.sdca_square import SquareSDCA
from engine.optimizers.sgd_square import SquareSGD

nomFichier = "datasets\Heart_disease\heart_disease_data.pkl"


# importation of data
with open(nomFichier, 'rb') as input:
    X = pickle.load(input)
    Y = pickle.load(input)

# dividing data in 2 classes
Y = np.where(Y == 1, 1, -1)

if False:
    # make estimator
    sgd = LogisticSGD(c=1, eps=1e-6)
    sgd_clf = LogisticRegression(optimizer=sgd)
    
    # train estimator with history
    sgd_hist_w, sgd_hist_loss = sgd_clf.fit(X, Y, epochs=20, save_hist=True)
    sgd_hist_w = np.array(sgd_hist_w)
    
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
    
    # final accuracy
    print("final accuracy SGD :", sgd_clf.score_accuracy(X, Y))
    
    
    # do it again with SDCA !
    
    sdca = LogisticSDCA(c=1)
    sdca_clf = LogisticRegression(optimizer=sdca)
    
    sdca_hist_w, sdca_hist_loss, sdca_hist_alpha = sdca_clf.fit(X, Y, epochs=20, save_hist=True)
    sdca_hist_w = np.array(sdca_hist_w)
    sdca_hist_alpha = np.array(sdca_hist_alpha)
    
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
    
    # final accuracy
    print("final accuracy SDCA :", sdca_clf.score_accuracy(X, Y))


if True:
    smooth_sgd_hist_loss = [min(sgd_hist_loss[:i]) for i in range(1, len(sgd_hist_loss))]
    smooth_sdca_hist_loss = [min(sdca_hist_loss[:i]) for i in range(1, len(sdca_hist_loss))]
    
    plt.figure()
    plt.plot(smooth_sgd_hist_loss, c='b', label="SGD")
    plt.plot(smooth_sdca_hist_loss, c='g', label="SDCA")
    plt.title("Evolution of the loss with iterations")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    