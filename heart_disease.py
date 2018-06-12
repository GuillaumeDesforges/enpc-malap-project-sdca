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

## Global param

nb_epoch = 5

## Data

# importation of data
with open(nomFichier, 'rb') as input:
    X = pickle.load(input)
    Y = pickle.load(input)

# dividing data in 2 classes
Y = np.where(Y == 1, 1, -1)

# normalisation
def normalize(mat):
    N, dim = mat.shape
    m = np.mean(mat, axis=0)
    z = mat
    for i in range(dim):
        if m[i] != 0:
            z[:,i] = mat[:,i] / m[i]
    return z

Xnorm = normalize(X)
    
## Estimators

# make estimator
sgd = LogisticSGD(c=1, eps=1e-20)
sgd_clf = LogisticRegression(optimizer=sgd)

sdca = LogisticSDCA(c=1)
sdca_clf = LogisticRegression(optimizer=sdca)

## Training

if True:
    # train estimator with history
    sgd_hist_w, sgd_hist_loss = sgd_clf.fit(Xnorm, Y, epochs=nb_epoch, save_hist=True)
    sgd_hist_w = np.array(sgd_hist_w)
    
    # plot histories
    '''plt.figure()
    plt.title("Evolution of the weights : SGD")
    for d in range(sgd_hist_w.shape[1]):
        plt.plot(sgd_hist_w[:, d])
    plt.show()'''
    
    plt.figure()
    plt.title("Evolution of the loss : SGD")
    plt.plot(sgd_hist_loss)
    plt.show()
    
    # final accuracy
    print("final accuracy SGD :", sgd_clf.score_accuracy(Xnorm, Y))
    
    
    # do it again with SDCA !
    
    sdca_hist_w, sdca_hist_loss, sdca_hist_alpha = sdca_clf.fit(Xnorm, Y, epochs=nb_epoch, save_hist=True)
    sdca_hist_w = np.array(sdca_hist_w)
    sdca_hist_alpha = np.array(sdca_hist_alpha)
    
    '''plt.figure()
    plt.title("Evolution of the weights : SDCA")
    for d in range(sdca_hist_w.shape[1]):
        plt.plot(sdca_hist_w[:, d])
    plt.show()
    
    plt.figure()
    plt.title("Evolution of the dual variables : SDCA")
    for n in range(sdca_hist_alpha.shape[1]):
        plt.plot(sdca_hist_alpha[:, n])
    plt.show()'''
    
    plt.figure()
    plt.title("Evolution of the loss : SDCA")
    plt.plot(sdca_hist_loss)
    plt.show()
    
    # final accuracy
    print("final accuracy SDCA :", sdca_clf.score_accuracy(Xnorm, Y))


# smoothed loss : without oscillations of convergence

def calc_smooth(liste):
    return [min(liste[:i]) for i in range(1, len(liste))]

if False:
    smooth_sgd_hist_loss = calc_smooth(sgd_hist_loss)
    smooth_sdca_hist_loss = calc_smooth(sdca_hist_loss)
    
    plt.figure()
    plt.plot(smooth_sgd_hist_loss, c='b', label="SGD")
    plt.plot(smooth_sdca_hist_loss, c='g', label="SDCA")
    plt.title("Evolution of the loss with iterations")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

def proj_degr2(X):
    N, dim = X.shape
    Z = np.zeros((N, dim**2))
    # xi * xj
    k = 0
    for i in range(dim):
        for j in range(i, dim):
            Z[:,k] = np.multiply(X[:,i], X[:,j])
            k += 1
    return Z

if False:
    np.random.seed(1)
    X_proj = proj_degr2(X)
    X_proj_norm = normalize(X_proj)
    # train estimator with history
    sgd_hist_w_proj, sgd_hist_loss_proj = sgd_clf.fit(X_proj_norm, Y, epochs=1, save_hist=True)
    sgd_hist_w_proj = np.array(sgd_hist_w_proj)
    
    smooth_sgd_hist_loss_proj = calc_smooth(sgd_hist_loss_proj)
    plt.figure()
    plt.plot(sgd_hist_loss_proj, c='b', label="SGD")
    plt.title("Evolution of the loss with iterations : projection degree 2")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.show()