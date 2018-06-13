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

nb_epoch = 10

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



## Training

if False:
    # make estimator
    sgd = LogisticSGD(c=1, eps=1e-6)
    sgd_clf = LogisticRegression(optimizer=sgd)
    
    sdca = LogisticSDCA(c=1)
    sdca_clf = LogisticRegression(optimizer=sdca)
    
    # train estimator with history
    sgd_hist_w, sgd_hist_loss = sgd_clf.fit(Xnorm, Y, epochs=nb_epoch, save_hist=True)
    sgd_hist_w = np.array(sgd_hist_w)
    
    # plot histories
    '''plt.figure()
    plt.title("Evolution of the weights : SGD")
    for d in range(sgd_hist_w.shape[1]):
        plt.plot(sgd_hist_w[:, d])'''
    
    plt.figure()
    plt.title("Evolution of the loss : SGD")
    plt.plot(sgd_hist_loss)
    
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
    
    plt.figure()
    plt.title("Evolution of the dual variables : SDCA")
    for n in range(sdca_hist_alpha.shape[1]):
        plt.plot(sdca_hist_alpha[:, n])'''
    
    plt.figure()
    plt.title("Evolution of the loss : SDCA")
    plt.plot(sdca_hist_loss)
    
    # final accuracy
    print("final accuracy SDCA :", sdca_clf.score_accuracy(Xnorm, Y))


# smoothed loss : without oscillations of convergence

def calc_smooth(liste):
    list_smooth = np.zeros(len(liste))
    list_smooth[0] = liste[0]
    for i in range(1, len(liste)):
        list_smooth[i] = min(list_smooth[i-1], liste[i])
    return list_smooth

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

def proj_degr2(X):
    N, dim = X.shape
    new_dim = int(dim*(dim+3)/2)
    Z = np.zeros((N, new_dim))
    # xi * xj
    k = 0
    for i in range(dim):
        Z[:,k] = X[:,i]
        Z[:,k+1] = np.multiply(X[:,i], X[:,i])
        k += 2
        for j in range(i+1, dim):
            Z[:,k] = np.multiply(X[:,i], X[:,j])
            k += 1
    return Z

if True:
    # make estimator
    sgd = LogisticSGD(c=1, eps=1e-35)
    sgd_clf = LogisticRegression(optimizer=sgd)
    
    sdca = LogisticSDCA(c=1)
    sdca_clf = LogisticRegression(optimizer=sdca)
    
    #np.random.seed(1)
    X_proj = proj_degr2(X)
    X_proj_norm = normalize(X_proj)
    # train estimator with history
    sgd_hist_w_proj, sgd_hist_loss_proj = sgd_clf.fit(X_proj_norm, Y, epochs=nb_epoch, save_hist=True)
    sgd_hist_w_proj = np.array(sgd_hist_w_proj)
    
    sdca_hist_w_proj, sdca_hist_loss_proj, sdca_hist_alpha_proj = sdca_clf.fit(X_proj_norm, Y, epochs=nb_epoch, save_hist=True)
    sdca_hist_w_proj = np.array(sdca_hist_w_proj)
    
    smooth_sgd_hist_loss_proj = calc_smooth(sgd_hist_loss_proj)
    smooth_sdca_hist_loss_proj = calc_smooth(sdca_hist_loss_proj)
    plt.figure()
    plt.plot(sgd_hist_loss_proj)
    plt.title("Evolution of the loss SGD : projection degree 2")
    plt.figure()
    plt.plot(sdca_hist_loss_proj)
    plt.title("Evolution of the loss SDCA : projection degree 2")
    plt.figure()
    plt.plot(smooth_sgd_hist_loss_proj, c='b', label="SGD")
    plt.plot(smooth_sdca_hist_loss_proj, c='g', label="SDCA")
    plt.title("Evolution of the loss with iterations : projection degree 2")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()


plt.show()