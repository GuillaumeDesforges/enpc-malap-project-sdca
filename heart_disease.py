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

from engine.utils.normalize import Normalizer

from sklearn.model_selection import train_test_split

nomFichier = "datasets\Heart_disease\heart_disease_data.pkl"

## Global param

nb_epoch = 30

## Data

# importation of data
with open(nomFichier, 'rb') as input:
    X = pickle.load(input)
    Y = pickle.load(input)

# dividing data in 2 classes
Y = np.where(Y == 1, 1, -1)

# normalisation
normalizer = Normalizer(X)
Xnorm = normalizer.normalize(X)


## paramètre C par validation croisée

if False:
    X_train, X_test, y_train, y_test = train_test_split(Xnorm, Y, test_size=0.1)
    vect_param = 10**np.linspace(-2, 4, 30)
    
    vect_train_accuracy_sgd = []
    vect_train_accuracy_sdca = []
    
    vect_test_accuracy_sgd = []
    vect_test_accuracy_sdca = []
    
    for param in vect_param:
        # make estimator
        sgd = LogisticSGD(c=param, eps=1e-6)
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
    plt.plot(np.log10(vect_param), vect_train_accuracy_sgd, 'b', label="train")
    plt.plot(np.log10(vect_param), vect_test_accuracy_sgd, 'r', label="test")
    plt.title("Accuracy SGD")
    plt.legend()
    
    plt.figure()
    plt.plot(np.log10(vect_param), vect_train_accuracy_sdca, 'b', label="train")
    plt.plot(np.log10(vect_param), vect_test_accuracy_sdca, 'r', label="test")
    plt.title("Accuracy SDCA")
    plt.legend()

'''
Bon paramètre => c=10
'''
    

## Training

if False:
    # make estimator
    sgd = LogisticSGD(c=10, eps=1e-7)
    sgd_clf = LogisticRegression(optimizer=sgd)
    
    sdca = LogisticSDCA(c=10)
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

if False:
    # make estimator
    sgd = LogisticSGD(c=10, eps=1e-38)
    sgd_clf = LogisticRegression(optimizer=sgd)
    
    sdca = LogisticSDCA(c=10)
    sdca_clf = LogisticRegression(optimizer=sdca)
    
    nb_epoch = 5
    
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


## Gaussian projection

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

if False:
    X_train, X_test, y_train, y_test = train_test_split(Xnorm, Y, test_size=0.1)
    vect_param = np.linspace(0.02, 0.8, 20)
    
    vect_train_accuracy_sgd = []
    vect_train_accuracy_sdca = []
    
    vect_test_accuracy_sgd = []
    vect_test_accuracy_sdca = []
    
    h = 10**2
    
    for param in vect_param:
        # make estimator
        sgd = LogisticSGD(c=10, eps=1e-7)
        sgd_clf = LogisticRegression(optimizer=sgd)
        
        sdca = LogisticSDCA(c=10)
        sdca_clf = LogisticRegression(optimizer=sdca)
        
        # base of projection
        Base_proj = create_base(X_train, prop=param)
        X_train_proj = gaussian_proj(X_train, Base_proj, h)
        X_test_proj = gaussian_proj(X_test, Base_proj, h)
        
        nb_epoch = 20
        
        # train estimators without history
        sgd_clf.fit(X_train_proj, y_train, epochs=nb_epoch, save_hist=False)
        sdca_clf.fit(X_train_proj, y_train, epochs=nb_epoch, save_hist=False)
        
        vect_train_accuracy_sgd.append(sgd_clf.score_accuracy(X_train_proj, y_train))
        vect_train_accuracy_sdca.append(sdca_clf.score_accuracy(X_train_proj, y_train))
        
        vect_test_accuracy_sgd.append(sgd_clf.score_accuracy(X_test_proj, y_test))
        vect_test_accuracy_sdca.append(sdca_clf.score_accuracy(X_test_proj, y_test))
    
    plt.figure()
    plt.plot(vect_param, vect_train_accuracy_sgd, 'b', label="train")
    plt.plot(vect_param, vect_test_accuracy_sgd, 'r', label="test")
    plt.title("Accuracy SGD")
    plt.legend()
    
    plt.figure()
    plt.plot(vect_param, vect_train_accuracy_sdca, 'b', label="train")
    plt.plot(vect_param, vect_test_accuracy_sdca, 'r', label="test")
    plt.title("Accuracy SDCA")
    plt.legend()


plt.show()