import numpy as np
import matplotlib.pyplot as plt


class classifier_SGD:
    def __init__(self, maxIter=500, eps=10**-4, C=1, nb_coefs=10):
        self.maxIter = maxIter
        self.eps = eps
        self.C = C
        self.w = 0
        self.nb_coefs = nb_coefs
        
        #stockage des itérations en cours
        self.vect_coefs = []
        self.vect_w = []
        
        #score au fil de l'apprentissage
        self.vect_score_learn = []
        
        #risque à minimiser
        self.risk = []
    
    def fit(self, X, y):
        N, dim = X.shape
        self.w = np.zeros((dim, 1))
        
        index = np.arange(dim)
        np.random.shuffle(index)
        self.index_coefs = index[:self.nb_coefs]
        
        i = 0
        j = 0
        ind = np.arange(N)
        np.random.shuffle(ind)
        self.vect_w.append(self.w)
        while i < self.maxIter:
            if (j == N):
                np.random.shuffle(ind)
                j = 0
            self.w += self.eps*(self.C * y[j] * X[j,:].reshape(-1,1) / (1 + np.exp(y[j] * X[j,:].dot(self.w))) - self.w)
            self.vect_coefs.append(self.w[self.index_coefs])
            self.vect_score_learn.append(self.score(X, y))
            self.risk.append(self.calc_risque(X, y))
            i += 1
            j += 1
            self.vect_w.append(self.w.copy())
    
    def predict(self, x):
        x = x.reshape(-1,1)
        y_proba = 1/(1 + np.exp(-self.w.transpose().dot(x)))
        if y_proba > 0.5:
            y_predict = 1
        else:
            y_predict = -1
        return y_predict
    
    def score(self, X_test, y_test):
        n, dim = X_test.shape
        y_predict = np.zeros((n, 1))
        for i in range(n):
            y_predict[i, 0] = self.predict(X_test[i,:])
        return np.sum(np.where(y_predict == y_test, 1, 0))/len(y_test)
    
    def calc_risque(self, X_test, y_test):
        z = X_test.dot(self.w)
        zz = np.multiply(z, y_test)
        r = self.C * np.sum(np.log(1 + np.exp(-zz))) + 0.5*self.w.transpose().dot(self.w)
        return r[0,0]


