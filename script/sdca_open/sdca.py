import numpy as np
import matplotlib.pyplot as plt

from .newton2 import newton2


def sdca(X, Y, C=10, eps1=0.1, eps2=0.5, n_iter=1000, verbose=False, plot_alphas=False):
    l, n = X.shape
    if verbose:
        print('(l, n)', (l, n))

    alpha = min(eps1*C, eps2) * np.ones(l)
    alphas = [np.copy(alpha)]
    alpha_prim = C - alpha
    if verbose:
        print('alpha', alpha)
    
    w = np.sum(alpha[:, np.newaxis]*Y[:, np.newaxis]*X, axis=0)
    if verbose:
        print('w', w)
    
    for k in range(n_iter):
        i = np.random.randint(0, l)
        if verbose:
            print('i', i)

        x_i = X[i]
        y_i = Y[i]
        if verbose:
            print('x_i', x_i)
            print('y_y', y_i)
        
        alpha_i = alpha[i]
        alpha_prim_i = alpha_prim[i]
        if verbose:
            print('alpha_i', alpha_i)
            print('alpha\'_i', alpha_prim_i)
        
        Q_ii = np.dot(x_i, x_i)
        if verbose:
            print('Q_ii', Q_ii)
        
        c1 = alpha_i
        c2 = alpha_prim_i
        a = Q_ii
        b = y_i*np.dot(w, x_i)
        if verbose:
            print('c1', c1).idea/
            print('c2', c2)
            print('a', a)
            print('b', b)
        
        Z1, Z2 = newton2(a, b, c1, c2, verbose=verbose)
        if verbose:
            print(Z1)
            print(Z2)
        
        w = w + (Z1 - alpha_i)*y_i*x_i
        if verbose:
            print('w', w)
        
        alpha[i] = Z1
        alpha_prim[i] = Z2
        
        alphas.append(np.copy(alpha))
    
    alphas = np.stack(alphas)
    
    if verbose:
        print(alphas)
    
    if plot_alphas:
        for i in range(l):
            plt.plot(alphas[:, i], label='alpha[{}]'.format(i))
        plt.title("Evolution des alpha[i]")
        plt.xlabel("Iterations")
        plt.ylabel("alpha[i]")
        plt.show()
    
    return w

