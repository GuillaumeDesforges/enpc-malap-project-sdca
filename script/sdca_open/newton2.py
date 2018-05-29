#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


def newton2(a, b, c1, c2, xi=0.01, threshold=1e-6, max_iter=10, verbose=False, plot=False):
    if verbose:
        print("Newton 2")
    s = c1 + c2
    if verbose:
        print('s', s)
    zm = (c2-c1)/2
    if verbose:
        print('zm', zm)
    
    t = 1 if zm >= -b/a else 2

    bt = b  if t == 1 else -b
    ct = c1 if t == 1 else c2
    if verbose:
        print('bt', bt)
        print('ct', ct)

    gt = lambda Zt: Zt*np.log(Zt)+(s-Zt)*np.log(s-Zt)+(a/2)*((Zt-ct)**2)+bt*(Zt-ct)
    def gt_diff1(Zt):
        return np.log(Zt/(s-Zt))+a*(Zt-ct)+bt
    def gt_diff2(Zt):
        return a+s/(Zt*(s-Zt))
    
    Zt = s/2
    grad = gt_diff1(Zt)
    k = 0
    while abs(grad) > threshold:
        if k > max_iter:
            # raise Exception("Max iter reached !")
            break
        
        if verbose:
            print('loss', gt(Zt))

        grad = gt_diff1(Zt)
        if verbose:
            print('grad', grad)

        d = - grad/gt_diff2(Zt)
        Zt_knext = xi*Zt if Zt+d <= 0 else Zt+d
        if verbose:
            print('Zt_knext', Zt_knext)
        
        if plot:
            z = np.linspace(0.0001, c1+c2-0.0001)
            plt.plot(z, gt(z), c='blue')
            plt.plot(z, (z-Zt)*gt_diff1(Zt)+gt(Zt), c='red', linestyle='--')
            plt.scatter([Zt], [gt(Zt)], c='red')
            plt.scatter([Zt_knext], [gt(Zt_knext)], c='green')
            plt.show()
        
        Zt = Zt_knext
        k += 1

    if t == 1:
        Z1 = Zt
        Z2 = s-Z1
    else:
        Z2 = Zt
        Z1 = s-Z2

    return Z1, Z2

