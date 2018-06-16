#!/usr/bin/python

from numpy import log, ceil


def compute(L, n, c, eps):
    return n + max(0, ceil(n*log( (c*n)/(2*L**2) ))) + (20*L**2)/(c*eps)


if __name__ == '__main__':
    L = 1
    n = 452
    c = 5e-2
    eps = 1e-3
    
    T = compute(L, n, c, eps)

    print('T =', T)
