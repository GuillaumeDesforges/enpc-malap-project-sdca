from itertools import product

import numpy as np


def identity_projection(x: np.ndarray):
    return x


def build_polynomial_projection(degree: int):
    def polynomial_projection(x: np.ndarray):
        if x.ndim == 1:
            n, d = len(x), 1
            x = np.reshape(x, (n, d))
        else:
            n, d = x.shape
        polys = []
        for superscripts in product(range(degree+1), repeat=d):
            if sum(superscripts) > degree:
                continue
            polys.append(np.prod([x[:, dim]**superscripts[dim] for dim in range(d)], axis=0))
        return np.stack(polys, axis=-1)

    return polynomial_projection