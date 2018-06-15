from itertools import product
from typing import Callable

import numpy as np


def identity_projection(x: np.ndarray):
    return x


def build_polynomial_projection(degree: int) -> Callable[[np.ndarray], np.ndarray]:
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


def build_gaussian_projection(data: np.ndarray, sampling_rate: float=0.1, kernel_h=1) -> Callable[[np.ndarray], np.ndarray]:
    n = data.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    sampling_size = int(n * sampling_rate)
    sampling_indices = indices[:sampling_size]

    sample = data[sampling_indices]

    def gaussian_projection(x: np.ndarray):
        # size of the current input
        n_x, d = x.shape
        # we repeat each row enough times to compute gaussians
        x = np.repeat(x, sampling_size, axis=0)
        # we tile the sample enough times to compute gaussian
        x_sample = np.tile(sample, (n_x, 1))
        # compute norm, each row is a pair (input_i, sample_j)
        norm = np.linalg.norm(x - x_sample, axis=1)
        # regroup distances to match data shape
        norm = np.reshape(norm, (n_x, -1))
        # compute through gaussian
        output = np.exp(norm**2 / kernel_h)

        return output

    return gaussian_projection
