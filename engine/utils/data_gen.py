import numpy as np


def gen_circle(center=(0, 0), radius=1, noise=0.1, size=100):
    angles = 2*np.pi * np.random.random(size=size)
    radius = radius * np.random.random(size=size)
    noises = np.random.normal(loc=0, scale=noise, size=(size, 2))
    x = np.stack([center[0] + radius*np.cos(angles), center[1] + radius*np.sin(angles)], axis=-1) + noises
    return x


def gen_circle_data(n1=100, n2=100, r1=3, r2=1):
    np.random.seed(0)

    x1 = gen_circle(radius=r1, size=n1)
    x2 = gen_circle(radius=r2, size=n2)
    x = np.concatenate([x1, x2])

    y1 = np.ones(shape=n1)
    y2 = -np.ones(shape=n2)
    y = np.concatenate([y1, y2])

    return x, y


def gen_gaussian_data(n1=100, n2=100):
    np.random.seed(0)

    x1 = np.random.normal(loc=(-1, -1), scale=(1, 1), size=(n1, 2))
    x2 = np.random.normal(loc=(1, 1), scale=(1, 1), size=(n2, 2))
    x = np.concatenate([x1, x2])

    y1 = -np.ones(shape=n1)
    y2 = np.ones(shape=n2)
    y = np.concatenate([y1, y2])

    return x, y