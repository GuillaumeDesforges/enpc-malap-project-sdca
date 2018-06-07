import numpy as np


def gen_circle(center=(0, 0), radius=1, noise=0.1, size=100):
    angles = 2*np.pi * np.random.random(size=size)
    radius = radius * np.random.random(size=size)
    noises = np.random.normal(loc=0, scale=noise, size=(size, 2))
    x = np.stack([center[0] + radius*np.cos(angles), center[1] + radius*np.sin(angles)], axis=-1) + noises
    return x

