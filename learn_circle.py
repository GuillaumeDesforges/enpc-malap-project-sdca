import numpy as np

from plot_learning import plot_learning
from engine.utils.data_gen import gen_circle

np.random.seed(222)

n1, n2 = 100, 100
r1, r2 = 3, 1
x1 = gen_circle(radius=r1, size=n1)
x2 = gen_circle(radius=r2, size=n2)
x = np.concatenate([x1, x2])

y1 = np.ones(shape=n1)
y2 = -np.ones(shape=n2)
y = np.concatenate([y1, y2])

plot_learning(x, y)
