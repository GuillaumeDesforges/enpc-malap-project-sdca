import numpy as np
import matplotlib.pyplot as plt
import engine.utils.malaptools as malaptools

from engine.estimators.logistic_regression import LogisticRegression
from engine.optimizers.sgd_logistic import LogisticSGD

# make data
np.random.seed(0)

n1, n2 = 100, 100
x1 = np.random.normal(loc=(-1, -1), scale=(1, 1), size=(n1, 2))
x2 = np.random.normal(loc=(1, 1), scale=(1, 1), size=(n2, 2))
x = np.concatenate([x1, x2])

y1 = -np.ones(shape=n1)
y2 = np.ones(shape=n2)
y = np.concatenate([y1, y2])

# make estimator
sgd = LogisticSGD(1, 1e-3)
sgd_clf = LogisticRegression(optimizer=sgd)

# train estimator with history
hist_w, hist_loss = sgd_clf.fit(x, y, epochs=20, save_hist=True)
hist_w = np.array(hist_w)

# verify result
malaptools.plot_frontiere(x, sgd_clf.predict)
malaptools.plot_data(x, y)
plt.show()

# plot histories
for d in range(hist_w.shape[1]):
    plt.plot(hist_w[:, d])
plt.show()

plt.plot(hist_loss)
plt.show()
