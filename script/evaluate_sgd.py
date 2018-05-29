from sgd.classifier_sgd import classifier_SGD
import numpy as np
import matplotlib.pyplot as plt



# génération de données gaussiennes
d = 30
N = 50
sigma = 5
T = 700

# data
data = np.zeros((N, d))
labels = np.zeros((N, 1))
for i in range(N):
    a = np.random.randint(2)
    if a == 0:
        data[i,:] = np.random.normal(loc=-1, scale=sigma, size=d)
        labels[i,0] = -1
    elif a == 1:
        data[i,:] = np.random.normal(loc=1, scale=sigma, size=d)
        labels[i,0] = 1


# apprentissage
clf = classifier_SGD()
clf.fit(data, labels)
print(clf.score(data, labels))


# évolution de quelques coefficients
plt.figure()
z = clf.vect_coefs[0]
print(len(z))
for i in range(len(z)):
    plt.plot([x[i] for x in clf.vect_coefs])
plt.show()

# évolution du score
plt.figure()
plt.plot(clf.vect_score_learn)
plt.show()


# évolution du risque empirique
plt.figure()
plt.plot(clf.risk)
plt.show()