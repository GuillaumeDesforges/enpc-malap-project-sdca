import numpy as np
import pickle

nomFichier = "arrhythmia.data"

with open(nomFichier, 'r') as input:
    data_brut = input.read()

data_lines = data_brut.splitlines()
data_all = [x.split(',') for x in data_lines]

print("Number of instances :", len(data_all))

# conversion adaptée des données

data = []
dim_problem = set()
for x in data_all:
    if not('?' in x):
        data.append(list(map(float, x)))
    else:
        for i in range(len(x)):
            if x[i] == '?':
                x[i] = '0'
        data.append(list(map(float, x)))

print("nb of valid instances :", len(data))

# conversion to numpy array
X = np.array(data)
Y = X[:,-1]
X = X[:,:-1]
print(X.shape)
print(Y.shape)

# save data
with open("heart_disease_data.pkl", 'wb') as output:
    pickle.dump(X, output)
    pickle.dump(Y, output)