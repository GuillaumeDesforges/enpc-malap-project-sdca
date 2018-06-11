import numpy as np
import pickle

with open("heart_disease_data.pkl", 'rb') as input:
    X = pickle.load(input)
    Y = pickle.load(input)