# trial_minsom.py

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# Minisom library and module is used for performing Self Organizing Maps

from minisom import MiniSom
# Loading Data

data = pd.read_csv('Credit_Card_Applications.csv')

# Defining X variables for the input of SOM
X = data.iloc[:, 1:14].values
y = data.iloc[:, -1].values
# X variables:
# print(pd.DataFrame(X))

sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Set the hyper parameters
som_grid_rows = 5
som_grid_columns = 5
iterations = 20000
sigma = 1
learning_rate = 0.5

# define SOM:

som = MiniSom(x = som_grid_rows, y = som_grid_columns, input_len=13, sigma=sigma, learning_rate=learning_rate)

# Initializing the weights

som.random_weights_init(X)

# Training

som.train_random(X, iterations)
# Weights are:
wts = som.get_weights()

# Get the network weights
nw = som.winner(X[0])
print(nw)
print(type(wts))
# # Shape of the weight are:
print(wts.shape)
# Returns the distance map from the weights
print(som.distance_map())

from pylab import plot, axis, show, pcolor, colorbar, bone
bone()
pcolor(som.distance_map().T)       # Distance map as background
colorbar()
show()