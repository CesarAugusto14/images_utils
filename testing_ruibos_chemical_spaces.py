import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utils.io import data
from os.path import join as pj
from sklearn.manifold import MDS
from scipy.interpolate import griddata, Rbf

# Loading CHEMBL target. 
dwd = 'data/CHEMBL2231'
raw_data, distance = data(dwd)
#%% Checking the chemical space using MDS. 
target = raw_data["pChEMBL Value"]
mds = MDS(random_state=2021, dissimilarity='precomputed')

# Saving it into a dataframe.
xy = pd.DataFrame(
        mds.fit_transform(distance),
        index=distance.index,
        columns=['x', 'y']
    )

x = xy['x'].values
y = xy['y'].values
z = target.values

# For Plotting: 
xi = np.linspace(min(x), max(x), 500)
yi = np.linspace(min(y), max(y), 500)

xi, yi = np.meshgrid(xi, yi)

# NOTE: What is this? Why are we mapping this to a gaussian? 
func = Rbf(x, y, z, kind='gaussian', epsilon=0.02, smooth=0.1)
zi = func(xi, yi)

plt.figure(figsize=(8, 6))
plt.contourf(xi, yi, zi, levels=25)
plt.colorbar()
plt.title("The chemical space of target " + dwd.strip("/").split("/")[-1])
plt.scatter(xy["x"], xy["y"], s=1)
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.show()
# NOTE: As I am going to use the same code for other targets & other dim reduction 
# methods, I will save this into a function. (Later)
