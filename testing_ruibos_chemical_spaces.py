import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utils.io import data
from os.path import join as pj
from sklearn.manifold import MDS, TSNE
from scipy.interpolate import griddata, Rbf
from sklearn.metrics import pairwise_distances

#Import KFOLD
from sklearn.model_selection import KFold
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
N = 100
xi = np.linspace(min(x), max(x), N)
yi = np.linspace(min(y), max(y), N)

xi, yi = np.meshgrid(xi, yi)

# NOTE: What is this? Why are we mapping this to a gaussian? 
func = Rbf(x, y, z, kind='gaussian', epsilon = 0.02, smooth = 0.1)
zi = func(xi, yi)

plt.figure(figsize=(8, 6))
plt.contourf(xi, yi, zi, levels=25)
plt.colorbar()
plt.title("The chemical space of target " + dwd.strip("/").split("/")[-1])
plt.scatter(xy["x"], xy["y"], s=10)
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.show()
# NOTE: As I am going to use the same code for other targets & other dim reduction 
# methods, I will save this into a function. (Later)

# Now we will use the KNN to find the nearest neighbors of a given compound.

cv_fold = 3
kf = KFold(n_splits=cv_fold, shuffle=True, random_state=2021)
i_fold = 1
top_k = 3

for train_idx, test_idx in kf.split(target):
    
#%%

# NOTE: How can I perform the dimensionality reduction if I just have the precoputed distances?
#
# Ans: check the data_ECFP4.parquet file

# df = pd.read_parquet(pj(dwd, 'data_ECFP4.parquet'))
# # Perform tsne

# tsne = TSNE(n_components=2, random_state=2021,)
# xy = pd.DataFrame(
#         tsne.fit_transform(df),
#         index=df.index,
#         columns=['x', 'y']
#     )

# x = xy['x'].values
# y = xy['y'].values
# z = target.values

# # For Plotting:
# N = 200
# xi = np.linspace(min(x), max(x), N)
# yi = np.linspace(min(y), max(y), N)

# xi, yi = np.meshgrid(xi, yi)

# # NOTE: What is this? Why are we mapping this to a gaussian?
# func = Rbf(x, y, z, kind='gaussian', epsilon = 0.02, smooth = 0.1)
# zi = func(xi, yi)

# plt.figure(figsize=(8, 6))
# plt.contourf(xi, yi, zi, levels=25)
# plt.colorbar()
# plt.title("The chemical space of target " + dwd.strip("/").split("/")[-1])
# plt.scatter(xy["x"], xy["y"], s=10)
# plt.xlabel("tsne1")
# plt.ylabel("tsne2")
# plt.show()
