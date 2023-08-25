import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import umap
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
N = 500
xi = np.linspace(min(x), max(x), N)
yi = np.linspace(min(y), max(y), N)

xi, yi = np.meshgrid(xi, yi)

# NOTE: What is this? Why are we mapping this to a gaussian? 
func = Rbf(x, y, z, kind='gaussian', epsilon = 0.02, smooth = 0.1)
zi = func(xi, yi)

plt.figure(figsize=(8, 6))
plt.contourf(xi, yi, zi, levels=25)
plt.colorbar()
plt.title("The chemical space of target " + dwd.strip("/").split("/")[-1] + " using Jaccard")
plt.scatter(xy["x"], xy["y"], s=10)
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.savefig("MDS_Jaccard.png", dpi=300)
# NOTE: As I am going to use the same code for other targets & other dim reduction 
# methods, I will save this into a function. (Later)
df = pd.read_parquet(pj(dwd, 'data_ECFP4.parquet'))
# print(df.values)
distance2 = pairwise_distances(df.values, metric='euclidean')
xy = pd.DataFrame(
        mds.fit_transform(distance2),
        columns=['x', 'y']
    )

x = xy['x'].values
y = xy['y'].values
z = target.values

# For Plotting:
xi = np.linspace(min(x), max(x), N)
yi = np.linspace(min(y), max(y), N)

xi, yi = np.meshgrid(xi, yi)

# NOTE: What is this? Why are we mapping this to a gaussian?
func = Rbf(x, y, z, kind='gaussian', epsilon = 0.02, smooth = 0.1)
zi = func(xi, yi)

plt.figure(figsize=(8, 6))
plt.contourf(xi, yi, zi, levels=25)
plt.colorbar()
plt.title("The chemical space of target " + dwd.strip("/").split("/")[-1] + " using Euclidean")
plt.scatter(xy["x"], xy["y"], s=10)
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.savefig("MDS_Euclidean.png", dpi=300)
#%% Checking the chemical space using Taxicab

distance2 = pairwise_distances(df.values, metric='cityblock')
xy = pd.DataFrame(
        mds.fit_transform(distance2),
        columns=['x', 'y']
    )

x = xy['x'].values
y = xy['y'].values
z = target.values

# For Plotting:
xi = np.linspace(min(x), max(x), N)
yi = np.linspace(min(y), max(y), N)

xi, yi = np.meshgrid(xi, yi)

# NOTE: What is this? Why are we mapping this to a gaussian?
func = Rbf(x, y, z, kind='gaussian', epsilon = 0.02, smooth = 0.1)
zi = func(xi, yi)

plt.figure(figsize=(8, 6))
plt.contourf(xi, yi, zi, levels=25)
plt.colorbar()
plt.title("The chemical space of target " + dwd.strip("/").split("/")[-1] + " using Taxicab")
plt.scatter(xy["x"], xy["y"], s=10)
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.savefig("MDS_cityblock.png", dpi=300)


# USING TSNE
tsne = TSNE(random_state=1311)
xy = pd.DataFrame(
        tsne.fit_transform(df),
        columns=['x', 'y']
    )

x = xy['x'].values
y = xy['y'].values
z = target.values

# For Plotting:
xi = np.linspace(min(x), max(x), N)
yi = np.linspace(min(y), max(y), N)

xi, yi = np.meshgrid(xi, yi)

# NOTE: What is this? Why are we mapping this to a gaussian?
func = Rbf(x, y, z, kind='gaussian', epsilon = 0.02, smooth = 0.1)
zi = func(xi, yi)

plt.figure(figsize=(8, 6))
plt.contourf(xi, yi, zi, levels=25)
plt.colorbar()
plt.title("The chemical space of target " + dwd.strip("/").split("/")[-1] + " using TSNE")
plt.scatter(xy["x"], xy["y"], s=10)
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.savefig("TSNE.png", dpi=300)


#%% Using UMAP
reducer = umap.UMAP(random_state=1311)
xy = pd.DataFrame(
        reducer.fit_transform(df),
        columns=['x', 'y']
    )

x = xy['x'].values
y = xy['y'].values
z = target.values

# For Plotting:
xi = np.linspace(min(x), max(x), N)
yi = np.linspace(min(y), max(y), N)

xi, yi = np.meshgrid(xi, yi)

# NOTE: What is this? Why are we mapping this to a gaussian?
func = Rbf(x, y, z, kind='gaussian', epsilon = 0.02, smooth = 0.1)
zi = func(xi, yi)

plt.figure(figsize=(8, 6))
plt.contourf(xi, yi, zi, levels=25)
plt.colorbar()
plt.title("The chemical space of target " + dwd.strip("/").split("/")[-1] + " using UMAP")
plt.scatter(xy["x"], xy["y"], s=10)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.savefig("UMAP.png", dpi=300)

# MLKR
from metric_learn import MLKR
mlkr = MLKR(n_components=2, max_iter=2000, verbose=False, tol=1e-9, random_state=2021)
xy = pd.DataFrame(
        mlkr.fit_transform(df, target),
        columns=['x', 'y']
    )

x = xy['x'].values
y = xy['y'].values
z = target.values

# For Plotting:
xi = np.linspace(min(x), max(x), N)
yi = np.linspace(min(y), max(y), N)

xi, yi = np.meshgrid(xi, yi)
func = Rbf(x, y, z, kind='gaussian', epsilon = 0.02, smooth = 0.1)
zi = func(xi, yi)

plt.figure(figsize=(8, 6))
plt.contourf(xi, yi, zi, levels=25)
plt.colorbar()
plt.title("The chemical space of target " + dwd.strip("/").split("/")[-1] + " using MLKR")
plt.scatter(xy["x"], xy["y"], s=10)
plt.xlabel("MLKR1")
plt.ylabel("MLKR2")
plt.savefig("MLKR.png", dpi=300)
