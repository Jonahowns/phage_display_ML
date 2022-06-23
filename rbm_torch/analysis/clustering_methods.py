import sys
sys.path.append("../")
from rbm import RBM
import utils
from crbm import CRBM

import math
import pandas as pd
from glob import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# X is data fed directly to sklearn PCA instance
def pca_component_check(X, components=10):
    # First PCA pass to decide number of components
    pca = PCA(n_components=components) # This seems like a good number
    principal_components = pca.fit_transform(X)
    # PCA_components = pd.DataFrame(principal_components)

    features = range(1, pca.n_components_ + 1)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.show()

def view_components(components):
    fig, axs = plt.subplots(1, 1)
    pos = axs.imshow(components, cmap='plasma', interpolation='none')
    axs.set_title("Principal Components")
    axs.set_xlabel(r"$I_{h}$")
    axs.set_ylabel("PCs")
    xticks = np.arange(int(0.5), int(components.shape[1]) + 0.5, 5)
    yticks = np.arange(int(0.5), int(components.shape[0]) + 0.5, 1)
    axs.set_xticks(xticks)
    axs.set_yticks(yticks)
    axs.set_xticklabels(xticks, fontsize=12)
    axs.set_yticklabels(yticks, fontsize=12)
    fig.colorbar(pos, ax=axs, anchor=(0, 0.3), shrink=0.2)
    plt.show()

