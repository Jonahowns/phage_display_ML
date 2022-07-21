# import sys
# # sys.path.append("../")
# from rbm_torch.rbm import RBM
# from rbm_torch import utils
# from rbm_torch.crbm import CRBM
#
# import math
# import pandas as pd
# from glob import glob
# import seaborn as sns
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


def estimate_dbscan_eps(input, nn=20, top_dists=None, title=None):
    """Graphs epsilon vs our dataset based off each sequences distance to a given number of nearest neighbors.

    Parameters
    ----------
    input: np.ndarray,
        categorical representation of all sequences you are clustering
    nn: int, optional, default = 20,
        number of nearest neighbors to consider when calculating the avg. distance
    top_dists: int, optional, default = None,
        focuses graph on the top distances as that is normally where the best eps can be found
    title: str, optional, default = None,
        title of the produced graph

    Returns
    -------
    Nothing

    Notes
    -----
    Plots graph using plt.show

    """
    from sklearn.neighbors import NearestNeighbors
    from matplotlib import pyplot as plt

    neighbors = NearestNeighbors(n_neighbors=nn, metric="hamming")
    neighbors_fit = neighbors.fit(input)
    distances, indices = neighbors_fit.kneighbors(input)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    if top_dists is not None:
        distances = distances[-top_dists:]
        plt.xlabel(f"Top {top_dists} Sequences")
    else:
        plt.xlabel("Sequence")

    if title is not None:
        plt.suptitle(title)
    plt.plot(distances)
    plt.ylabel(f"Avg. Dist of {nn} Nearest Neighbors")
    plt.show()


def cluster_seqs(input, cluster_method="db", min_samples=120, eps=0.5):
    """Clusters the provided sequences and returns their cluster labels.

    Parameters
    ----------
    input: np.ndarray,
        categorical representation of all sequences you are clustering
    cluster_method: str, optional, default = "db",
        which clustering method to use can be {"db", "op"}
    min_samples: int, optional, default = 120,
        parameter of dbscan and optics, controls min number of samples to be considered a cluster
    eps: float, optional, default = 0.5,
        the max distance sequences of the same cluster can be from one another, parameter of dbscan

    Returns
    -------
    labels: list of str,
        the cluster label to which each provided sequence was assigned
    """

    from sklearn.cluster import OPTICS
    from sklearn.cluster import DBSCAN

    if cluster_method == "db":
        alg = DBSCAN(eps=eps, metric='hamming', min_samples=min_samples).fit(input)
    elif cluster_method == "op":
        alg = OPTICS(min_samples=min_samples, metric="hamming").fit(input)

    labels = alg.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    cluster_id = set(labels)
    for clust in cluster_id:
        print('Clust', clust, 'Length', list(labels).count(clust))

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    return labels

def hierarchy_dendrogram(input):
    from matplotlib import pyplot as mtp
    import scipy.cluster.hierarchy as shc
    dendro = shc.dendrogram(shc.linkage(input, method="ward", metric="hamming"))
    plt.title("Dendrogram Plot")
    plt.ylabel("Hamming Distances")
    mtp.xlabel("Sequences")
    mtp.show()

def hierarchy_clustering(input, n_clusters):
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='hamming', linkage='ward')
    y_pred = hc.fit_predict(input)
    return y_pred
