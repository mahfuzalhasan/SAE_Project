import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt # Visulization
import matplotlib.cm as cm


nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.shape == y_true.shape
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.shape[0]):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    # print(ind)

    # for i,j in zip(ind[0], ind[1]):
    #     print(i,j)
    
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0]

def plot_score(data, labels, y_true, num_clusters = 10, name="train"):
    
    df_embedded = TSNE(n_components = 2).fit_transform(data)
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(18,7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, data.shape[0] + (num_clusters + 1)*10])
  
    if len(np.unique(np.array(labels))) == 1:
        print("This time, no good.")
    else:
        silhouette_avg = silhouette_score(data, labels)
        sample_silhouette_values = silhouette_samples(data, labels)
        y_lower = 10
        for i in range(num_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i)/num_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor = color, edgecolor = color, alpha = 0.7)
            ax1.text(-0.05, y_lower+0.5*size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette plot for the various clussters.")
        ax1.set_ylabel("Cluster label.")
        ax1.axvline(x = silhouette_avg, color = 'red', linestyle = '--')
        ax1.set_yticks([])
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(labels.astype(float)/num_clusters)
        ax2.scatter(df_embedded[:,0], df_embedded[:,1], marker = '.', s = 60, lw = 0, alpha = 0.7, c = colors,
                    edgecolor = 'k')
        ax2.set_title("The TSNE visualisation of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature.")
        ax2.set_ylabel("Feature space for the 2nd feature.")
        plt.suptitle(("Silhouette analysis for clustering on sampling data"
                      "with n_clusters = %d" %num_clusters), fontsize = 14, fontweight = 'bold')
        
        plt.savefig(f'./output/{name}.jpg')