import numpy as np
from scipy.spatial.distance import pdist, cdist

def calculate_dvis(data, labels):
    unique_labels = np.unique(labels)
    dvi1_sum = 0
    intra_cluster_density_sum = 0
    inter_cluster_density_sum = 0
    dvi3_sum = 0
    tightness_sum = 0

    for v in range(len(data)):
        cluster = labels[v]
        Cv = data[labels.flatten() == cluster]
        V_minus_Cv = data[labels.flatten() != cluster]

        intra_cluster_density = np.max(pdist(Cv)) if len(Cv) > 1 else 0
        inter_cluster_density = np.max(cdist([data[v]], V_minus_Cv)) if len(V_minus_Cv) > 0 else 0

        dvi1_sum += inter_cluster_density - intra_cluster_density
        intra_cluster_density_sum += intra_cluster_density
        inter_cluster_density_sum += inter_cluster_density
        dvi3_sum += inter_cluster_density / intra_cluster_density if intra_cluster_density > 0 else 0
        tightness_sum += np.mean(pdist(Cv)) if len(Cv) > 1 else 0

    dvi2 = inter_cluster_density_sum / intra_cluster_density_sum if intra_cluster_density_sum > 0 else 0
    tightness_avg = tightness_sum / len(unique_labels)
    return dvi1_sum, dvi2, dvi3_sum, tightness_avg
