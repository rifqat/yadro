import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Sample data
x1 = np.array([
    [1, 2], [1, 3], [2, 2], [2, 3], [3, 4], [7, 2], [7, 3], [8, 2], [8, 3], [9, 4],
    [8, 8], [8, 9], [9, 8], [9, 9], [10, 10], [4, 5], [6, 5], [6, 7],
])

# Function to create and normalize the distance matrix
def create_distance_matrix(zscores, metric):
    distance_matrix = pdist(zscores, metric=metric)
    A = squareform(distance_matrix)
    max_vals = np.max(A, axis=1, keepdims=True)
    D = (max_vals - A) / max_vals
    np.fill_diagonal(D, 0)
    return D


# Function to initialize clustering variables
def initialize_clustering_variables(D):
    n1 = D.shape[1] - 1
    C = np.zeros((n1 + 1, 4))
    return n1, C



# Function to perform the main clustering loop
def perform_clustering(D, n1):
    C = np.zeros((n1 + 1, 4))
    A = np.copy(D)
    for m in range(n1):
        B = np.sum(A, axis=1)
        nonzero_indices = np.nonzero(B)[0]
        if len(nonzero_indices) == 0:
            break
        min_value = np.min(B[nonzero_indices])
        min_indices = np.where(B == min_value)[0]
        C[m, 0] = min_indices[0]
        C[m, 1] = min_value
        A[min_indices[0], :] = 0
        A[:, min_indices[0]] = 0
    return C
	
	
# Function to calculate cluster thresholds and update C matrix
def calculate_cluster_thresholds(C, alfa, n1):
    C[0:n1, 2] = (C[0:n1, 1] - C[1:n1+1, 1]) / C[0:n1, 1]
    C[:, 3] = (C[:, 2] > alfa)
    return C

# Function to assign initial clusters based on thresholds
def assign_initial_clusters(C, n1):
    n, sigma = 1, 0
    S = np.zeros((C.shape[0], 1))
    for x1 in range(n1):
        if C[x1, 3] - C[x1 + 1, 3] == 1:
            S[int(C[x1, 0]), 0] = n
            n += 1
            if sigma + 1 > 1:
                sigma_max = np.max(C[x1 - sigma + 1:x1 + 1, 1])
                for i_sigma in range(x1 - sigma, x1 + 1):
                    if C[i_sigma, 1] != sigma_max:
                        S[int(C[i_sigma, 0]), 0] = 0
                        C[i_sigma, 3] = 0
                sigma = 0
        if C[x1, 3] + C[x1 + 1, 3] == 2:
            S[int(C[x1, 0]), 0] = n
            sigma += 1    
    m = np.sum(C[:, 3] == 0)
    return S, m, n
	
	
# Function to assign initial clusters based on thresholds
def assign_initial_clusters(C, n1):
    n, sigma = 1, 0
    S = np.zeros((C.shape[0], 1))
    for x1 in range(n1):
        if C[x1, 3] - C[x1 + 1, 3] == 1:
            S[int(C[x1, 0]), 0] = n
            n += 1
            if sigma + 1 > 1:
                sigma_max = np.max(C[x1 - sigma + 1:x1 + 1, 1])
                for i_sigma in range(x1 - sigma, x1 + 1):
                    if C[i_sigma, 1] != sigma_max:
                        S[int(C[i_sigma, 0]), 0] = 0
                        C[i_sigma, 3] = 0
                sigma = 0
        if C[x1, 3] + C[x1 + 1, 3] == 2:
            S[int(C[x1, 0]), 0] = n
            sigma += 1    
    m = np.sum(C[:, 3] == 0)
    return S, m, n
	

# Function to calculate DVI and Tightness indices
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
	
	
# Main clustering function
def yadro(zscores, metrika, alfa, visualize=False):
    D = create_distance_matrix(zscores, metrika)
    n1, C = initialize_clustering_variables(D)
    C = perform_clustering(D, n1)
    C = calculate_cluster_thresholds(C, alfa, n1)
    S, m, n = assign_initial_clusters(C, n1)
    S = finalize_clusters(S, D, m, n)
    dvi1, dvi2, dvi3, tightness_avg = calculate_dvis(zscores, S)
    
    print(f'tightness_avg: {tightness_avg}')
    print(f'DVI1: {dvi1}')
    print(f'DVI2: {dvi2}')
    print(f'DVI3: {dvi3}')

    if visualize:
        visualize_clusters(zscores, S)

    return S, dvi1, dvi2, dvi3,tightness_avg

# Function to visualize clusters
def visualize_clusters(data, cluster_labels):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        plt.scatter(data[cluster_labels.flatten() == label, 0], data[cluster_labels.flatten() == label, 1], label=f'Cluster {int(label)}')
    plt.title('Cluster Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
# Function to find optimal clustering parameters
def find_optimal_parameters(graph, metrika, delta_values):
    best_params = None
    best_score = -np.inf

    for delta in delta_values:
        clusters, dvi1, dvi2, dvi3,tightness_avg = yadro(graph, metrika, delta)
         # Normalize and combine DVI scores and tightness_avg to get a single score
        combined_score = dvi1 - dvi2 - dvi3 + tightness_avg  # Adjust weights as needed
        if combined_score > best_score:
            best_score = combined_score
            best_params = delta

    return best_params, best_score
# Example usage
alfa = np.arange(0, 0.8, 0.1)
best_params, best_score = find_optimal_parameters(x1, 'euclidean', alfa)
print("Best Parameters:", best_params)
print("Best Score:", best_score)

zscores = x1  # Example data
S, dvi1, dvi2, dvi3,tightness_avg = yadro(zscores, 'euclidean', best_params, visualize=True)