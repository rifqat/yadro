import matplotlib.pyplot as plt
import numpy as np

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
