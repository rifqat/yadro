import numpy as np

class Clustering:
    def __init__(self, metric='euclidean',alfa = 0):
        self.metric = metric
        self.alfa = alfa
        self.labels_ = None  # To store the cluster labels after fitting
        self.dvi1_ = None
        self.dvi2_ = None
        self.dvi3_ = None
        self.tightness_avg_ = None

    def create_distance_matrix(self, zscores):
        from scipy.spatial.distance import pdist, squareform
        distance_matrix = pdist(zscores, metric=self.metric)
        A = squareform(distance_matrix)
        max_vals = np.max(A, axis=1, keepdims=True)
        D = (max_vals - A) / max_vals
        np.fill_diagonal(D, 0)
        return D

    def initialize_clustering_variables(self, D):
        n1 = D.shape[1] - 1
        C = np.zeros((n1 + 1, 4))
        return n1, C

    def perform_clustering(self, D, n1):
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

    def calculate_cluster_thresholds(self, C, alfa, n1):
        C[0:n1, 2] = (C[0:n1, 1] - C[1:n1+1, 1]) / C[0:n1, 1]
        C[:, 3] = (C[:, 2] > alfa)
        return C

    def assign_initial_clusters(self, C, n1):
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
    
    # Function to finalize cluster assignments
    def finalize_clusters(self, S, A, m, n):
        zero_indices = np.where(S == 0)[0]
        m = len(zero_indices)
        for i in range(m):
            D = np.zeros((m - i, n - 1))
            for j in range(n - 1):
                c = S == (j + 1)
                r = np.where(S == 0)[0]
                D[:, j] = np.mean(A[np.ix_(r, c.flatten())], axis=1)
            ra, ca = np.where(D == np.max(D))
            S[r[ra[0]], 0] = ca[0] + 1
        return S
    
    def fit(self, zscores):
        """
        Perform clustering on the provided zscores with a given threshold alfa.
        
        Parameters:
        zscores : ndarray
            The input data to be clustered.
        alfa : float
            The threshold value for clustering.
        
        Returns:
        self : object
            Fitted clustering object with labels and metrics.
        """
        D = self.create_distance_matrix(zscores)
        n1, C = self.initialize_clustering_variables(D)
        C = self.perform_clustering(D, n1)
        C = self.calculate_cluster_thresholds(C, self.alfa, n1)
        S, m, n = self.assign_initial_clusters(C, n1)
        S = self.finalize_clusters(S, D, m, n)
        
        from .metrics import calculate_dvis
        self.dvi1_, self.dvi2_, self.dvi3_, self.tightness_avg_ = calculate_dvis(zscores, S)
        self.labels_ = S.flatten()
        
        return self

    def yadro(self, zscores, visualize=False):        
        if visualize:
            from .visualization import visualize_clusters
            visualize_clusters(zscores, self.labels_)
        return self.labels_, self.dvi1_, self.dvi2_, self.dvi3_, self.tightness_avg_

    def find_optimal_parameters(self, zscores, delta_values):
        """
        Find the optimal alfa parameter by evaluating the DVI scores and tightness.

        Parameters:
        zscores : ndarray
            The input data to be clustered.
        delta_values : array-like
            A range of possible alfa values to evaluate.

        Returns:
        best_params : float
            The alfa value that yields the best score.
        best_score : float
            The best score achieved using the best_params.
        """
        best_params = None
        best_score = -np.inf

        for delta in delta_values:
            self.alfa=delta
            self.fit(zscores)
            combined_score = self.dvi1_ - self.dvi2_ - self.dvi3_ # + self.tightness_avg_
            if combined_score > best_score:
                best_score = combined_score
                best_params = delta

        self.alfa=best_params
        print(best_params)
        return best_params, best_score