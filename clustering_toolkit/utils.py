import numpy as np

def find_optimal_parameters(graph, metrika, delta_values):
    from .clustering import Clustering
    best_params = None
    best_score = -np.inf
    clustering = Clustering(metrika)

    for delta in delta_values:
        clusters, dvi1, dvi2, dvi3, tightness_avg = clustering.yadro(graph, delta)
        combined_score = dvi1 - dvi2 - dvi3 + tightness_avg
        if combined_score > best_score:
            best_score = combined_score
            best_params = delta

    return best_params, best_score
