"""
Comprehensive Experiment Script for IJAI Paper Revision (v2 - corrected)
=========================================================================
Correctly classifies datasets:
  - Synthetic: sklearn-generated + code/data/ (data1-data4)
  - Real-world: code/real-world/ (ARFF benchmark datasets from UCI)

Addresses all Reviewer L concerns:
  L1: Parameter sensitivity analysis
  L2: Scalability and runtime benchmarks
  L3: Multi-metric evaluation (ARI, Silhouette, NMI, Davies-Bouldin) + statistical tests
  L4: Noise handling analysis
  L5: Real-world dataset validation (UCI benchmark datasets)
"""

import time
import warnings
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    normalized_mutual_info_score,
    davies_bouldin_score,
)
from scipy.stats import wilcoxon, friedmanchisquare

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segmentation import YadroSegmentation

# ============================================================
# Paths
# ============================================================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CODE_DIR, '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYNTHETIC_DATA_DIR = os.path.join(CODE_DIR, 'data')       # data1-data4 (synthetic 2D)
REAL_WORLD_DIR = os.path.join(CODE_DIR, 'real-world')       # ARFF benchmark datasets


# ============================================================
# ARFF Parser
# ============================================================

def parse_arff(filepath):
    """Parse ARFF file, return (X, y) with integer-encoded labels."""
    data_section = False
    X_rows = []
    y_vals = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.upper().startswith(('@RELATION', '@ATTRIBUTE')):
                continue
            if line.upper().startswith('@DATA'):
                data_section = True
                continue
            if data_section:
                vals = line.split(',')
                if '?' in vals:
                    continue
                try:
                    features = [float(v.strip().strip("'\"")) for v in vals[:-1]]
                    label = vals[-1].strip().strip("'\"")
                    X_rows.append(features)
                    y_vals.append(label)
                except (ValueError, IndexError):
                    continue

    if not X_rows:
        return None, None

    X = np.array(X_rows)
    le = LabelEncoder()
    y = le.fit_transform(y_vals)
    return X, y


# ============================================================
# Helper functions
# ============================================================

def load_txt_dataset(file_path):
    """Load synthetic dataset from text file (x, y, label)."""
    data = np.loadtxt(file_path)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y


def safe_silhouette(X, labels):
    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return float('nan')
    mask = np.array(labels) >= 0
    if mask.sum() < 2:
        return float('nan')
    try:
        return silhouette_score(X[mask], np.array(labels)[mask])
    except Exception:
        return float('nan')


def safe_nmi(y_true, y_pred):
    try:
        return normalized_mutual_info_score(y_true, y_pred)
    except Exception:
        return float('nan')


def safe_davies_bouldin(X, labels):
    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return float('nan')
    mask = np.array(labels) >= 0
    if mask.sum() < 2:
        return float('nan')
    try:
        return davies_bouldin_score(X[mask], np.array(labels)[mask])
    except Exception:
        return float('nan')


def get_n_clusters(y):
    unique = set(y)
    unique.discard(-1)
    return len(unique)


# ============================================================
# Dataset Loading
# ============================================================

# Selected real-world datasets (diverse dimensions, sizes, cluster counts)
SELECTED_REAL_WORLD = [
    'iris',           # 150 × 4,  k=3   (classic, low-dim)
    'glass',          # 214 × 9,  k=6   (medium-dim, multi-class)
    'ecoli',          # 336 × 7,  k=8   (imbalanced)
    'vehicle',        # 846 × 18, k=4   (medium-dim, medium-size)
    'wisc',           # 699 × 9,  k=2   (binary, medical)
    'balance-scale',  # 625 × 4,  k=3   (low-dim)
    'heart-statlog',  # 270 × 13, k=2   (medical)
    'zoo',            # 101 × 16, k=7   (small, multi-class)
]


def load_all_datasets():
    """Load synthetic, custom synthetic, and real-world benchmark datasets."""
    n_samples = 500
    seed = 30

    # --- Sklearn synthetic datasets ---
    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
    )
    noisy_moons = datasets.make_moons(
        n_samples=n_samples, noise=0.05, random_state=seed
    )
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)

    random_state = 170
    X_blobs, y_blobs = datasets.make_blobs(
        n_samples=n_samples, random_state=random_state
    )
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_blobs, transformation)
    aniso = (X_aniso, y_blobs)

    varied = datasets.make_blobs(
        n_samples=n_samples,
        cluster_std=[1.0, 2.5, 0.5],
        random_state=random_state,
    )

    all_datasets = {
        'noisy_circles': {'data': noisy_circles, 'n_clusters': 2, 'type': 'synthetic', 'dim': 2},
        'noisy_moons':   {'data': noisy_moons,   'n_clusters': 2, 'type': 'synthetic', 'dim': 2},
        'varied':        {'data': varied,         'n_clusters': 3, 'type': 'synthetic', 'dim': 2},
        'aniso':         {'data': aniso,          'n_clusters': 3, 'type': 'synthetic', 'dim': 2},
        'blobs':         {'data': blobs,          'n_clusters': 3, 'type': 'synthetic', 'dim': 2},
    }

    # --- Custom synthetic datasets (code/data/) ---
    for i in range(1, 5):
        fpath = os.path.join(SYNTHETIC_DATA_DIR, f'data{i}.txt')
        if os.path.exists(fpath):
            X, y = load_txt_dataset(fpath)
            all_datasets[f'data{i}'] = {
                'data': (X, y),
                'n_clusters': get_n_clusters(y),
                'type': 'synthetic',
                'dim': 2
            }

    # --- Real-world benchmark datasets (code/real-world/) ---
    for name in SELECTED_REAL_WORLD:
        fpath = os.path.join(REAL_WORLD_DIR, f'{name}.arff')
        if os.path.exists(fpath):
            X, y = parse_arff(fpath)
            if X is not None:
                all_datasets[name] = {
                    'data': (X, y),
                    'n_clusters': get_n_clusters(y),
                    'type': 'real-world',
                    'dim': X.shape[1]
                }
            else:
                print(f"  WARNING: Failed to parse {name}.arff")
        else:
            print(f"  WARNING: {name}.arff not found")

    return all_datasets


# ============================================================
# Algorithm Setup
# ============================================================

def create_algorithms(X_scaled, X_raw, n_clusters, n_neighbors=3, quantile=0.3,
                      eps=0.3, damping=0.9, preference=-200, random_state=42):
    bandwidth = cluster.estimate_bandwidth(X_scaled, quantile=quantile)
    if bandwidth <= 0:
        bandwidth = 0.1

    connectivity = kneighbors_graph(
        X_scaled, n_neighbors=min(n_neighbors, len(X_scaled) - 1), include_self=False
    )
    connectivity = 0.5 * (connectivity + connectivity.T)

    algorithms = {
        'MiniBatchKMeans': cluster.MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state
        ),
        'AffinityPropagation': cluster.AffinityPropagation(
            damping=damping, preference=preference, random_state=random_state
        ),
        'MeanShift': cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True),
        'SpectralClustering': cluster.SpectralClustering(
            n_clusters=n_clusters, eigen_solver='arpack',
            affinity='nearest_neighbors', random_state=random_state
        ),
        'Ward': cluster.AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward', connectivity=connectivity
        ),
        'AgglomerativeClustering': cluster.AgglomerativeClustering(
            linkage='average', metric='cityblock',
            n_clusters=n_clusters, connectivity=connectivity
        ),
        'DBSCAN': cluster.DBSCAN(eps=eps),
        'HDBSCAN': cluster.HDBSCAN(
            min_samples=3, min_cluster_size=15, allow_single_cluster=True
        ),
        'OPTICS': cluster.OPTICS(
            min_samples=7, xi=0.05, min_cluster_size=0.1
        ),
        'BIRCH': cluster.Birch(n_clusters=n_clusters),
        'GaussianMixture': mixture.GaussianMixture(
            n_components=n_clusters, covariance_type='full', random_state=random_state
        ),
        'CoreClustering': YadroSegmentation(X_raw, epsilon=0.85),
    }
    return algorithms


# ============================================================
# EXPERIMENT 1: Full Benchmark (all datasets)
# ============================================================

def run_full_benchmark(all_datasets):
    print("\n" + "="*70)
    print("EXPERIMENT 1: Full Multi-Metric Benchmark")
    print("="*70)

    results = []

    for ds_name, ds_info in all_datasets.items():
        X_raw, y_true = ds_info['data']
        n_clusters = ds_info['n_clusters']
        ds_type = ds_info['type']
        ds_dim = ds_info['dim']

        X_scaled = StandardScaler().fit_transform(X_raw)

        algorithms = create_algorithms(X_scaled, X_raw, n_clusters)

        print(f"\n  Dataset: {ds_name} ({ds_type}, n={len(X_raw)}, d={ds_dim}, k={n_clusters})")

        for alg_name, algorithm in algorithms.items():
            t0 = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                try:
                    if alg_name == 'CoreClustering':
                        algorithm.full_pipeline(
                            delta=0.9, beta=5, theta=0.1,
                            lambda_value=0.5, visualize=False
                        )
                    else:
                        algorithm.fit(X_scaled)
                except Exception as e:
                    print(f"    {alg_name}: FAILED ({e})")
                    results.append({
                        'Dataset': ds_name, 'Type': ds_type, 'Dim': ds_dim,
                        'Algorithm': alg_name,
                        'ARI': float('nan'), 'Silhouette': float('nan'),
                        'NMI': float('nan'), 'DaviesBouldin': float('nan'),
                        'Time': float('nan'), 'n_clusters_found': 0
                    })
                    continue

            t1 = time.time()

            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(X_scaled)

            min_len = min(len(y_true), len(y_pred))
            y_t = y_true[:min_len]
            y_p = y_pred[:min_len]
            X_eval = X_scaled[:min_len]

            ari = adjusted_rand_score(y_t, y_p)
            sil = safe_silhouette(X_eval, y_p)
            nmi = safe_nmi(y_t, y_p)
            dbi = safe_davies_bouldin(X_eval, y_p)
            n_found = len(set(y_p) - {-1})

            results.append({
                'Dataset': ds_name, 'Type': ds_type, 'Dim': ds_dim,
                'Algorithm': alg_name,
                'ARI': round(ari, 4),
                'Silhouette': round(sil, 4) if not np.isnan(sil) else float('nan'),
                'NMI': round(nmi, 4) if not np.isnan(nmi) else float('nan'),
                'DaviesBouldin': round(dbi, 4) if not np.isnan(dbi) else float('nan'),
                'Time': round(t1 - t0, 4),
                'n_clusters_found': n_found
            })

            print(f"    {alg_name:25s}: ARI={ari:.3f}  SI={sil:.3f}  NMI={nmi:.3f}  DB={dbi:.3f}  t={t1-t0:.3f}s")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'full_benchmark.csv'), index=False)

    for metric in ['ARI', 'Silhouette', 'NMI', 'DaviesBouldin']:
        pivot = df.pivot_table(index='Dataset', columns='Algorithm', values=metric, aggfunc='first')
        pivot.to_csv(os.path.join(OUTPUT_DIR, f'pivot_{metric}.csv'))

    generate_heatmaps(df)
    generate_heatmaps_by_type(df)

    return df


def generate_heatmaps(df):
    metrics = ['ARI', 'NMI', 'Silhouette']
    titles = {
        'ARI': 'Adjusted Rand Index (ARI)',
        'NMI': 'Normalized Mutual Information (NMI)',
        'Silhouette': 'Silhouette Score'
    }

    for metric in metrics:
        pivot = df.pivot_table(
            index='Dataset', columns='Algorithm', values=metric, aggfunc='first'
        )

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(
            pivot.astype(float), annot=False, cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': titles[metric]}, ax=ax
        )

        highlight = pivot.apply(lambda row: row == row.max(), axis=1)
        for y_idx in range(pivot.shape[0]):
            for x_idx in range(pivot.shape[1]):
                value = pivot.iloc[y_idx, x_idx]
                if pd.notna(value):
                    is_max = highlight.iloc[y_idx, x_idx]
                    color = 'red' if is_max else 'black'
                    weight = 'bold' if is_max else 'normal'
                    ax.text(
                        x_idx + 0.5, y_idx + 0.5, f"{value:.2f}",
                        ha='center', va='center', fontsize=7,
                        color=color, fontweight=weight
                    )

        ax.set_title(f'Clustering Algorithms - {titles[metric]}', fontsize=14)
        ax.set_xlabel('Algorithm', fontsize=11)
        ax.set_ylabel('Dataset', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_{metric}.png'), dpi=150)
        plt.close()

    print(f"\n  Heatmaps saved to {OUTPUT_DIR}")


def generate_heatmaps_by_type(df):
    """Generate separate heatmaps for synthetic vs real-world datasets."""
    for ds_type, label in [('synthetic', 'Synthetic Datasets'), ('real-world', 'Real-World Datasets')]:
        subset = df[df['Type'] == ds_type]
        if subset.empty:
            continue

        pivot = subset.pivot_table(
            index='Dataset', columns='Algorithm', values='ARI', aggfunc='first'
        )

        fig, ax = plt.subplots(figsize=(16, max(4, len(pivot) * 0.7)))
        sns.heatmap(
            pivot.astype(float), annot=True, fmt='.2f', cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': 'ARI'}, ax=ax,
            annot_kws={'size': 8}
        )

        ax.set_title(f'ARI - {label}', fontsize=14)
        ax.set_xlabel('Algorithm', fontsize=11)
        ax.set_ylabel('Dataset', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'heatmap_ARI_{ds_type}.png'), dpi=150)
        plt.close()


# ============================================================
# EXPERIMENT 2: Parameter Sensitivity (L1)
# ============================================================

def run_parameter_sensitivity(all_datasets):
    print("\n" + "="*70)
    print("EXPERIMENT 2: Parameter Sensitivity Analysis")
    print("="*70)

    defaults = {
        'epsilon': 0.85, 'delta': 0.9, 'beta': 5,
        'theta': 0.1, 'lambda_value': 0.5,
    }

    param_ranges = {
        'epsilon': np.arange(0.5, 1.0, 0.05),
        'delta': np.arange(0.1, 1.0, 0.1),
        'beta': [2, 3, 4, 5, 6, 7, 8, 10],
        'theta': np.arange(0.05, 0.5, 0.05),
        'lambda_value': np.arange(0.1, 1.0, 0.1),
    }

    # Test on a mix of synthetic and real-world datasets
    test_datasets = ['noisy_circles', 'noisy_moons', 'varied',
                     'data1', 'data2',
                     'iris', 'glass', 'ecoli', 'wisc']
    sensitivity_results = []

    for param_name, param_values in param_ranges.items():
        print(f"\n  Varying: {param_name}")

        for ds_name in test_datasets:
            if ds_name not in all_datasets:
                continue
            X_raw, y_true = all_datasets[ds_name]['data']

            for pval in param_values:
                params = defaults.copy()
                params[param_name] = pval

                try:
                    seg = YadroSegmentation(X_raw, epsilon=params['epsilon'])
                    seg.full_pipeline(
                        delta=params['delta'],
                        beta=int(params['beta']),
                        theta=params['theta'],
                        lambda_value=params['lambda_value'],
                        visualize=False
                    )
                    y_pred = seg.labels_.astype(int)
                    min_len = min(len(y_true), len(y_pred))
                    ari = adjusted_rand_score(y_true[:min_len], y_pred[:min_len])
                    nmi = safe_nmi(y_true[:min_len], y_pred[:min_len])
                except Exception:
                    ari = float('nan')
                    nmi = float('nan')

                sensitivity_results.append({
                    'Parameter': param_name,
                    'Value': round(float(pval), 3),
                    'Dataset': ds_name,
                    'DatasetType': all_datasets[ds_name]['type'],
                    'ARI': round(ari, 4) if not np.isnan(ari) else float('nan'),
                    'NMI': round(nmi, 4) if not np.isnan(nmi) else float('nan'),
                })

        print(f"    Done.")

    df_sens = pd.DataFrame(sensitivity_results)
    df_sens.to_csv(os.path.join(OUTPUT_DIR, 'parameter_sensitivity.csv'), index=False)
    generate_sensitivity_plots(df_sens)
    return df_sens


def generate_sensitivity_plots(df_sens):
    params = df_sens['Parameter'].unique()
    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 4))
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        subset = df_sens[df_sens['Parameter'] == param]
        for ds_name in subset['Dataset'].unique():
            ds_data = subset[subset['Dataset'] == ds_name]
            ds_type = ds_data['DatasetType'].iloc[0]
            linestyle = '-' if ds_type == 'synthetic' else '--'
            ax.plot(ds_data['Value'], ds_data['ARI'], marker='o', markersize=3,
                    label=ds_name, linewidth=1, linestyle=linestyle)
        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel('ARI', fontsize=10)
        ax.set_title(f'Sensitivity: {param}', fontsize=11)
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_sensitivity.png'), dpi=150)
    plt.close()
    print(f"  Sensitivity plots saved.")


# ============================================================
# EXPERIMENT 3: Scalability (L2)
# ============================================================

def run_scalability_benchmark():
    print("\n" + "="*70)
    print("EXPERIMENT 3: Scalability Analysis")
    print("="*70)

    sizes = [100, 200, 500, 800, 1000, 1500, 2000, 3000]
    scalability_results = []

    for n in sizes:
        print(f"  n={n}...")
        X, y = datasets.make_blobs(n_samples=n, centers=5, random_state=42)

        seg = YadroSegmentation(X, epsilon=0.85)

        t0 = time.time()
        seg.create_graph_with_weights()
        t_graph = time.time() - t0

        t0 = time.time()
        seg.compute_density_variation_sequence()
        t_density = time.time() - t0

        t0 = time.time()
        core_pixels = seg.identify_core_pixels(
            seg.Dt_sequence, seg.Mt_sequence, delta=0.9, beta=5
        )
        t_core = time.time() - t0

        t0 = time.time()
        core_segments = seg.partition_core_pixels(seg.G, core_pixels, theta=0.1)
        segments, _ = seg.expand_segments(
            seg.G, seg.Mt_sequence, core_segments, lambda_value=0.5
        )
        t_expand = time.time() - t0

        t_total = t_graph + t_density + t_core + t_expand

        scalability_results.append({
            'n': n, 'algorithm': 'CoreClustering',
            'graph_construction': round(t_graph, 4),
            'density_sequence': round(t_density, 4),
            'core_identification': round(t_core, 4),
            'segment_expansion': round(t_expand, 4),
            'total': round(t_total, 4),
        })

        X_scaled = StandardScaler().fit_transform(X)
        for alg_name, alg in [('DBSCAN', cluster.DBSCAN(eps=0.3)),
                               ('HDBSCAN', cluster.HDBSCAN(min_samples=3, min_cluster_size=15))]:
            t0 = time.time()
            alg.fit(X_scaled)
            t1 = time.time()
            scalability_results.append({
                'n': n, 'algorithm': alg_name,
                'graph_construction': 0, 'density_sequence': 0,
                'core_identification': 0, 'segment_expansion': 0,
                'total': round(t1 - t0, 4),
            })

    df_all = pd.DataFrame(scalability_results)
    df_all.to_csv(os.path.join(OUTPUT_DIR, 'scalability.csv'), index=False)

    # Plot
    df_core = df_all[df_all['algorithm'] == 'CoreClustering']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df_core['n'], df_core['graph_construction'], 'o-', label='Graph construction')
    ax1.plot(df_core['n'], df_core['density_sequence'], 's-', label='Density sequence')
    ax1.plot(df_core['n'], df_core['core_identification'], '^-', label='Core identification')
    ax1.plot(df_core['n'], df_core['segment_expansion'], 'D-', label='Segment expansion')
    ax1.plot(df_core['n'], df_core['total'], 'k*-', label='Total', linewidth=2)
    ax1.set_xlabel('Number of points (n)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Core Clustering: Stage Breakdown')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    for alg_name in ['CoreClustering', 'DBSCAN', 'HDBSCAN']:
        subset = df_all[df_all['algorithm'] == alg_name]
        ax2.plot(subset['n'], subset['total'], 'o-', label=alg_name, linewidth=2)
    ax2.set_xlabel('Number of points (n)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Runtime Comparison')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scalability.png'), dpi=150)
    plt.close()
    print(f"  Scalability results saved.")
    return df_all


# ============================================================
# EXPERIMENT 4: Statistical Significance Tests (L3)
# ============================================================

def run_statistical_tests(df_benchmark):
    print("\n" + "="*70)
    print("EXPERIMENT 4: Statistical Significance Tests")
    print("="*70)

    comparison_algorithms = [
        'DBSCAN', 'HDBSCAN', 'OPTICS',
        'MiniBatchKMeans', 'SpectralClustering', 'Ward',
        'AgglomerativeClustering', 'BIRCH', 'GaussianMixture',
        'AffinityPropagation', 'MeanShift'
    ]

    stat_results = []
    core_data = df_benchmark[df_benchmark['Algorithm'] == 'CoreClustering']
    if core_data.empty:
        print("  No CoreClustering results!")
        return pd.DataFrame()

    core_ari = core_data.set_index('Dataset')['ARI']

    for alg in comparison_algorithms:
        alg_data = df_benchmark[df_benchmark['Algorithm'] == alg]
        alg_ari = alg_data.set_index('Dataset')['ARI']
        common = core_ari.index.intersection(alg_ari.index)

        if len(common) < 3:
            stat_results.append({
                'Comparison': f'CoreClustering vs {alg}',
                'n_datasets': len(common),
                'Core_mean_ARI': float('nan'),
                'Other_mean_ARI': float('nan'),
                'p_value': float('nan'),
                'significant': 'N/A',
                'wins': 0, 'ties': 0, 'losses': 0,
            })
            continue

        core_vals = core_ari.reindex(common).values.astype(float)
        alg_vals = alg_ari.reindex(common).values.astype(float)

        wins = int(np.sum(core_vals > alg_vals + 0.01))
        ties = int(np.sum(np.abs(core_vals - alg_vals) <= 0.01))
        losses = int(np.sum(core_vals < alg_vals - 0.01))

        diffs = core_vals - alg_vals
        nonzero_diffs = diffs[np.abs(diffs) > 1e-10]
        if len(nonzero_diffs) >= 3:
            try:
                stat, p_val = wilcoxon(nonzero_diffs)
                significant = 'Yes' if p_val < 0.05 else 'No'
            except Exception:
                p_val = float('nan')
                significant = 'N/A'
        else:
            p_val = float('nan')
            significant = 'N/A'

        stat_results.append({
            'Comparison': f'CoreClustering vs {alg}',
            'n_datasets': len(common),
            'Core_mean_ARI': round(float(core_ari.reindex(common).mean()), 4),
            'Other_mean_ARI': round(float(alg_ari.reindex(common).mean()), 4),
            'p_value': round(float(p_val), 4) if not np.isnan(p_val) else float('nan'),
            'significant': significant,
            'wins': wins, 'ties': ties, 'losses': losses,
        })

        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        print(f"  vs {alg:25s}: W={wins} T={ties} L={losses}  p={p_str}")

    # Friedman test
    print("\n  Friedman test (all algorithms):")
    try:
        pivot_ari = df_benchmark.pivot_table(
            index='Dataset', columns='Algorithm', values='ARI', aggfunc='first'
        ).dropna(axis=1, how='all').dropna(axis=0, how='any')

        if pivot_ari.shape[1] >= 3 and pivot_ari.shape[0] >= 3:
            groups = [pivot_ari[col].values for col in pivot_ari.columns]
            friedman_stat, friedman_p = friedmanchisquare(*groups)
            print(f"    Chi-square = {friedman_stat:.4f}, p = {friedman_p:.6f}")
        else:
            friedman_stat, friedman_p = float('nan'), float('nan')
    except Exception as e:
        friedman_stat, friedman_p = float('nan'), float('nan')
        print(f"    Friedman test failed: {e}")

    df_stat = pd.DataFrame(stat_results)
    df_stat.to_csv(os.path.join(OUTPUT_DIR, 'statistical_tests.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'friedman_test.json'), 'w') as f:
        json.dump({
            'friedman_chi_square': round(float(friedman_stat), 4) if not np.isnan(friedman_stat) else None,
            'friedman_p_value': round(float(friedman_p), 6) if not np.isnan(friedman_p) else None,
        }, f, indent=2)

    print(f"  Statistical test results saved.")
    return df_stat


# ============================================================
# EXPERIMENT 5: Noise Handling Analysis (L4)
# ============================================================

def run_noise_analysis(all_datasets):
    print("\n" + "="*70)
    print("EXPERIMENT 5: Noise Handling Analysis")
    print("="*70)

    noise_results = []
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]

    for noise_frac in noise_levels:
        n_samples = 500
        n_noise = int(n_samples * noise_frac)
        n_clean = n_samples - n_noise

        X_clean, y_clean = datasets.make_moons(n_samples=n_clean, noise=0.05, random_state=42)

        if n_noise > 0:
            rng = np.random.RandomState(42)
            X_noise = rng.uniform(
                low=X_clean.min(axis=0) - 0.5,
                high=X_clean.max(axis=0) + 0.5,
                size=(n_noise, 2)
            )
            y_noise = np.full(n_noise, -1)
            X = np.vstack([X_clean, X_noise])
            y = np.concatenate([y_clean, y_noise])
        else:
            X, y = X_clean, y_clean

        try:
            seg = YadroSegmentation(X, epsilon=0.85)
            seg.full_pipeline(delta=0.9, beta=5, theta=0.1, lambda_value=0.5, visualize=False)
            y_pred_core = seg.labels_.astype(int)
            min_len = min(len(y), len(y_pred_core))
            ari_core = adjusted_rand_score(y[:min_len], y_pred_core[:min_len])
        except Exception:
            ari_core = float('nan')

        X_scaled = StandardScaler().fit_transform(X)
        hdb = cluster.HDBSCAN(min_samples=3, min_cluster_size=15, allow_single_cluster=True)
        hdb.fit(X_scaled)
        ari_hdb = adjusted_rand_score(y, hdb.labels_)

        dbs = cluster.DBSCAN(eps=0.3)
        dbs.fit(X_scaled)
        ari_dbs = adjusted_rand_score(y, dbs.labels_)

        noise_results.append({
            'noise_fraction': noise_frac,
            'n_total': len(X),
            'n_noise_true': int((y == -1).sum()),
            'CoreClustering_ARI': round(ari_core, 4) if not np.isnan(ari_core) else float('nan'),
            'HDBSCAN_ARI': round(ari_hdb, 4),
            'DBSCAN_ARI': round(ari_dbs, 4),
        })

    # Also on synthetic datasets with noise (data1-data4)
    for ds_name in ['data1', 'data2', 'data3', 'data4']:
        if ds_name not in all_datasets:
            continue
        X_raw, y_true = all_datasets[ds_name]['data']
        n_noise_true = int((y_true == -1).sum())

        try:
            seg = YadroSegmentation(X_raw, epsilon=0.85)
            seg.full_pipeline(delta=0.9, beta=5, theta=0.1, lambda_value=0.5, visualize=False)
            y_pred = seg.labels_.astype(int)
            min_len = min(len(y_true), len(y_pred))
            ari_core = adjusted_rand_score(y_true[:min_len], y_pred[:min_len])
        except Exception:
            ari_core = float('nan')

        X_scaled = StandardScaler().fit_transform(X_raw)
        hdb = cluster.HDBSCAN(min_samples=3, min_cluster_size=15, allow_single_cluster=True)
        hdb.fit(X_scaled)
        ari_hdb = adjusted_rand_score(y_true, hdb.labels_)

        dbs = cluster.DBSCAN(eps=0.3)
        dbs.fit(X_scaled)
        ari_dbs = adjusted_rand_score(y_true, dbs.labels_)

        noise_results.append({
            'noise_fraction': f'{ds_name} (synthetic)',
            'n_total': len(X_raw),
            'n_noise_true': n_noise_true,
            'CoreClustering_ARI': round(ari_core, 4) if not np.isnan(ari_core) else float('nan'),
            'HDBSCAN_ARI': round(ari_hdb, 4),
            'DBSCAN_ARI': round(ari_dbs, 4),
        })

    df_noise = pd.DataFrame(noise_results)
    df_noise.to_csv(os.path.join(OUTPUT_DIR, 'noise_analysis.csv'), index=False)

    # Plot
    synthetic_data = df_noise[df_noise['noise_fraction'].apply(lambda x: isinstance(x, float))]
    if not synthetic_data.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(synthetic_data['noise_fraction'], synthetic_data['CoreClustering_ARI'],
                'o-', label='Core Clustering', linewidth=2, markersize=8)
        ax.plot(synthetic_data['noise_fraction'], synthetic_data['HDBSCAN_ARI'],
                's-', label='HDBSCAN', linewidth=2, markersize=8)
        ax.plot(synthetic_data['noise_fraction'], synthetic_data['DBSCAN_ARI'],
                '^-', label='DBSCAN', linewidth=2, markersize=8)
        ax.set_xlabel('Noise Fraction', fontsize=12)
        ax.set_ylabel('ARI', fontsize=12)
        ax.set_title('Noise Robustness Comparison', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'noise_robustness.png'), dpi=150)
        plt.close()

    print(f"  Noise analysis results saved.")
    return df_noise


# ============================================================
# EXPERIMENT 6: Real-World vs Synthetic Summary (L5)
# ============================================================

def generate_summary_by_type(df_benchmark):
    """Generate summary statistics comparing performance on synthetic vs real-world."""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Synthetic vs Real-World Summary")
    print("="*70)

    summary = []
    for ds_type in ['synthetic', 'real-world']:
        subset = df_benchmark[df_benchmark['Type'] == ds_type]
        if subset.empty:
            continue
        for alg in subset['Algorithm'].unique():
            alg_data = subset[subset['Algorithm'] == alg]
            summary.append({
                'Type': ds_type,
                'Algorithm': alg,
                'Mean_ARI': round(alg_data['ARI'].mean(), 4),
                'Std_ARI': round(alg_data['ARI'].std(), 4),
                'Mean_NMI': round(alg_data['NMI'].mean(), 4),
                'Mean_Silhouette': round(alg_data['Silhouette'].mean(), 4),
                'n_datasets': len(alg_data),
            })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, 'summary_by_type.csv'), index=False)

    # Print summary table
    for ds_type in ['synthetic', 'real-world']:
        sub = df_summary[df_summary['Type'] == ds_type]
        if not sub.empty:
            print(f"\n  {ds_type.upper()} datasets mean ARI:")
            for _, row in sub.sort_values('Mean_ARI', ascending=False).iterrows():
                print(f"    {row['Algorithm']:25s}: ARI={row['Mean_ARI']:.3f} ± {row['Std_ARI']:.3f}  "
                      f"NMI={row['Mean_NMI']:.3f}  SI={row['Mean_Silhouette']:.3f}")

    # Bar chart comparing mean ARI
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, ds_type in zip(axes, ['synthetic', 'real-world']):
        sub = df_summary[df_summary['Type'] == ds_type].sort_values('Mean_ARI', ascending=True)
        if sub.empty:
            continue
        colors = ['#e74c3c' if alg == 'CoreClustering' else '#3498db' for alg in sub['Algorithm']]
        ax.barh(sub['Algorithm'], sub['Mean_ARI'], color=colors, edgecolor='white')
        ax.set_xlabel('Mean ARI', fontsize=11)
        ax.set_title(f'{ds_type.title()} Datasets', fontsize=13)
        ax.set_xlim(0, 1.0)
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_by_type.png'), dpi=150)
    plt.close()

    print(f"  Summary saved.")
    return df_summary


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("IJAI Paper Revision - Comprehensive Experiments (v2 - corrected)")
    print("=" * 70)
    start_time = time.time()

    print("\nLoading datasets...")
    all_datasets = load_all_datasets()
    print(f"  Loaded {len(all_datasets)} datasets")

    synthetic_count = sum(1 for v in all_datasets.values() if v['type'] == 'synthetic')
    real_count = sum(1 for v in all_datasets.values() if v['type'] == 'real-world')
    print(f"  Synthetic: {synthetic_count}, Real-world: {real_count}")

    for name, info in all_datasets.items():
        X, y = info['data']
        print(f"    {name:20s}: n={len(X):>5}, d={info['dim']:>2}, k={info['n_clusters']}, type={info['type']}")

    # Run all experiments
    df_benchmark = run_full_benchmark(all_datasets)
    df_sensitivity = run_parameter_sensitivity(all_datasets)
    df_scalability = run_scalability_benchmark()
    df_stats = run_statistical_tests(df_benchmark)
    df_noise = run_noise_analysis(all_datasets)
    df_type_summary = generate_summary_by_type(df_benchmark)

    # Save final summary
    total_time = time.time() - start_time
    summary = {
        'total_experiments': 6,
        'total_datasets': len(all_datasets),
        'synthetic_datasets': synthetic_count,
        'real_world_datasets': real_count,
        'total_algorithms': 12,
        'total_runtime_seconds': round(total_time, 2),
        'output_directory': OUTPUT_DIR,
        'dataset_details': {
            name: {
                'n_samples': len(info['data'][0]),
                'n_features': info['dim'],
                'n_clusters': info['n_clusters'],
                'type': info['type']
            }
            for name, info in all_datasets.items()
        }
    }
    with open(os.path.join(OUTPUT_DIR, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time:.1f} seconds")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")
