"""
Regenerate publication-quality figures from saved result CSVs.
========================================================================
Reads the deterministic benchmark outputs in ../results/*.csv and redraws
every paper figure at high resolution (300 dpi PNG for the dense heatmaps,
vector PDF for the line plots) with enlarged, print-legible fonts.

No clustering is re-run: the numbers are taken verbatim from the CSVs that
run_experiments.py produced, so every value is identical to the reviewed
results. Only matplotlib/numpy/pandas are required (no seaborn/hdbscan).

Outputs (in ../results/):
  fig1_ari_heatmap.png        Figure 1  (ARI heatmap)
  fig2_parameter_sensitivity.pdf Figure 2
  fig3_nmi_heatmap.png        Figure 3  (NMI heatmap)
  fig4_silhouette_heatmap.png Figure 4  (Silhouette heatmap)
  fig5_noise_robustness.pdf   Figure 5
  fig6_scalability.pdf        Figure 6
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,   # embed TrueType, editable text in PDF
    'ps.fonttype': 42,
})


def _heatmap(csv_name, out_name, cbar_label, title):
    """Dense dataset x algorithm heatmap, bold-red best (row max).

    Rendered with pcolormesh + vector text so the PDF output stays fully
    vector (crisp cell edges and annotations at any zoom), unlike imshow
    which rasterizes the grid.
    """
    df = pd.read_csv(os.path.join(RESULTS, csv_name), index_col=0)
    data = df.values.astype(float)
    rows = list(df.index)
    cols = list(df.columns)
    nrows, ncols = data.shape

    fig, ax = plt.subplots(figsize=(15, 10))
    cmap = cm.get_cmap('YlGnBu').copy()
    cmap.set_bad(color='#f0f0f0')            # NaN cells (undefined metric)
    masked = np.ma.masked_invalid(data)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # rows drawn top-to-bottom: flip vertical axis
    x = np.arange(ncols + 1)
    y = np.arange(nrows + 1)
    mesh = ax.pcolormesh(x, y, masked, cmap=cmap,
                         norm=Normalize(vmin=vmin, vmax=vmax),
                         edgecolors='white', linewidth=1.3)
    mesh.set_rasterized(False)               # keep vector in PDF
    ax.invert_yaxis()
    ax.set_aspect('auto')

    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_xticklabels(cols, rotation=40, ha='right', fontsize=12)
    ax.set_yticklabels(rows, fontsize=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # annotate every cell; bold red = best (max) in the row.
    # NaN = metric undefined (e.g., a single cluster): show a dash so the
    # cell reads as "not applicable" rather than a missing value.
    for i in range(nrows):
        row = data[i, :]
        best = np.nanmax(row)
        for j in range(ncols):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j + 0.5, i + 0.5, '–',
                        ha='center', va='center',
                        fontsize=11, color='#888888')
                continue
            is_best = np.isclose(v, best)
            norm_v = (v - vmin) / (vmax - vmin + 1e-9)
            base = 'white' if norm_v > 0.6 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{v:.2f}',
                    ha='center', va='center',
                    fontsize=11,
                    color='red' if is_best else base,
                    fontweight='bold' if is_best else 'normal')

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)
    ax.set_title(title, fontsize=15, pad=12)
    fig.savefig(os.path.join(RESULTS, out_name))   # .pdf -> vector
    plt.close(fig)
    print('wrote', out_name)


def fig_parameter_sensitivity(out_name):
    df = pd.read_csv(os.path.join(RESULTS, 'parameter_sensitivity.csv'))
    params = ['epsilon', 'delta', 'beta', 'theta', 'lambda_value']
    pretty = {'epsilon': r'$\varepsilon$ (edge threshold)',
              'delta': r'$\delta$ (percentile)',
              'beta': r'$\beta$ (consecutive count)',
              'theta': r'$\theta$ (subgraph threshold)',
              'lambda_value': r'$\lambda$ (confidence)'}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    datasets = list(df.Dataset.unique())
    cmap = cm.get_cmap('tab10')
    for k, p in enumerate(params):
        ax = axes[k]
        sub = df[df.Parameter == p]
        for di, ds in enumerate(datasets):
            d = sub[sub.Dataset == ds].sort_values('Value')
            if d.empty:
                continue
            style = '-' if d.DatasetType.iloc[0] == 'synthetic' else '--'
            ax.plot(d.Value, d.ARI, style, marker='o', markersize=3,
                    linewidth=1.3, color=cmap(di % 10), label=ds)
        ax.set_xlabel(pretty[p], fontsize=11)
        ax.set_ylabel('ARI', fontsize=11)
        ax.set_title(pretty[p], fontsize=12)
        ax.grid(True, alpha=0.3)
    # shared legend in the empty 6th cell
    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', fontsize=10,
                   title='Dataset (solid=synthetic, dashed=real-world)',
                   title_fontsize=10, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, out_name))
    plt.close(fig)
    print('wrote', out_name)


def fig_noise(out_name):
    df = pd.read_csv(os.path.join(RESULTS, 'noise_analysis.csv'))
    # keep only the numeric noise-fraction sweep rows
    df = df[pd.to_numeric(df['noise_fraction'], errors='coerce').notna()].copy()
    df['noise_fraction'] = df['noise_fraction'].astype(float)
    df = df.sort_values('noise_fraction')
    x = df['noise_fraction'] * 100
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x, df['CoreClustering_ARI'], 'o-', linewidth=2, markersize=8,
            label='Core Clustering', color='#e74c3c')
    ax.plot(x, df['HDBSCAN_ARI'], 's-', linewidth=2, markersize=8,
            label='HDBSCAN', color='#2980b9')
    ax.plot(x, df['DBSCAN_ARI'], '^-', linewidth=2, markersize=8,
            label='DBSCAN', color='#27ae60')
    ax.set_xlabel('Noise fraction (%)', fontsize=12)
    ax.set_ylabel('ARI', fontsize=12)
    ax.set_title('Noise robustness on Noisy Moons', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.savefig(os.path.join(RESULTS, out_name))
    plt.close(fig)
    print('wrote', out_name)


def fig_scalability(out_name):
    df = pd.read_csv(os.path.join(RESULTS, 'scalability.csv'))
    core = df[df.algorithm == 'CoreClustering'].sort_values('n')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    # left: stage breakdown for Core Clustering
    stages = ['graph_construction', 'density_sequence',
              'core_identification', 'segment_expansion']
    labels = ['Graph construction', 'Density sequence',
              'Core identification', 'Segment expansion']
    for s, lab in zip(stages, labels):
        ax1.plot(core['n'], core[s], 'o-', linewidth=1.8, markersize=5, label=lab)
    ax1.set_xlabel('Number of points $n$', fontsize=12)
    ax1.set_ylabel('Time (s)', fontsize=12)
    ax1.set_title('(a) Core Clustering stage breakdown', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    # right: total runtime comparison
    for alg, color, mk in [('CoreClustering', '#e74c3c', 'o'),
                           ('DBSCAN', '#27ae60', '^'),
                           ('HDBSCAN', '#2980b9', 's')]:
        d = df[df.algorithm == alg].sort_values('n')
        ax2.plot(d['n'], d['total'], mk + '-', color=color,
                 linewidth=1.8, markersize=6, label=alg)
    ax2.set_xlabel('Number of points $n$', fontsize=12)
    ax2.set_ylabel('Total runtime (s)', fontsize=12)
    ax2.set_title('(b) Total runtime vs. baselines', fontsize=13)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, out_name))
    plt.close(fig)
    print('wrote', out_name)


def main():
    _heatmap('pivot_ARI.csv', 'fig1_ari_heatmap.pdf', 'ARI',
             'Adjusted Rand Index (ARI) across algorithms and datasets')
    fig_parameter_sensitivity('fig2_parameter_sensitivity.pdf')
    _heatmap('pivot_NMI.csv', 'fig3_nmi_heatmap.pdf', 'NMI',
             'Normalized Mutual Information (NMI) across algorithms and datasets')
    _heatmap('pivot_Silhouette.csv', 'fig4_silhouette_heatmap.pdf', 'Silhouette',
             'Silhouette Score across algorithms and datasets')
    fig_noise('fig5_noise_robustness.pdf')
    fig_scalability('fig6_scalability.pdf')


if __name__ == '__main__':
    main()
