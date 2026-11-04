"""Compare three steering seeds. Set CSV paths and seed labels below."""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Replace these paths and labels with the actual experiments.
CSV_FILES = [Path('/Users/daniele/Desktop/esperimenti/1_steering_compliant_subset_seed42/risultati importanti/new_results.csv'),
             Path('/Users/daniele/Desktop/esperimenti/3_steering_compliant_subset_seed_43/risultati importanti/new_results.csv'),
             Path('/Users/daniele/Desktop/esperimenti/4_steering_compliant_subset_seed_44/risultati importanti/new_results.csv')]
SEED_LABELS = ['Seed 42', 'Seed 43', 'Seed 44']
OUTPUT_DIR = (CSV_FILES[1]).parent 


def read_results(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    rows.sort(key=lambda row: float(row['alpha']))
    columns = ['alpha', 'compliance', 'non_vacuous_compliance (support)',
               'dl_score_original', 'dl_score_compliant_subset']
    return {key: [float(row[key]) if row[key] else float('nan') for row in rows]
            for key in columns}


def plot_three_seeds(csv_files, seed_labels, output_dir=OUTPUT_DIR):
    """Read three CSVs and return (figure, axes); save PNG and PDF."""
    if len(csv_files) != 3 or len(seed_labels) != 3:
        raise ValueError('Provide exactly three CSV paths and three seed labels.')
    data = [read_results(path) for path in csv_files]
    colors = ['#0072F5', '#F28C00', '#B536DA']
    markers = ['o', 's', '^']
    metrics = [
        [('compliance', 'Compliance'),
         ('dl_score_original', 'DLS — Full test set')],
        [('non_vacuous_compliance (support)', 'Non-vacuous compliance (support)'),
         ('dl_score_compliant_subset', 'DLS — Compliant test subset')],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True)
    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            key, label = metrics[row][col]
            for dataset, seed, color, marker in zip(data, seed_labels, colors, markers):
                ax.plot(dataset['alpha'], dataset[key], color=color, marker=marker,
                        linewidth=2, markersize=5, label=seed)
            ax.set_ylabel(label)
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='x', labelbottom=True)
            ax.margins(x=0.03)
            if col == 0:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    # For each column: maximum excursion across BOTH metrics and ALL seeds.
    # The two axes have equal spans, but are centered on their own values.
    for col in range(2):
        bounds = []
        for row in range(2):
            key = metrics[row][col][0]
            finite = [v for dataset in data for v in dataset[key] if math.isfinite(v)]
            if not finite:
                raise ValueError(f'No finite values for {key}')
            bounds.append((min(finite), max(finite)))
        span = min(1.0, max(max(hi - lo for lo, hi in bounds) * 1.16, 0.01))
        for row, (lo, hi) in enumerate(bounds):
            center = (lo + hi) / 2
            bottom = max(0.0, min(center - span / 2, 1.0 - span))
            axes[row, col].set_ylim(bottom, bottom + span)

    ticks = sorted({a for dataset in data for a in dataset['alpha']})
    for ax in axes[1, :]:
        ax.set_xlabel(r'Steering coefficient $\alpha$')
        ax.set_xticks(ticks)
    fig.tight_layout()
    for ax in axes.flat:
        ax.set_box_aspect(0.27)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        path = output_dir / f'three_seeds_metrics.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig, axes


if __name__ == '__main__':
    # Relative paths are resolved from the script's directory.
    csv_files = [p if p.is_absolute() else Path(__file__).resolve().parent / p
                 for p in CSV_FILES]
    fig, axes = plot_three_seeds(csv_files, SEED_LABELS)
    plt.show()
