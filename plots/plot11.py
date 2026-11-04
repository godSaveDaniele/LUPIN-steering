"""Compare real steering and a random subset of the same cardinality."""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Set the two actual paths here.
STEERING_CSV = Path('/Users/daniele/Desktop/esperimenti/1_steering_compliant_subset_seed42/risultati importanti/new_results.csv')
RANDOM_CSV = Path('/Users/daniele/Desktop/esperimenti/2_steering_random_subset_seed_42/risultati importanti/new_results.csv')
OUTPUT_DIR = RANDOM_CSV.parent


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


def plot_comparison(steering_csv, random_csv, output_dir=OUTPUT_DIR):
    steering = read_results(steering_csv)
    random = read_results(random_csv)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 2]})
    metrics = [
        (0, 'compliance', 'Compliance', '#0072F5'),
        (1, 'non_vacuous_compliance (support)',
         'Non-vacuous compliance (support)', '#F28C00'),
        (2, 'dl_score_original', 'DLS — Full test set', '#00A878'),
        (2, 'dl_score_compliant_subset', 'DLS — Compliant test subset', '#B536DA'),
    ]
    for panel, key, label, color in metrics:
        for data, method, style in [(steering, 'Steering', '-'),
                                     (random, 'Random subset', '--')]:
            legend = method if panel < 2 else f'{label} — {method}'
            axes[panel].plot(data['alpha'], data[key], linestyle=style,
                             marker='o', color=color, linewidth=2, markersize=5,
                             markerfacecolor=color if style == '-' else 'white',
                             label=legend)

    axes[0].set_ylabel('Compliance')
    axes[1].set_ylabel('Non-vacuous compliance\n(support)')
    axes[2].set_ylabel('Damerau–Levenshtein similarity')

    # Same y-axis span for compliance and support, with different offsets.
    # Include both experiments when computing each metric's excursion.
    bounds = []
    for key in ('compliance', 'non_vacuous_compliance (support)'):
        finite = [v for data in (steering, random) for v in data[key]
                  if math.isfinite(v)]
        if not finite:
            raise ValueError(f'No finite values for {key}')
        bounds.append((min(finite), max(finite)))
    span = min(1.0, max(max(hi - lo for lo, hi in bounds) * 1.16, 0.01))
    for ax, (lo, hi) in zip(axes[:2], bounds):
        center = (lo + hi) / 2
        bottom = max(0.0, min(center - span / 2, 1.0 - span))
        ax.set_ylim(bottom, bottom + span)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', labelbottom=True)
        ax.margins(x=0.03)
    axes[2].margins(y=0.15)
    axes[2].set_xlabel(r'Steering coefficient $\alpha$')
    axes[2].set_xticks(sorted(set(steering['alpha'] + random['alpha'])))
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        path = output_dir / f'steering_vs_random.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    return fig, axes


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    steering_path = STEERING_CSV if STEERING_CSV.is_absolute() else folder / STEERING_CSV
    random_path = RANDOM_CSV if RANDOM_CSV.is_absolute() else folder / RANDOM_CSV
    fig, axes = plot_comparison(steering_path, random_path)
    plt.show()
