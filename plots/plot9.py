import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_results(csv_path):
    """Read new_results.csv and save three vertically aligned plots as PNG/PDF."""
    csv_path = Path(csv_path)
    with csv_path.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('The input CSV is empty.')
    rows.sort(key=lambda row: float(row['alpha']))

    def values(column):
        return [float(row[column]) if row[column] else float('nan') for row in rows]

    alpha = values('alpha')
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 2]})
    metrics = [
        ('compliance', 'Compliance', '#0072F5'),
        ('non_vacuous_compliance (support)', 'Non-vacuous compliance\n(support)', '#F28C00'),
        ('dl_score_original', 'DLS\nFull test set', '#00A878'),
        ('dl_score_compliant_subset', 'DLS\nCompliant test subset', '#B536DA'),
    ]

    for ax, (column, label, color) in zip(axes[:2], metrics[:2]):
        ax.plot(alpha, values(column), 'o-', color=color,
                linewidth=2, markersize=5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', labelbottom=True)
        ax.margins(x=0.03, y=0.15)

    # Equal y-axis span within each pair, but different vertical offsets.
    for first, second in ((0, 1),):
        bounds = []
        for index in (first, second):
            finite_values = [v for v in values(metrics[index][0]) if math.isfinite(v)]
            if not finite_values:
                raise ValueError('No finite values available for a metric.')
            bounds.append((min(finite_values), max(finite_values)))

        max_span = max(high - low for low, high in bounds)
        # Same 8% margin on each side; minimum span for constant curves.
        span = min(1.0, max(max_span * 1.16, 0.01))
        for index, (low, high) in zip((first, second), bounds):
            center = (low + high) / 2
            bottom = max(0.0, min(center - span / 2, 1.0 - span))
            axes[index].set_ylim(bottom, bottom + span)

    # Both DLS curves share the taller bottom panel.
    for column, label, color in metrics[2:]:
        axes[2].plot(alpha, values(column), 'o-', color=color,
                     linewidth=2, markersize=5, label=label.replace('\n', ' — '))
    axes[2].set_ylabel('Damerau–Levenshtein similarity')
    axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.25)
    axes[2].spines[['top', 'right']].set_visible(False)
    axes[2].margins(x=0.03, y=0.15)

    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[-1].set_xlabel(r'Steering coefficient $\alpha$')
    axes[-1].set_xticks(sorted(set(alpha)))
    fig.tight_layout()
    output_stem = csv_path.parent / 'plot'
    for extension in ('png', 'pdf'):
        output_path = output_stem.with_suffix(f'.{extension}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {output_path}')
    return fig, axes


if __name__ == '__main__':
    # Set the CSV path here; no command-line arguments are needed.
    CSV_PATH = '/Users/daniele/Desktop/esperimenti/8_steering_secondo_vincolo/risultati importanti/new_results.csv'
    fig, axes = plot_results(CSV_PATH)
    plt.show()
