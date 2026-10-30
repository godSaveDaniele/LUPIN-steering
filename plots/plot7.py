"""Uso: python plot7.py constrained.csv random_subset.csv"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def read_results(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Il CSV è vuoto: {path}")
    columns = ("alpha", "compliance", "non_vacuous_compliance",
               "activation_rate", "dl_score_original")
    rows.sort(key=lambda row: float(row["alpha"]))
    return {column: [float(row[column]) for row in rows] for column in columns}


def plot_results(csv_path, random_csv_path, output_dir):
    constrained = read_results(csv_path)
    random_subset = read_results(random_csv_path)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    def plot_metric(ax, column, label, color, marker="o"):
        ax.plot(constrained["alpha"], constrained[column],
                linestyle="-", marker=marker, color=color,
                label=f"{label} — Constrained")
        ax.plot(random_subset["alpha"], random_subset[column],
                linestyle="--", marker=marker, markerfacecolor="white",
                color=color, label=f"{label} — Random subset")

    plot_metric(axes[0], "compliance", "Compliance", "#315B7D")
    plot_metric(axes[0], "non_vacuous_compliance",
                "Non-vacuous compliance", "#749B78", marker="s")
    axes[0].set_ylabel("Compliance")

    plot_metric(axes[1], "activation_rate", "Activation rate", "#B08C46")
    axes[1].set_ylabel("Activation rate")

    plot_metric(axes[2], "dl_score_original", "DL score", "#9B718C")
    axes[2].set_ylabel("DL score")
    axes[2].set_xlabel(r"Steering Coefficient $\alpha$")
    axes[2].set_xticks(sorted(set(constrained["alpha"] + random_subset["alpha"])))

    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.03, y=0.15)
        ax.tick_params(axis="x", labelbottom=True)

    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

    fig.suptitle("Steering: constrained e random subset", fontsize=16)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"steering_comparison.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Salvato: {path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Risultati constrained")
    parser.add_argument("random_csv_path", type=Path, help="Risultati random subset")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/results"))
    args = parser.parse_args()
    plot_results(args.csv_path, args.random_csv_path, args.output_dir)
