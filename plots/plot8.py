"""Confronta compliance, non-vacuous compliance, activation rate e DL score per tre seed.
Modifica i percorsi e le etichette qui sotto, poi esegui questo file.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# Inserisci i tre CSV e i seed effettivamente utilizzati.
# I percorsi relativi vengono risolti rispetto alla cartella di questo script.
STEERING = [
    ("Seed 42", Path("/Users/daniele/Desktop/projects/LUPIN/plots/metrics_to_plot/results.csv"), "#4878A8"),
    ("Seed 43", Path("/Users/daniele/Desktop/projects/LUPIN/plots/metrics_to_plot/results_seed_43.csv"), "#719B75"),
    ("Seed 44", Path("/Users/daniele/Desktop/projects/LUPIN/plots/metrics_to_plot/results_seed_44.csv"), "#BC827A"),
]
OUTPUT_DIR = Path(__file__).resolve().parent


def read_results(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Il CSV è vuoto: {path}")
    rows.sort(key=lambda row: float(row["alpha"]))
    return {
        column: [float(row[column]) for row in rows]
        for column in ("alpha", "compliance", "non_vacuous_compliance",
                       "activation_rate", "dl_score_original")
    }


def main():
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    metrics = [
        ("compliance", "Compliance"),
        ("non_vacuous_compliance", "Non-vacuous compliance"),
        ("activation_rate", "Activation rate"),
        ("dl_score_original", "DL score"),
    ]
    all_alphas = set()

    for label, path, color in STEERING:
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        data = read_results(path)
        all_alphas.update(data["alpha"])

        for ax, (column, _) in zip(axes, metrics):
            ax.plot(data["alpha"], data[column],
                    "o-", color=color, label=label)

    for ax, (column, title) in zip(axes, metrics):
        ax.set_title(title)
        ax.set_ylabel(title)
        if column != "dl_score_original":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.03, y=0.15)
        ax.tick_params(axis="x", labelbottom=True)

    axes[-1].set_xticks(sorted(all_alphas))
    axes[-1].set_xlabel(r"Coefficiente $\alpha$")
    fig.suptitle("Steering con tre seed diversi", fontsize=16)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = OUTPUT_DIR / f"steering_three_seeds.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Salvato: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
