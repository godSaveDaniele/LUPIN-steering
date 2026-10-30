import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_results(csv_path, output_dir):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    # Lettura dei dati CSV
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Ordinamento per alpha crescente
    rows.sort(key=lambda row: float(row["alpha"]))

    def values(column):
        return [float(row[column]) for row in rows]

    alpha = values("alpha")

    # Figura con tre pannelli
    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 10),
        sharex=True
    )

    # Compliance
    axes[0].plot(
        alpha, values("compliance"),
        "o-",
        color="#315B7D",
        linewidth=2,
        markersize=5,
        label="Compliance"
    )

    axes[0].plot(
        alpha, values("non_vacuous_compliance"),
        "s-",
        color="#749B78",
        linewidth=2,
        markersize=5,
        label="Non-vacuous compliance"
    )

    axes[0].set_ylabel("Compliance", fontsize=11)
    axes[0].legend(frameon=False)

    # Activation rate
    axes[1].plot(
        alpha, values("activation_rate"),
        "o-",
        color="#B08C46",
        linewidth=2,
        markersize=5
    )

    axes[1].set_ylabel("Activation rate", fontsize=11)

    # DL score
    axes[2].plot(
        alpha, values("dl_score_original"),
        "o-",
        color="#9B718C",
        linewidth=2,
        markersize=5
    )

    axes[2].set_ylabel("DL score", fontsize=11)
    axes[2].set_xlabel(r"Steering coefficient $\alpha$", fontsize=11)
    axes[2].set_xticks(sorted(set(alpha)))

    # Impostazioni comuni
    for ax in axes:
        ax.set_yscale("linear")

        ax.grid(
            True,
            linestyle="--",
            alpha=0.25
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.margins(x=0.03, y=0.15)
        ax.tick_params(axis="x", labelbottom=True)

    # Percentuali per compliance e activation rate
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1, decimals=1)
        )

    # Layout
    fig.tight_layout()

    # Salvataggio
    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        path = output_dir / f"steering_secondo_vincolo.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Salvato: {path.resolve()}")

    plt.show

if __name__=='__main__':
    plot_results("/Users/daniele/Desktop/projects/LUPIN/plots/metrics_to_plot/results copy.csv", "plots/results")