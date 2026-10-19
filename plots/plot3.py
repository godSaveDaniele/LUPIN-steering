
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator

# Lettura dei dati
with open("/Users/daniele/Desktop/projects/LUPIN/plots/metrics_to_plot/losses_semantic_finetuning.json", "r") as f:
    data = json.load(f)

epochs = [item["epoch"] for item in data]
train_loss = [item["train_semantic_loss"] for item in data]
val_loss = [item["val_semantic_loss"] for item in data]

# Colori
TRAIN_COLOR = "#3274A1"
VAL_COLOR = "#D68C45"

# Figura con due pannelli
fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 4.5),
    sharex=True
)

for ax in axes:
    ax.plot(
        epochs, train_loss,
        color=TRAIN_COLOR,
        marker="o",
        markersize=5,
        linewidth=2,
        label="Training loss"
    )

    ax.plot(
        epochs, val_loss,
        color=VAL_COLOR,
        marker="s",
        markersize=5,
        linewidth=2,
        label="Validation loss"
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Fuzzy Semantic Loss", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(epochs)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Scala lineare
axes[0].set_title("Linear scale", fontsize=12)
axes[0].set_yscale("linear")

# Scala logaritmica
axes[1].set_title("Logarithmic scale", fontsize=12)
axes[1].set_yscale("log")
axes[1].yaxis.set_major_locator(LogLocator(base=10))
axes[1].yaxis.set_major_formatter(LogFormatterMathtext())

# Legenda condivisa
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    ncol=2,
    frameon=False
)

plt.tight_layout(rect=[0, 0.07, 1, 1])

# Salvataggio
plt.savefig("semantic_loss.pdf", bbox_inches="tight")
plt.savefig("semantic_loss.png", dpi=300, bbox_inches="tight")

plt.show()
