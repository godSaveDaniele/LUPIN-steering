import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Lettura dei dati
with open("/Users/daniele/Desktop/projects/LUPIN/plots/metrics_to_plot/losses_finetuning1.json", "r") as f:
    data = json.load(f)

epochs = [item["epoch"] for item in data]
train_loss = [item["train_loss"] for item in data]
val_loss = [item["val_loss"] for item in data]

# Colori
TRAIN_COLOR = "#3274A1"
VAL_COLOR = "#D68C45"

# Figura singola
fig, ax = plt.subplots(figsize=(7, 4.5))

# Training loss
ax.plot(
    epochs, train_loss,
    color=TRAIN_COLOR,
    marker="o",
    markersize=5,
    linewidth=2,
    label="Training loss"
)

# Validation loss
ax.plot(
    epochs, val_loss,
    color=VAL_COLOR,
    marker="s",
    markersize=5,
    linewidth=2,
    label="Validation loss"
)

# Impostazioni degli assi
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Cross Entropy Loss", fontsize=11)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(epochs)

# Scala lineare
ax.set_yscale("linear")

# Griglia
ax.grid(
    True,
    linestyle="--",
    alpha=0.25
)

# Rimozione bordi superiore e destro
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legenda
ax.legend(
    loc="best",
    frameon=False,
    fontsize=10
)

plt.tight_layout()

# Salvataggio
plt.savefig("cross_entropy_loss_1.pdf", bbox_inches="tight")
plt.savefig("cross_entropy_loss_1.png", dpi=300, bbox_inches="tight")

plt.show()