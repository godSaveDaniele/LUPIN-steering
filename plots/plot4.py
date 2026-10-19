import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


JSON_PATH = '/Users/daniele/Desktop/projects/LUPIN/plots/metrics_to_plot/metrics_classic_semantic_losses.json'

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
records= sorted(data, key=lambda item: float(item['alpha']))


alpha = [float(r['alpha']) for r in records]
compliance = [r['predicted']['compliance'] for r in records]
non_vacuous = [r['predicted']['non_vacuous_compliance'] for r in records]
activation = [r['predicted']['activation_rate'] for r in records]
dl = [r['dl_score_original'] for r in records]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True, gridspec_kw={'hspace': 0.16})
regions = [
    (0.0, 0.4, '#E1F0E5'),
    (0.4, max(alpha), "#E8DDB5"),
]
for ax in (ax1, ax2):
    for start, end, color in regions:
        ax.axvspan(start, end, color=color, alpha=0.8, zorder=0)
    boundary=0.4
    ax.axvline(boundary, color='#777777', linestyle='--', linewidth=1, alpha=0.8)
    ax.grid(axis='y', alpha=0.2, linestyle=':')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(min(alpha)-0.025, max(alpha)+0.025)

ax1.plot(alpha, compliance, '-o', label='Compliance', color='#2879B8', lw=2, ms=4)
ax1.plot(alpha, non_vacuous, '-s', label='Non-vacuous compliance', color='#248655', lw=2, ms=4)
ax1.plot(alpha, activation, '-^', label='Activation rate', color='#C17A16', lw=2, ms=4)
ax1.set_ylabel('Rate', fontsize=11)
ax1.set_ylim(0.18, 1.08)
ax1.set_title(' Constraint evaluation', loc='left', fontsize=13, weight='bold')
ax1.legend(loc='lower left', ncol=3, fontsize=9, frameon=False)

ax2.plot(alpha, dl, '-o', color='#565D9B', label='Damerau–Levenshtein similarity', lw=2.5, ms=4)
ax2.set_ylabel('Similarity', fontsize=11)
ax2.set_ylim(0.35, 0.93)
ax2.set_title('Prediction Evaluation', loc='left', fontsize=13, weight='bold')
ax2.legend(loc='lower left', frameon=False, fontsize=9)
ax2.set_xlabel('Alpha', fontsize=11)
ax2.set_xticks(alpha)
ax2.set_xticklabels([f'{a:g}' for a in alpha], fontsize=9)


fig.subplots_adjust(top=0.87, bottom=0.09, left=0.09, right=0.98)

for ext in ('png', 'pdf'):
    fig.savefig(f'plots/results/three_regions.{ext}', dpi=300, bbox_inches='tight')
plt.show()