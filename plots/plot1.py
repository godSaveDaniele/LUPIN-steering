import json
import matplotlib.pyplot as plt

JSON_FILE = '/Users/daniele/Desktop/projects/LUPIN/plots/metrics_to_plot/metrics_semantic_finetuning_1.json'

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)



data = sorted(data, key=lambda item: float(item['alpha']))
alphas = [str(item['alpha']) for item in data]
fulfilled = [item['predicted']['fulfilled'] for item in data]
vacuous = [item['predicted']['vacuous'] for item in data]
violated = [item['predicted']['violated'] for item in data]
totals = [f + v + e for f, v, e in zip(fulfilled, vacuous, violated)]

fulfilled_pct = [100 * f / t for f, t in zip(fulfilled, totals)]
vacuous_pct = [100 * v / t for v, t in zip(vacuous, totals)]
violated_pct = [100 * e / t for e, t in zip(violated, totals)]


colori = {"verde": '#2E8B57', "giallo": '#F4C542' , "rosso":'#D9534F'}
colori = {"verde": '#91B493', "giallo": '#D8BF75' , "rosso":'#CE8E89'}
x = range(len(alphas))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, fulfilled_pct, color= colori["verde"], label='Activated and fulfilled')
ax.bar(x, vacuous_pct, bottom=fulfilled_pct, color= colori["giallo"], label='Vacuously satisfied')
ax.bar(x, violated_pct, bottom=[f + v for f, v in zip(fulfilled_pct, vacuous_pct)], color= colori["rosso"], label='Violated')
ax.set_xticks(list(x), alphas)
ax.set_xlabel('Alpha')
ax.set_ylabel('Predicted suffixes (%)')
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.savefig('plots/results/plot1.png', dpi=300)
plt.savefig('plots/results/plot1.pdf')
plt.show()
