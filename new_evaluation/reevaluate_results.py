import csv
import json
import math
from collections import Counter
from pathlib import Path
from jellyfish import damerau_levenshtein_distance

INPUT_DIR = Path('/Users/daniele/Desktop/esperimenti/8_steering_secondo_vincolo/risultati parziali/')
# Stesso ordine usato durante lo steering originale.

ALPHAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.5]
OUTPUT_FILE = INPUT_DIR.parent / 'risultati importanti/new_results.csv'

def dls(predicted, real):
    # Un carattere per attività: la distanza conta attività, non lettere o cifre.
    vocabulary = {activity: chr(i) for i, activity in enumerate(dict.fromkeys(predicted + real))}
    a = ''.join(vocabulary[x] for x in predicted)
    b = ''.join(vocabulary[x] for x in real)
    length = max(len(a), len(b))
    return 1 - damerau_levenshtein_distance(a, b) / length if length else 1.0


def main():
    # Il timestamp YYYYMMDD_HHMMSS nei nomi permette l'ordine cronologico.
    files = sorted(INPUT_DIR.rglob('examples.jsonl'), key=lambda path: path.parent.name)
    if not files:
        raise ValueError(f'Nessun examples.jsonl in {INPUT_DIR}')
    if len(files) != len(ALPHAS):
        raise ValueError(f'Trovati {len(files)} file, ma specificati {len(ALPHAS)} alpha.')
    if len(set(ALPHAS)) != len(ALPHAS):
        raise ValueError('La lista ALPHAS contiene duplicati.')
    results, reference = [], None
    for alpha, path in zip(ALPHAS, files):
        counts = Counter()
        scores, compliant_scores = [], []
        examples = {}
        with path.open(encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                index = item['test_index']
                if index in examples:
                    raise ValueError(f'test_index duplicato in {path}: {index}')
                state, real_state = item['constraint_predicted'], item['constraint_real']
                if state not in {'fulfilled', 'violated', 'vacuous'} or real_state not in {'fulfilled', 'violated', 'vacuous'}:
                    raise ValueError(f'Stato del vincolo non valido in {path}')
                examples[index] = (tuple(item['prefix_activities']), tuple(item['true_suffix']), real_state)
                counts[state] += 1
                score = dls(item['predicted_suffix'], item['true_suffix'])
                scores.append(score)
                if real_state in {'fulfilled', 'vacuous'}:
                    compliant_scores.append(score)
        if reference is None:
            reference = examples
        elif examples != reference:
            raise ValueError(f'Test set o stati reali diversi in {path}')

        n = len(scores)
        f, v, u = (counts[k] for k in ('fulfilled', 'violated', 'vacuous'))
        results.append({
            'alpha': alpha,
            'examples': n,
            'dl_score_original': math.fsum(scores) / n,
            'dl_score_compliant_subset': math.fsum(compliant_scores) / len(compliant_scores) if compliant_scores else None,
            'compliance': (f + u) / n,
            'non_vacuous_compliance (support)': f / n,
            'confidence': f / (f + v) if f + v else None,
            'activation_rate': (f + v) / n,
            'fulfilled': f,
            'violated': v,
            'vacuous': u,
        })
        print(f'alpha={alpha:g}: {n} esempi, {len(compliant_scores)} reali compliant; {path.parent.name}')
    results.sort(key=lambda row: row['alpha'])
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'Salvato: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
