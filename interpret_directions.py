"""Interpreta un task vector heads_only tramite Delta W h + Delta b, senza softmax."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from eval_model import BASE_MODEL, normalize, read_pickle
from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from preprocessing.log_to_history import Log
from utility import reproducibility
from weight_steering import check_weights

GROUPS = {
    'satisfied': 'A presente: obblighi già soddisfatti',
    'pending': 'A presente: obbligo aperto',
    'absent': 'A assente',
}


def prefix_group(prefix, activation, target):
    """Stato di Response(A,B) nel solo prefisso: non usa il suffisso reale."""
    seen_a = False
    pending = False
    for activity in prefix:
        if activity == activation:
            seen_a = True
            pending = True
        elif activity == target:
            pending = False
    return 'pending' if pending else 'satisfied' if seen_a else 'absent'


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    # Compatibilità con checkpoint salvati tramite DataParallel.
    return {name.removeprefix('module.'): value for name, value in state.items()}


def head_directions(base, destination, positions):
    check_weights(base, destination)
    head_keys = {f'output_layers.{t}.{kind}' for t in range(positions) for kind in ('weight', 'bias')}
    if not head_keys.issubset(base):
        raise ValueError('Numero di teste non coerente con i file di preprocessing.')
    for name in base:
        if name not in head_keys and not torch.equal(base[name], destination[name]):
            raise ValueError(f'Non è un adattamento heads_only: è cambiato {name}')
    # Double per sottrazione e accumulo: l'estrazione BERT resta nella precisione del modello.
    weights = torch.stack([destination[f'output_layers.{t}.weight'].double()
                           - base[f'output_layers.{t}.weight'].double() for t in range(positions)])
    biases = torch.stack([destination[f'output_layers.{t}.bias'].double()
                          - base[f'output_layers.{t}.bias'].double() for t in range(positions)])
    return weights, biases


def delta_logits(pooled, weights, biases):
    """[batch, hidden] × [posizioni, attività, hidden] -> [batch, posizioni, attività]."""
    return torch.einsum('bh,tah->bta', pooled.double(), weights) + biases.unsqueeze(0)


def save_matrices(output_dir, means, counts, activities):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    for group, matrix in means.items():
        with (output_dir / f'{group}_mean_delta_logits.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['future_position'] + activities)
            for t, row in enumerate(matrix, start=1):
                writer.writerow([t] + [float(x) if np.isfinite(x) else '' for x in row])

    finite = np.concatenate([m[np.isfinite(m)] for m in means.values()])
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    scale = scale or 1.0
    cmap = LinearSegmentedColormap.from_list('muted', ['#CE8E89', '#FAF8EF', '#91B493'])
    fig, axes = plt.subplots(1, 3, figsize=(max(17, len(activities) * 2), 6.5),
                             layout='constrained', sharey=True)
    for ax, (group, title) in zip(axes, GROUPS.items()):
        matrix = means[group]
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-scale, vmax=scale, aspect='auto')
        ax.set_title(f'{title}\nn = {counts[group]:,}')
        ax.set_xticks(range(len(activities)), activities, rotation=55, ha='right')
        ax.set_yticks(range(matrix.shape[0]), range(1, matrix.shape[0] + 1))
        ax.set_xlabel('Attività (END incluso)')
        if counts[group] == 0:
            ax.text(.5, .5, 'Nessun prefisso: media non definita', transform=ax.transAxes, ha='center')
        elif matrix.size <= 150:
            for (t, a), value in np.ndenumerate(matrix):
                ax.text(a, t, f'{value:+.2f}', ha='center', va='center', fontsize=8)
    axes[0].set_ylabel('Posizione futura / testa')
    fig.suptitle('Variazione media dei logits: ΔW h + Δb (nessuna softmax)')
    fig.colorbar(im, ax=list(axes), label='Δ logit — scala comune centrata su zero', shrink=.75)
    fig.savefig(output_dir / 'mean_delta_logits.png', dpi=180)
    fig.savefig(output_dir / 'mean_delta_logits.pdf')
    plt.close(fig)


def interpret_direction(args):
    if args.batch_size < 1:
        raise ValueError('batch-size deve essere positivo.')
    activation, target = normalize(args.activation), normalize(args.target)
    if activation == target:
        raise ValueError('Questa analisi richiede A e B distinte.')
    reproducibility.set_seed()
    base = load_checkpoint(args.base_checkpoint)
    destination = load_checkpoint(args.destination_checkpoint)
    folder = Path('log_history') / args.dataset
    kinds = ['test', 'suffix_test', 'prefix_activities_test', 'len_test', 'id2label', 'label2id']
    paths = {kind: folder / f'{args.dataset}_{kind}_all.pkl' for kind in kinds}
    # Genera tutti i file con lo split temporale originale, senza filtri sui vincoli.
    if args.regenerate_logs or any(not path.is_file() for path in paths.values()):
        Log(args.dataset, 'all')
    data = {kind: read_pickle(path) for kind, path in paths.items()}
    texts, prefixes, labels = data['test'], data['prefix_activities_test'], data['suffix_test']
    id2label, label2id = data['id2label']['activity'], data['label2id']['activity']
    if not texts or len(prefixes) != len(texts) or len(data['len_test']) != len(texts):
        raise ValueError('Test vuoto o prefissi e testi non allineati.')
    if any(len(p) != n for p, n in zip(prefixes, data['len_test'])):
        raise ValueError('Lunghezze dei prefissi incoerenti.')
    if not labels or any(len(v) != len(texts) for v in labels.values()):
        raise ValueError('Label dei suffissi non allineate.')
    for activity in (activation, target):
        if activity not in label2id or activity == 'ENDactivity':
            raise ValueError(f'Attività non valida: {activity}')
    positions = len(labels)
    dw, db = head_directions(base, destination, positions)
    if dw.shape[1] != len(id2label):
        raise ValueError('Vocabolario non coerente con i checkpoint.')
    activities = [id2label[i] for i in range(len(id2label))]
    groups = [prefix_group(p, activation, target) for p in prefixes]
    counts = {g: groups.count(g) for g in GROUPS}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, truncation_side='left')
    loader = DataLoader(CustomDataset(texts, {}, tokenizer, 512), batch_size=args.batch_size, shuffle=False)
    model = BertMultiOutputClassificationHeads(AutoModel.from_pretrained(BASE_MODEL), [len(id2label)] * positions)
    model.load_state_dict(base)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    # Calcolo piccoli delta e somme su CPU double; BERT eseguito sulla GPU se disponibile.
    sums = {g: torch.zeros_like(db) for g in GROUPS}
    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Pooled vector e delta logits'):
            pooled = model.gpt_model(input_ids=batch['input_ids'].to(device),
                                     attention_mask=batch['attention_mask'].to(device)).pooler_output.cpu()
            delta = delta_logits(pooled, dw, db)
            if not torch.isfinite(delta).all():
                raise ValueError('Delta logits non finiti.')
            batch_groups = groups[offset:offset + len(pooled)]
            for group in GROUPS:
                mask = torch.tensor([g == group for g in batch_groups], dtype=torch.bool)
                sums[group] += delta[mask].sum(dim=0)
            offset += len(pooled)
    means = {g: (sums[g] / counts[g]).numpy() if counts[g]
             else np.full(tuple(db.shape), np.nan) for g in GROUPS}
    run = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = Path('outputs') / f'interpret_direction_{args.dataset}_{run}'
    output_dir.mkdir(parents=True)
    config = dict(vars(args), template='Response', normalized_activation=activation,
                  normalized_target=target, counts=counts, examples=len(texts),
                  base_model=BASE_MODEL, seed=reproducibility.SEED,
                  note='Ogni prefisso pesa ugualmente. Gruppi sul prefisso completo; testo troncato a 512 token a sinistra.')
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2))
    with (output_dir / 'prefix_groups.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['test_index', 'prefix_length', 'group'])
        writer.writerows((i, len(p), g) for i, (p, g) in enumerate(zip(prefixes, groups)))
    save_matrices(output_dir, means, counts, activities)
    print('Prefissi per gruppo:', counts)
    print('Risultati:', output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--destination-checkpoint', required=True)
    parser.add_argument('--activation', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--regenerate-logs', action='store_true', help='Rigenera il preprocessing originale.')
    interpret_direction(parser.parse_args())


if __name__ == '__main__':
    main()
