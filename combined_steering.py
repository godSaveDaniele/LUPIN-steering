"""Combina due task vector e valuta entrambi i vincoli sul test originale."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from jellyfish import damerau_levenshtein_distance
from tqdm.auto import tqdm

from constraints.extraction import EVALUATORS
from eval_model import BASE_MODEL, read_pickle, trim_suffix, normalize
from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from utility import reproducibility
from weight_steering import check_weights, print_nonzero_directions


def combine_weights(base, checkpoint_1, checkpoint_2, alpha, beta):
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError('Alpha e beta devono essere finiti.')
    # Conserva esattamente i tre checkpoint agli estremi.
    if alpha == 0 and beta == 0:
        return base
    if alpha == 1 and beta == 0:
        return checkpoint_1
    if alpha == 0 and beta == 1:
        return checkpoint_2
    weights = {}
    for name, value in base.items():
        if not value.is_floating_point():
            weights[name] = value
            continue
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        start = value.to(dtype)
        direction_1 = checkpoint_1[name].to(dtype) - start
        direction_2 = checkpoint_2[name].to(dtype) - start
        weights[name] = (start + alpha * direction_1 + beta * direction_2).to(value.dtype)
        if not torch.isfinite(weights[name]).all():
            raise ValueError(f'Pesi non finiti: {name}, alpha={alpha}, beta={beta}')
    return weights


def metrics(counts):
    total = sum(counts.values())
    activated = counts['fulfilled'] + counts['violated']
    return dict(
        fulfilled=counts['fulfilled'], violated=counts['violated'], vacuous=counts['vacuous'],
        compliance=(counts['fulfilled'] + counts['vacuous']) / total,
        non_vacuous_compliance=counts['fulfilled'] / activated if activated else None,
        activation_rate=activated / total,
    )


def update_counts(trace, constraints, counts, joint):
    states = [EVALUATORS[c['template']](trace, c['activation'], c['target']) for c in constraints]
    for counter, state in zip(counts, states):
        counter[state] += 1
    ok1, ok2 = [state != 'violated' for state in states]
    key = 'both' if ok1 and ok2 else 'only_1' if ok1 else 'only_2' if ok2 else 'neither'
    joint[key] += 1
    return states


def summarize(counts, joint):
    return {
        'constraint_1': metrics(counts[0]),
        'constraint_2': metrics(counts[1]),
        'joint_compliance': joint['both'] / sum(joint.values()),
        'contingency': {key: joint[key] for key in ('both', 'only_1', 'only_2', 'neither')},
    }


def evaluate_combination(model, loader, texts, prefixes, id2label, end_id,
                         constraints, device, examples_path):
    counts, real_counts = [Counter(), Counter()], [Counter(), Counter()]
    joint, real_joint = Counter(), Counter()
    score_sum = 0.0
    index = 0
    model.eval()
    with torch.no_grad(), examples_path.open('w', encoding='utf-8') as file:
        for batch in tqdm(loader, desc='Evaluation', unit='batch'):
            outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            predictions = torch.stack([head.argmax(dim=1) for head in outputs], dim=1).cpu().tolist()
            for row, raw_ids in enumerate(predictions):
                pred_ids = trim_suffix(raw_ids, end_id)
                true_ids = trim_suffix([batch['labels'][p][row].item() for p in range(len(outputs))], end_id)
                pred = [id2label[i] for i in pred_ids]
                truth = [id2label[i] for i in true_ids]
                prefix = prefixes[index]
                states = update_counts(prefix + pred, constraints, counts, joint)
                real_states = update_counts(prefix + truth, constraints, real_counts, real_joint)
                a, b = ''.join(map(chr, pred_ids)), ''.join(map(chr, true_ids))
                length = max(len(a), len(b))
                score = 1 - damerau_levenshtein_distance(a, b) / length if length else 1.0
                score_sum += score
                file.write(json.dumps(dict(
                    test_index=index, prefix_text=texts[index], prefix_activities=prefix,
                    predicted_suffix=pred, true_suffix=truth, dl_score_original=score,
                    constraint_predicted=states, constraint_real=real_states,
                ), ensure_ascii=False) + '\n')
                index += 1
    return dict(examples=index, dl_score_original=score_sum / index,
                predicted=summarize(counts, joint), real=summarize(real_counts, real_joint))


def evaluate_grid(args):
    if not args.alphas or not args.betas or not all(math.isfinite(x) for x in args.alphas + args.betas):
        raise ValueError('Le liste alpha e beta devono essere non vuote e finite.')
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('Batch size e limit devono essere positivi.')
    reproducibility.set_seed()
    base, first, second = [torch.load(path, map_location='cpu', weights_only=True) for path in
                           (args.base_checkpoint, args.checkpoint_1, args.checkpoint_2)]
    check_weights(base, first)
    check_weights(base, second)
    print('Direzione vincolo 1:')
    print_nonzero_directions(base, first)
    print('Direzione vincolo 2:')
    print_nonzero_directions(base, second)

    folder = Path('log_history') / args.dataset
    def load(kind):
        return read_pickle(folder / f'{args.dataset}_{kind}_all.pkl')
    texts, labels, prefixes = load('test'), load('suffix_test'), load('prefix_activities_test')
    id2label, label2id = load('id2label')['activity'], load('label2id')['activity']
    if not texts or len(prefixes) != len(texts) or not labels or any(len(v) != len(texts) for v in labels.values()):
        raise ValueError('Test vuoto o testi, prefissi e label non allineati.')
    constraints = [dict(template=getattr(args, f'template_{i}'),
                        activation=normalize(getattr(args, f'activation_{i}')),
                        target=normalize(getattr(args, f'target_{i}'))) for i in (1, 2)]
    for c in constraints:
        for activity in (c['activation'], c['target']):
            if activity not in label2id or activity == 'ENDactivity':
                raise ValueError(f'Attività non valida: {activity}')
    limit = min(args.limit or len(texts), len(texts))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, truncation_side='left')
    dataset = CustomDataset(texts[:limit], {p: v[:limit] for p, v in labels.items()}, tokenizer, 512)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertMultiOutputClassificationHeads(AutoModel.from_pretrained(BASE_MODEL),
                                               [len(id2label)] * len(labels)).to(device)
    run = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = Path('outputs') / f'combined_steering_{args.dataset}_{run}'
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))
    print('Combinazioni:', len(args.alphas) * len(args.betas), '| Device:', device)
    results, summaries = [], []
    for alpha in args.alphas:
        for beta in args.betas:
            print(f'\n===== ALPHA={alpha}, BETA={beta} =====', flush=True)
            weights = combine_weights(base, first, second, alpha, beta)
            model.load_state_dict(weights)
            del weights
            pair_dir = output_dir / f'pair_{len(results):03d}'
            pair_dir.mkdir()
            summary = evaluate_combination(model, loader, texts, prefixes, id2label,
                                           label2id['ENDactivity'], constraints, device,
                                           pair_dir / 'examples.jsonl')
            summary.update(alpha=alpha, beta=beta, constraints=constraints)
            (pair_dir / 'metrics.json').write_text(json.dumps(summary, indent=2))
            summaries.append(summary)
            row = dict(alpha=alpha, beta=beta, examples=summary['examples'],
                       dl_score_original=summary['dl_score_original'],
                       joint_compliance=summary['predicted']['joint_compliance'])
            for i in (1, 2):
                for name, value in summary['predicted'][f'constraint_{i}'].items():
                    row[f'c{i}_{name}'] = value
            row.update(summary['predicted']['contingency'])
            results.append(row)
            with (output_dir / 'results.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(results)
            (output_dir / 'metrics.json').write_text(json.dumps(summaries, indent=2))
            print(json.dumps(row, indent=2), flush=True)
    print('Risultati:', output_dir / 'results.csv')
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--checkpoint-1', required=True)
    parser.add_argument('--checkpoint-2', required=True)
    for i in (1, 2):
        parser.add_argument(f'--template-{i}', choices=list(EVALUATORS), default='Response')
        parser.add_argument(f'--activation-{i}', required=True)
        parser.add_argument(f'--target-{i}', required=True)
    parser.add_argument('--alphas', nargs='+', type=float, default=[0, 0.1, 0.2, 0.3])
    parser.add_argument('--betas', nargs='+', type=float, default=[0, 0.1, 0.2, 0.3])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int)
    evaluate_grid(parser.parse_args())


if __name__ == '__main__':
    main()
