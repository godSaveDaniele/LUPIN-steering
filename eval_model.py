import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from jellyfish import damerau_levenshtein_distance
from tqdm import tqdm

from constraints.extraction import EVALUATORS
from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from utility import reproducibility

#permette di valutare i modelli di suffix_generation
#Dato un test set, per ogni modello si valuta, la correttezza del
# suffisso rispetto alla ground truth e la compliance rispetto
# ai vincoli. 

BASE_MODEL = 'prajjwal1/bert-medium'


def read_pickle(path):
    with path.open('rb') as file:
        return pickle.load(file)


def trim_suffix(ids, end_id):
    result = []
    for value in ids:
        value = int(value)
        if value == end_id:
            break
        result.append(value)
    return result


def normalize(activity):
    for char in (' ', '+', '-', '_'):
        activity = activity.replace(char, '')
    return activity


def constraint_metrics(counts):
    total = sum(counts.values())
    activated = counts['fulfilled'] + counts['violated']
    return {
        'fulfilled': counts['fulfilled'],
        'violated': counts['violated'],
        'vacuous': counts['vacuous'],
        'compliance': (counts['fulfilled'] + counts['vacuous']) / total,
        'non_vacuous_compliance': counts['fulfilled'] / activated,
        'activation_rate': activated / total,
    }


def evaluate(args):
    reproducibility.set_seed()
    checkpoint = Path(args.checkpoint or f'models/{args.dataset}_all.pth')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    
    # Entrambi i modelli usano il test originale , mai un test filtrato.
    folder = Path('log_history') / args.dataset
    def load(kind):
        return read_pickle(folder / f'{args.dataset}_{kind}_all.pkl')

    texts = load('test')
    labels = load('suffix_test')
    prefixes = load('prefix_activities_test')
    #per valutare la compliance è necessario saperen la lista di attività del prefisso e non solo il contenuto testuale
    lengths = load('len_test')
    id2label = load('id2label')['activity']
    label2id = load('label2id')['activity']
    if len(prefixes) != len(texts) or len(lengths) != len(texts):
        raise ValueError('Prefixes and texts not aligned')
    if any(len(p) != n for p, n in zip(prefixes, lengths)):
        raise ValueError('Prrefixes lenght not coherent')
    if any(len(values) != len(texts) for values in labels.values()):
        raise ValueError('Labels and texts not aligned.')
  

    evaluator = EVALUATORS[args.template]
    activation, target = normalize(args.activation), normalize(args.target)
    for activity in (activation, target):
        if activity == 'ENDactivity' or activity not in label2id:
            raise ValueError(f'Attività non valida: {activity}')
    end_id = label2id['ENDactivity']
    limit = min(args.limit or len(texts), len(texts))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, truncation_side='left')
    dataset = CustomDataset(texts[:limit], {p: v[:limit] for p, v in labels.items()}, tokenizer, 512)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    bert = AutoModel.from_pretrained(BASE_MODEL)
    model = BertMultiOutputClassificationHeads(bert, [len(id2label)] * len(labels))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()

    run = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = Path('outputs') / f'eval_{args.dataset}_{run}'
    output_dir.mkdir(parents=True, exist_ok=False)
    predicted_counts, actual_counts = Counter(), Counter()
    dl_scores = []
    with (output_dir / 'examples.jsonl').open('w', encoding='utf-8') as file, torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc='Evaluation')):
            outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            raw_pred = [head.argmax(dim=1).item() for head in outputs] #prendi il suffisso più probabile
            pred_ids = trim_suffix(raw_pred, end_id)
            true_ids = trim_suffix([batch['labels'][p].item() for p in range(len(outputs))], end_id)
            predicted_suffix = [id2label[i] for i in pred_ids]
            true_suffix = [id2label[i] for i in true_ids]
            prefix = prefixes[index]

            # Il vincolo si applica alla sequenza COMPLETA, senza END o padding.
            predicted_state = evaluator(prefix + predicted_suffix, activation, target)
            actual_state = evaluator(prefix + true_suffix, activation, target)
            predicted_counts[predicted_state] += 1
            actual_counts[actual_state] += 1

            
            #pred_string = ' '.join(map(str, pred_ids))
            #true_string = ' '.join(map(str, true_ids))
            pred_string = "".join(chr(activity_id) for activity_id in pred_ids)
            true_string = "".join(chr(activity_id) for activity_id in true_ids)
            denominator = max(len(pred_string), len(true_string))
            score = 1 - damerau_levenshtein_distance(pred_string, true_string) / denominator if denominator else 1.0
            dl_scores.append(score)
            record = {
                'test_index': index,
                'prefix_text': texts[index], 
                'prefix_activities': prefix,
                'predicted_suffix': predicted_suffix, 
                'true_suffix': true_suffix,
                'predicted_end': end_id in raw_pred,
                'exact_match': predicted_suffix == true_suffix,
                'constraint_predicted': predicted_state, 
                'constraint_real': actual_state,
                'dl_score_original': score,
            }
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
            if index < 5:
                tqdm.write(f'{index}: previsto={predicted_state}, reale={actual_state}, DL={score:.3f}')

    summary = {
        'dataset': args.dataset, 
        'checkpoint': str(checkpoint), 
        'examples': limit,
        'constraint': {'template': args.template, 'activation': activation, 'target': target},
        'dl_score_original': float(np.mean(dl_scores)),
        'predicted': constraint_metrics(predicted_counts),
        'real': constraint_metrics(actual_counts),
    }
    (output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('Risultati:', output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--checkpoint')
    parser.add_argument('--template', required=True, choices=list(EVALUATORS))
    parser.add_argument('--activation', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--limit', type=int)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
