"""Valuta theta(alpha) = theta_base + alpha * (theta_constrained - theta_base).
Riusa evaluate() di eval_model.py senza modificare le metriche o il test.
"""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import tempfile

import torch
from constraints.extraction import EVALUATORS
from eval_model import evaluate


def check_weights(base, constrained):
    if base.keys() != constrained.keys():
        raise ValueError('I checkpoint non hanno gli stessi parametri.')
    for name in base:
        a, b = base[name], constrained[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f'Shape o dtype diversi per {name}.')
        if a.is_floating_point():
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f'Pesi non finiti in {name}.')
        elif not torch.equal(a, b):
            # Buffer interi/bool non sono una direzione continua da interpolare.
            raise ValueError(f'Buffer non floating diverso: {name}.')


def interpolate(base, constrained, alpha):
    if not math.isfinite(alpha):
        raise ValueError('Alpha deve essere finito.')
    # Gli estremi usano esattamente i checkpoint originali.
    if alpha == 0:
        return base
    if alpha == 1:
        return constrained

    weights = {}
    for name, base_weight in base.items():
        if base_weight.is_floating_point():
            # Calcolo almeno in float32 per checkpoint eventualmente in half precision.
            dtype = torch.float64 if base_weight.dtype == torch.float64 else torch.float32
            start = base_weight.to(dtype)
            direction = constrained[name].to(dtype) - start
            weights[name] = (start + alpha * direction).to(base_weight.dtype)
            if not torch.isfinite(weights[name]).all():
                raise ValueError(f'Interpolazione non finita: {name}, alpha={alpha}.')
        else:
            weights[name] = base_weight
    return weights


def sweep(args):
    if not all(math.isfinite(alpha) for alpha in args.alphas):
        raise ValueError('Tutti gli alpha devono essere finiti.')
    base = torch.load(args.base_checkpoint, map_location='cpu', weights_only=True)
    constrained = torch.load(args.constrained_checkpoint, map_location='cpu', weights_only=True)
    check_weights(base, constrained)

    run = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = Path('outputs') / f'steering_{args.dataset}_{run}'
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))
    results = []
    summaries = []

    # Un solo checkpoint temporaneo, riscritto per ciascun alpha.
    # I due checkpoint di partenza non vengono mai modificati.
    with tempfile.TemporaryDirectory(prefix='lupin-steering-') as directory:
        temporary_checkpoint = Path(directory) / 'interpolated.pth'
        for alpha in args.alphas:
            print(f'\n===== ALPHA = {alpha} =====', flush=True)
            weights = interpolate(base, constrained, alpha)
            torch.save(weights, temporary_checkpoint)
            del weights

            summary = evaluate(argparse.Namespace(
                dataset=args.dataset,
                checkpoint=str(temporary_checkpoint),
                template=args.template,
                activation=args.activation,
                target=args.target,
                limit=args.limit,
            ))
            # L'evaluation restituisce il percorso temporaneo: registra invece
            # qui i checkpoint reali e alpha per identificare il modello valutato.
            summary.pop('checkpoint', None)
            summary['alpha'] = alpha
            summary['base_checkpoint'] = args.base_checkpoint
            summary['constrained_checkpoint'] = args.constrained_checkpoint
            summaries.append(summary)
            metrics = summary['predicted']
            row = {
                'alpha': alpha,
                'examples': summary['examples'],
                'dl_score_original': summary['dl_score_original'],
                'satisfaction_rate': metrics['satisfaction_rate'],
                'non_vacuous_support': metrics['non_vacuous_support'],
                'activation_rate': metrics['activation_rate'],
                'fulfilled': metrics['fulfilled'],
                'violated': metrics['violated'],
                'vacuous': metrics['vacuous'],
            }
            results.append(row)
            with (output_dir / 'results.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(results)
            (output_dir / 'metrics.json').write_text(json.dumps(summaries, indent=2))
            # Libera la cache inutilizzata dopo ogni chiamata all'evaluator.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print('\nRisultati dello sweep:', output_dir / 'results.csv')
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--constrained-checkpoint', required=True)
    parser.add_argument('--template', required=True, choices=list(EVALUATORS))
    parser.add_argument('--activation', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--alphas', nargs='+', type=float, default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument('--limit', type=int)
    sweep(parser.parse_args())


if __name__ == '__main__':
    main()
