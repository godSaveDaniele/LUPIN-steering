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
        raise ValueError('Checkpoints must be the same parameters')
    for name in base:
        a, b = base[name], constrained[name]
        #itera su tutti i tensori, ad esempio potrebbero essere due matrici dei pesi della testa
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f'Shape or dtype different for {name}.')
        if a.is_floating_point():
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f'Weights are not finite in {name}.')
        elif not torch.equal(a, b):
            #il checkpoint è costituito anche da matrici non floating point,
            # booleani/interi che quindi non costituiscono una direzine da sommare
            raise ValueError(f'Buffer non floating diverso: {name}.')

def print_nonzero_directions(base, constrained):
    print("\nTensors modified trough fine tuning:")

    for name, base_weight in base.items():
        if not base_weight.is_floating_point():
            continue

        direction = (
            constrained[name].double() - base_weight.double()
        )
        nonzero = torch.count_nonzero(direction).item()

        if nonzero > 0:
            print(
                f"{name} | "
                f"shape={tuple(direction.shape)} | "
                f"not null components={nonzero}/{direction.numel()} | "
                f"norm={direction.norm().item():.6g}"
            )


def add_steering_vector(base, constrained, alpha):

    # Gli estremi usano esattamente i checkpoint originali.
    if alpha == 0:
        return base
    if alpha == 1:
        return constrained

    weights = {}
    for name, base_weight in base.items():
        if base_weight.is_floating_point():
            
            dtype = torch.float64 if base_weight.dtype == torch.float64 else torch.float32
            start = base_weight.to(dtype)
            direction = constrained[name].to(dtype) - start
            weights[name] = (start + alpha * direction).to(base_weight.dtype)
            if not torch.isfinite(weights[name]).all():
                raise ValueError(f'Not finite value in: {name}, alpha={alpha}.')
        else:
            weights[name] = base_weight
    return weights


def evaluate_alphas(args):
    if not all(math.isfinite(alpha) for alpha in args.alphas):
        raise ValueError('Alpha values must be finite.')
    
    base = torch.load(args.base_checkpoint, map_location='cpu', weights_only=True)
    constrained = torch.load(args.constrained_checkpoint, map_location='cpu', weights_only=True)
    check_weights(base, constrained)
    print_nonzero_directions(base, constrained)
    #sono due state_dict 

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

            #calcolo dei pesi temporaneo
            weights = add_steering_vector(base, constrained, alpha)
            torch.save(weights, temporary_checkpoint)
            del weights

            #valutazione del modello steered
            summary = evaluate(argparse.Namespace(
                dataset=args.dataset,
                checkpoint=str(temporary_checkpoint),
                template=args.template,
                activation=args.activation,
                target=args.target,
                limit=args.limit,
            ))

            
            summary.pop('checkpoint', None)
            summary['alpha'] = alpha
            summary['base_checkpoint'] = args.base_checkpoint
            summary['constrained_checkpoint'] = args.constrained_checkpoint
            #modifico le info, aggiungendo base_checkpoint e constrained_checkpoint

            summaries.append(summary)
            metrics = summary['predicted']
            row = {
                'alpha': alpha,
                'examples': summary['examples'],
                'dl_score_original': summary['dl_score_original'],
                'support': metrics['support'],
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True) 
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--constrained-checkpoint', required=True)
    parser.add_argument('--template', required=True, choices=list(EVALUATORS))
    parser.add_argument('--activation', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--alphas', nargs='+', type=float, default=[0, 0.1, 0.2, 0.3, 0.75, 1])
    parser.add_argument('--limit', type=int)
    evaluate_alphas(parser.parse_args())


if __name__ == '__main__':
    main()
