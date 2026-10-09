"""Confronta due direzioni di fine-tuning rispetto allo stesso checkpoint base."""
import argparse
import math


import torch


def new_stats():
    return {"dot": 0.0, "squared_a": 0.0, "squared_b": 0.0}
#inizializza il calcolo della similarità del coseno, tenendo conto del prodotto scalare a numeratore, 
# e dei quadrati delle norme a denominatore. 


def add_stats(stats, a, b):
    stats["dot"] += torch.dot(a, b).item()
    stats["squared_a"] += torch.dot(a, a).item()
    stats["squared_b"] += torch.dot(b, b).item()
    # somma al prodotto scalare parziale il prodotto scalare tra una nuova coppia di tensori
    # il prodotto scalare di un vettore per sè stesso corrisponde alla norma al quadrato. 
    # item() serve a concertore da tensore di un solo numero a cifra python



def summarize(stats):
    norm_a = math.sqrt(stats["squared_a"])
    norm_b = math.sqrt(stats["squared_b"])
    # Una direzione nulla non ha un angolo definito.
    cosine = None
    if norm_a > 0 and norm_b > 0:
        cosine = max(-1.0, min(1.0, (stats["dot"] / norm_a) / norm_b))
    return {"cosine": cosine, "norm_a": norm_a, "norm_b": norm_b}


def compare_directions(base, checkpoint_a, checkpoint_b):
    if base.keys() != checkpoint_a.keys() or base.keys() != checkpoint_b.keys():
        raise ValueError("Checkpoints must contain the same tensors")

    global_stats = new_stats()

    for name, original in base.items():
        a = checkpoint_a[name]
        b = checkpoint_b[name]

        for value in (a, b):
            if value.shape != original.shape or value.dtype != original.dtype:
                raise ValueError(f"Different Shape or dtype: {name}")

        if not original.is_floating_point():
            if not torch.equal(original, a) or not torch.equal(original, b):
                raise ValueError(f"Not-floating tensor is modified: {name}")
            continue

        for value in (original, a, b):
            if not torch.isfinite(value).all():
                raise ValueError(f"Not finite values: {name}")

        # Salta il tensore solo se non è cambiato in nessuno dei due modelli.
        if torch.equal(original, a) and torch.equal(original, b):
            continue

        direction_a = (a.double() - original.double()).reshape(-1)
        direction_b = (b.double() - original.double()).reshape(-1)

        tensor_stats = new_stats()
        add_stats(tensor_stats, direction_a, direction_b)
        print_result(name, summarize(tensor_stats))

        add_stats(global_stats, direction_a, direction_b)

    print_result("Globale", summarize(global_stats))


def print_result(name, result):
    cosine = result["cosine"]
    text = "non definito (direzione nulla)" if cosine is None else f"{cosine:.6f}"
    print(
        f"{name}: coseno={text} | "
        f"norma A={result['norm_a']:.6g} | norma B={result['norm_b']:.6g}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    args = parser.parse_args()

    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    a = torch.load(args.checkpoint_a, map_location="cpu", weights_only=True)
    b = torch.load(args.checkpoint_b, map_location="cpu", weights_only=True)
    compare_directions(base, a, b)


if __name__ == "__main__":
    main()

