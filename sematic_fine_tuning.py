"""Fine-tuning LUPIN sul training originale con loss soft Response e cross-entropy opzionale."""
import argparse
import json
import math
import pickle
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from constrained_fine_tuning import freeze_model
from semantic_loss.response_loss import ResponseSoftLoss
from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from preprocessing.log_to_history import Log
from utility import reproducibility

BASE_MODEL = "prajjwal1/bert-medium"
MAX_LENGTH = 512
VALIDATION_SIZE = 0.2


def read_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


class SemanticDataset(CustomDataset):
    """Riusa la tokenizzazione originale e aggiunge le attività reali del prefisso."""
    def __init__(self, texts, prefixes, labels, tokenizer):
        super().__init__(texts, labels, tokenizer, MAX_LENGTH)
        self.prefixes = prefixes
        self.prefix_width = max(map(len, prefixes))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        prefix = self.prefixes[index]
        item["prefix_ids"] = torch.tensor(
            prefix + [-1] * (self.prefix_width - len(prefix)), dtype=torch.long
        )
        item["prefix_length"] = len(prefix)
        return item


def compute_losses(model, batch, criterion, device, dual_objective, semantic_weight):
    outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    logits = torch.stack(outputs, dim=1)
    semantic = criterion(
        logits, batch["prefix_ids"].to(device), batch["prefix_length"].to(device)
    )
    # Come nel training originale: somma delle CE medie delle singole teste.
    ce = sum(
        torch.nn.functional.cross_entropy(output, batch["labels"][position].to(device))
        for position, output in enumerate(outputs)
    )
    total = ce + semantic_weight * semantic if dual_objective else semantic
    return {"total": total, "ce": ce, "semantic": semantic}


def train_fn(model, train_loader, optimizer, device, criterion, mode, top_k,
             dual_objective, semantic_weight):
    model.train()
    # Niente dropout nei blocchi congelati.
    if mode != "full_short":
        model.gpt_model.eval()
        if mode == "top_k":
            for layer in model.gpt_model.encoder.layer[-top_k:]:
                layer.train()
            model.gpt_model.pooler.train()
    totals = {"total": 0.0, "ce": 0.0, "semantic": 0.0}
    total_examples = 0
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        optimizer.zero_grad()
        losses = compute_losses(model, batch, criterion, device, dual_objective, semantic_weight)
        if any(not torch.isfinite(value) for value in losses.values()):
            raise RuntimeError("Loss di training non finita.")
        losses["total"].backward()
        optimizer.step()
        size = batch["input_ids"].size(0)
        for name, value in losses.items():
            totals[name] += value.item() * size
        total_examples += size
    return {name: value / total_examples for name, value in totals.items()}


def evaluate_fn(model, data_loader, criterion, device, dual_objective, semantic_weight):
    model.eval()
    totals = {"total": 0.0, "ce": 0.0, "semantic": 0.0}
    total_examples = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation", unit="batch"):
            losses = compute_losses(model, batch, criterion, device, dual_objective, semantic_weight)
            if any(not torch.isfinite(value) for value in losses.values()):
                raise RuntimeError("Loss di validation non finita.")
            size = batch["input_ids"].size(0)
            for name, value in losses.items():
                totals[name] += value.item() * size
            total_examples += size
    return {name: value / total_examples for name, value in totals.items()}


def train_llm(model, train_loader, val_loader, optimizer, epochs,
              criterion, device, mode, top_k, output_dir, dual_objective, semantic_weight):
    best_loss = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train = train_fn(model, train_loader, optimizer, device, criterion, mode, top_k,
                         dual_objective, semantic_weight)
        torch.save(model.state_dict(), output_dir / f"epoch_{epoch:03d}.pth")
        val = evaluate_fn(model, val_loader, criterion, device, dual_objective, semantic_weight)
        if val["total"] < best_loss:
            best_loss = val["total"]
            torch.save(model.state_dict(), output_dir / "best.pth")
        row = {"epoch": epoch}
        for split, losses in (("train", train), ("val", val)):
            for name, value in losses.items():
                row[f"{split}_{name}_loss"] = value
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoca {epoch}/{epochs}", flush=True)
        for split, losses in (("Training", train), ("Validation", val)):
            print(f"  {split}: totale={losses['total']:.8g} | "
                  f"CE={losses['ce']:.8g} | semantica={losses['semantic']:.8g}", flush=True)
    model.load_state_dict(torch.load(output_dir / "best.pth", map_location=device, weights_only=True))
    return model


def normalize(activity):
    for char in (" ", "+", "-", "_"):
        activity = activity.replace(char, "")
    return activity


def fine_tune(args):
    if not math.isfinite(args.semantic_weight) or args.semantic_weight < 0:
        raise ValueError("--lambda deve essere finito e non negativo.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs e batch-size devono essere positivi.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate deve essere positivo e finito.")
    if args.mode == "top_k" and (args.top_k is None or args.top_k < 1):
        raise ValueError("Con top_k specifica --top-k positivo.")
    if args.mode != "top_k" and args.top_k is not None:
        raise ValueError("--top-k si usa soltanto con --mode top_k.")
    checkpoint = Path(args.checkpoint or f"models/{args.dataset}_all.pth")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    reproducibility.SEED = args.seed
    reproducibility.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nessun constraint passato a Log: tutti i casi del training temporale originale.
    Log(args.dataset, "all")
    folder = Path("log_history") / args.dataset
    texts = read_pickle(folder / f"{args.dataset}_train_all.pkl")
    prefixes = read_pickle(folder / f"{args.dataset}_prefix_activities_train_all.pkl")
    suffixes = read_pickle(folder / f"{args.dataset}_suffix_train_all.pkl")
    id2label = read_pickle(folder / f"{args.dataset}_id2label_all.pkl")
    label2id = read_pickle(folder / f"{args.dataset}_label2id_all.pkl")["activity"]
    if len(texts) < 2 or len(texts) != len(prefixes):
        raise ValueError("Servono almeno due esempi con testi e prefissi allineati.")
    if not suffixes or any(len(v) != len(texts) for v in suffixes.values()):
        raise ValueError("Suffissi e testi non allineati.")
    activation, target = normalize(args.activation), normalize(args.target)
    if any(a not in label2id or a == "ENDactivity" for a in (activation, target)):
        raise ValueError("Attività del vincolo non valide.")
    criterion = ResponseSoftLoss(label2id[activation], label2id[target], label2id["ENDactivity"])
    prefixes = [[label2id[a] for a in prefix] for prefix in prefixes]
    train_indices, val_indices = train_test_split(
        list(range(len(texts))), test_size=VALIDATION_SIZE, random_state=args.seed
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, truncation_side="left")
    def make_loader(indices):
        labels = {position: [values[i] for i in indices]
                  for position, values in suffixes.items()}
        dataset = SemanticDataset([texts[i] for i in indices],
                                  [prefixes[i] for i in indices], labels, tokenizer)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    train_loader, val_loader = make_loader(train_indices), make_loader(val_indices)

    bert = AutoModel.from_pretrained(BASE_MODEL)
    model = BertMultiOutputClassificationHeads(bert, [len(id2label["activity"])] * len(suffixes))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    freeze_model(model, args.mode, args.top_k)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.01
    )
    output_dir = Path("models") / args.dataset / "semantic_response" / f"{args.mode}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), checkpoint=str(checkpoint), loss="ce_plus_response_soft" if args.dual_objective else "response_soft_only",
                  base_model=BASE_MODEL, max_length=MAX_LENGTH,
                  validation_size=VALIDATION_SIZE, weight_decay=0.01,
                  best_selection="validation_total_loss")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    with (output_dir / "id2label.pkl").open("wb") as file:
        pickle.dump(id2label, file)
    print("Device:", device)
    print("Seed:", args.seed)
    print("Obiettivo:", f"CE + {args.semantic_weight} * Response"
          if args.dual_objective else "Solo Response (CE monitorata, lambda ignorato)")
    print("Prefissi training/validation:", len(train_indices), len(val_indices))
    print("Checkpoint in:", output_dir)
    train_llm(model, train_loader, val_loader, optimizer, args.epochs,
              criterion, device, args.mode, args.top_k, output_dir,
              args.dual_objective, args.semantic_weight)
    return output_dir / "best.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--activation", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--mode", choices=["heads_only", "full_short", "top_k"], default="heads_only")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dual-objective", action="store_true",
                        help="Ottimizza CE + lambda * Response; default: sola Response.")
    parser.add_argument("--lambda", dest="semantic_weight", type=float, default=1.0,
                        help="Peso della loss semantica nel doppio obiettivo (default: 1).")
    parser.add_argument("--seed", type=int, default=reproducibility.SEED)
    print("Checkpoint migliore:", fine_tune(parser.parse_args()))


if __name__ == "__main__":
    main()
