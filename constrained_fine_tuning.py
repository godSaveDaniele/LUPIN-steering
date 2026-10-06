import argparse
import json
import math
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from constraints.extraction import EVALUATORS
from preprocessing.log_to_history import Log
from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads

# Valori fissi, come nel training originale.
BASE_MODEL = "prajjwal1/bert-medium"
MAX_LENGTH = 512
VALIDATION_SIZE = 0.2
SEED = 42


def read_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def freeze_model(model, mode, top_k):
    # Tutto congelato, tranne in full_short.
    for parameter in model.parameters():
        parameter.requires_grad = (mode == "full_short")

    # Le teste sono sempre aggiornabili.
    for parameter in model.output_layers.parameters():
        parameter.requires_grad = True

    if mode == "top_k":
        layers = model.gpt_model.encoder.layer
        if not 1 <= top_k <= len(layers):
            raise ValueError(f"top-k deve essere tra 1 e {len(layers)}")
        for layer in layers[-top_k:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in model.gpt_model.pooler.parameters():
            parameter.requires_grad = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_fn(model, train_loader, optimizer, device, criterion, mode, top_k):
    model.train()

    # Disabilita il dropout soltanto nelle parti congelate di BERT.
    if mode != "full_short":
        model.gpt_model.eval()
        if mode == "top_k":
            for layer in model.gpt_model.encoder.layer[-top_k:]:
                layer.train()
            model.gpt_model.pooler.train()

    total_loss = 0.0
    total_examples = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        loss = 0
        for position, output in enumerate(outputs):
            target = batch["labels"][position].to(device)
            loss = loss + criterion[position](output, target)

        if not torch.isfinite(loss):
            raise RuntimeError("Loss di training non finita.")
        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)

            loss = 0
            for position, output in enumerate(outputs):
                target = batch["labels"][position].to(device)
                loss = loss + criterion[position](output, target)

            if not torch.isfinite(loss):
                raise RuntimeError("Loss di validation non finita.")
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / total_examples


def train_llm(model, train_loader, val_loader, optimizer, epochs,
              criterion, device, mode, top_k, output_dir):
    best_valid_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_fn(
            model, train_loader, optimizer, device, criterion, mode, top_k
        )
        valid_loss = evaluate_fn(model, val_loader, criterion, device)

        torch.save(model.state_dict(), output_dir / f"epoch_{epoch:03d}.pth")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), output_dir / "best.pth")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": valid_loss})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"Epoca {epoch}/{epochs}: train={train_loss:.4f}, val={valid_loss:.4f}",
            flush=True,
        )

    # Restituisce il modello con i pesi della migliore epoca, come inteso in main.py.
    model.load_state_dict(
        torch.load(output_dir / "best.pth", map_location=device, weights_only=True)
    )
    return model


def fine_tune(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs e batch-size devono essere positivi.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate deve essere positivo e finito.")
    if args.mode == "top_k" and (args.top_k is None or args.top_k < 1):
        raise ValueError("Con top_k specifica --top-k con un valore positivo.")
    if args.mode != "top_k" and args.top_k is not None:
        raise ValueError("--top-k si usa soltanto con --mode top_k.")

    checkpoint = Path(args.checkpoint or f"models/{args.dataset}_all.pth")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Genera i dati filtrati usando la tua classe Log.
    constraint = {
        "template": args.template,
        "activation": args.activation,
        "target": args.target,
    }
    Log(args.dataset, "all", constraint=constraint)

    #lettura dei file di pre-processing
    folder = Path("log_history") / args.dataset / "constrained"
    train = read_pickle(folder / f"{args.dataset}_train_all.pkl")
    y_train_suffix = read_pickle(folder / f"{args.dataset}_suffix_train_all.pkl")
    id2label = read_pickle(folder / f"{args.dataset}_id2label_all.pkl")

    # Un controllo piccolo ma utile: stessi ID delle attività del modello iniziale.
    original_mapping = folder.parent / f"{args.dataset}_id2label_all.pkl"
    if original_mapping.exists() and read_pickle(original_mapping) != id2label:
        raise ValueError("Il mapping delle attività differisce da quello originale.")


 
    train_input, val_input = train_test_split(
        train, test_size=VALIDATION_SIZE, random_state=SEED
    )

    train_label = {}
    val_label = {}
    for position in y_train_suffix:
        train_label[position], val_label[position] = train_test_split(
            y_train_suffix[position], test_size=VALIDATION_SIZE, random_state=SEED
        ) #genera esattamente la stessa suddivisione

    print(train_input[0])
    for position in train_label:
        print(train_label[position][0])
        #controllo da cancellare

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, truncation_side="left")
    train_dataset = CustomDataset(train_input, train_label, tokenizer, MAX_LENGTH)
    val_dataset = CustomDataset(val_input, val_label, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

  
    bert = AutoModel.from_pretrained(BASE_MODEL)
    output_sizes = [len(id2label["activity"])] * len(y_train_suffix)
    model = BertMultiOutputClassificationHeads(bert, output_sizes)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    freeze_model(model, args.mode, args.top_k)
    model.to(device)
    
    print('TRAINING START...')

    criterion = {}
    for position in y_train_suffix:
        criterion[position] = torch.nn.CrossEntropyLoss()

    parameters = [p for p in model.parameters() if p.requires_grad]
    # 0.01 è il default di AdamW, usato anche dal main originale.
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)

    # Ogni esecuzione ha una cartella automatica distinta.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("models") / args.dataset / "constrained" / f"{args.mode}_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), checkpoint=str(checkpoint), seed=SEED,
                  max_length=MAX_LENGTH, validation_size=VALIDATION_SIZE,
                  base_model=BASE_MODEL, weight_decay=0.01)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    with (output_dir / "id2label.pkl").open("wb") as file:
        pickle.dump(id2label, file)

    print("Device:", device)
    print("Prefissi training/validation:", len(train_input), len(val_input))
    print("Checkpoint in:", output_dir)

    # Il ciclo delle epoche è gestito da train_llm
    model = train_llm(
        model, train_loader, val_loader, optimizer, args.epochs,
        criterion, device, args.mode, args.top_k, output_dir,
    )
    return output_dir / "best.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--template", required=True, choices=list(EVALUATORS))
    parser.add_argument("--activation", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint", help="Default: models/<dataset>_all.pth")
    parser.add_argument("--mode", choices=["heads_only", "full_short", "top_k"], default="heads_only")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    best = fine_tune(parser.parse_args())
    print("Checkpoint migliore:", best)


if __name__ == "__main__":
    main()
