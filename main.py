import pickle
from torch.utils.data import DataLoader
from neural_network.HistoryDataset import CustomDataset
from transformers import AutoModel, AutoTokenizer
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from sklearn.model_selection import train_test_split
from preprocessing.log_to_history import Log
from utility import reproducibility
from pathlib import Path
from datetime import datetime
import argparse

import torch
import random
import numpy as np
import sys


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)




def train_fn(model, train_loader, optimizer, device, criterion):
    model.train() #metti il modello in modalità training 
    total_loss = 0
    for X_train_batch in train_loader:
        input_ids = X_train_batch['input_ids'].to(device) #[8*512 interi che rappresnetano token]
        attention_mask = X_train_batch['attention_mask'].to(device)
        optimizer.zero_grad() #prima di aggiornare i gradienti del bathc corrente vengono aggiornati quelli del batch precedente
        output = model(input_ids, attention_mask) #avviene la fase di forward per quel batch
        # l'output dovrebbe essere della forma [B * MAX_LEN * NUM_ACTIVITIES]
        loss = 0 
        for o, c, l in zip(output, criterion, X_train_batch['labels']):
            loss += criterion[c](o.to(device), X_train_batch['labels'][l].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids, attention_mask)
            for o, c, l in zip(output, criterion, batch['labels']):
                total_loss += criterion[c](o.to(device), batch['labels'][l].to(device))
            total_loss = total_loss.item()
    return total_loss / len(data_loader)


def train_llm(
    model, train_data_loader, valid_data_loader,
    optimizer, EPOCHS, criterion, output_dir
):
    best_valid_loss = float("inf")
    early_stop_counter = 0
    patience = 5

    # Modello senza il wrapper: checkpoint compatibili con eval e steering.
    model_to_save = (
        model.module
        if isinstance(model, torch.nn.DataParallel)
        else model
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_fn( model, train_data_loader, optimizer, device, criterion)

        # Salva subito dopo il training, prima della validation.
        torch.save( model_to_save.state_dict(), output_dir / f"epoch_{epoch:03d}.pth",
        )

        valid_loss = evaluate_fn( model, valid_data_loader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stop_counter = 0

            torch.save( model_to_save.state_dict(), output_dir / "best.pth" )
        else:
            early_stop_counter += 1


        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {valid_loss:.4f}",
            flush=True,
        )

        if early_stop_counter >= patience:
            print("Early stopping.", flush=True)
            break

    # Ripristina effettivamente l'epoca migliore.
    model_to_save.load_state_dict(
        torch.load(
            output_dir / "best.pth",
            map_location=device,
            weights_only=True,
        )
    )

    return model_to_save




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--seed", type=int, default=reproducibility.SEED)
    args = parser.parse_args()

    reproducibility.SEED = args.seed
    reproducibility.set_seed()

    csv_log = args.dataset
    print("Seed:", reproducibility.SEED)

    MAX_LEN = 512  
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    EPOCHS = 50
    TYPE = 'all'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device-->', device)

    Log(csv_log, TYPE)  #effettuare il pre-processing

    with open('log_history/'+csv_log+'/'+csv_log+'_id2label_'+TYPE+'.pkl', 'rb') as f:
        id2label = pickle.load(f)
        #dizionario che ha per chiave il nome degli attributi e per valore un dizionario ch associa ad ogni id uno dei valore che assume quell'attributo

    with open('log_history/'+csv_log+'/'+csv_log+'_label2id_'+TYPE+'.pkl', 'rb') as f:
        label2id = pickle.load(f)

    with open('log_history/'+csv_log+'/'+csv_log+'_train_'+TYPE+'.pkl', 'rb') as f:
        train = pickle.load(f)
        #lista che contiene tutti i prefissi di training trasformati in formato testaule

    with open('log_history/'+csv_log+'/'+csv_log+'_label_train_'+TYPE+'.pkl', 'rb') as f:
        y_train = pickle.load(f)
        #contiene una lista con tutte le next activity (una per ogni prefisso)
        #attenzione non contiene il suffisso per intero
        #le activity sono sottoforma di id

    with open('log_history/'+csv_log+'/'+csv_log+'_suffix_train_'+TYPE+'.pkl', 'rb') as f:
        y_train_suffix = pickle.load(f)
        #dizionario che contiene per ogni posizione futura la lista di activity che assumerà in corrispondenza dei diversi prefissi



    train_input, val_input = train_test_split(train, test_size=0.2, random_state=reproducibility.SEED)
    train_label = {} 
    val_label = {}
    # equivalenti y_train_suffix ma splitate per training e validation

    for key in y_train_suffix.keys():
        train_label[key], val_label[key] = train_test_split(y_train_suffix[key], test_size=0.2, random_state=reproducibility.SEED)


    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-medium', truncation_side='left')
    # se devo troncare qualcosa, tronco la parte di sinistra (predilifo gli eventi più recenti)
    model = AutoModel.from_pretrained('prajjwal1/bert-medium')

    output_sizes = []

    for i in range(len(y_train_suffix)):   #max length
        output_sizes.append(len(id2label['activity']))
        #alla fine ho una lista di max_length volte il numero di etichette da predire. 

    train_dataset = CustomDataset(train_input, train_label, tokenizer, MAX_LEN)
    val_dataset = CustomDataset(val_input, val_label, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('TRAINING START...')
    # Initialize model
    model = BertMultiOutputClassificationHeads(model, output_sizes).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    print("GPU utilizzate:", torch.cuda.device_count())
    criterion = {}

    for l in y_train_suffix:
        criterion[l] = torch.nn.CrossEntropyLoss()
        #crea tanyte istanze di cross Entropy Loss (una per ogni posizione futura)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    #Il fine tuning avviene su tutti i parametri del modello, incluse le teste di classificazione

    
    output_dir = Path("models") / csv_log / f"seed_{reproducibility.SEED}"
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Checkpoint salvati in:", output_dir, flush=True)


    bert_model = train_llm( model, train_loader, val_loader, optimizer, EPOCHS, criterion, output_dir )
