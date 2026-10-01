import sys
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from neural_network.HistoryDataset import CustomDataset
from neural_network.llamp_multiout import BertMultiOutputClassificationHeads
from jellyfish._jellyfish import damerau_levenshtein_distance
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm

def clean_sequence(sequence_str):
    sequence_list = sequence_str.split()
    end_id = str(label2id["activity"]["ENDactivity"])

    if end_id in sequence_list:
        sequence_list = sequence_list[:sequence_list.index(end_id)]

    return " ".join(sequence_list)
        #rimuove tutti gli id corrispondenti ad end activity

def remove_word(sentence, word):
    words = sentence.split()
    words = [w for w in words if w != word]
    new_sentence = ' '.join(words)
    return new_sentence
    #rimuove l'ultimo ENDactivity

def decode_suffix(activity_ids, id2label):
    activities = []

    for activity_id in activity_ids:
        activity = id2label["activity"][int(activity_id)]

        if activity == "ENDactivity":
            break

        activities.append(activity)

    return activities

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device-->', device)
    csv_log = sys.argv[1]
    TYPE = 'all'
    with open('log_history/'+csv_log+'/'+csv_log+'_test_'+TYPE+'.pkl', 'rb') as f:
        test = pickle.load(f)

    with open('log_history/'+csv_log+'/'+csv_log+'_label_test_'+TYPE+'.pkl', 'rb') as f:
        y_test = pickle.load(f)

    with open('log_history/' + csv_log + '/' + csv_log + '_id2label_'+TYPE+'.pkl', 'rb') as f:
        id2label = pickle.load(f)

    with open('log_history/' + csv_log + '/' + csv_log + '_label2id_'+TYPE+'.pkl', 'rb') as f:
        label2id = pickle.load(f)

    with open('log_history/'+csv_log+'/'+csv_log+'_suffix_train_'+TYPE+'.pkl', 'rb') as f:
        y_train_suffix = pickle.load(f)

    with open('log_history/'+csv_log+'/'+csv_log+'_suffix_test_'+TYPE+'.pkl', 'rb') as f:
        y_test_suffix = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-medium', truncation_side='left')
    model = AutoModel.from_pretrained('prajjwal1/bert-medium')
    MAX_LEN = 512

    test_dataset = CustomDataset(test, y_test_suffix, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #nota che la batch size è pari ad uno
    output_sizes = []

    dict_pred = {}
    dict_truth = {}
    for i in range(len(y_train_suffix)):
        output_sizes.append(len(id2label['activity']))

    model = BertMultiOutputClassificationHeads(model, output_sizes)

    # Load the state dictionary into the model
    #model.load_state_dict(torch.load('models/'+csv_log+'_'+TYPE+'.pth'))
    #model = model.to(device)
    state_dict = torch.load(
        f"models/{csv_log}_{TYPE}.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    model=model.to(device)

    # Make sure to set the model in evaluation mode if you're not training it further
    model.eval()
    dict_pred = {}
    dict_truth = {}

    """
    list_dl_distance =[] #viene calcolato uno score per ogni prefisso di test
    file_dl = open('suffix_'+csv_log+'_'+TYPE+'.txt','w')
    file_dl.write('pred,truth,dl_score\n')
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) #forward
            output = model(input_ids, attention_mask)
            #lista di max_length tensori [1* num_activities]
            #print(id2label['activity'])
            l_pred = [] #suffisso predetto
            l_true = [] #suffisso vero
            for i in range(len(y_train_suffix)):
                pred = output[i].argmax(dim=1).cpu().numpy()
                #non serve fare softmax perchè è una funziona monotona
                l_pred.append(str(pred[0]))
                l_true.append(str(batch['labels'][i].item()))
            seq_pred = ' '.join(l_pred)
            seq_true = ' '.join(l_true)

            seq_pred = clean_sequence(seq_pred)
            seq_true = clean_sequence(seq_true)
            seq_pred = remove_word(seq_pred, str(label2id['activity']['ENDactivity']))
            seq_true = remove_word(seq_true, str(label2id['activity']['ENDactivity']))
            if seq_pred == '' and seq_true == '':
                seq_pred = 'end'
                seq_true = 'end'
            dl_distance = 1 - (damerau_levenshtein_distance(seq_pred, seq_true) / max(len(seq_pred), len(seq_true)))
            #normalizza la distanza
            #potrebbero esserci dei problemi sul fatto che la distanza viene passata sottoforma di stringa
            
            file_dl.write(seq_pred+','+seq_true+','+str(dl_distance)+'\n')
            list_dl_distance.append(dl_distance)
    print(f"DL--> {np.mean(list_dl_distance):.3f}")
    """
    # Se impostato, valuta soltanto i primi N prefissi.
    # Senza secondo argomento, valuta tutto il test set.
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else len(test_dataset)

    if limit <= 0:
        raise ValueError("Il numero di esempi deve essere positivo.")

    limit = min(limit, len(test_dataset))
    list_dl_distance = []

    # Usa nomi diversi per le prove parziali.
    run_name = f"{csv_log}_{TYPE}"
    if limit < len(test_dataset):
        run_name += f"_first{limit}"

    scores_path = f"outputs/suffix_{run_name}.txt"
    examples_path = f"outputs/examples_{run_name}.jsonl"

    def display_suffix(activities):
        return " → ".join(activities) if activities else "(nessuna attività futura)"

    with (
        open(scores_path, "w", encoding="utf-8") as scores_file,
        open(examples_path, "w", encoding="utf-8") as examples_file,
        torch.no_grad(),
    ):
        scores_file.write("pred,truth,dl_score\n")

        progress = tqdm(total=limit, desc="Evaluation")

        for example_index, batch in enumerate(test_loader):
            if example_index >= limit:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output = model(input_ids, attention_mask)

            pred_ids = [
                head.argmax(dim=1).item()
                for head in output
            ]
            true_ids = [
                batch["labels"][position].item()
                for position in range(len(output))
            ]

            predicted_suffix = decode_suffix(pred_ids, id2label)
            true_suffix = decode_suffix(true_ids, id2label)
            prefix_text = batch["text"][0]

            # Mantiene la metrica originale del repository,
            # calcolata sulle stringhe degli ID.
            seq_pred = clean_sequence(" ".join(map(str, pred_ids)))
            seq_true = clean_sequence(" ".join(map(str, true_ids)))

            if not seq_pred and not seq_true:
                dl_score = 1.0
            else:
                dl_score = 1 - (
                    damerau_levenshtein_distance(seq_pred, seq_true)
                    / max(len(seq_pred), len(seq_true))
                )

            list_dl_distance.append(dl_score)
            scores_file.write(f"{seq_pred},{seq_true},{dl_score}\n")

            record = {
                "test_index": example_index,
                "prefix_text": prefix_text,
                "predicted_suffix": predicted_suffix,
                "true_suffix": true_suffix,
                "predicted_end": (
                    label2id["activity"]["ENDactivity"] in pred_ids
                ),
                "exact_match": predicted_suffix == true_suffix,
                "dl_score_original": dl_score,
            }

            examples_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            # Mostra a terminale i primi cinque esempi.
            if example_index < 5:
                tqdm.write(
                    f"\nESEMPIO {example_index}\n"
                    f"PREFISSO TESTUALE:\n{prefix_text}\n\n"
                    f"SUFFISSO PREVISTO:\n"
                    f"{display_suffix(predicted_suffix)}\n\n"
                    f"SUFFISSO REALE:\n"
                    f"{display_suffix(true_suffix)}\n\n"
                    f"Corrispondenza esatta: {record['exact_match']}\n"
                )

            progress.update(1)

        progress.close()

    print(f"Prefissi valutati: {len(list_dl_distance)}")
    if list_dl_distance:
        print(f"DL originale: {np.mean(list_dl_distance):.3f}")

    print(f"Score salvati in: {scores_path}")
    print(f"Esempi leggibili salvati in: {examples_path}")