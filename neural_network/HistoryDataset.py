import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts  #prefissi prodotti dal log
        self.labels = {}
        for v in labels:
            self.labels[v] = labels[v] #sta facendo la copia del dizionario

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts) #numero di esempi del dataset

    def __getitem__(self, idx):    #
        text = str(self.texts[idx])
        label = {}
        for v in self.labels:
            label[v] = self.labels[v][idx]
            #prende tutte le label per quel prefisso

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  #Aggiunge [CLS] [SEP]
            max_length=self.max_len,  #512
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',    #deve restituire tensori Pytorch
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(), # [1 * 512] -> [512]
            'labels': label
        }
    #Questo codice non fa altro ch ricostruire come si prende un item. Quello del prefisso é bello e pronto
    #Per quello del suffisso bisogna iterare su titte le posizione future e prendere l'elemento in posizione idx.
    