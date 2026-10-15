import torch
from torch import nn
from torch.nn import functional as F

"""Loss fuzzy Direct-Soft per Response(A, B)"""

#Questa loss va a penalizzare la generazione di suffissi che rispetto al prefisso
# non rispettano il vincolo. 

# E' specifica per il vincolo Response



class ResponseSoftLoss(nn.Module):
    def __init__(self, activation_id, target_id, end_id, eps=1e-7):
        super().__init__()
        if len({activation_id, target_id, end_id}) != 3:
            raise ValueError("A, B and END must correspond to distinct IDs")
        if not 0 < eps < 1:
            raise ValueError("eps should be between 0 and 1.")
        self.activation_id = activation_id
        self.target_id = target_id
        self.end_id = end_id
        self.eps = eps


    # Restituisce la loss media sul batch.
    def forward(self, suffix_logits, prefix_ids, prefix_lengths):
        # suffix_logits -> tensore numpy [batch_size * num_teste * num_activities]
        # prefix_ids -> matrice che contiene gli id dei prefissi del batch [batch, max_prefix]
        # prefix_lengths -> lista che contiene le lunghezze dei prefissi.

        if not suffix_logits.is_floating_point():
            raise ValueError("Logits must be floating point.")
        logits = suffix_logits if suffix_logits.dtype == torch.float64 else suffix_logits.float()
        probabilities = logits.softmax(dim=-1)
        return self.evaluate_probabilities(probabilities, prefix_ids, prefix_lengths)["loss"]

    

    def evaluate_probabilities(self, probabilities, prefix_ids, prefix_lengths):
        #Valuta probabilità già normalizzate, anche one-hot, senza training.

        # prefix_ids: [batch, max_prefix], -> id delle activities dei prefissi 
        # Il padding fino a max_prefix è fatto da -1
        # Restituisce score, loss per esempio e loss media.
        # Lo score è fuzzy, non una probabilità esatta di compliance.
    
       
        p = probabilities if probabilities.dtype == torch.float64 else probabilities.float()

        batch, horizon, vocabulary = p.shape

   
        if max(self.activation_id, self.target_id, self.end_id) >= vocabulary:
            raise ValueError("There is an ID grater than the numebr of targt activities")
      

        prefix_ids = torch.as_tensor(prefix_ids, device=p.device)
        lengths = torch.as_tensor(prefix_lengths, device=p.device)

        #prefix_ids deve avere come dimensione [batch, max_prefix]
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != batch or lengths.shape != (batch,):
            raise ValueError("Prefixes shape not aligned to batch size.")
        if prefix_ids.dtype not in (torch.int32, torch.int64) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("IDs and lenghts must be int.")
        
        width = prefix_ids.shape[1]
        if (lengths < 0).any() or (lengths > width).any():
            raise ValueError("Not valid prefix lenghts.")

        positions = torch.arange(width, device=p.device).unsqueeze(0) #tensore [1, max_prefix]
        prefix_lengths = lengths.unsqueeze(1)  #tensore [batch_size, 1]

        #unsqueeze aggiunge un dimensione

        valid = positions < prefix_lengths
        # si ottiene una maschera che vale True se la posizione j per il prefisso i è valida
        observed = prefix_ids[valid] #serve solo per controllare gli IDs (mischia tutti i prefissi)

        if ((observed < 0) | (observed >= vocabulary) | (observed == self.end_id)).any():
            raise ValueError("Real prefix must contain valid activities (not ENDactivity).")


        

    
        safe_ids = prefix_ids.masked_fill(~valid, 0).long()
        #Prende gli id, mette 0 dove c'è il padding e trasforma in log.
        #Serve perchè  la funzione one_hot non accetta l'ID = -1

        prefix = F.one_hot(safe_ids, num_classes=vocabulary).to(p.dtype)
        #Trasforma ciascuna attività in un vettore one_hot
        # [batch, max_prefix, num_activities]

        prefix = prefix * valid.unsqueeze(-1)
        #nella moltiplicazione True vale 1 e false vale 0.
        #Come risultato vengono annullati i one_hot vector di padding


        full = torch.cat((prefix, p), dim=1)
        #Uniamo le probabilità certe di prefix con le probabilità predette di p
        # prefix [batch_size, max_prefix, num_activities]
        # p [batch_size, num_teste, activites]
        # risultato [batch, size, max_prefix + num_teste, activities]


        # s_t: probabilità di non avere END nelle posizioni PRECEDENTI.
        no_end = 1 - full[..., self.end_id]
        # per ogni prefisso e per ogni istate di tempo hai la probabilita di END
        # poi fai la probabilità complementare

        ones = p.new_ones((batch, 1))
        survival = torch.cat((ones, no_end[:, :-1].cumprod(dim=1)), dim=1)


        mass_a = survival * full[..., self.activation_id]
        mass_b = survival * full[..., self.target_id]
        # shape [batch_size, max_prefix + num_teste]

        # OR delle B strettamente SUCCESSIVE a t (prodotto inverso esclusivo).
        no_b_from_here = torch.flip(
            torch.cumprod(torch.flip(1 - mass_b, dims=[1]), dim=1), dims=[1]
        )
        #ogni elemento contiene il prodotto (1 - massa_B) dall'istante corrente fino alla fine


        no_b_after = torch.cat((no_b_from_here[:, 1:], ones), dim=1)
        future_b = 1 - no_b_after

        implications = (1 - mass_a * (1 - future_b)).clamp(0, 1)
        #per ogni prefisso, e per ogni istante di tempo, ho la probabilità che 
        # l'implicazione sia verificata


        score = implications.prod(dim=1)
        #calcolo lo score come prodottoria sui diversi istanti di tempo,
        #perchè l'implicazione deve essere sempre verificata


        # -log del prodotto = somma dei -log.
        # eps limita la penalità delle violazioni certe a un valore finito.

        per_example = -implications.clamp_min(self.eps).log().sum(dim=1)
        #per calcolare la loss, visto che è logaritmata, evito di passare 
        #per la produttoria e calcolo direttamente la somma dei logaritmi, cambiata di segno.

        #La loss finale si ottiene come media sul batch

        return {"satisfaction": score, "loss_per_example": per_example, "loss": per_example.mean()}
