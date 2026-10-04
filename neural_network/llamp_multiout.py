import torch.nn as nn 

class BertMultiOutputClassificationHeads(nn.Module):  #la classe eredita da nn.Module che è la classe base utilizzata da Pytorch per creare modelli neurali
    def __init__(self, gpt_model, output_sizes):
        super(BertMultiOutputClassificationHeads, self).__init__()
        self.gpt_model = gpt_model
        self.output_layers = nn.ModuleList([nn.Linear(gpt_model.config.hidden_size, output_sizes[i]) for i in range(len(output_sizes))])
        #Viene costruita una lista di moduli pytorch. Ciascun modulo ha in input un vettore di dimensione pari alla hidden_size del modello pretrained
        #In ouput ciascuna testa lineare ha un numero di logit, in base a come specificato in input. 
        #nella pratica ilo numero di logit è pari al numero di activities


    def forward(self, input_ids, attention_mask):
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        out = []
        for output_layer in self.output_layers:
            out.append(output_layer(pooled_output))

        return out



#Il costruttore riceve in input un modello pre-addestrato. 
#output_sizes, è una lista che dice quanti classi predire per ogni testa di classificazione

#input_ids è una matrice [batch_size x seq_len] che contiene per gli id dei token di ciascuna frase in input. 
# attention mask serve per comunicare quali token sono token di padding della sentence.
# Nel risultato finale vado ad inserire i logit di ciascuna testa di classificazione


