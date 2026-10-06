from utility import log_config as lg
from jinja2 import Template
import pandas as pd
import numpy as np
import pickle
import torch
from itertools import chain, repeat, islice
from pathlib import Path
from constraints.extraction import EVALUATORS

# Prende un event log csv e lo trasforma nei dati che verranno passati a BERT.
# Genera i prefissi delle tracce, le label, divide in train e test set
# e serializza tutto in file.pkl

class Log():
    def __init__(self, log, setting, constraint=None):
        self.__log_name = log
        self.__constraint = (
            dict(constraint) if constraint is not None else None
        )
        self.__output_dir = Path("log_history") / log
        self.__log = pd.read_csv('event_log/'+log+'.csv')


        if self.__constraint is not None:
            self.__output_dir = self.__output_dir / "constrained"
        self.__output_dir.mkdir(parents=True, exist_ok=True)


        self.__train = []
        self.__test = []
        self.__len_prefix_train = []
        self.__len_prefix_test = []
        self.__history_train = [] #input che finiscono nel training
        self.__history_test = [] 
        self.__dict_label_train = [] #etichette
        self.__dict_label_test = []
        self.__id2label = {} #mappa gli id nel nome della classe
        self.__label2id = {} #mappa il nome della classe negli id
        self.__setting = setting
        self.__max_length = 0
        self.__cont_trace = 0
        self.__max_trace = 0
        self.__mean_trace = 0
        self.__split_log() #fa partire tutto il pre-processing

    def pad_infinite(self, iterable, padding=None):
        return chain(iterable, repeat(padding))

    def pad(self, iterable, size, padding=None):
        return islice(self.pad_infinite(iterable, padding), size)



    #Questo metodo verifica una traccia completa per volta, e mantiene
    # tutte le righe degli eventi appartenenti ai casi selezionati.
    def __filter_training_cases(self):
        template = self.__constraint["template"]

        # Stessa normalizzazione usata nel preprocessing.
        def normalize(activity):
            for char in (" ", "+", "-", "_"):
                activity = activity.replace(char, "")
            return activity

        activation = normalize(self.__constraint["activation"])
        target = normalize(self.__constraint["target"])

        vocabulary = self.__label2id["activity"]

        for activity in (activation, target):
            if activity == "ENDactivity" or activity not in vocabulary:
                raise ValueError(f"Attività non valida: {activity}")

        evaluator = EVALUATORS[template]

        selected_cases = []
        counts = {
            "fulfilled": 0,
            "violated": 0,
            "vacuous": 0,
        }

        for case_id, group in self.__train.groupby("case", sort=False):
            trace = group["activity"].tolist()

            state = evaluator(trace, activation, target)
            counts[state] += 1

            # Mantieni solo i casi in cui il vincolo è attivato e rispettato.
            if state == "fulfilled" or state=="vacuous":
                selected_cases.append(case_id)

        print("Risultati del vincolo sul training:", counts)
        print("Casi selezionati:", len(selected_cases))

        if not selected_cases:
            raise ValueError(
                "Nessun caso di training soddisfa il vincolo "
                "in modo non vacuo."
            )

        self.__train = self.__train[
            self.__train["case"].isin(selected_cases)
        ].copy()


    #prende ogni case del frame e li trasforma in più storie temporali,
    #costruendo gli appositi suffissi
    def __gen_prefix_history(self, df):
            list_seq = []
            prefix_activities = []
            # lista contenente i prefissi trasformati in testo
            list_len_prefix = []
            #lista contenente la lunghezza dei prefissi 
            sequence = df.groupby('case', sort=False)
            event_template = Template(lg.log[self.__log_name]['event_template'])
            #rappresentazione testuale di un evento insieme ai suoi attributi. 
            trace_template = Template(lg.log[self.__log_name]['trace_template'])
            #rappresentazione testuale degli attributi della trace

            dict_event_label = {}
            for v in lg.log[self.__log_name]['event_attribute']:
                dict_event_label[v] = []
            dict_trace_label = {}
            for v in lg.log[self.__log_name]['trace_attribute']:
                dict_trace_label[v] = []
            # per ogni attributo di evento e per ogni attributo di traccia crea una lista vuota 
            # nel dizionario

            dict_len_label = {}
            for i in range(self.__max_length):
                dict_len_label[i] = []
            #si crea un dizionario con una lista vuota per ogni posizione futura da predire



            for group_name, group_data in sequence:
                #prende un case per volta
                event_dict_hist = {}
                trace_dict_hist = {}
                event_text = ''
                len_prefix = 1
                activity_list = []
                for index, row in group_data.iterrows():
                    #iteriamo sugli eventi del case corrente
                    activity_list.append(row['activity'])
                    prefix_activities.append(activity_list.copy())
                    for v in lg.log[self.__log_name]['event_attribute']:
                        value = row[v]
                        if isinstance(value, str):
                            event_dict_hist[v] = value.replace(' ','')
                        else:
                            event_dict_hist[v] = value
                            #Questo dizionario rappresenta un singolo evento. Per ogni attributo prendo il suo valore
                    event_text = event_text + event_template.render(event_dict_hist) + ' '
                    #Di volta in volta incremento questa stringa con il template relativo all'evento i-esimo
                    for w in lg.log[self.__log_name]['trace_attribute']:
                        value = row[w]
                        if isinstance(value, str):
                            trace_dict_hist[w] = value.replace(' ','')
                        else:
                            trace_dict_hist[w] = value
                    trace_text = trace_template.render(trace_dict_hist)
                    #Faccio la stessa cosa per gli attributi di traccia


                    prefix_hist = event_text + trace_text
                    list_seq.append(prefix_hist)
                    #per ogni evento della traccia salvo il prefix history ottenuto fino a quel momento
                    list_len_prefix.append(len_prefix)
                    len_prefix = len_prefix + 1
                suffixes = []
                activity_list.pop(0)
                activity_list.append('ENDactivity')
                for i in range(len(activity_list)):
                    suffixes.append(list(self.pad(activity_list[i:], self.__max_length, 'ENDactivity')))
                for s in suffixes:
                    for i in range(len(s)):
                        dict_len_label[i].append(self.__label2id['activity'][s[i]])
                #GENERAZIONE DEI SUFFISSI
                #1. SUFFIXES è UNA MATRICE IN CUI OGNI RIGA CONTIENE UN SUFFISSO
                #2. OGNI SUFFISSO VIENE RIEMPITO CON 'ENDActivity' FINO A MAX_LENGTH
                #3. dict_len_label è un dizionario che associa a ciascuna posizione futura
                #.  la lista dei valori che assume quella posizione per i diversi prefissi.

                for v in lg.log[self.__log_name]['event_attribute']:
                    if v!='timesincecasestart':
                        dict_event_label[v].extend(group_data[v].shift(-1).fillna('END'+v).tolist())
                    else:
                        dict_event_label[v].extend(group_data[v].shift(-1).fillna(0).tolist())
            return list_seq, dict_event_label, list_len_prefix, dict_len_label, prefix_activities



    def __extract_timestamp_features(self, group):
        timestamp_col = 'timestamp'
        group = group.sort_values(timestamp_col, ascending=True)
        # end_date = group[timestamp_col].iloc[-1]
        start_date = group[timestamp_col].iloc[0]

        timesincelastevent = group[timestamp_col].diff()
        timesincelastevent = timesincelastevent.fillna(pd.Timedelta(seconds=0))
        group["timesincelastevent"] = timesincelastevent.apply(
            lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds


        elapsed = group[timestamp_col] - start_date
        elapsed = elapsed.fillna(pd.Timedelta(seconds=0))
        group["timesincecasestart"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds
        return group
        #  Riceve un gruppo di eventi relativi ad un case,
        # e aggiunge due colonne, una con il tempo relativo dall'inizio del case,
        # e un'altra con il tempo dall'ultimo evento


    def __split_log(self):
        self.__log['activity']= self.__log['activity'].str.replace(' ', '')
        self.__log['activity']= self.__log['activity'].str.replace('+', '')
        self.__log['activity']= self.__log['activity'].str.replace('-', '')
        self.__log['activity']= self.__log['activity'].str.replace('_', '')
        #pulisce il nome delle activity


        if self.__log_name !='sepsis':
            self.__log['resource'] = self.__log['resource'].astype(str)
            self.__log['resource']= self.__log['resource'].str.replace(' ', '')
            self.__log['resource'] = self.__log['resource'].str.replace('+', '')
            self.__log['resource'] = self.__log['resource'].str.replace('-', '')
            self.__log['resource'] = self.__log['resource'].str.replace('_', '')
        #converte e pulisce il nome degli attributi resource di ogni stringa
        


        self.__log.fillna('UNK', inplace=True)
        #sostiuisce i valori NaN 

        self.__cont_trace = self.__log['case'].value_counts(dropna=False)
        self.__max_trace = max(self.__cont_trace)
        #lunghezza massima di una traccia

        self.__mean_trace = int(round(np.mean(self.__cont_trace)))
        #lunghezza media di una traccia

        self.__log['timestamp'] = pd.to_datetime(self.__log['timestamp'])
        #conversione in oggetti datetime Pandas

        for c in lg.log[self.__log_name]['event_attribute']:
            if c!='timesincecasestart':#c!='timesincelastevent' or
                ALL_LABEL = list(self.__log[c].unique())
                ALL_LABEL.append('END' + c)
                self.__id2label[c] = {k: l for k, l in enumerate(ALL_LABEL)}
                self.__label2id[c] = {l: k for k, l in enumerate(ALL_LABEL)}
                #Gli output sono dizionari di dizionari. Il dizionario esterno ha per chiave i nomi degli attributi
                # Mentre il dizionario interno associa a ciascun valore unico assunto dall'attributo un id numerico e viceversa
     

    

        cont_trace = self.__log['case'].value_counts(dropna=False)
        self.__max_length = max(cont_trace)
        #print("Max lenght trace", max_trace)

        self.__log = self.__log.groupby('case', group_keys=False).apply(self.__extract_timestamp_features)
        self.__log = self.__log.reset_index(drop=True)
        self.__log['timesincecasestart'] = (self.__log['timesincecasestart'])#.round(3)
        self.__log['timesincecasestart'] = self.__log['timesincecasestart'].astype(int)
        #definisce un tempo relatiuvo all'inizio di ogni trace

        grouped = self.__log.groupby("case")
        start_timestamps = grouped["timestamp"].min().reset_index()
        start_timestamps = start_timestamps.sort_values("timestamp", ascending=True, kind="mergesort")
        #ordina le tracce in base al timestamp di partenza.
        #start_timestamps associa a ciascun case il minimo  timestamp (normale dataframe)
        
        train_ids = list(start_timestamps["case"])[:int(0.66 * len(start_timestamps))]
        self.__train = self.__log[self.__log["case"].isin(train_ids)].sort_values("timestamp", ascending=True,kind='mergesort')
        self.__test = self.__log[~self.__log["case"].isin(train_ids)].sort_values("timestamp", ascending=True,kind='mergesort')
        #Prende il primo 66% delle tracce e le mette nel training set
        #Prende il restante 33% e lo mette nel test_set

        if self.__constraint is not None:
            self.__filter_training_cases()

        self.__history_train, self.__dict_label_train, self.__len_prefix_train, dict_suffix_train, prefix_train = self.__gen_prefix_history(self.__train)
        self.__history_test, self.__dict_label_test, self.__len_prefix_test, dict_suffix_test, prefix_test = self.__gen_prefix_history(self.__test)
        #Sia per il training set che per il test set, ho 4 strutture dati finali
        #1. __history_train: una lista che contiene un elemento per ogni prefisso del template già convertito in testo
        #2. __dict_label_train: dizionario che associa ad ogni attributo una lista con il prossimo valore che assume ogni attributo nello step successivo ad un certo prefisso
        #3. __len_prefix : lunghezza di ciascun prefisso
        #4. __dict_suffix_train dizionario che associa a ciascuna posizione futura, la lista di activity che assumerà sottoforma di id. 

        for v in self.__dict_label_train:
            if v!='timesincecasestart':
                temp_list = []
                for key in self.__dict_label_train[v]:
                        temp_list.append(self.__label2id[v].get(key))
                self.__dict_label_train[v] = torch.tensor(temp_list)
            else:
                self.__dict_label_train[v] = torch.tensor(self.__dict_label_train[v]).view(-1, 1)
        #dict_label_train continua ad avere per ogni prefisso del training set, gli attributi dell'evento
        # immediatamente successivo ma li converte negli appositi id e poi in tensori. 

        for v in self.__dict_label_test:
            if v!='timesincecasestart':
                temp_list = []

                for key in self.__dict_label_test[v]:
                        temp_list.append(self.__label2id[v].get(key))
                self.__dict_label_test[v] = torch.tensor(temp_list)
            else:
                self.__dict_label_test[v] = torch.tensor(self.__dict_label_test[v]).view(-1, 1)

        self.__serialize_object(prefix_train, 'prefix_activities_train')
        self.__serialize_object(prefix_test, 'prefix_activities_test')

        self.__serialize_object(self.__history_train, 'train')
        self.__serialize_object(self.__history_test, 'test')
        self.__serialize_object(self.__len_prefix_train, 'len_train')
        self.__serialize_object(self.__len_prefix_test, 'len_test')
        self.__serialize_object(dict_suffix_train, 'suffix_train')
        self.__serialize_object(dict_suffix_test, 'suffix_test')

        self.__serialize_object(self.__dict_label_train[lg.log[self.__log_name]['target']], 'label_train')
        self.__serialize_object(self.__dict_label_test[lg.log[self.__log_name]['target']], 'label_test')
        #DEL DIZIONARIO VAI A PRENDERE SOLO LE ACTIVITY. QUINDI AVRAI UNA LISTA CON LA NEXT ACTIVITY PER OGNI PREFISSO


        self.__serialize_object(self.__id2label, "id2label")
        self.__serialize_object(self.__label2id, "label2id")

    def __utility_function(self,list_seq,dict_event_label):
        for l, a, r, t in zip(list_seq, dict_event_label['activity'], dict_event_label['resource'],
                              dict_event_label['timesincecasestart']):
            print(l, 'label-->', a, r, t)
            print('-------------------------')



    def __serialize_object(self, obj, kind):
        filename = (
            f"{self.__log_name}_{kind}_{self.__setting}.pkl"
        )

        with (self.__output_dir / filename).open("wb") as f:
            pickle.dump(obj, f)

    def get_id2label(self):
        return self.__id2label

    def get_label2id(self):
        return self.__label2id
