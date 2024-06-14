import numpy as np
import csv
import os
import viterbi

# Variabili per memorizzare i dati
tags = []
words = []
emission_P = []
transition_P = []

###################################
###LETTURA FILE 'PoS_Probabilities'
# Leggi array e matrici con le probabilità di riferimento dal file CSV
# with open('probabilities1.csv', 'r', encoding='utf-8') as file:
with open('probabilities.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    section = None
    
    for row in reader:
        if row:
            if row[0] == 'tags':
                tags = next(reader)
            elif row[0] == 'words':
                words = next(reader)
            elif row[0] == 'emissione':
                section = 'emissione'
                emission_P = []
            elif row[0] == 'transizione':
                section = 'transizione'
                transition_P = []
            elif section == 'emissione':
                emission_P.append([float(x) for x in row])
            elif section == 'transizione':
                transition_P.append([float(x) for x in row])

# conversione delle matrici python in matrici numpy per accedervi più facilmente
transition_P = np.array(transition_P)
emission_P = np.array(emission_P)

#######################
###LETTURA FILE DI TEST
file_name = 'test.conllu'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'wikineural_en', file_name)

with open(file_path, 'r', encoding='utf-8') as test:
   prime_100_righe = test.readlines()[:100]

# un array che contiene le parole della frase da analizzare
sequence = []

# un ciclo for generale legge tutte le frasi del file test
for riga in prime_100_righe:
    riga = riga.strip()
    if riga:
        riga = riga.split()
        sequence.append(riga[1])

    else:     # riga vuota => end of sentence
        final_sequence = viterbi.viterbi(sequence, emission_P, transition_P, tags, words)
        print("frase: ")
        print(sequence)
        print("tags: ")
        print(final_sequence)
        sequence = []
        final_sequence = []

