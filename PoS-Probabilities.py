import numpy as np
import csv
import os

###LETTURA FILE DI TRAIN
file_name = 'train.conllu'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'wikineural_en', file_name)

with open(file_path, 'r', encoding='utf-8') as train:
   prime_100_righe = train.readlines()[:100]


emission_P = [[], []]     #matrice iniziale composta da due righe vuote
transition_P = [[0, 0], [0, 0]]  #matrice iniziale 2x2
tags = ['START', 'END']
words = []
word = 0
tag = 0
tag_prec = 'START'
tpi = 0

#due funzioni per aggiungere dinamicamente una riga o una colonna alle matrici
def aggiungi_riga(matrice):
    nuova_riga = [0 for _ in range(len(matrice[0]))]
    matrice.append(nuova_riga)

def aggiungi_colonna(matrice):
    for row in matrice:
        row.append(0)

#Gli array tags e words contengono una lista dei tag e delle parole presenti nel corpus;
#La matrice transition_P viene aggiornata dinamicamente e avrà un tag per ogni riga e uno per ogni colonna;
#in questo modo ogni cella funzionerà come contatore delle occorrenze di transizione da tag(riga-1) -> tag(riga).
#La matrice emission_P avrà un tag per ogni riga e una word per ogni colonna; ogni cella conterà il numero di
#occorrenze di emissione tag(riga) -> word(riga)  
#conta le occorrenze di tag_prec->tag e di tag->parola e le inserisce nelle relative matrici
for riga in prime_100_righe:
    riga = riga.strip()
    if riga:
        riga = riga.split()
        if riga[1] not in words:    #se la parola è nuova, aggiorna le dimensioni di words e di emission_P
           words.append(riga[1])
           word = len(words) - 1
           aggiungi_colonna(emission_P)
        else:   
           word = words.index(riga[1])

        if riga[2] not in tags:  #se il tag è nuovo, aggiorna le dimensioni di tags e delle due matrici 
           tags.append(riga[2])
           tag = len(tags) - 1
           aggiungi_riga(transition_P)
           aggiungi_colonna(transition_P)
           aggiungi_riga(emission_P)
        else:
           tag = tags.index(riga[2]) 
        
        #in tag abbiamo l'indice della riga in cui operare
        #in word abbiamo l'indice della parola in cui operare
        emission_P[tag][word] += 1 
        #emission_P[riga[2]][riga[1]] = emission_P[riga[2]].get(riga[1], 0) + 1
        tpi = tags.index(tag_prec)
        transition_P[tpi][tag] += 1
        tag_prec = riga[2]
    else: 
        #in questo caso la riga è vuota => il tag è END, si fanno i calcoli necessari e poi il tag_prec diventa START
        tpi = tags.index(tag_prec)
        transition_P[tpi][1] += 1  #indice 1 perchè END è il secondo elemento dell'array tags
        tag_prec = "START"         

#print(emission_P)
#print(transition_P)

#CALCOLO PROB. EMISSIONE
emission_P = emission_P / np.sum(emission_P, axis=1, keepdims=True)
#CALCOLO PROB. TRANSIZIONE
transition_P = transition_P / np.sum(emission_P, axis=1, keepdims=True)

#print(emission_P)
#print(transition_P)

# Salva i dati in un unico file CSV
with open('probabilities.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Scrivi array1
    writer.writerow(['tags'])
    writer.writerow(tags)
    
    # Scrivi array2
    writer.writerow(['words'])
    writer.writerow(words)
    
    # Scrivi matrix1
    writer.writerow(['emissione'])
    writer.writerows(emission_P)
    
    # Scrivi matrix2
    writer.writerow(['transizione'])
    writer.writerows(transition_P)