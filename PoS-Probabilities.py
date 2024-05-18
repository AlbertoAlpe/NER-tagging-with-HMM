import numpy as np
import csv
import os
file_name = 'train.conllu'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'wikineural_en', file_name)

train = open(file_path, 'r', encoding='utf-8')
prime_100_righe = train.readlines()[:1000]

emission_P = {}
transition_P = np.zeros((11, 11), dtype=int)
tags = ['BEGIN', 'END']
words = []
word = 0
tag = 0
tag_prec = None
tpi = 0

#conta le occorrenze di tag_prec->tag e di tag->parola e le inserisce nelle relative matrici
for riga in prime_100_righe:
    riga = riga.strip()
    if riga:
        riga = riga.split()
        if riga[2] not in tags:
           tags.append(riga[2])
           tag = len(tags) - 1
           emission_P[riga[2]] = {}
        else:
           tag = tags.index(riga[2]) 

        if riga[1] not in words:
           words.append(riga[1])
           word = len(words) - 1
        else:   
           word = words.index(riga[1])
        
        
        emission_P[riga[2]][riga[1]] = emission_P[riga[2]].get(riga[1], 0) + 1

        if(tag_prec):
          tpi = tags.index(tag_prec)
          transition_P[tpi][tag] += 1
        tag_prec = riga[2]    
    else: 
        #in questo caso il tag è END, si fanno i rispettivi calcoli e poi il tag_prec dicenta BEGIN
        tag = "END"
        tpi = tags.index(tag_prec)
        transition_P[tpi][1] += 1  #indice 1 perchè END è il secondo elemento dell'array tags
        tag_prec = "BEGIN"         

print(tags)
print(transition_P)


###PROBABILITA' DI EMISSIONE E TRANSIZIONE###
for riga in emission_P:
   tagTot = 0
   #in ogni riga abbiamo tutte le occorrenze di un tag divise per ogni parola, quindi le sommiamo in tagTot
   for i in len(riga)-1:
      tagTot = tagTot + emission_P[riga][i]
   for j in len(riga)-1:   #CALCOLO PROB. EMISSIONE
      emission_P[riga][j] /= tagTot
   for t in range(11):     #CALCOLO PROB. TRANSIZIONE (su matrice transition_P)    
      transition_P[riga][t] /= tagTot

#salva le matrici risultanti in file .csv
emiss_csv = "wikineural_en/emissione_en.csv"
transiz_csv = "wikineural_en/transizione_en.csv"

with open(emiss_csv, mode='w', newline='') as emiss:
    writer = csv.writer(emiss)
    writer.writerows(emission_P)

with open(transiz_csv, mode='w', newline='') as transiz:
    writer = csv.writer(transiz)
    writer.writerows(transition_P)
