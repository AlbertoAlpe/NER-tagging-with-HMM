import csv
import os
file_name = 'train.conllu'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'wikineural_en', file_name)

train = open(file_path, 'r', encoding='utf-8')
prime_100_righe = train.readlines()[:100]


emission_P = [[], []]           #matrice iniziale composta da due righe vuote
transition_P = [[0, 0], [0, 0]]
#emission_P = [[], []]
#transition_P = [[], []]  
tags = ['BEGIN', 'END']
words = []
word = 0
tag = 0
tag_prec = 'BEGIN'
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
#La matrice emission_P avrà un tagp per ogni riga e una word per ogni colonna; ogni cella conterà il numero di
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
        #in questo caso la riga è vuota => il tag è END, si fanno i calcoli necessari e poi il tag_prec diventa BEGIN
        tpi = tags.index(tag_prec)
        transition_P[tpi][1] += 1  #indice 1 perchè END è il secondo elemento dell'array tags
        tag_prec = "BEGIN"         

print(tags)
print(transition_P)


###PROBABILITA' DI EMISSIONE E TRANSIZIONE###
#Una volta ottenute le matrici con le occorrenze di trasmissione e di transizione,
#in ogni riga avremo tutte le occorrenze di un tag divise per ogni parola, quindi le sommiamo in tagTot
#la variabile tagTot verrà usata nei due cicli for incapsulati nel primo, 
#per calcolare le probabilità di emissione e tansizione
for r, row in enumerate(emission_P):
   tagTot = 0
   for i in row:
      tagTot += i
   if (tagTot>0): 
      for idx, value in enumerate(row):     #CALCOLO PROB. EMISSIONE
            emission_P[r][idx] = value / tagTot

      for t in transition_P[r]:      #CALCOLO PROB. TRANSIZIONE    
         t = t/tagTot

print(transition_P)

'''
#salva le matrici risultanti in due files .csv
emiss_csv = "emissione_en.csv"
transiz_csv = "transizione_en.csv"

with open(emiss_csv, mode='w', newline='') as emiss:
    writer = csv.writer(emiss)
    writer.writerows(emission_P)

with open(transiz_csv, mode='w', newline='') as transiz:
    writer = csv.writer(transiz)
    writer.writerows(transition_P)

'''