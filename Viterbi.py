'''
Viterbi:

per ogni frase di n parole -> le dobbiamo salvare in un array parole[]
creiamo matrice di (n+2) colonne (cioè n parole + begin e end) e t righe(array tags già fatto) -> dobbiamo importarlo in un csv?
possiamo crearla con tutte le celle contenenti il valore 0
col=1, per ogni cella si calcola solo P(tag-cella|'Begin') * P(tag-cella|parola-colonna)
col=n+1(End) = nella cella 'end' si tiene il migliore di tutti i P(in arrivo)*P('End'|tag-cella-1)
per ogni colonna in mezzo invece si fa così: 

per ogni colonna da 1 a n (le parole della frase)
   per ogni cella di quella colonna
	prob_max=0
	calcola tutte le probabilità (scorrendo ogni cella della colonna precedente)
	salva la prob_max in quella cella
	bisogna salvare anche un puntatore alla cella che le ha permesso di avere quel valore(come?)

Così ogni cella della matrice salverà UN VALORE (il più alto)
Alla fine dovremo fare backtrace:
partiremo dall'ultima colonna, con un solo valore(le altre celle saranno a 0)
andiamo indietro di una colonna e vogliamo sapere CHI ci ha portato lì; quindi non ci interessano più i valor nella colonne, ma solo i puntatori
es. partiremo dalla cella end, avrà un solo puntatore (alla cella che l'ha portata ad avere quel valore), quindi andremo in quella cella della colonna precedente (le altre non le guardiamo più), e così via fino ad arrivare a begin

->come salvare i puntatori? ne serve uno per cella; creare un'altra matrice di righe=t(il numero dei tag) per ogni parola della frase?Però il numero di colonne cambia ogni volte(es. frase da 10 parole o da 15). Ci sono metodi più efficienti?
'''
import csv
import os


# Variabili per memorizzare i dati
tags = []
words = []
emission_P = []
transition_P = []

###################################
###LETTURA FILE 'PoS_Probabilities'
#Leggi array e matrici con i pesi di riferimento dal file CSV
with open('probabilities.csv', 'r') as file:
    reader = csv.reader(file)
    section = None
    
    for row in reader:
        if row:
            if row[0] == 'tags':
                section = 'tags'
                tags = next(reader)
                tags = [int(x) for x in emission_P]
            elif row[0] == 'words':
                section = 'words'
                words = next(reader)
                words = [int(x) for x in transition_P]
            elif row[0] == 'emissione':
                section = 'emissione'
                emission_P = []
            elif row[0] == 'transizione':
                section = 'transizione'
                transition_P = []
            elif section == 'matrix1':
                emission_P.append([int(x) for x in row])
            elif section == 'matrix2':
                transition_P.append([int(x) for x in row])

# Visualizza i dati
print("Array tags:", tags)
print("Array words:", words)
print("emissione:", emission_P)
print("transizione:", transition_P)

#######################
###LETTURA FILE DI TEST
file_name = 'test.conllu'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'wikineural_en', file_name)

with open(file_path, 'r', encoding='utf-8') as train:
   prime_100_righe = train.readlines()[:100]

#un array che contiene le parole della frase da analizzare
parole = ['start', 'end']
#una matrice viterbi che a ogni iterazione conterrà i risultati dei calcoli
viterbi = [0][0]
#e una matrice puntatori di uguale dimensione per poter fare backtrace
puntatori = [0][0]

#nota: le due matrici contengono già i due tag 'start' e 'end' 
#ATTENZIONE ALLA GESTIONE DEGLI INDICI! END ANDREBBE ALLA FINE

#una funzione per aggiungere dinamicamente una colonna a una matrice
def aggiungi_colonna(matrice):
    for row in matrice:
        row.append(0)

#un ciclo for generale legge tutte le frasi del file test
for riga in prime_100_righe:
    riga = riga.strip()
    if riga:
        parole.append(riga[1])
        aggiungi_colonna(viterbi)
        aggiungi_colonna(puntatori)

    else:     #riga vuota => end of sentence
        #MATRICE COMPLETA; qui si fanno i calcoli per la singola frase
        #...
        #Fine calcoli, si reinizializzano matrici e array
        parole = ['start', 'end']
        puntatori = [0][0]
        parole = [0][0]


