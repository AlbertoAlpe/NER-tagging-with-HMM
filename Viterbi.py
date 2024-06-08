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

from PoS-Probabilities.py import tags, words

viterbi = []

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