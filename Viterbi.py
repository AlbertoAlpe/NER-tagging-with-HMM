import numpy as np

# Algoritmo di Viterbi
def viterbi(sequence, emission_P, transition_P, tags, words):
    n = len(sequence)
    m = len(tags)
    viterbi_matrix = np.zeros((m, n))
    backpointer = np.zeros((m, n), dtype=int)
    Oindex = tags.index('O')  #queste variabili serviranno per lo smoothing
    MISCindex = tags.index('B-MISC')

    # Initialization step
    for tag in range(2, m):  # Ignoro START e END nei calcoli
        if sequence[0] in words:
            viterbi_matrix[tag, 0] = transition_P[0, tag] * emission_P[tag, words.index(sequence[0])]
        else:       #se la parola è sconosciuta
        #versione base, solo transizione    
            viterbi_matrix[tag, 0] = transition_P[0, tag]
        #SMOOTHING
            #vers. 1: P(unk|O)=1
            viterbi_matrix[Oindex, 0] = transition_P[0, Oindex] * 1
            #vers. 2: P(unk|O)=P(unk|B-MISC)=0.5 
            viterbi_matrix[Oindex, 0] = transition_P[0, Oindex] * 0.5
            viterbi_matrix[MISCindex, 0] = transition_P[0, MISCindex] * 0.5
            #vers. 3: uniforme-> P(unk|Ti)= 1/len(tags)
            viterbi_matrix[tag, 0] = transition_P[0, tag] * 1/m    
            #aggiungere statistica development set(parole che compaiono una sola volta)

    # Recursion step
    for word in range(1, n):
        if sequence[word] in words:
            for tag in range(2, m):
                max_tr_prob = viterbi_matrix[:, word-1] * transition_P[:, tag]
                backpointer[tag, word] = np.argmax(max_tr_prob)
                viterbi_matrix[tag, word] = np.max(max_tr_prob) * emission_P[tag, words.index(sequence[word])]
        else:       #se la parola è sconosciuta
        #versione base
            viterbi_matrix[tag, word] = np.max(max_tr_prob)
        #SMOOOTHING
            #vers. 1: P(unk|O)=1
            max_tr_prob = viterbi_matrix[:, word-1] * transition_P[:, Oindex]
            backpointer[Oindex, word] = np.argmax(max_tr_prob)
            viterbi_matrix[Oindex, word] = np.max(max_tr_prob)
            #vers. 2: P(unk|O)=P(unk|B-MISC)=0.5    
            max_tr_prob = viterbi_matrix[:, word-1] * transition_P[:, Oindex]
            backpointer[Oindex, word] = np.argmax(max_tr_prob)
            viterbi_matrix[Oindex, word] = np.max(max_tr_prob) * 0.5
            max_tr_prob = viterbi_matrix[:, word-1] * transition_P[:, MISCindex]
            backpointer[MISCindex, word] = np.argmax(max_tr_prob)
            viterbi_matrix[MISCindex, word] = np.max(max_tr_prob) * 0.5
            #vers. 3: uniforme-> P(unk|Ti)= 1/len(tags)
            for tag in range(2, m):
                max_tr_prob = viterbi_matrix[:, word-1] * transition_P[:, tag]
                backpointer[tag, word] = np.argmax(max_tr_prob)
                viterbi_matrix[tag, word] = np.max(max_tr_prob) * 1/len(m)
        
    
    # Termination step
    max_prob = viterbi_matrix[:, n-1] * transition_P[:, 1]
    best_path_pointer = np.argmax(max_prob)
    best_path = [best_path_pointer]

    for word in range(n-1, 0, -1):
        best_path_pointer = backpointer[best_path_pointer, word]
        best_path.insert(0, best_path_pointer)

    best_tag_sequence = [tags[i] for i in best_path]

    return best_tag_sequence        
