import numpy as np

# Algoritmo di Viterbi
def viterbi(sentence, emission_P, transition_P, tags, words):
    log_emission_P = np.log(emission_P)
    log_transition_P = np.log(transition_P)
    n = len(sentence)
    m = len(tags)
    viterbi_matrix = np.full((m, n), -np.inf)   # viterbi_matrix = np.zeros(m, n)
    backpointer = np.zeros((m, n), dtype=int)
    Oindex = tags.index('O')  #queste variabili serviranno per lo smoothing
    MISCindex = tags.index('B-MISC')

    # Initialization step
    for tag in range(2, m):  # Ignoro START e END nei calcoli
        if sentence[0] in words:
            viterbi_matrix[tag, 0] = log_transition_P[0, tag] + log_emission_P[tag, words.index(sentence[0])]
        else:       #se la parola è sconosciuta
        #vers. 3: uniforme-> P(unk|Ti)= 1/len(tags)
            viterbi_matrix[tag, 0] = log_transition_P[0, tag] * 1/m
        

    # Recursion step
    for word in range(1, n):
        if sentence[word] in words:
            for tag in range(2, m):
                max_tr_prob = viterbi_matrix[:, word-1] + log_transition_P[:, tag]
                backpointer[tag, word] = np.argmax(max_tr_prob)
                viterbi_matrix[tag, word] = np.max(max_tr_prob) + log_emission_P[tag, words.index(sentence[word])]
        else:       #se la parola è sconosciuta
        #vers. 3: uniforme-> P(unk|Ti)= 1/len(tags)
            for tag in range(2, m):
                max_tr_prob = viterbi_matrix[:, word-1] + log_transition_P[:, tag]
                backpointer[tag, word] = np.argmax(max_tr_prob)
                viterbi_matrix[tag, word] = np.max(max_tr_prob) * 1/m

    
    # Termination step
    max_prob = viterbi_matrix[:, n-1] + log_transition_P[:, 1]
    best_path_pointer = np.argmax(max_prob)
    best_path = [best_path_pointer]

    for word in range(n-1, 0, -1):
        best_path_pointer = backpointer[best_path_pointer, word]
        best_path.insert(0, best_path_pointer)

    best_tag_sequence = [tags[i] for i in best_path]

    return best_tag_sequence        

### ALTERNATIVE PER SMOOTHING  --> da sostituire dopo gli else nei primi due step di Viterbi ###
### Initialization ###
    '''
        #versione base, solo transizione    
        viterbi_matrix[tag, 0] = log_transition_P[0, tag]
    
        #vers. 1: P(unk|O)=1
        viterbi_matrix[Oindex, 0] = log_transition_P[0, Oindex]
            
        #vers. 2: P(unk|O)=P(unk|B-MISC)=0.5
        viterbi_matrix[Oindex, 0] = log_transition_P[0, Oindex] * 0.5
        viterbi_matrix[MISCindex, 0] = log_transition_P[0, MISCindex] * 0.5
            
        #vers. 3: uniforme-> P(unk|Ti)= 1/len(tags)
        viterbi_matrix[tag, 0] = log_transition_P[0, tag] * 1/m    
        
        
         #statistiche su parole che compaiono una volta sola
            for t in array_dev_set:
                #viterbi_matrix[tag]
        
            
    '''
    
### Recursion ###
    '''
        #versione base
            for tag in range(2, m):
                max_tr_prob = viterbi_matrix[:, word-1] + log_transition_P[:, tag]
                backpointer[tag, word] = np.argmax(max_tr_prob)
                viterbi_matrix[tag, word] = np.max(max_tr_prob)
    
        #vers. 1: P(unk|O)=1
        max_tr_prob = viterbi_matrix[:, word-1] + log_transition_P[:, Oindex]
        backpointer[Oindex, word] = np.argmax(max_tr_prob)
        viterbi_matrix[Oindex, word] = np.max(max_tr_prob)
        
        #vers. 2: P(unk|O)=P(unk|B-MISC)=0.5    
        max_tr_prob = viterbi_matrix[:, word-1] + log_transition_P[:, Oindex]
        backpointer[Oindex, word] = np.argmax(max_tr_prob)
        viterbi_matrix[Oindex, word] = np.max(max_tr_prob) * 0.5
        max_tr_prob = viterbi_matrix[:, word-1] + log_transition_P[:, MISCindex]
        backpointer[MISCindex, word] = np.argmax(max_tr_prob)
        viterbi_matrix[MISCindex, word] = np.max(max_tr_prob) * 0.5
        
        #vers. 3: uniforme-> P(unk|Ti)= 1/len(tags)
        for tag in range(2, m):
            max_tr_prob = viterbi_matrix[:, word-1] + log_transition_P[:, tag]
            backpointer[tag, word] = np.argmax(max_tr_prob)
            viterbi_matrix[tag, word] = np.max(max_tr_prob) * 1/m
            
        #4: statistiche su parole che compaiono una volta sola

    '''