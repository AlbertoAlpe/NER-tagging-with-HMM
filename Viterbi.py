import numpy as np

# Algoritmo di Viterbi
def viterbi(sequence, emission_P, transition_P, tags, words):
    n = len(sequence)
    m = len(tags)
    viterbi_matrix = np.zeros((m, n))
    backpointer = np.zeros((m, n), dtype=int)

    # Initialization step
    for s in range(2, m):  # Ignoro START e END nei calcoli
        if sequence[0] in words:
            viterbi_matrix[s, 0] = transition_P[0, s] * emission_P[s, words.index(sequence[0])]
        else:
            viterbi_matrix[s, 0] = transition_P[0, s]   #se la parola non è nel corpus

    # Recursion step
    for t in range(1, n):
        for s in range(2, m):
            max_tr_prob = viterbi_matrix[:, t-1] * transition_P[:, s]
            backpointer[s, t] = np.argmax(max_tr_prob)
            if sequence[t] in words:
                viterbi_matrix[s, t] = np.max(max_tr_prob) * emission_P[s, words.index(sequence[t])]
            else:
                viterbi_matrix[s, t] = np.max(max_tr_prob)

    # Termination step
    max_prob = viterbi_matrix[:, n-1] * transition_P[:, 1]
    best_path_pointer = np.argmax(max_prob)
    best_path = [best_path_pointer]

    for t in range(n-1, 0, -1):
        best_path_pointer = backpointer[best_path_pointer, t]
        best_path.insert(0, best_path_pointer)

    best_tag_sequence = [tags[i] for i in best_path]

    return best_tag_sequence        
