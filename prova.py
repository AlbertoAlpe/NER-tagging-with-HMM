import pickle 
import numpy as np 
import csv 
import os 
file_name = 'train.conllu' 
current_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(current_dir, 'wikineural_en', file_name) 
 
train = open(file_path, 'r', encoding='utf-8') 
prime_100_righe = train.readlines() 
 
 
emission_P = [[], []]     #matrice iniziale composta da due righe vuote 
transition_P = [[0, 0], [0, 0]]
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
            viterbi_matrix[s, 0] = transition_P[0, s] 
 
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
 
def read_conllu(file_path): 
    sentences = [] 
    current_sentence = [] 
    with open(file_path, 'r', encoding='utf-8') as f: 
        for line in f: 
            line = line.strip() 
            if line: 
                parts = line.split('\t') 
                if len(parts) == 3: 
                    current_sentence.append((parts[1], parts[2])) 
            else: 
                if current_sentence: 
                    sentences.append(current_sentence) 
                    current_sentence = [] 
        if current_sentence:  # Add last sentence if file doesn't end with newline 
            sentences.append(current_sentence) 
    return sentences 
 
def evaluate(test_sentences, emission_P, transition_P, tags, words): 
    correct = 0 
    total = 0 
    true_positives = 0 
    false_positives = 0 
    false_negatives = 0 
     
    for sentence in test_sentences: 
        sequence = [word for word, true_tag in sentence] 
        true_tags = [true_tag for word, true_tag in sentence] 
        predicted_tags = viterbi(sequence, emission_P, transition_P, tags, words) 
         
        for true_tag, predicted_tag in zip(true_tags, predicted_tags): 
            if true_tag == predicted_tag: 
                correct += 1 
            total += 1 
 
            if predicted_tag.startswith("B-") or predicted_tag.startswith("I-"): 
                if predicted_tag == true_tag: 
                    true_positives += 1 
                else: 
                    false_positives += 1
            if true_tag.startswith("B-") or true_tag.startswith("I-"): 
                if true_tag != predicted_tag: 
                    false_negatives += 1 
 
    accuracy = correct / total 
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0 
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0 
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 
 
    return accuracy, precision, recall, f1_score 
 
 
 
 
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
        #in questo caso la riga è vuota => il tag è END, si fanno i calcoli necessari e poi il tag_prec diventa START 
        tpi = tags.index(tag_prec) 
        transition_P[tpi][1] += 1  #indice 1 perchè END è il secondo elemento dell'array tags 
        tag_prec = "START"           
 
# Calcolo delle probabilità di emissione 
emission_P = np.array(emission_P, dtype=float) 
row_sums = np.sum(emission_P, axis=1, keepdims=True) 
non_zero_rows = row_sums.squeeze() != 0 
emission_P[non_zero_rows] = emission_P[non_zero_rows] / row_sums[non_zero_rows] 
 
# Calcolo delle probabilità di transizione 
transition_P = np.array(transition_P, dtype=float) 
row_sums = np.sum(transition_P, axis=1, keepdims=True) 
non_zero_rows = row_sums.squeeze() != 0 
transition_P[non_zero_rows] = transition_P[non_zero_rows] / row_sums[non_zero_rows] 
 
 
print(emission_P) 
print(transition_P) 
 
sequence = ["This", "division", "also", "contains", "the"] 
 
print("Input sequence:", sequence) 
predicted_tags = viterbi(sequence, emission_P, transition_P, tags, words) 
print("Predicted tags:", predicted_tags) 
 
file_name = 'test.conllu' 
current_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(current_dir, 'wikineural_en', file_name) 
test_sentences = read_conllu(file_path) 
 
accuracy, precision, recall, f1_score = evaluate(test_sentences, emission_P, transition_P, tags, words) 
 
print(f"Accuracy: {accuracy:.4f}") 
print(f"Precision: {precision:.4f}") 
print(f"Recall: {recall:.4f}") 
print(f"F1 Score: {f1_score:.4f}") 
 
 
 
 
 
''' 
row_sums = np.sum(emission_P, axis=1, keepdims=True)       
non_zero_rows = row_sums != 0 
emission_P[non_zero_rows] = emission_P[non_zero_rows] / row_sums[non_zero_rows] 
 
row_sums = np.sum(transition_P, axis=1, keepdims=True)       
non_zero_rows = row_sums != 0 
transition_P[non_zero_rows] = transition_P[non_zero_rows] / row_sums[non_zero_rows] 
 
 
###PROBABILITA' DI EMISSIONE E TRANSIZIONE### 
#Una volta ottenute le matrici con le occorrenze di emissione e di transizione, 
#in ogni riga avremo tutte le occorrenze di un tag divise per ogni parola, quindi le sommiamo in tagTot 
#la variabile tagTot verrà usata nei due cicli for incapsulati nel primo,  
#per calcolare le probabilità di emissione e transizione 
for r, row in enumerate(emission_P): 
   tagTot = 0 
   for i in row: 
      tagTot += i 
   if (tagTot>0):  
      for emissCol, value in enumerate(row):     #CALCOLO PROB. EMISSIONE 
         emission_P[r][emissCol] = value/tagTot     
      for transCol, val in enumerate(transition_P[r]):        #CALCOLO PROB. EMISSIONE    
         transition_P[r][transCol] = val/tagTot   
              
 
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
'''