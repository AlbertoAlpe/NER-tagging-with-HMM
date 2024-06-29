import numpy as np

tag_sequence = []

# baseline più semplice, a ogni parola assegna il tag più frequente (o MISC se sconosciuta)
def easy_baseline(sequence, emission_P, tags, words):
    tag_sequence = []
    for word in sequence:
        if word in words:
            word_index = words.index(word)
            best_tag = np.argmax(emission_P[:, word_index])
            tag_sequence.append(tags[best_tag])
        else:
            tag_sequence.append('MISC')
        
    return tag_sequence