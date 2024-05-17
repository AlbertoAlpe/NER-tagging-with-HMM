import numpy as np
import os
file_name = 'train.conllu'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'wikineural_en', file_name)

train = open(file_path, 'r', encoding='utf-8')
prime_100_righe = train.readlines()[:34]

emission_P = {}
transition_P = np.zeros((9, 9), dtype=int)
tags = []
words = []
word = 0
tag = 0
tag_prec = None
tpi = 0

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
        tag_prec = None         

print(transition_P)





#Probabilità di emissione



#Probabilità di transizione


