import numpy as np
import os
file_name = 'train.conllu'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'wikineural_en', file_name)

train = open(file_path, 'r', encoding='utf-8')
#prime_100_righe = file.readlines()[:100]

trasm = {}
tags = []
words = []

for riga in train:
    riga = riga.strip()
    if riga:
        riga = riga.split()
        if riga[1] not in words:
           words.append(riga[1])
        if riga[2] not in tags:
           tags.append(riga[2])
        word = riga[1]
        tag = riga[2]
        trasm[tag][word] += 1

print(tags)
print(words)



#Probabilità di emissione



#Probabilità di transizione


close