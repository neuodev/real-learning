import numpy as np
from tensorflow.keras.layers import (
    Dense, LSTM
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import string
import os
import sys
import re

def load_doc(path):
    with open(path, mode='rt', encoding='utf-8') as f:
        doc = f.read()
    return doc

def to_pair(doc: str):
    lines = doc.strip().split('\n')
    pairs = []
    for line in lines:
        line = line.split('\t')
        if len(line) > 2:
            pairs.append(line[:2])
    return pairs

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
        
		cleaned.append(clean_pair)
	return np.array(cleaned)

def remove_empty_pairs(pairs):
    new_pairs = list()
    for pair in pairs:
        if pair[0] and pair[1]:
            new_pairs.append(pair)
    return new_pairs

def print_sample(pairs, n_samples):
    file_name = './sample.txt'
    lines = list()
    for i in range(n_samples):
       lines.append(' => '.join(pairs[i]))
    with open(file_name, mode='wt') as f:
        f.writelines('\n'.join(lines))
    print(f'Open {file_name}')

DATASET_FILE = './ara.txt'
if not os.path.exists(DATASET_FILE):
    print(f"'{DATASET_FILE}' doesn't exist.")
    sys.exit(1)

doc = load_doc(DATASET_FILE)
pairs = to_pair(doc)
cleaned = clean_pairs(pairs)
cleaned = remove_empty_pairs(cleaned)

if '--sample' in sys.argv:
    print_sample(cleaned, 1000)