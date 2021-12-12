import numpy as np
from tensorflow.keras.layers import (
    Dense, LSTM
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import os
import sys

def load_doc(path):
    with open(path, mode='r') as f:
        doc = f.read()
    # select only arabic and english columns
    lines = doc.split('\n')
    new_lines = list()
    for line in lines:
        line = line.split('\t')
        if len(line) > 2:
            new_lines.append('\t'.join(line[:2]))
    return '\n'.join(new_lines)


DATASET_FILE = './ara.txt'

if not os.path.exists(DATASET_FILE):
    print(f"'{DATASET_FILE}' doesn't exist.")
    sys.exit(1)

doc = load_doc(DATASET_FILE)

print(doc)