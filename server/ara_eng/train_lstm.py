import numpy as np
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, RepeatVector, TimeDistributed
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import string
import os
import sys
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from pick import pick

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

def print_sample(pairs, n_samples, root_path):
    file_name = os.path.join(root_path, 'ara_eng', 'sample.txt')
    lines = list()
    for i in range(n_samples):
       lines.append(' => '.join(pairs[i]))
    with open(file_name, mode='wt') as f:
        f.writelines('\n'.join(lines))
    print(f'Open {file_name}')

def save_pairs(pairs, file):
    np.savez_compressed(file, pairs)
    print(f"Open {file}")

def load_cleaned_dataset(file, n_samples):
    raw_data = np.load(file)['arr_0']
    print("Dataset: ", raw_data.shape)
    if not n_samples:
        n_samples = len(raw_data)
    dataset = raw_data[:n_samples, :]
    np.random.shuffle(dataset)
    train_size = int(n_samples * 0.9)
    train = dataset[:train_size, :] 
    test = dataset[train_size:, :]
    return dataset, train, test
    
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
	return max(len(line.split()) for line in lines)

def encode_sequence(tokenizer,length, lines):
    sequences = tokenizer.texts_to_sequences(lines)
    sequences = pad_sequences(sequences, maxlen=length, padding='post')
    return sequences
def encode_output(sequences, vocab_size):
    y_list = list()
    for sequence in sequences:
        sequence = to_categorical(sequence, num_classes=vocab_size)
        y_list.append(sequence)
    y = np.array(y_list)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

def define_model(src_vocb, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocb, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model

def learning_curve(history, epochs, path):
    _, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes = axes.flatten()

    axes[0].plot(history['loss'], label="Traning Loss")
    axes[0].plot(history['val_loss'], label="Validation Loss")
    axes[0].set_title("Losses")
    axes[0].legend()
    axes[1].plot(history['accuracy'], label="Traning Accuracy")
    axes[1].plot(history['val_accuracy'], label="Validation Accuracy")
    axes[1].set_title("Accuracy")
    plt.legend()
    path = os.path.join(path, 'ara_eng', f'history-{epochs}.png')
    plt.savefig(path)

def format_output_sequence(tokenizer, sequences):
    # Sequences: (n_examples, timesteps, one hot encoding)
    decoded = list()
    for sequence in sequences:
        sequence = [np.argmax(timestep) for timestep in sequence]
        decoded.append(sequence)
    return np.array(decoded)

def save_test_sample(root_path, X, y_true, y_pred, src_tokenizer, tar_tokenizer):
    X = src_tokenizer.sequences_to_texts(X)
    y_true = tar_tokenizer.sequences_to_texts(y_true)
    y_pred = tar_tokenizer.sequences_to_texts(y_pred)
    lines = list()
    for i in range(len(X)):
        lines.append(' => '.join([X[i], y_true[i], y_pred[i]]))
    file_path = os.path.join(root_path, 'ara_eng', 'test_sample.txt')
    with open(file_path, mode='wt') as f:
        f.writelines('\n'.join(lines))
    

def test_model(test_dir,root_path, testX, testY, src_tokenizer, tar_tokenizer):
    models = [model for model in os.listdir(test_dir) if model.endswith('.h5')]
    if len(models) == 0:
        print("You should train a model first.\n try `flask nmt_lstm --train=True --epochs=100`")
        sys.exit(0)
    model_name = pick(models, 'Choose a model.')[0]
    model_path = os.path.join(test_dir, model_name)
    model = load_model(model_path)
    y_preds = model.predict(testX)
    y_preds = format_output_sequence(tar_tokenizer, y_preds)
    y_true = format_output_sequence(tar_tokenizer, testY)
    save_test_sample(root_path, testX, y_true, y_preds,src_tokenizer, tar_tokenizer )

instance_path = './instance/datasets'
DATASET_FILE = os.path.join(instance_path, 'ara.txt')
CLEANED_FILE = os.path.join(instance_path, 'cleaned.npz')
n_sentences = 15000

if not os.path.exists(DATASET_FILE):
    print(f"'{DATASET_FILE}' doesn't exist.")
    sys.exit(1)

if not os.path.exists(CLEANED_FILE):
    print(f"'{CLEANED_FILE}' doesn't exist.\nWill create it...")
    doc = load_doc(DATASET_FILE)
    pairs = to_pair(doc)
    cleaned = clean_pairs(pairs)
    cleaned = remove_empty_pairs(cleaned)
    save_pairs(cleaned, CLEANED_FILE)

def encode_data(dataset):
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1 # +1 for unknow words
    eng_length = max_length(dataset[:, 0])
    print("English Vocab. Size: %s" % (eng_vocab_size))
    print("Max length for english sentences: %s" % (eng_length))
    ara_tokenizer = create_tokenizer(dataset[:, 1])
    ara_vocab_size = len(ara_tokenizer.word_index) + 1 
    ara_length = max_length(dataset[:, 1])
    print("Arabic Vocab. Size: %s" % (ara_vocab_size))
    print("Max length for Arabic sentences: %s" % (ara_length))

    return (
        ara_tokenizer,
        ara_length,
        ara_vocab_size
    ), (
        eng_tokenizer,
        eng_length,
        eng_vocab_size
    )

def split_data(ara_tokenizer, ara_length, eng_tokenizer, eng_length, eng_vocab_size, train ,test):
    # X: Arabic, Y: English
    X_idx = 1
    y_idx = 0
    trainX = encode_sequence(ara_tokenizer, ara_length, train[:, X_idx])
    trainY = encode_sequence(eng_tokenizer, eng_length, train[:, y_idx])
    trainY = encode_output(trainY, eng_vocab_size)
    testX = encode_sequence(ara_tokenizer, ara_length, test[:, X_idx])
    testY = encode_sequence(eng_tokenizer, eng_length, test[:, y_idx])
    testY = encode_output(testY, eng_vocab_size)

    print("Input: ", trainX.shape)
    print("Output: ", trainY.shape)
    return trainX, trainY, testX, testY


def train_lstm(summary, sample, train, epochs, save, test, instance_path, root_path,):
    dataset, train_set, test_set = load_cleaned_dataset(CLEANED_FILE, n_sentences)
    (ara_tokenizer, ara_length, ara_vocab_size), (eng_tokenizer, eng_length, eng_vocab_size) = encode_data(dataset)
    trainX, trainY, testX, testY = split_data(ara_tokenizer, ara_length, eng_tokenizer, eng_length, eng_vocab_size, train_set, test_set)


    model = define_model(
        src_vocb=ara_vocab_size, 
        tar_vocab=eng_vocab_size, 
        src_timesteps=ara_length, 
        tar_timesteps=eng_length,
        n_units=256
    )
    if summary:
        print(model.summary())
    if sample:
        print_sample(dataset, 1000, root_path)
    if train:
        history = model.fit(
            trainX, trainY,
            epochs=epochs, batch_size=64, validation_data=(testX, testY)
        ).history

        learning_curve(history, epochs, root_path)

        if save == 'local':
            model_path = os.path.join(root_path, 'ara_eng', f'lstm_{epochs}.h5')
        elif save == 'replace':
            models_path = os.path.join(instance_path, 'models')
            if not models_path:
                os.mkdir(models_path)
            model_path = os.path.join(models_path, f'lstm_{epochs}.h5')
        
        model.save(model_path)
        # model.
    if test:
        test_model(
            os.path.join(root_path, 'ara_eng'), root_path,
            trainX[:10], trainY[:10], 
            tar_tokenizer=eng_tokenizer,
            src_tokenizer=ara_tokenizer
        )

