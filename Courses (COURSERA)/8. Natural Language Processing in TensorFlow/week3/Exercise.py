import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import csv
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.keras.models import Model

def hs_plot(history):
    ''' history plot '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10,3))
    ax[0].plot(epochs, loss, color='red', label='Training loss')
    ax[0].plot(epochs, val_loss, color='deeppink', label='Validation loss')
    ax[1].plot(epochs, accuracy, color='green', label='accuracy')
    ax[1].plot(epochs, val_accuracy, color='lime', label='val_accurace')
    ax[0].set_title('Training and validation loss & Metrics')
    ax[1].set_title('Training and validation Metrics(Accuracy)')
    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Metrics')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    plt.savefig('hist.png')
    plt.show()

embedding_dim = 100
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<oov>'
training_size = 180000
test_portion = 0.2
corpus = []

num_sentences = 0
with open('training_cleaned.csv', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        list_item=[]
        list_item.append(row[5])
        this_label = row[0]
        if this_label == '0':
            list_item.append(0)
        else:
            list_item.append(1)
        num_sentences = num_sentences + 1
        corpus.append(list_item)

print(num_sentences)
print(len(corpus))
print(corpus[1])
print('--------------------')

sentences = []
labels = []
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size = len(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences = np.array(padded[0:int(split/2)])
valid_sequences = np.array(padded[int(split/2):split])
training_sequences = np.array(padded[split:training_size])

test_labels = np.array(labels[0:int(split/2)])
valid_labels = np.array(labels[int(split/2):split])
training_labels = np.array(labels[split:training_size])



print(vocab_size)
print(word_index['i'])
print(len(training_sequences))
print(len(test_sequences))
print('-------------------------')

embeddings_index = {};
with open('glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector
print(len(embeddings_matrix))
print('----------------------')

# model
inpt = Input(shape=(max_length,))
X = Embedding(vocab_size+1,
              embedding_dim,
              input_length=max_length,
              weights=[embeddings_matrix],
              trainable=False)(inpt)
X = Bidirectional(LSTM(64))(X)
#X = GlobalAveragePooling1D()(X)
X = Dense(16, activation='relu')(X)
outp = Dense(1, activation='sigmoid')(X)
model = Model(inpt, outp)

# compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(training_sequences, training_labels,
                    epochs=50, verbose=2,
                    validation_data=(valid_sequences, valid_labels))
print('----------')
model.evaluate(test_sequences, test_labels, verbose=2)
hs_plot(history)
