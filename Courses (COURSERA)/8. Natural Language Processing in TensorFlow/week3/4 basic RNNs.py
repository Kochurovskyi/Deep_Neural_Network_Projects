import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Input, Bidirectional, GRU, LSTM, Conv1D


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

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
tr_dt, ts_dt = imdb['train'], imdb['test']
tr_st = []
tr_lb = []
ts_st = []
ts_lb = []

for st, lb in tr_dt:
    tr_st.append(str(st.numpy()))
    tr_lb.append(lb.numpy())

for st, lb in ts_dt:
    ts_st.append(str(st.numpy()))
    ts_lb.append(lb.numpy())

tr_lb = np.array(tr_lb)
ts_lb = np.array(ts_lb)

vl_st = ts_st[15000:]
vl_lb = ts_lb[15000:]
ts_st = ts_st[:15000]
ts_lb = ts_lb[:15000]

voc_size = 10000
embedding_dim = 16
max_length = 120

# preprocessing
toke_er = Tokenizer(voc_size, oov_token='<oov>')
toke_er.fit_on_texts(tr_st)
word_index = toke_er.word_index
tr_seq = toke_er.texts_to_sequences(tr_st)
tr_pad = pad_sequences(tr_seq, maxlen=max_length, padding='post', truncating='post')
ts_seq = toke_er.texts_to_sequences(ts_st)
ts_pad = pad_sequences(ts_seq, maxlen=max_length, padding='post', truncating='post')
vl_seq = toke_er.texts_to_sequences(vl_st)
vl_pad = pad_sequences(vl_seq, maxlen=max_length, padding='post', truncating='post')

# model ------------------------------------------------------- Vanila
inpt = Input(shape=(max_length,))
X = Embedding(voc_size, embedding_dim, input_length=max_length)(inpt)
X = GlobalAveragePooling1D()(X)
X = Dense(8, activation='relu')(X)
outpt = Dense(1, activation='sigmoid')(X)
model = Model(inpt, outpt)
# compile-fit-evalueate
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(tr_pad, tr_lb, epochs=3, validation_data=(vl_pad, vl_lb), verbose=2)
model.evaluate(ts_pad, ts_lb, verbose=2)
hs_plot(history)

# model ---------------------------------------------------------- Conv1D
inpt = Input(shape=(max_length,))
X = Embedding(voc_size, embedding_dim, input_length=max_length)(inpt)
X = Conv1D(128, 5, activation='relu')(X)
X = GlobalAveragePooling1D()(X)
X = Dense(8, activation='relu')(X)
outpt = Dense(1, activation='sigmoid')(X)
model = Model(inpt, outpt)
# compile-fit-evalueate
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(tr_pad, tr_lb, epochs=3, validation_data=(vl_pad, vl_lb), verbose=2)
model.evaluate(ts_pad, ts_lb, verbose=2)
hs_plot(history)


# model ---------------------------------------------------------- LSTM
inpt = Input(shape=(max_length,))
X = Embedding(voc_size, embedding_dim, input_length=max_length)(inpt)
X = Bidirectional(LSTM(32))(X)
X = Dense(8, activation='relu')(X)
outpt = Dense(1, activation='sigmoid')(X)
model = Model(inpt, outpt)
# compile-fit-evalueate
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(tr_pad, tr_lb, epochs=3, validation_data=(vl_pad, vl_lb), verbose=2)
model.evaluate(ts_pad, ts_lb, verbose=2)
hs_plot(history)


# model ---------------------------------------------------------- GRU
inpt = Input(shape=(max_length,))
X = Embedding(voc_size, embedding_dim, input_length=max_length)(inpt)
X = Bidirectional(GRU(32))(X)
X = Dense(8, activation='relu')(X)
outpt = Dense(1, activation='sigmoid')(X)
model = Model(inpt, outpt)
# compile-fit-evalueate
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(tr_pad, tr_lb, epochs=3, validation_data=(vl_pad, vl_lb), verbose=2)
model.evaluate(ts_pad, ts_lb, verbose=2)
hs_plot(history)











