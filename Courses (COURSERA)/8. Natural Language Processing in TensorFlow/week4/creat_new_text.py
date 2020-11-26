import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import csv
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.keras.models import Model

def hs_plot(history):
    ''' history plot '''
    loss = history.history['loss']
    #val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    #val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].plot(epochs, loss, color='red', label='Training loss')
    #ax[0].plot(epochs, val_loss, color='deeppink', label='Validation loss')
    ax[1].plot(epochs, accuracy, color='green', label='accuracy')
    #ax[1].plot(epochs, val_accuracy, color='lime', label='val_accurace')
    ax[0].set_title('Training and validation loss')
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

tok_er = Tokenizer()
data = "In the town of Athy one Jeremy Lanigan \n " \
       "Battered away til he hadnt a pound. \n" \
       "His father died and made him a man again \n " \
       "Left him a farm and ten acres of ground. \n" \
       "He gave a grand party for friends and relations \n" \
       "Who didnt forget him when come to the wall, \n" \
       "And if youll but listen Ill make your eyes glisten \n" \
       "Of the rows and the ructions of Lanigans Ball. \n" \
       "Myself to be sure got free invitation, \n" \
       "For all the nice girls and boys I might ask, \n" \
       "And just in a minute both friends and relations \n" \
       "Were dancing round merry as bees round a cask. \n" \
       "Judy ODaly, that nice little milliner, \n" \
       "She tipped me a wink for to give her a call, \n" \
       "And I soon arrived with Peggy McGilligan \n" \
       "Just in time for Lanigans Ball. \n" \
       "There were lashings of punch and wine for the ladies, \n" \
       "Potatoes and cakes; there was bacon and tea, \n" \
       "There were the Nolans, Dolans, OGradys \n" \
       "Courting the girls and dancing away. \n" \
       "Songs they went round as plenty as water, \n" \
       "The harp that once sounded in Taras old hall,\n" \
       "Sweet Nelly Gray and The Rat Catchers Daughter,\n" \
       "All singing together at Lanigans Ball. \n" \
       "They were doing all kinds of nonsensical polkas \n" \
       "All round the room in a whirligig. \n" \
       "Julia and I, we banished their nonsense \n" \
       "And tipped them the twist of a reel and a jig. \n" \
       "Ach mavrone, how the girls got all mad at me \n" \
       "Danced til youd think the ceiling would fall. \n" \
       "For I spent three weeks at Brooks Academy \n" \
       "Learning new steps for Lanigans Ball. \n" \
       "Three long weeks I spent up in Dublin, \n" \
       "Three long weeks to learn nothing at all,\n " \
       "Three long weeks I spent up in Dublin, \n" \
       "Learning new steps for Lanigans Ball. \n" \
       "She stepped out and I stepped in again, \n" \
       "I stepped out and she stepped in again, \n" \
       "She stepped out and I stepped in again, \n" \
       "Learning new steps for Lanigans Ball. \n" \
       "Boys were all merry and the girls they were hearty \n" \
       "And danced all around in couples and groups, \n" \
       "Til an accident happened, young Terrance McCarthy \n" \
       "Put his right leg through miss Finnertys hoops. \n" \
       "Poor creature fainted and cried Meelia murther, \n" \
       "Called for her brothers and gathered them all. \n" \
       "Carmody swore that hed go no further \n" \
       "Til he had satisfaction at Lanigans Ball. \n" \
       "In the midst of the row miss Kerrigan fainted, \n" \
       "Her cheeks at the same time as red as a rose. \n" \
       "Some of the lads declared she was painted, \n" \
       "She took a small drop too much, I suppose. \n" \
       "Her sweetheart, Ned Morgan, so powerful and able, \n" \
       "When he saw his fair colleen stretched out by the wall, \n" \
       "Tore the left leg from under the table \n" \
       "And smashed all the Chaneys at Lanigans Ball. \n" \
       "Boys, oh boys, twas then there were runctions. \n" \
       "Myself got a lick from big Phelim McHugh. \n" \
       "I soon replied to his introduction \n" \
       "And kicked up a terrible hullabaloo. \n" \
       "Old Casey, the piper, was near being strangled. \n" \
       "They squeezed up his pipes, bellows, chanters and all. \n" \
       "The girls, in their ribbons, they got all entangled \n" \
       "And that put an end to Lanigans Ball."

corpus = data.lower().split('\n')
tok_er.fit_on_texts(corpus)
word_index = tok_er.word_index
total_words = len(word_index) + 1
#print(word_index)
print(total_words)

input_sequences = []
for line in corpus:
    token_list = tok_er.texts_to_sequences([line])[0]
    for i in range (1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

inp_len = max_sequence_len-1
inp = Input(shape=(inp_len,))
X = Embedding(total_words, 64, input_length=inp_len)(inp)
X = Bidirectional(LSTM(20))(X)
out = Dense(total_words, activation='softmax')(X)
model = Model(inp, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(xs, ys, epochs=300, verbose=2)

hs_plot(history)

seed_text = "Laurence went to dublin"
next_words = 100

for _ in range(next_words):
    token_list = tok_er.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    output_word = ""
    for word, index in tok_er.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)