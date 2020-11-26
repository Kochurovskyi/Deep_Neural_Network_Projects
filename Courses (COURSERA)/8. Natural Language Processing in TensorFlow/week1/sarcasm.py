import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

datastore = []
for line in open("sarcasm.json", 'r'):
    datastore.append(json.loads(line))
sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
print(len(sentences), len(labels), len(urls))
print(sentences[0], labels[0])

tokenizer = Tokenizer(oov_token='<OOV')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))
print(word_index)

seque = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(seque, padding='post')
print(seque[1])
print(padded.shape)