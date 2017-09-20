from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
import pickle
from keras.layers import Bidirectional
import os
from sklearn.metrics import f1_score, classification_report


N_SENTENCES = 42255
MAX_SENTENCE_LENGTH = 11
EMBEDDING_VECTOR_DIM = 400
VOCABULARY_SIZE = 7679 #?????

from nltk.tokenize import TweetTokenizer
import pymorphy2
import nltk.data

twtk = TweetTokenizer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
morph = pymorphy2.MorphAnalyzer(lang='uk')



print("Loading word vectors...")
w2v_model = Word2Vec.load("../vectors/300features_20minwords_10context")
vectors = w2v_model.wv
toponims = set()


with open("../dictionaries/toponims.txt", "r") as file:
    toponims.update([w.strip() for w in file.readlines()])



to_open = ["/home/dzvinka/PycharmProjects/eleks_ds/data/VelikaIstoriyaYkrajni_1412180965.txt"]
sentences = []
#processed_sentences = []

for file in os.listdir("/home/dzvinka/PycharmProjects/eleks_ds/data/lang-uk-data/data"):
    if file.endswith(".txt"):
       to_open.append(os.path.join("../data/lang-uk-data/data/", file))


for filename in to_open:
    with open(filename, "r") as file:
        data = file.read()
        sentences.extend(tokenizer.tokenize(data))

X = np.zeros((N_SENTENCES, MAX_SENTENCE_LENGTH, EMBEDDING_VECTOR_DIM))
Y = []
#Y = np.zeros((N_SENTENCES, MAX_SENTENCE_LENGTH, 1))
# embedding_matrix = np.zeros((VOCABULARY_SIZE, EMBEDDING_VECTOR_DIM))
#
# i = 0
# for word in vectors.vocab:
#     embedding_matrix[i] = vectors[word]
#     i += 1
#
for i in range(len(sentences)):
    words = [morph.parse(word)[0].normal_form for word in twtk.tokenize(sentences[i]) if word.isalpha()]
    labels = [0] * MAX_SENTENCE_LENGTH
    j = 0
    for word in words:
        if j >= MAX_SENTENCE_LENGTH:
            break
        if word in vectors.vocab:
            X[i, j] = vectors[word]
            if word in toponims:
                labels[j] = 1
            j += 1

    Y.append(labels)

#
Y = np.array(Y)
Y = Y.reshape(N_SENTENCES, MAX_SENTENCE_LENGTH, 1)
#
#
# print(X.shape)
# print(Y.shape)
#
print(X[0, 0])
print(Y[5])



data_train, data_test, labels_train, labels_test = \
    train_test_split(X, Y,
                     test_size=0.2, random_state=42)

print("Building model...")
model = Sequential()


model.add(Bidirectional(LSTM(100,  return_sequences=True), input_shape=(MAX_SENTENCE_LENGTH, EMBEDDING_VECTOR_DIM),))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("Training model...")
model.fit(data_train, labels_train, nb_epoch=1, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(data_test, labels_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_predict = model.predict(data_test)
labels_test_r = labels_test.reshape(labels_test.shape[0] * labels_test.shape[1])
y_predict = y_predict.reshape(y_predict.shape[0] * y_predict.shape[1])
pred = []
print(y_predict.shape)
print(y_predict)
for i in range(y_predict.shape[0]):
    if y_predict[i] >= 0.5:
        pred.append(1)
    else:
        pred.append(0)

print(f1_score(labels_test_r, pred))
print(classification_report(labels_test_r, pred))


# f = open('../models/lstm_my_embeddings.pickle', 'wb')
# pickle.dump(model, f)
# f.close()



