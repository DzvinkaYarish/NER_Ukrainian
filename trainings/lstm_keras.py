from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Embedding
import os

N_SENTENCES = 42255
MAX_SENTENCE_LENGTH = 15
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
embedding_matrix = np.zeros((VOCABULARY_SIZE, EMBEDDING_VECTOR_DIM))

i = 0
for word in vectors.vocab:
    embedding_matrix[i] = vectors[word]
    i += 1

for i in range(len(sentences)):
    words = [morph.parse(word)[0].normal_form for word in twtk.tokenize(sentences[i]) if word.isalpha()]
    j = 0
    labels = [0] * MAX_SENTENCE_LENGTH
    for word in words:
        if j > MAX_SENTENCE_LENGTH - 1:
            break
        if word in vectors.vocab:
            X[i,j] = vectors[word]
            if word in toponims:
                labels[j] = 1
            j += 1

    Y.extend(labels)

Y = np.array(Y)
Y = Y.reshape(N_SENTENCES, MAX_SENTENCE_LENGTH, 1)


print(X.shape)
print(Y.shape)


embedding_layer = Embedding(
        VOCABULARY_SIZE, # how many words are mapped into vectors
        EMBEDDING_VECTOR_DIM, # size of output vector dimension (we use pre-trained model with vectors of 300 values)
        weights=[embedding_matrix], # we initialize weight from pre-trained model
        input_length=MAX_SENTENCE_LENGTH, # how many words in the sentence we process
        trainable=False) # we will not update this layer

data_train, data_test, labels_train, labels_test = \
    train_test_split(X, Y,
                     test_size=0.1, random_state=42)

print("Building model...")
model = Sequential()
model.add(embedding_layer)

model.add(LSTM(100))
#model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print("Training model...")
model.fit([data_train], labels_train, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(data_test, labels_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))





