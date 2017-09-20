from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.metrics import f1_score, classification_report
from sklearn.grid_search import GridSearchCV
import os
from nltk.tokenize import TweetTokenizer
import pymorphy2
import nltk.data
import numpy as np
from feature_engineering import extract_features



numb_of_words_dict = 5000


twtk = TweetTokenizer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
morph = pymorphy2.MorphAnalyzer(lang='uk')

to_open = ['/home/dzvinka/PycharmProjects/eleks_ds/data/VelikaIstoriyaYkrajni_1412180965.txt']
toponims = set()




with open('../dictionaries/toponims.txt', 'r') as file:
    toponims.update([w.strip() for w in file.readlines()])

print(len(toponims))



for file in os.listdir("/home/dzvinka/PycharmProjects/eleks_ds/data/lang-uk-data/data"):
    if file.endswith(".txt"):
        to_open.append(os.path.join("../data/lang-uk-data/data/", file))

X = []
Y = []
sentences = []
processed_sentences = []
dictionary = set()

print('Processing data...')
for filename in to_open:
    with open(filename, 'r') as file:
        data = file.read()
        sentences.extend(tokenizer.tokenize(data))

for sentence in sentences:
    words = [word for word in twtk.tokenize(sentence) if word.isalpha()]
    dictionary.update([morph.parse(word)[0].normal_form for word in words])
    processed_sentences.append(words)




#vocabulary =  nltk.FreqDist(dictionary)
#vocabulary = [word[0] for word in vocabulary.most_common(numb_of_words_dict)]
dict = {}
i = 0
for word in dictionary:
    dict[word] = i
    i += 1



for sent in processed_sentences:
    features, labels = extract_features(sent, dict, toponims, lemmatize=True)
    X.extend(features)
    Y.extend(labels)


X = np.array(X)
Y = np.array(Y)
print(Y.reshape((Y.shape[0], 1)))
print(X[:3])





data_train, data_test, labels_train, labels_test = \
    train_test_split(X, Y,
                     test_size=0.2, random_state=42, stratify=Y)


print('Training the model...')

clf = RandomForestClassifier(n_jobs=-1, n_estimators=25)

parameters_grid = {'min_samples_leaf': [2, 25],
                   'min_samples_split': [10, 20, 30]}
                   #"rf__n_estimators":  [10, 15, 20, 25]}

clf_grid = GridSearchCV(estimator=clf, param_grid=parameters_grid)


clf_grid.fit(data_train, labels_train)

print("Best params: "  + str(clf_grid.best_params_))

y_predict = clf_grid.predict(data_test)

print(f1_score(labels_test, y_predict))
print(classification_report(labels_test, y_predict))

f = open('../models/random_forest.pickle', 'wb')
pickle.dump(clf_grid.best_estimator_, f)
f.close()
