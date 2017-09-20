from nltk.tokenize import TweetTokenizer
import pymorphy2

NUMB_OF_FEATURES = 6

twtk = TweetTokenizer()
morph = pymorphy2.MorphAnalyzer(lang='uk')

POS_l = ["Unk","NOUN", "ADJF", "ADJS","COMP","VERB","INFN","PRTS","GRND","NUMR","ADVB","NPRO","PRED","PREP","CONJ","PRCL","INTJ",]
cases_l = ['Unk', 'nomn','gent',	'datv',	'accs',	'ablt',	'loct',	'voct',	'gen2',	'acc2',	'loc2']
POS = {}
cases = {}
for i in range(len(POS_l)):
    POS[POS_l[i]] = i
for i in range(len(cases_l)):
    cases[cases_l[i]] = i



def extract_features(sent, voc, entities, lemmatize=False, nwords_to_consider=3):  #sent= list of words
    f_vectors = []
    labels = [0] * len(sent)
    for i in range(len(sent)):
        f_vector = []
        word = sent[i]
        word_morph = morph.parse(sent[i])[0]
        if lemmatize:
            word = word_morph.normal_form

        if word in entities:
            labels[i] = 1

        f_vector.append(voc[word])    #in voc
        f_vector.append(int(sent[i][0].isupper()))  #is capital
        try:
            f_vector.append(POS[word_morph.tag.POS]) #part of speech
        except KeyError:
            f_vector.append(POS["Unk"])
        try:
            f_vector.append(cases[word_morph.tag.case]) #case of the word
        except KeyError:
            f_vector.append(cases["Unk"])
        f_vector.append(len(word))                   #length
        f_vector.append(i)      #position
        f_vectors.append(f_vector)
    features = []
    f_vectors.insert(0, [0] * NUMB_OF_FEATURES)
    f_vectors.append([0] * NUMB_OF_FEATURES)
    for i in range(1, len(f_vectors) - 1):
       features.append(f_vectors[i - 1] + f_vectors[i] + f_vectors[i + 1])
    assert(len(features) == len(labels))
    return features, labels










