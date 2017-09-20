import nltk.data
from nltk.tokenize import TweetTokenizer
import csv, pymorphy2

twtk = TweetTokenizer(preserve_case=False, strip_handles=True)
morph = pymorphy2.MorphAnalyzer(lang='uk')


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = []
locations = set()

with open("../data/Боргардт_-_Аналітична_історія_України.txt", "r", encoding ='utf') as file:
    data = file.read()
    #print(data)
    data = data.decode("utf-8")
    sentences.extend(tokenizer.tokenize(data))

with open("../dictiponaries/locations_analytistoriya.txt", "r") as file:
    locations.update([word.strip() for word in file.readlines()])

dict = {}

for sent in sentences:
    label = ""
    for word in [morph.parse(w)[0].normal_form for w in twtk.tokenize(sent) if w.isalpha()]:
        if word.strip() in locations:
            label += "1"
        else:
            label += "0"
    dict[sent] = label

with open("../data/labeled_Боргардт_-_Аналітична_історія_України.csv", "w") as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerow(["sentence", "NEL_label"])
    for key in dict:
        writer.writerow([key, dict[key]])
