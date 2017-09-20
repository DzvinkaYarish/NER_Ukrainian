import os
from nltk.tokenize import TweetTokenizer
import pymorphy2

twtk = TweetTokenizer()
morph = pymorphy2.MorphAnalyzer(lang='uk')


to_open = []
locations = []
for file in os.listdir("/home/dzvinka/PycharmProjects/eleks_ds/data/lang-uk-data/data"):
    if file.endswith(".tok.ann"):
       to_open.append(os.path.join("../data/lang-uk-data/data/", file))

count = 0
for filename in to_open:
    if count > 2:
        break
    with open(filename, "r") as file:
        for line in file.readlines():
            if "ЛОК" in line.split():
                locations.append(" ".join(line.split()[4:]))
    #count += 1
dict = set()
with open("../dictionaries/toponims.txt", "r") as file:
    dict.update(file.readlines())

for loc in locations:
    loc = " ".join([morph.parse(w)[0].normal_form for w in loc.split()])
    #if loc not in dict:
    dict.add(loc)

dict = sorted(list(dict))
with open("../dictionaries/toponims.txt", "w") as file:
    for toponim in dict:
        file.write(toponim.strip().lower() + "\n")

print(len(dict))


dict = set()
with open("../dictionaries/toponims.txt", "r") as file:
    dict.update(file.readlines())

print(len(list(dict)))
dict = sorted(list(dict))

with open("../dictionaries/toponims.txt", "w") as file:
    for toponim in dict:
        file.write(toponim)



