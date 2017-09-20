from nltk.tokenize import TweetTokenizer
import pymorphy2

twtk = TweetTokenizer()
morph = pymorphy2.MorphAnalyzer(lang='uk')


words = []
# with open("../data/VelikaIstoriyaYkrajni_1412180965.txt", "r") as file:
#     c = 0
#     for line in file.readlines():
#         #if c > 1000:
#             #break
#         words.extend([morph.parse(word)[0].normal_form for word in twtk.tokenize(line) if word.isalpha()])
#         c += 1
#
# with open("../data/words_lemmatized.txt", "w") as file:
#     for w in words:
#         file.write(w + "\n")

with open("../data/words_lemmatized.txt", "r") as file:
     words.extend(file.readlines())

# print(words)
print("written")

dict = []
with open("../dictionaries/toponims.txt", "r") as file:
    dict.extend(file.readlines())

match = set()
print(dict[43])
words.sort()
for word in words:
    for loc in dict:
        if loc > word:
            break
        if loc.lower().strip() == word.strip():
                #print(word)
                match.add(word)

print(match)
print(len(match))

with open("../dictionaries/locations_analytistoriya.txt", "w") as file:
    for m in match:
        file.write(m)







