import wikipedia
wikipedia.set_lang("uk")

from bs4 import BeautifulSoup




article = wikipedia.page(pageid=30094)  #cities


soup = BeautifulSoup(article.html(), 'html.parser')
#print(soup.prettify())

toponims = set()

for sct in soup.find_all("td"):
    try:
        txt = sct.a["title"].strip()
    except TypeError:
        continue
    if not txt.isdigit() and "область" not in txt.split() and "століття" not in txt.split():
        toponims.add(txt)

print(toponims)
print(len(toponims))




article = wikipedia.page(pageid=29071) #rivers


soup = BeautifulSoup(article.html(), 'html.parser')
#print(soup.prettify())
for sct in soup.find_all("a"):
    try:
        txt = sct["title"].strip()
    except TypeError and KeyError:
        continue
    if len(txt.split()) < 3:
        toponims.add(txt)

print(toponims)
print(len(toponims))


article = wikipedia.page(pageid=30018) #lakes


soup = BeautifulSoup(article.html(), 'html.parser')
#print(soup.prettify())
for sct in soup.find_all("a"):
    try:
        txt = sct["title"].strip()
    except TypeError:
        continue
    except KeyError:
        continue
    if len(txt.split()) < 3:
        toponims.add(txt)

print(toponims)
print(len(toponims))

article = wikipedia.page(pageid=33702) #towns


soup = BeautifulSoup(article.html(), 'html.parser')
#print(soup.prettify())
for sct in soup.find_all("td"):
    try:
        txt = sct.a["title"].strip()
    except TypeError:
        continue
    except KeyError:
        continue
    if not txt.isdigit() and "область" not in txt.split() and "століття" not in txt.split():
        toponims.add(txt)

print(toponims)
print(len(toponims))

for name in ["Говерла","Бребенескул","Піп Іван" ,"Петрос","Гутин Томнатик","Ребра","Роман-Кош","Грофа",
             "Демір-Капу","Зейтін-Кош","Кемаль-Егерек","Сивуля","Еклізі-Бурун","Ангара-Бурун", "Довбушанка"]:
    toponims.add(name)



# article = wikipedia.page(pageid=243230) #national parks
#
#
# soup = BeautifulSoup(article.html(), 'html.parser')
# print(soup.prettify())
# for sct in soup.find_all("td"):
#     try:
#         txt = sct.a["title"].strip()
#     except TypeError:
#         continue
#     except KeyError:
#         continue
#     if not txt.isdigit() and "область" not in txt.split() and "століття" not in txt.split():
#         toponims.add(txt)
#
#
#
toponims = sorted(list(toponims))

with open("../dictionaries/toponims.txt", "w") as file:
    for toponim in toponims:
        file.write(toponim.split("(")[0] + "\n")










