from nltk.corpus import wordnet

data = wordnet.synsets("happy")

print(data[0].examples())
print(data[0].definition())

for w in data:
    print("for word :")
    for l in w.lemmas():
        print(l.name())


synonym = []
antonym = []

for w in data:
    print("for synset: ",w)
    for l in w.lemmas():
        synonym.append(l.name())
        if l.antonyms():
            antonym.append(l.antonyms()[0].name())

print(set(synonym))
print(set(antonym))
