from nltk.corpus import wordnet

w1 = wordnet.synset("cat.n.01")
w2= wordnet.synset("dog.n.01")

print(w1.wup_similarity(w2))