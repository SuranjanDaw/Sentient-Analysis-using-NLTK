from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

sentence = "I am a CSE engg. I Like programming in Java. Because i like to be programmed by a programmer"

words=word_tokenize(sentence)
ps = PorterStemmer()

for w in words:
    port = ps.stem(w)
    print(port)