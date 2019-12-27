from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sentence = "I am a CSE engg. I Like programming in Java"

stopwords_set = set(stopwords.words('english'))
print(stopwords_set)

print(word_tokenize(sentence))
new_sent = word_tokenize(sentence)
for w in new_sent:
    if w not in stopwords_set:
        print(w)