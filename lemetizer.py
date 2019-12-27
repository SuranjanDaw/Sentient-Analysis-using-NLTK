from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

sent = "democracy democratic organize organise organisation organising orgasm much better"

ps = PorterStemmer()
leme = WordNetLemmatizer()
tokenized = word_tokenize(sent)
print(tokenized)
for w in tokenized:
    print(leme.lemmatize(w,"a") )
    print("Stemmer :"+ps.stem(w))