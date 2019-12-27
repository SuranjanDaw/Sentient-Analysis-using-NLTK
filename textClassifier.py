from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import nltk
import random

document = []

#list of all stopwords
st = stopwords.words("english")
#cl]print(st)


#extracting the words from files and shufflin them
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        document.append((list(movie_reviews.words(fileid)) ,category))
random.shuffle(document)

count = 0
count1 = 0
all_words = []
stop_word_removed = []

#making a list of all the words and counting them
for w in movie_reviews.words():
    all_words.append(w.lower())
    count = count +1
print(count)

all_words_freq = nltk.FreqDist(all_words)
#print(all_words_freq.most_common(20))

#making a list of all words except stop words and counting them and printing them
for w in all_words:
    if w not in st:
        stop_word_removed.append(w.lower())
        count1 = count1+1

print(count1)

stop_word_removed_freq = nltk.FreqDist(stop_word_removed)
#print(stop_word_removed_freq.most_common(20))



#word features

word_feature = all_words_freq.keys()[:3000]
print(list(word_feature))

#to find features
def find_feature(document):
    features = {}
    words =set(document)
    for w in word_feature:
        features[w] = (w in words)
    
    return features