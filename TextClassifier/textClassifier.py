from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
import nltk
import random
import pickle

document = []

#list of all stopwords
#st = stopwords.words("english")
#print(st)


#extracting the words from files and shuffling them
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
print("Number of words in total: ",count)

all_words_freq = nltk.FreqDist(all_words)
'''print(all_words_freq.most_common(20))'''

#making a list of all words except stop words and counting them and printing them
'''
for w in all_words:
    if w not in st:
        stop_word_removed.append(w.lower())
        count1 = count1+1

print("Number of words except the stop-words :",count1)

stop_word_removed_freq = nltk.FreqDist(stop_word_removed)
print(stop_word_removed_freq.most_common(20))
print("plot over all words removed stop wors")
stop_word_removed_freq.plot(30, cumulative = False)
'''



#plotting the frequency distribution curve
print("plot over all words")
all_words_freq.plot(30, cumulative =False)


#word features
word_feature = list(all_words_freq.keys())[:3000]
#print(list(word_feature))

#this func is taking a list of words and checking if they are in 
#word_feature and lableing them as Ture or False
#this function always returns a vector of size of word_feature
def find_feature(document):
    features = {}
    words =set(document)
    for w in word_feature:
        features[w] = (w in words)
    
    return features

#sending a review from negetive set to find features and prinitng its result

'''print(find_feature(movie_reviews.words("neg/cv000_29416.txt")))'''

feature_set = [(find_feature(rev),category) for (rev,category) in document]




#Naive Bayes Classifer

train_set = feature_set[:1900]
test_set = feature_set[1900:]

#training the data set and saving it in pickel
bayeClassifier = nltk.NaiveBayesClassifier.train(train_set)
bayeClassifierSave = open("NaiveBayesClassifier.pickle","wb")
pickle.dump(bayeClassifier,bayeClassifierSave)
bayeClassifierSave.close()


# to load from pickle 
'''
bayeClassifier_file = open("NaiveBayesClassifier.pickle", "rb")
bayeClassifier = pickle.load(bayeClassifier_file)
bayeClassifier_file.close()
'''



#testing it on test set

print(nltk.classify.accuracy(bayeClassifier, test_set))

bayeClassifier.show_most_informative_features(15)


