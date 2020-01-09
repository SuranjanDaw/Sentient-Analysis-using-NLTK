from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
import nltk
import random
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC , LinearSVC , NuSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.classify import ClassifierI
from statistics import mode

class VotedClassifier(ClassifierI):

    def __init__(self, *classifier):
        self._classifier = classifier
    
    def classify(self,feature):
        votes = []
        for c in self._classifier:
            v = c.classify(feature)
            votes.append(v)
        return mode(votes)
    def confidence(self,feature):
        votes = []
        for c in self._classifier:
            v = c.classify(feature)
            votes.append(v)
        

        return (votes.count(mode(votes)) / len(votes))



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
'''all_words_freq.plot(30, cumulative =False)'''


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

train_set = feature_set[100:]
test_set = feature_set[:100]

#training the data set 
bayeClassifier = nltk.NaiveBayesClassifier.train(train_set)

#saving it in pickel
'''
bayeClassifierSave = open("NaiveBayesClassifier.pickle","wb")
pickle.dump(bayeClassifier,bayeClassifierSave)
bayeClassifierSave.close()
'''

# to load from pickle 
'''
bayeClassifier_file = open("NaiveBayesClassifier.pickle", "rb")
bayeClassifier = pickle.load(bayeClassifier_file)
bayeClassifier_file.close()
'''



#testing it on test set

print("Original nltk bayes classifier accuracy: ",nltk.classify.accuracy(bayeClassifier, test_set))

#To see the most top 15 words and ratio of their classification 
'''bayeClassifier.show_most_informative_features(15)'''


#trianing using sklearn library for MultiNomial Naive Bayes

multiNB_classifer = SklearnClassifier(MultinomialNB())
multiNB_classifer.train(train_set)
print("MultiNomialNB accuracy", nltk.classify.accuracy(multiNB_classifer, test_set))

#trianing using sklearn library for Gaussian Naive Bayes
'''
gaussNB_classifer = SklearnClassifier(GaussianNB())
gaussNB_classifer.train(train_set)
print("GaussianNB accuracy", nltk.classify.accuracy(gaussNB_classifer, test_set))
'''


#trianing using sklearn library for Bernollis Naive Bayes

berNB_classifer = SklearnClassifier(BernoulliNB())
berNB_classifer.train(train_set)
print("BernolliNB accuracy", nltk.classify.accuracy(berNB_classifer, test_set))


#trianing using sklearn library for Linear model Linear Regression
logisticReg_classifier = SklearnClassifier(LogisticRegression())
logisticReg_classifier.train(train_set)
print("Linear Regression accuracy: ", nltk.classify.accuracy(logisticReg_classifier, test_set))


#trianing using sklearn library for Linear model Schocastic Gradient Discent Model
grad_dist_classifier = SklearnClassifier(SGDClassifier(max_iter = 1000))
grad_dist_classifier.train(train_set)
print("Gradient Discent Accuracy: ", nltk.classify.accuracy(grad_dist_classifier, test_set))

#trianing using sklearn library for SVM SVC Model
'''
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(train_set)
print("SVC Accuracy: ", nltk.classify.accuracy(SVC_classifier, test_set))
'''

#trianing using sklearn library for SVM LinearSVC Model
linearSVC_classifier = SklearnClassifier(LinearSVC())
linearSVC_classifier.train(train_set)
print("Linear SVC Accuracy: ", nltk.classify.accuracy(linearSVC_classifier, test_set))

#trianing using sklearn library for SVM NuSVM Model
nuSVC_classifier = SklearnClassifier(NuSVC())
nuSVC_classifier.train(train_set)
print("NuSVC Accuracy: ", nltk.classify.accuracy(nuSVC_classifier, test_set))

#training using VoteClassifier
voted_classifier = VotedClassifier(bayeClassifier,multiNB_classifer, berNB_classifer,logisticReg_classifier,grad_dist_classifier,linearSVC_classifier,nuSVC_classifier)
print("Voted Accuracy: ", nltk.classify.accuracy(voted_classifier, test_set))

for i in range(5):
    print("Sentiment for is =",voted_classifier.classify(test_set[i][0]),"with a confidence of = ",voted_classifier.confidence(test_set[i][0]))