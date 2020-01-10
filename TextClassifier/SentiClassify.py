import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import random
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
from nltk import ClassifierI, SklearnClassifier
import pickle

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

f = open("/home/suranjan/Documents/WebCrawler/NLP/TextClassifier/positive.txt", "r")
pos = []
pos = f.read().split("\n")
f.close()

f = open("/home/suranjan/Documents/WebCrawler/NLP/TextClassifier/negetive.txt", "r")
neg = []
neg = f.read().split("\n")
f.close()

'''
print("Positive :\n", pos[:10])
print("Negetive :\n", neg[:10])
'''

sent = []
for p in pos:
    sent.append((p,"pos"))
for n in neg:
    sent.append((n,"neg"))
#random.shuffle(sent)


#saving sent as pickle
sent_file = open("pickle/sent_file.pickle","wb")
pickle.dump(sent, sent_file)
sent_file.close()

#print("all:", sent[:100])


word = []

word_type =["J"]
for p in pos:
    word_list = word_tokenize(p)
    positive = nltk.pos_tag(word_list)
    for w in positive:
        if w[1][0] in word_type:
            word.append(w[0].lower())
for n in neg:
    word_list = word_tokenize(n)
    negetive = nltk.pos_tag(word_list)

    for w in negetive:
        if w[1][0] in word_type:
            word.append(w[0].lower())

print("\n\n")
word_list_freq = FreqDist(word)
#word_list_freq.plot(30)


word_features = list(word_list_freq.keys())[:5000]
#saving word_features in pickle
word_features_file = open("pickle/wordfeature.pickle","wb")
pickle.dump(word_features, word_features_file)
word_features_file.close()



def getfeature(document):
    feature = {}
    words = word_tokenize(document)
    for w in word_features:
        feature[w] = w in words

    return feature

feature_set = [(getfeature(rev),category) for (rev,category) in sent]

random.shuffle(feature_set)

#saving feature set in pickle
feature_set_file = open("pickle/feature_set.pickle", "wb")
pickle.dump(feature_set,feature_set_file)
feature_set_file.close()


#Naive Bayes Classifer

train_set = feature_set[10000:]
test_set = feature_set[:10000]

#training the data set 
bayeClassifier = nltk.NaiveBayesClassifier.train(train_set)
print("Original nltk bayes classifier accuracy: ",nltk.classify.accuracy(bayeClassifier, test_set))

#saving Naive Bayes Classifier 
NaiveBayesFile = open("pickle/naive_bayes_file.pickle","wb")
pickle.dump(bayeClassifier, NaiveBayesFile)
NaiveBayesFile.close()

#trianing using sklearn library for MultiNomial Naive Bayes

multiNB_classifer = SklearnClassifier(MultinomialNB())
multiNB_classifer.train(train_set)
print("MultiNomialNB accuracy", nltk.classify.accuracy(multiNB_classifer, test_set))

#saving Multinomial Bayes Classifier 
MultiNomialNaiveFile = open("pickle/MNB_file.pickle","wb")
pickle.dump(multiNB_classifer, MultiNomialNaiveFile)
MultiNomialNaiveFile.close()

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

#saving BernolliNB Classifier 
BernolliNbFile = open("pickle/bernolli_NB_file.pickle","wb")
pickle.dump(berNB_classifer, BernolliNbFile)
BernolliNbFile.close()


#trianing using sklearn library for Linear model Linear Regression
logisticReg_classifier = SklearnClassifier(LogisticRegression())
logisticReg_classifier.train(train_set)
print("Logistic Regression accuracy: ", nltk.classify.accuracy(logisticReg_classifier, test_set))

#saving Logistic regression Classifier 
LogisticRegFile = open("pickle/logistic_redg_file.pickle","wb")
pickle.dump(logisticReg_classifier, LogisticRegFile)
LogisticRegFile.close()


#trianing using sklearn library for Linear model Schocastic Gradient Discent Model
grad_dist_classifier = SklearnClassifier(SGDClassifier(max_iter = 1000))
grad_dist_classifier.train(train_set)
print("Gradient Discent Accuracy: ", nltk.classify.accuracy(grad_dist_classifier, test_set))

#saving Schocastic Gradient Discent Classifier 
SGDCFile = open("pickle/SGDC_file.pickle","wb")
pickle.dump(grad_dist_classifier, SGDCFile)
SGDCFile.close()



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

#saving LinearSVC Classifier 
LinearSVCFile = open("pickle/linear_SVC_file.pickle","wb")
pickle.dump(linearSVC_classifier,LinearSVCFile)
LinearSVCFile.close()




#trianing using sklearn library for SVM NuSVM Model
nuSVC_classifier = SklearnClassifier(NuSVC())
nuSVC_classifier.train(train_set)
print("NuSVC Accuracy: ", nltk.classify.accuracy(nuSVC_classifier, test_set))

#saving NuSVC Classifier 
NuSVCFile = open("pickle/nu_SVC_file.pickle","wb")
pickle.dump(nuSVC_classifier,NuSVCFile)
NuSVCFile.close()

 
#training using VoteClassifier
voted_classifier = VoteClassifier(bayeClassifier,multiNB_classifer, berNB_classifer,logisticReg_classifier,grad_dist_classifier,linearSVC_classifier,nuSVC_classifier)
print("Voted Accuracy: ", nltk.classify.accuracy(voted_classifier, test_set))

'''for i in range(5):
    print("Sentiment for is =",voted_classifier.classify(test_set[i][0]),"with a confidence of = ",voted_classifier.confidence(test_set[i][0]))

'''


def sentiment(sentence):
    features = getfeature(sentence)

    return voted_classifier.classify(features),voted_classifier.confidence(features)


