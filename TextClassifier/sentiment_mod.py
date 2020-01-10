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

#loading word_feature from pickle
word_feature_file = open("pickle/wordfeature.pickle","rb")
word_features = pickle.load(word_feature_file)
word_feature_file.close()

def getfeature(document):
    feature = {}
    words = word_tokenize(document)
    for w in word_features:
        feature[w] = w in words

    return feature


#loading sent as pickle
sent_file = open("pickle/sent_file.pickle","rb")
sent = pickle.load(sent_file)
sent_file.close()


#loading feature set in pickle
feature_set_file = open("pickle/feature_set.pickle", "rb")
feature_set = pickle.load(feature_set_file)
feature_set_file.close()


#Naive Bayes Classifer

train_set = feature_set[10000:]
test_set = feature_set[:10000]

#loading Naive Bayes Classifier 
NaiveBayesFile = open("pickle/naive_bayes_file.pickle","rb")
bayeClassifier = pickle.load(NaiveBayesFile)
NaiveBayesFile.close()


#loading Multinomial Bayes Classifier 
MultiNomialNaiveFile = open("pickle/MNB_file.pickle","rb")
multiNB_classifer = pickle.load(MultiNomialNaiveFile)
MultiNomialNaiveFile.close()




#loading BernolliNB Classifier 
BernolliNbFile = open("pickle/bernolli_NB_file.pickle","rb")
berNB_classifer = pickle.load(BernolliNbFile)
BernolliNbFile.close()



#loading Logistic regression Classifier 
LogisticRegFile = open("pickle/logistic_redg_file.pickle","rb")
logisticReg_classifier = pickle.load(LogisticRegFile)
LogisticRegFile.close()



#loading Schocastic Gradient Discent Classifier 
SGDCFile = open("pickle/SGDC_file.pickle","rb")
grad_dist_classifier = pickle.load(SGDCFile)
SGDCFile.close()



#loading LinearSVC Classifier 
LinearSVCFile = open("pickle/linear_SVC_file.pickle","rb")
linearSVC_classifier = pickle.load(LinearSVCFile)
LinearSVCFile.close()




#loading NuSVC Classifier 
NuSVCFile = open("pickle/nu_SVC_file.pickle","rb")
nuSVC_classifier = pickle.load(NuSVCFile)
NuSVCFile.close()

 
#training using VoteClassifier
voted_classifier = VotedClassifier(bayeClassifier,multiNB_classifer, berNB_classifer,logisticReg_classifier,grad_dist_classifier,linearSVC_classifier,nuSVC_classifier)
#print("Voted Accuracy: ", nltk.classify.accuracy(voted_classifier, test_set))


def sentiment(sentence):
    features = getfeature(sentence)

    return voted_classifier.classify(features),voted_classifier.confidence(features)


