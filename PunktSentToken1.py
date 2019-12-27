from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
from nltk.tokenize import word_tokenize
import nltk

train_data = state_union.raw("2005-GWBush.txt")
sample_data = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_data)

tokenized_sent = custom_sent_tokenizer.tokenize(sample_data)

def process():
    try:

        for sent in tokenized_sent:
            words = word_tokenize(sent)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

process()
    