from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
import nltk


train_data = state_union.raw("2005-GWBush.txt")
sample_data = state_union.raw("2006-GWBush.txt")


custom_tokenizer = PunktSentenceTokenizer(train_data)

tokenized = custom_tokenizer.tokenize(sample_data)

def process():
    try:
        for sent in tokenized:
            words = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(words)

            named_er = nltk.ne_chunk(tagged, binary=True)
            named_er.draw()
    except Exception as e:
        print(str(e))


process()
