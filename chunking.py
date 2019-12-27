import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_tokenizer.tokenize(sample_text)

print(tokenized)

def process():
    try :
        for sent in tokenized:
            words = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*}"""
            chunkPar = nltk.RegexpParser(chunkGram)
            chunked = chunkPar.parse(tagged)
            chunked.draw()
    except Exception as e:
        print("exception")            


process()