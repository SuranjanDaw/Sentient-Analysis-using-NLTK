import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

train_data = state_union.raw("2005-GWBush.txt")
sample_data = state_union.raw("2006-GWBush.txt")

custom_tokenizer = PunktSentenceTokenizer(train_data)

tokenized = custom_tokenizer.tokenize(sample_data)

def process():
    try:
        for sent in tokenized:
            words = word_tokenize(sent)
            tagged = pos_tag(words)


            chunkGram = r"""chunk:{<.*>+} 
                            }<VB.?|IN|TO|DT>+{ """
            chunkPar = nltk.RegexpParser(chunkGram)

            chunked = chunkPar.parse(tagged)

            chunked.draw()
    except Exception as e:
        print(str(e))


process()