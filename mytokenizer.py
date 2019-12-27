import nltk
#nltk.data.path.append("/home/nltk_data")
from nltk.tokenize import sent_tokenize, word_tokenize

sentnce = "I am a CSE engg. I Like programming in Java"

print(word_tokenize(sentnce))
print(sent_tokenize(sentnce))