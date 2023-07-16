import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re

paragraph = '''The Telugu film industry, also known as Tollywood, is one of the largest and most prominent film industries in India, producing movies primarily in the Telugu language. It is based in the state of Andhra Pradesh and the newly formed state of Telangana, with Hyderabad serving as its hub. Telugu cinema has a rich history and has made significant contributions to the Indian film industry.

Tollywood emerged in the early 20th century with the release of its first silent film, "Bhishma Pratigna," in 1921. Over the years, the industry has witnessed tremendous growth and transformation, both in terms of artistic quality and commercial success. Today, Telugu cinema is renowned for its diverse range of genres, encompassing everything from family dramas and romantic comedies to action-packed blockbusters and historical epics.

The Telugu film industry has produced several notable actors, directors, and technicians who have achieved immense fame and recognition across India. Actors like N.T. Rama Rao, Akkineni Nageswara Rao, Chiranjeevi, and Mahesh Babu have enjoyed massive popularity and have become household names. Directors such as S.S. Rajamouli, Trivikram Srinivas, and Sukumar have garnered critical acclaim for their unique storytelling and technical brilliance.'''
               
# text Preprocessing /cleaning the data

text = re.sub(r'\[[0-9]*\]',' ',paragraph)  # replace [any digit] e.g [346] with space
text = re.sub(r'\s+',' ',paragraph)  # replace any whitespace character (tab,newline,multiple whitespaces ) with single space
text = text.lower() # make all text in lower case
text = re.sub(r'\d',' ',text) # replaces any single digit with blank space
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
# converting cleaned text into sentences

sentences = nltk.sent_tokenize(text)

# converting sentences into words

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if not word in stopwords.words('english')]
    
# Training the Word2Vec model

model = Word2Vec(sentences,min_count=1)

'''The min_count=1 parameter is an optional argument that sets a threshold for the minimum frequency
 of a word in the training data. Words that occur less frequently than this threshold will be
 ignored and not included in the vocabulary of the Word2Vec model. In this case, since min_count 
 is set to 1, all words, regardless of their frequency, will be considered in the training process.'''
 
#words = model.wv.key_to_index
  
# in this paragrapb if we want to find the vocalbulary & create a object called words
# if you select then each & every word there may be vectors and dimensions associated to it

# Finding Word Vectors
vector = model.wv['film']
#if i want to find the vector of war word and if i want to find the relationship 

# Most similar words
similar = model.wv.most_similar('film')
#if i try to find most similar word related to the war 



#STILL MORE RESEARCH GOING ON REGARDS TO THE WORD2VEC