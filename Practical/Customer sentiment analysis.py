import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\hp\OneDrive\Data Science\Practical\NLP\Dataset\Restaurant_Reviews.tsv",delimiter='\t',quoting=3)

# Cleaning the texts

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

''' if you do not get accuracy minimum 80% by applying below option:
    1. All classification models
    2. Increasing decreasing test_size
    3. hyper parameter tunning of all classification model
 then try without removing stopwords from dataset.
still you do not get accuracy minimum 80 % then double or tripple the existing dataset
'''
'''
# dataset by removing stopwords
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i]) # remove all other characters except lower and upper alphabets
    review = review.lower() # make all in lower case
    review = review.split() # split at space
    ps = PorterStemmer()    # comment it for Word2Vector testing
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # took only those words which are not english stopwords(# comment it for Word2Vector testing)
    review = ' '.join(review)
    corpus.append(review)
 '''   
 # dataset without removing stopwords
corpus = []
for i in range(0, 1000):
                review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
                review = review.lower()
                review = review.split()
                ps = PorterStemmer()
                review = [ps.stem(word) for word in review]
                review = ' '.join(review)
                corpus.append(review) 
 # Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# Creating the TF-IDF model
'''
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
'''
# Creating the Word2Vec model
'''
from gensim.models import Word2Vec
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, sg=0)

# Prepare word vectors for training examples
x = []
for sentence in corpus:
    sentence_vector = []
    for word in sentence:
        sentence_vector.append(model.wv[word])
    x.append(sum(sentence_vector) / len(sentence_vector))
'''
# separating x and y

x = cv.fit_transform(corpus).toarray() # comment it for Word2Vec test
y = df.iloc[:,1].values

# split into train and test

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.20,random_state=0)

# training on all classification models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Define a list of classification models
models = [
    
   LogisticRegression(),
   DecisionTreeClassifier(),
   RandomForestClassifier(),
   SVC(),
   KNeighborsClassifier(),
   GaussianNB(),
   BernoulliNB(),
   MultinomialNB(),
   XGBClassifier(),
   GradientBoostingClassifier()
    
    
]


# empty dictionary for storing accuracy
accuracy_scores = {}

# Iterate over each model

for model in models:
    model_name = model.__class__.__name__
    print("Training", model_name)
    
    # Train the model
    model.fit(train_x,train_y)
    
    # Make predictions on the test set
    y_pred = model.predict(test_x)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_y, y_pred)
    # Print the accuracy
    print("Accuracy:", accuracy)
    print()
    
    # Calculate bias
    bias = model.score(train_x,train_y)
    # Print the bias
    print("bias:", bias)
    print()
    
    # Calculate variance
    variance = model.score(test_x,test_y)
    # Print the variance
    print("variance:", variance)
    print()
    
    # Calculate Confusion matrix
    cm = confusion_matrix(test_y, y_pred)
    # print confusion matrix
    
    print("Confusion matrix:")
    print(cm)
    
    # Store the accuracy score in the dictionary
    accuracy_scores[model] = accuracy
    # Sort the accuracy scores in descending order
    sorted_scores = sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True)
    # Print the ranking and accuracy scores of the models
    print("Model Rankings based on Accuracy Score:")
for rank, (model, accuracy) in enumerate(sorted_scores, start=1):
      print(f"Rank {rank}: {model} - Accuracy: {accuracy:.4f}")
