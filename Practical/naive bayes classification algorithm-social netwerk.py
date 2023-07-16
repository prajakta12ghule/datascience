# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\Users\hp\OneDrive\Data Science\Practical\Machine Learning\Dataset\Social_Network_Ads.csv")
df.head()

# sepearte dependant and independant variable


x = df.iloc[:,[2,3]].values
y = df.iloc[:,-1].values

# train test split

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.20,random_state=0)

# feature scaling(Normalizer)

from sklearn.preprocessing import Normalizer
sc = Normalizer()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

# feature scaling(standarization)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)


# Training the Naive Bayes model(MultinomialNB) on the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB() 
classifier.fit(train_x, train_y)

# Training the Naive Bayes model(BernoulliNB) on the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB() 
classifier.fit(train_x, train_y)

# Training the Naive Bayes model(GaussianNB) on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() 
classifier.fit(train_x, train_y)


# prediction on test data

y_pred = classifier.predict(test_x)


#confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y,y_pred)

# accuracy score

from sklearn.metrics import accuracy_score
acc = accuracy_score(test_y,y_pred)

# bias

bais = classifier.score(train_x,train_y)

# variance

variance = classifier.score(test_x,test_y)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


##########################output#######################################

"""with feature scaling(standard scaler):
    1] BernoulliNB=>
         accuracy :82.5%
         bias:70.93%
         variance:82.5%
             
    2] GaussianNB=>
         accuracy:91.25%
         bias:88.43%
         variance:91.25%
    3] MultinomialNB=> it gives error with standard scaler so use normalizer
         accuracy:72.5%
         bias:62.18%
         variance:72.5%
             
 without feature scaling(standard scaler):
       1] BernoulliNB=>
            accuracy :72.5%
            bias:62.18%
            variance:72.5%
                
       2] GaussianNB=>
            accuracy:92.5%
            bias:87.81%
            variance:92.5%
       3] MultinomialNB=>
            accuracy:56.25
            bias:67.18%
            variance:56.25%       

from above results we can say that with and without scaling result of """