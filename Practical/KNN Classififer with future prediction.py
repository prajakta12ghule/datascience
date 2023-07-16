import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\hp\OneDrive\Data Science\Practical\Machine Learning\Dataset\Social_Network_Ads.csv")

x = df.iloc[:,[2,3]]
y = df.iloc[:,-1]

df.isnull().sum()
x.info()

# no null values and all required fields in x are numerical so divide data in train and test

from sklearn.model_selection  import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.20,random_state = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
# feature scaling - standarization

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.fit_transform(test_x)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                          
#build  KNN model 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3,weights='distance') # we can check the accuracy by hyper parameter tunning here
knn.fit(train_x,train_y)

# prediction
y_pred = knn.predict(test_x)

from sklearn.metrics import confusion_matrix
cm =  confusion_matrix(test_y,y_pred)
print(cm)


from sklearn.metrics import accuracy_score
acc = accuracy_score(test_y,y_pred) 
print(acc)         

from sklearn.metrics import classification_report
cr = classification_report(test_y,y_pred)
print(cr)

bias = knn.score(train_x,train_y)
bias

variance = knn.score(test_x,test_y) 
variance    

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = train_x, train_y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = test_x, test_y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()      


##################################Future Prediction ########################3

dataset1 = pd.read_excel(r"C:\Users\hp\OneDrive\Data Science\Practical\Machine Learning\Dataset\future prediction _ 2.xlsx")
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[2,3]]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

d2['y_pred'] = knn.predict(M)
d2.to_csv('future pred_2.csv')

import os
os.getcwd()

         