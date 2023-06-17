# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:36:19 2023

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# reading the dataset
salary = pd.read_csv(r"C:\Users\hp\OneDrive\Data Science\Dataset\Salary_Data.csv")
# checking null values
salary.isnull().sum()
#checking datatypes of columns
salary.info()

#split data into independant variable

x=salary.iloc[:,:-1].values

#split data into dependant variable

y=salary.iloc[:,1].values

# split data into train and test 80-20%

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=0)

#we called simple linear regression algoriytm from sklearm framework 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# we build simple linear regression model regressor

regressor.fit(train_x,train_y)

# test the model & create a predicted table

y_pred = regressor.predict(test_x)

# visualize train data point ( 24 data)

plt.scatter(train_x, train_y, color='red')
plt.plot(train_x,regressor.predict(train_x),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visulaize test data point 

plt.scatter(test_x,test_y,color='blue')
plt.plot(train_x,regressor.predict(train_x),color='red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# slope is generrated from linear regress algorith which fit to dataset 

m=regressor.coef_
m
# interceppt also generatre by model.
 
c = regressor.intercept_
c

# predict or forcast the future the data which we not trained before 

y_12 = m * 12 + c
y_12

y_20 = m * 20 + c
y_20

# to check overfitting  ( low bias high variance)

bias = regressor.score(train_x,train_y)
bias

# to check underfitting (high bias low variance)

variance = regressor.score(test_x,test_y)
variance
