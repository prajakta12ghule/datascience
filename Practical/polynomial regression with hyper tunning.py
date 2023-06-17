import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


dataset = pd.read_csv(r"C:\Users\hp\OneDrive\Data Science\Practical\Machine Learning\Dataset\emp_sal.csv")


X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values
##############Comparision of linear and polynomial model####################
# linear model  -- linear algor ( degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# polynomial model  ( bydefeaut degree - 2)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

# prediction
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

################# Degree = 3  ###########################################

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

# prediction
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

################# Degree = 4  ###########################################

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

# prediction
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

################# Degree = 5  ###########################################

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

# prediction
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

################# Degree = 6  ###########################################3

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

# prediction
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

############################################################################
# linear regression visualizaton 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# poly nomial visualization 

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predicton 

#lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


########################  SVR  #######################################

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='poly', degree=7, gamma='auto')
regressor.fit(X, y)

y_pred_svr = regressor.predict([[6.5]])






