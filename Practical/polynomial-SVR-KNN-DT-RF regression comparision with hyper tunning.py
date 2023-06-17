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


########################  SVR(support vector Regressor)  #######################################
# restart the kernel and run only till separating dependant and independant
#variable and then run below part 

                ## kernel=rbf (by default) ##
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_svr = SVR(kernel = "sigmoid",degree=5,gamma="auto" )
regressor_svr.fit(X, y)

y_pred = regressor_svr.predict([[6.5]])
y_pred


                ## kernel=sigmoid ##,degree=5,gamma=auto##
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_svr = SVR(kernel = "sigmoid",degree=5,gamma="auto" )
regressor_svr.fit(X, y)

y_pred = regressor_svr.predict([[6.5]])
y_pred


                ## kernel=poly ##,degree=6,gamma=scale##
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_svr = SVR(kernel = "poly",degree=6,gamma="scale" )
regressor_svr.fit(X, y)

y_pred = regressor_svr.predict([[6.5]])
y_pred

#################################KNN(K neirest neighbours)#####################

# restart the kernel and run only till separating dependant and independant
#variable and then run below part 
        # default=5 #
from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor()
regressor_knn.fit(X,y)

y_pred_knn = regressor_knn.predict([[6.5]])

              # n_neighbors=2 #
from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=2)
regressor_knn.fit(X,y)

y_pred_knn = regressor_knn.predict([[6.5]])

             #   n_neighbors=3   #

from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=6)
regressor_knn.fit(X,y)

y_pred_knn = regressor_knn.predict([[6.5]])

########################## Decision Tree ###############################

from sklearn.tree import DecisionTreeRegressor
regressor_DT = DecisionTreeRegressor(criterion = 'friedman_mse',splitter = 'random')   
regressor_DT.fit(X, y)
y_pred = regressor_DT.predict([[6.5]])

       #### criterion=squared_error   #####
      
from sklearn.tree import DecisionTreeRegressor
regressor_DT = DecisionTreeRegressor(criterion = 'squared_error',splitter = 'random')   
regressor_DT.fit(X, y)
y_pred = regressor_DT.predict([[6.5]])
y_pred


       #### criterion=absolute_error   #####
      
from sklearn.tree import DecisionTreeRegressor
regressor_DT = DecisionTreeRegressor(criterion = 'absolute_error',splitter = 'random')   
regressor_DT.fit(X, y)
y_pred = regressor_DT.predict([[6.5]])
y_pred


     #### criterion=absolute_error   #####
      
from sklearn.tree import DecisionTreeRegressor
regressor_DT = DecisionTreeRegressor(criterion = 'squared_error',splitter = 'random')   
regressor_DT.fit(X, y)
y_pred = regressor_DT.predict([[6.5]])
y_pred


################################# Random Forest ###########################

from sklearn.ensemble import RandomForestRegressor 
reggressor_RF = RandomForestRegressor(n_estimators = 58, random_state = 0)
reggressor_RF.fit(X,y)
y_pred = reggressor_RF.predict([[6.5]])
y_pred
