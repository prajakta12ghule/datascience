import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

investment = pd.read_csv(r"C:\Users\hp\OneDrive\Data Science\Practical\Machine Learning\Dataset\Investment.csv")

x = investment.iloc[:,:-1]
y= investment.iloc[:,-1]

# converting categorical column(states) to numerical column
#x=pd.get_dummies(x)

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
x['State']=labelencoder_X.fit_transform(x['State'])
x


from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(train_x,train_y)

y_pred = regressor.predict(test_x)

regressor.coef_
regressor.intercept_


bias=regressor.score(train_x,train_y)
bias

variance = regressor.score(test_x,test_y)
variance

import statsmodels.formula.api as sm

x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
