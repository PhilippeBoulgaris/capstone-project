import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing


# read cars dataset, a clean data set
cars = pd.read_csv('auto-mpg.data.txt',header=None, sep='\s+')

# extract mpg values
y = cars.iloc[:,0].values

# extract horsepower and weight values, apply one-hot encoding for origin
X = pd.concat([cars.iloc[:,[3,4]], pd.get_dummies(cars[7])], axis = 1).values

# fit linear regression model
reg = linear_model.LinearRegression()
reg.fit(X,y)

# coefficients
reg.intercept_ # 43.974410233714622
reg.coef_ # [-0.05354417, -0.00484275, -1.2344519 , -0.27333471, 1.50778661]

# compute correlation coefficient
np.corrcoef(reg.predict(X),y) # 0.84810338
# compute mean squared error (MSE)
sum((reg.predict(X) - y)**2) / len(y) # 17.057355871889044

