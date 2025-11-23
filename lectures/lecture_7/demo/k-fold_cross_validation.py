import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, preprocessing, metrics, model_selection

### download the original data set -- it has a bunch of NaN's)
#cars = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',header=None, sep='\s+')

### Clean data set
cars = pd.read_csv('auto-mpg.data.txt',header=None, sep='\s+')

### Label columns
cars.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
print (cars.head())

### Some scatter plots
cols = ['mpg', 'horsepower', 'weight', 'acceleration']
sns.pairplot(cars[cols], size=2.5)
#plt.show()

### extract the fuel consumption
y = cars.iloc[:,0].values

### horsepower
X = cars.iloc[:,[3]].values

# Compute Polynomial Features (e.g., horsepower^2)
poly = preprocessing.PolynomialFeatures(2)
X= poly.fit_transform(X)

#5-fold Cross-validation
kf = model_selection.KFold(n_splits=5, shuffle=True)
mses = []
for train_index, test_index in kf.split(X):
    
    #Split into training and test data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #Linear regression
    reg = linear_model.LinearRegression()
    reg.fit(X_train,y_train)
    
    #Print Parameters
    print("Parameter: ")
    print('w0: %f' %reg.intercept_)
    print('w1: %f' %reg.coef_[0])
    print('w2: %f' %reg.coef_[1])

    # MSE 
    mse = sum((y_test - reg.predict(X_test))**2.0)/len(y_test)
    print("MSE: %f" %mse)
    mses.append(mse)
    
print("MSE (Average): %f" %(sum(mses)/len(mses)))