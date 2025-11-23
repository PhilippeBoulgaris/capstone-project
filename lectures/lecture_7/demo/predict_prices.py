import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics

### download the original data set -- it has a bunch of NaN's)
#cars = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',header=None, sep='\s+')

### Clean data set
cars = pd.read_csv('auto-mpg.data.txt',header=None, sep='\s+')

### Label columns
cars.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
print (cars.head())

#### Some scatter plots
cols = ['mpg', 'horsepower', 'weight', 'acceleration']
sns.pairplot(cars[cols], size=2.5)
plt.show()


### extract the fuel consumption
y = cars.iloc[:,0].values

### horsepower
X = cars.iloc[:,[3]].values
print(X.size)

### Plot 
g = sns.regplot(x=X, y=y, fit_reg=False)

#### Correlation
#cm = np.corrcoef(cars[cols].values.T)
#sns.set(font_scale=1.5)
#hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
#print("correlation", cm)


### Linear Regression
reg = linear_model.LinearRegression()
reg.fit(X,y)
plt.plot(X, reg.predict(X), color='red')

### Labels
plt.xlabel('Power [hp]')
plt.ylabel('Consumption [mpg]')
plt.show()

# Coefficients and R2
print('Parameters:')
print('w0: %f'%reg.intercept_)
print('w1: %f'%reg.coef_[0])
print('R2: %f'%metrics.r2_score(y,reg.predict(X)))

