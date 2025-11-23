import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import random

# read cars dataset, a clean data set
cars = pd.read_csv('auto-mpg.data.txt',header=None, sep='\s+')


# random sample of 20 cars
sample = random.sample(range(0,len(cars)), 10)
out_of_sample = list(set(range(0,len(cars))) - set(sample))

# extract mpg values for cars in sample
y = cars.iloc[sample, 0].values
y_oos = cars.iloc[out_of_sample, 0].values

# extract horsepower values for cars in sample
X = cars.iloc[sample, [3]].values
X.reshape(X.size, 1)

# precompute polynomial features for degree 5
poly = preprocessing.PolynomialFeatures(5)
Xp = poly.fit_transform(X)


for lmbd in [0.0, 0.0001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
# fit linear regression model
    reg = linear_model.Ridge(alpha=lmbd, normalize=True)
    reg.fit(Xp,y)
    # plot fitted function
    hp = cars.iloc[:,3].values
    mpg = cars.iloc[:,0].values
    hps = np.array(sorted(hp))
    hps = hps.reshape(hps.size, 1)
    hpsp = poly.fit_transform(hps)
    plt.title("lambda: " + str(lmbd))
    plt.scatter(hp, mpg, color='gray', marker='x')
    plt.scatter(X, y, color='blue', marker='o')
    plt.plot(hps, reg.predict(hpsp), color='red', lw=2)
    plt.xlabel('Power [hp]')
    plt.ylabel('Fuel consumption [miles/gallon]')
    plt.xlim([min(hp), max(hp)])
    plt.ylim([min(mpg), max(mpg)])
    plt.show()
