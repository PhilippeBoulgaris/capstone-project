import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

########################################################################

# load data
cars = pd.read_csv('auto-mpg.data.txt',header=None, sep='\s+')

# extract power and weight as data matrix X
X = cars.iloc[:, [3,4]].values

# extract origin as target value y
y = cars.iloc[:, 7].values

# normalize data
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X) # determine min and max
X_normalized = min_max_scaler.transform(X)

# DBSCAN
db = DBSCAN(eps=0.05,min_samples=5,metric='euclidean')
db.fit_predict(X_normalized)

# plot cars
# U.S. : o / Europe: x / Japan : +
m = ['o' if o==1 else 'x' if o==2 else '+' for o in y]

# Noise : black / Cluster 1 : red / Cluster 2 : blue /
# Cluster 3 : green / Cluster 4 : yellow
c = ['black' if l==-1 else 'red' if l==0 else 'blue' if l==1
else 'green' if l==2 else 'yellow' for l in db.labels_]
for i in range(0,len(X)):
    plt.scatter(X[i,0], X[i,1], color=c[i], marker=m[i])
    
plt.xlabel('Power [hp]')
plt.ylabel('Weight [lbs]')
plt.show()

