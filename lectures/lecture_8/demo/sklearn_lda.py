import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# 0. Load in the data and split the descriptive and the target feature
df = pd.read_csv('wine.txt',sep=',',
                 names=['target','Alcohol','Malic_acid','Ash','Akcakinity','Magnesium',
                        'Total_pheonols','Flavanoids','Nonflavanoids',
                        'Proanthocyanins','Color_intensity','Hue','OD280','Proline'])

X = df.iloc[:,1:].copy()
target = df['target'].copy()
X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.3,random_state=0) 

# 1. Instantiate the method and fit_transform the algotithm
LDA = LinearDiscriminantAnalysis(n_components=2) 
"""The n_components key word gives us the projection to the 
n most discriminative directions in the dataset. 
We set this parameter to two to get a transformation in two dimensional space. """

data_projected = LDA.fit_transform(X_train,y_train)
print(data_projected.shape)

# PLot the transformed data
markers = ['s','x','o']
colors = ['r','g','b']
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
for l,m,c in zip(np.unique(y_train),markers,colors):
    ax0.scatter(data_projected[:,0][y_train==l],data_projected[:,1][y_train==l],c=c,marker=m)
    
plt.show()