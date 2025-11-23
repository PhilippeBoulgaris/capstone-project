import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

########################################################################
# load data
cars = pd.read_csv('auto-mpg.data.txt',header=None, sep='\s+')


# extract power and weight as data matrix X
X = cars.iloc[:, [3,4]].values

# extract origin (0:Non-U.S. / 1:U.S.) as target vector y
y = cars.iloc[:, 7].values

# split into training data (80%) and test data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=0)

########################################################################
# use kNN with k = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_predicted = knn.predict(X_test)

# compute accuracy
print(accuracy_score(y_true=y_test, y_pred=y_predicted)) 


########################################################################
# normalize data
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X_train) # determine min and max
X_train_normalized = min_max_scaler.transform(X_train)
X_test_normalized = min_max_scaler.transform(X_test)

# use kNN with k = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_normalized, y_train)
y_predicted = knn.predict(X_test_normalized)

# compute accuracy
print(accuracy_score(y_true=y_test, y_pred=y_predicted)) 


########################################################################
# normalize data
scaler = StandardScaler()
scaler.fit(X_train) # determine mean and standard deviation
X_train_normalized = scaler.transform(X_train)
X_test_normalized = scaler.transform(X_test)

# use kNN with k = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_normalized, y_train)
y_predicted = knn.predict(X_test_normalized)

# compute accuracy
print(accuracy_score(y_true=y_test, y_pred=y_predicted)) 
