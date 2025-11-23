import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

########################################################################

# load data
cars = pd.read_csv('auto-mpg.data.txt',header=None, sep='\s+')

# extract power and weight as data matrix X
X = cars.iloc[:, [3,4]].values

# extract origin as target vector y
y = cars.iloc[:, 7].values

# split into training data (80%) and test data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit logistic regression model on training data
lr = LogisticRegression()

# use kNN with k = 3
knn = KNeighborsClassifier(n_neighbors=3)

# learn decision tree
tree = DecisionTreeClassifier(criterion='entropy')

# voting classifier
vc = VotingClassifier(estimators=[('lr',lr),('knn',knn), ('tree', tree)], voting='hard')
vc.fit(X_train, y_train)
vc_y_predicted = vc.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=vc_y_predicted)) # 0.683544303797


