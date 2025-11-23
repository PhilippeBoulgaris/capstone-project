import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

# learn decision tree of maximal depth 2
tree = DecisionTreeClassifier(criterion='entropy', max_depth=2)

# bagging classifier based on 10 decision trees
bc = AdaBoostClassifier(base_estimator=tree, n_estimators=30)
bc.fit(X_train, y_train)
bc_y_predicted = bc.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=bc_y_predicted)) # 0.620253164556962




