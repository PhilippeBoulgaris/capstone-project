import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# load data
reviews = pd.read_csv("Amazon_Unlocked_Mobile.csv", encoding='utf-8')

X = reviews.iloc[:, 4].values
X_clean = X[pd.notnull(X)]
y = reviews.iloc[:, 3].values
y_clean = y[pd.notnull(X)]

# convert documents into bags-of-words
vectorizer = CountVectorizer()
X_cnt = vectorizer.fit_transform(X_clean)

# split into training data (80%) and test data (20%)
X_train, X_test, y_train, y_test = train_test_split(X_cnt, y_clean, test_size=0.2, random_state=0)

# train naive Bayes classifier with a small non-zero alpha
nb = MultinomialNB(alpha=1e-10)
nb.fit(X_train, y_train)

# predict labels
y_predicted = nb.predict(X_test)

# compute confusion matrix and accuracy
print(confusion_matrix(y_true=y_test, y_pred=y_predicted))
print(accuracy_score(y_true=y_test, y_pred=y_predicted))

# print class priors
for c in range(len(nb.classes_)):
    print('Class: ' + str(c))
    print(str(math.exp(nb.class_log_prior_[c])))

# updated way to get feature names
feature_names = vectorizer.get_feature_names()


# print probabilities per class for specific words
words = ['android', 'apple', 'good', 'bad', 'terrible', 'error', 'crash']
for w in words:
    if w in feature_names:
        print('Word: ' + w)
        index = list(feature_names).index(w)
        for c in range(len(nb.classes_)):
            print(str(c) + " : " + str(math.exp(nb.feature_log_prob_[c][index])))
    else:
        print(f"Word '{w}' not found in the vocabulary.")
