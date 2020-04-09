import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


# #############################################################################
# Init data
train = pd.read_csv('data/train.csv', sep=';', header=0)
y_train = train.TYPE
X_train = train.drop('TYPE', axis=1)

test = pd.read_csv('data/test.csv', sep=';', header=0)
y_test = test.TYPE
X_test = test.drop('TYPE', axis=1)

# #############################################################################
# Fit model

# * Decision Tree Classifier
print('Start: "Decision tree" classifier')
params = {}
classifier = DecisionTreeClassifier(**params)

classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

f = open("result/DecisionTreeClassifier.result", "a")
f.write('Cross Validation: ' + str(np.mean(cross_val_score(classifier, X_train, y_train, cv=5))) + '\n\n')
f.write(classification_report(y_test, y_predict))
f.close()
