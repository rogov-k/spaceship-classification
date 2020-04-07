import csv

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
import xgboost as xgb


# #############################################################################
# Init data
X = []
y = []
sc = StandardScaler()
le = preprocessing.LabelEncoder()

with open('data/response.csv') as f:
    data = csv.reader(f)
    for line in data:
        row = line[0].split(';')
        X.append(map(lambda x: float(x), row[:10]))
        y.append(row[10:][0])

# Transform label to simple value (Label encoder)
le.fit(list(set(y)))
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Feature Scaling
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# #############################################################################
# Fit model
params = {}
classifier = xgb.XGBClassifier(**params)

classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

# #############################################################################
# Set metrics
f = open("result/XGBClassifier.result", "a")
f.write('Cross Validation: ' + str(np.mean(cross_val_score(classifier, X_train, y_train, cv=5))) + '\n')
f.write('Mean Absolute Error: ' + str(mean_absolute_error(y_test, y_predict)) + '\n')
f.write('Mean Squared Error: ' + str(mean_squared_error(y_test, y_predict)) + '\n')
f.write('Root Mean Squared Error: ' + str(np.sqrt(mean_squared_error(y_test, y_predict))) + '\n')
f.close()

