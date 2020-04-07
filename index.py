import csv

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# #############################################################################
# Init data
X = []
y = []
sc = StandardScaler()

with open('data/response.csv') as f:
    data = csv.reader(f)
    for line in data:
        row = line[0].split(';')
        X.append(map(lambda x: float(x), row[:10]))
        y.append(row[10:][0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Feature Scaling
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# #############################################################################
# Fit model
params = {'n_neighbors': 11}
classifier = KNeighborsClassifier(**params)

classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

# #############################################################################
# Set metrics
f = open("result/KNeighborsClassifier.result", "a")
f.write('Cross Validation: ' + str(np.mean(cross_val_score(classifier, X_train, y_train, cv=5))) + '\n\n')
f.write(classification_report(y_test, y_predict))
f.close()

