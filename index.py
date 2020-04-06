import csv

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# #############################################################################
# Init data
X = []
y = []

with open('data/response.csv') as f:
    data = csv.reader(f)
    for line in data:
        row = line[0].split(';')
        X.append(map(lambda x: float(x), row[:10]))
        y.append(row[10:][0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# #############################################################################
# Fit model
params = {}
classifier = DecisionTreeClassifier(**params)

classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

# #############################################################################
# Set metrics
f = open("result/DecisionTreeClassifier.result", "a")
f.write(classification_report(y_test, y_predict))
f.close()

