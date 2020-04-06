# coding=utf-8

import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score


def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)

    return rmse_val


# #############################################################################
body = []
label = []

with open('data/response.csv') as f:
    data = csv.reader(f)
    for line in data:
        row = map(lambda x: float(x), line[0].split(';'))
        body.append(row[:10])
        label.append(row[10:][0])

# #############################################################################
# Загружаем данные
#
# X_train - обучающие данные
# X_train - обучающие лейблы
#
# X_test - тестовые данные
# y_test - тестовые лейблы
#
X, y = shuffle(body, label, random_state=11)
offset = int(len(body) * .9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# #############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
X_test_predict = clf.predict(X_test)
mse = mean_squared_error(y_test, X_test_predict)
rmse = rmse(y_test, X_test_predict)
r2 = r2_score(y_test, clf.predict(X_test))

print("MSE: %.4f" % mse)
print("RMSE: %.4f" % rmse)
print("R^2: %.4f" % r2)

# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_predict in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_predict)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('result/learning_rate.png')
