import csv

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score


attributes = [
    'MEAN_MOTION',
    'ECCENTRICITY',
    'INCLINATION',
    'RA_OF_ASC_NODE',
    'ARG_OF_PERICENTER',
    'MEAN_ANOMALY',
    'BSTAR',
    'MEAN_MOTION_DOT',
    'MEAN_MOTION_DDOT'
]

result = {}

data = pd.read_csv('./data/data.csv', sep=';', header=0)
data = data.loc[data['TYPE'] == 'Human Spaceflight']

for attribute_name in attributes:
    X = data['EPOCH']
    y = data[attribute_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train = np.array(X_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    params = {
        'n_estimators': 500,
        'max_depth': 4,
        'min_samples_split': 2,
        'learning_rate': 0.01,
        'loss': 'ls'
    }
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(np.array(X_train), y_train.ravel())
    X_test_predict = clf.predict(X_test)

    mse = mean_squared_error(y_test, X_test_predict)
    r2 = r2_score(y_test, X_test_predict)
    rmse = sqrt(mse)

    result[attribute_name] = {
        'mse': mse,
        'r2': r2,
        'rmse': rmse
    }

    mse = "mse: %.4f " % mse
    r2 = "r2: %.4f " % r2
    rmse = "rmse: %.4f " % rmse

    print(attribute_name, rmse)

    fig, image = plt.subplots(figsize=(16, 9))

    image.set_title(mse + '; ' + rmse + '; ' + r2)
    image.plot(y_test, label="Data")
    image.plot(clf.predict(X_test), label="Regression")
    image.legend()

    fig.savefig('./result/' + attribute_name + '.png')
    plt.close(fig)
    plt.cla()

min_type = ''
min = 100

for attribute, values in result.items():
    if min > values['rmse']:
        min_type = attribute
        min = values['rmse']

print(min_type, min)
