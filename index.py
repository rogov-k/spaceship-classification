import pandas as pd
import numpy as np
import os


directory = './data/type/'
files = os.listdir(directory)
types = map(lambda file_name: file_name.split('.')[0], files)

for file in files:
    print(file)

exit(2)

# #############################################################################
# Read data
data = pd.read_csv('data/response.csv', sep=';', header=0)

# Determine correlation
correlation = data.corr()
correlation.to_csv("result/Correlation/base.csv")
correlation = correlation.drop(columns=['EPHEMERIS_TYPE'])
correlation = correlation.drop(['EPHEMERIS_TYPE'])
correlation.to_csv("result/Correlation/base_without_nan.csv")

# Minimum correlation
min_correlation = pd.DataFrame(correlation.min())
min_correlation.columns = ['Min']
min_correlation = min_correlation.sort_values(by='Min')
min_correlation.to_csv("result/Correlation/min.csv")

# Maximum correlation
max_correlation = pd.DataFrame(correlation.apply(lambda x:
                                                 np.max(filter(lambda x: x != 1., x)),
                                                 axis=1))
max_correlation.columns = ['Max']
max_correlation = max_correlation.sort_values(by='Max', ascending=False)
max_correlation.to_csv("result/Correlation/max.csv")
