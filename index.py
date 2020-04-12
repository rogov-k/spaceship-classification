# coding=utf-8
import os
import matplotlib.pyplot as plt
import pandas as pd


directory = './data/type/'
files = os.listdir(directory)
types = map(lambda file_name: file_name.split('.')[0], files)

for type in types:
    data = pd.read_csv(directory + type + '.csv', sep=';', header=0)
    headers = list(data.columns.values)
    headers.pop(-1)
    for index in headers:
        fig = plt.figure(figsize=(30, 10))
        plt.plot(data[index])
        plt.title(index)
        try:
            os.mkdir('result/graphic/' + type + '/')
        except OSError as error:
            print(error)
        plt.savefig('result/graphic/' + type + '/' + index + '.png', dpi=100)
