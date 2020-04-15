import matplotlib.pyplot as plt
import pandas as pd


classes = [
    'Amateur',
    'Globalstar',
    'Human Spaceflight',
    'Intelsat',
    'Iridium',
    'Navigation',
    'Orbcomm',
    'Weather'
]

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

graphics_data = {}

for attribute_name in attributes:
    graphics_data[attribute_name] = {}
    for class_name in classes:
        graphics_data[attribute_name][class_name] = []

data = pd.read_csv('data/response.csv', sep=';', header=0)

for i, j in data.iterrows():
    for attr in attributes:
        graphics_data.get(attr).get(j['TYPE']).append(j[attr])

for attribute_name, classes_ in graphics_data.items():
    fig, image = plt.subplots(figsize=(16, 9))
    for class_name, value in classes_.items():
        index = classes.index(class_name)
        image.scatter([index]*len(value), value, label=class_name, alpha=0.3, s=[10]*len(value))
    image.legend()
    image.set_title(attribute_name)
    image.set_xlabel("Classes")
    image.set_ylabel("Value")
    fig.savefig('result/' + attribute_name + '.png', dpi=100)
