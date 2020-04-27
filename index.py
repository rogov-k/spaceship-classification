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

data = pd.read_csv('./data/data.csv', sep=';', header=0)

for attribute_name in attributes:
    fig, image = plt.subplots(len(classes), figsize=(20, 9 * len(classes)))
    fig.suptitle(attribute_name)
    for index, class_name in enumerate(classes):
        class_data = data.loc[data['TYPE'] == class_name]
        norads = class_data['NORAD'].unique()
        for y, norad in enumerate(norads):
            norad_data = class_data.loc[data['NORAD'] == norad]
            image[index].scatter(norad_data['EPOCH'], norad_data[attribute_name], s=3)
        image[index].set_title(class_name)
        image[index].set_xlabel("Time (Min)")
        image[index].set_ylabel("Value")
    fig.savefig('result/' + attribute_name + '.png')
    plt.close(fig)
    plt.cla()
