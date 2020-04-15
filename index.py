import os

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


def get_norads(class_name):
    return map(lambda x: x.split('.')[0], os.listdir('data/' + class_name))


# class -> attribute -> norad
def get_empty_graphic_data():
    graphics_data = {}
    for class_name in classes:
        graphics_data[class_name] = {}
        norads = get_norads(class_name)
        for attribute_name in attributes:
            graphics_data[class_name][attribute_name] = {}
            for norad in norads:
                graphics_data[class_name][attribute_name][norad] = []

    return graphics_data


path = 'data/'
graphics_data = get_empty_graphic_data()
for class_name in classes:
    files = os.listdir(path + class_name)
    for file_name in files:
        data = pd.read_csv(path + class_name + '/' + file_name, sep=';', header=0)
        norad = file_name.split('.')[0]
        for i, row in data.iterrows():
            for attribute_name in attributes:
                graphics_data.get(class_name) \
                    .get(attribute_name) \
                    .get(norad) \
                    .append(row[attribute_name])

for class_name in classes:
    norads = get_norads(class_name)
    for attribute_name in attributes:
        last_index = 0
        for i, norad in enumerate(norads):
            index = i % 5
            last_index = index
            if index == 0:
                fig, image = plt.subplots(5, figsize=(16, 45))
                fig.suptitle(attribute_name)
            value = graphics_data.get(class_name).get(attribute_name).get(norad)
            image[index].scatter(range(len(value)), value)
            image[index].set_title(norad)
            image[index].set_xlabel("Time")
            image[index].set_ylabel("Value")
            if index == 4:
                if not os.path.exists('result/' + class_name + '/' + attribute_name):
                    os.mkdir('result/' + class_name + '/' + attribute_name)
                fig.savefig('result/' + class_name + '/' + attribute_name + '/' + str(i // 5) + '.png')
                plt.close(fig)
                plt.cla()
        if last_index != 4:
            if not os.path.exists('result/' + class_name + '/' + attribute_name):
                os.mkdir('result/' + class_name + '/' + attribute_name)
            fig.savefig('result/' + class_name + '/' + attribute_name + '/_.png')
            plt.close(fig)
            plt.cla()
