import pickle
import numpy as np
import csv
from numpy import genfromtxt
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data_and_labels():
    data = []
    labels = []

    for i in range(5):
        id = str(i + 1)
        filename = "cifar10/dataset/data_batch_" + id
        batch = unpickle(filename)

        labels_key = 'labels'.encode()
        batch_labels = batch[labels_key]
        labels.extend(batch_labels)

        data_key = 'data'.encode()
        raw_batch_data = batch[data_key]

        batch_data = []
        for i in range(len(raw_batch_data)):
            np_data = (np.array(raw_batch_data[i], dtype='float32') / 255) - 0.5
            batch_data.append(np_data)

        data.extend(batch_data)

    return (data, labels)

def get_segmented_data():
    (data, labels) = get_data_and_labels()

    num_data = len(labels)
    num_labels = 10

    segmented_data = [[] for i in range(num_labels)]
    for i in range(num_data):
        segmented_data[ labels[i] ].append(data[i])

    return segmented_data

def visualize_image(segmented_data, img_key, idx):
    flattened_img = segmented_data[img_key][idx]

    sz = math.sqrt(len(flattened_img) / 3)
    img = np.zeros((sz, sz, 3))
    area = sz * sz

    for (i, pix) in enumerate(flattened_img):
        x_idx = int((i % area) % sz)
        y_idx = int((i % area) / sz)
        color = int(i / area)
        img[y_idx][x_idx][color] = pix

    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    segmented_data = get_segmented_data()
    visualize_image(segmented_data, 2, 0)

