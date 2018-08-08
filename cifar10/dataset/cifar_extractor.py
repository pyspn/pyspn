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

def get_data_and_labels(filename):
    batch = unpickle(filename)

    labels_key = 'labels'.encode()
    labels = batch[labels_key]

    data_key = 'data'.encode()
    raw_batch_data = batch[data_key]

    data = []
    for i in range(len(raw_batch_data)):
        np_data = (np.array(raw_batch_data[i], dtype='float32') / 255)
        data.append(np_data)

    return (data, labels)

def get_train_data_labels():
    train_data = []
    train_labels = []

    for i in range(5):
        id = str(i + 1)
        train_filename = "cifar10/dataset/data_batch_" + id
        (batch_data, batch_labels) = get_data_and_labels( train_filename )

        train_data.extend(batch_data)
        train_labels.extend(batch_labels)

    return (train_data, train_labels)

def get_test_data_labels():
    return get_data_and_labels("cifar10/dataset/test_batch")

def get_segmented_data(data, labels):
    num_data = len(labels)
    num_labels = 10

    segmented_data = [[] for i in range(num_labels)]
    for i in range(num_data):
        segmented_data[ labels[i] ].append(data[i])

    return np.array(segmented_data, dtype='float32')

def get_cifar_10_train_test():
    (train_data_raw, train_labels_raw) = get_train_data_labels()
    (test_data_raw, test_labels_raw) = get_test_data_labels()

    segmented_training_data = get_segmented_data(train_data_raw, train_labels_raw)
    segmented_test_data = get_segmented_data(test_data_raw, test_labels_raw)

    return (segmented_training_data, segmented_test_data)

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
