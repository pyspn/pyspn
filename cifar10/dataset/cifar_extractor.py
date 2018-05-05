import pickle
import numpy as np

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
        batch_data = batch[data_key]

        for i in range(len(batch_data)):
            np_data = (np.array(batch_data[i], dtype='float32') / 255) - 0.5
            batch_data[i] = np_data

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

if __name__ == '__main__':
    get_segmented_data()
