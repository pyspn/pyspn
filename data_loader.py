from torch import optim, cuda
import math
import csv
import pdb
import numpy as np
import pickle
from numpy import genfromtxt
import time
import cProfile
import os.path
import sys

from collections import defaultdict, deque
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cifar10.dataset import cifar_extractor

mnist_16 = "mnist_16"
cifar_10 = "cifar_10"

def segment_data(data):
    segmented_data = []
    for i in range(10):
        i_examples = (data[data[:,0] == i][:,1:] / 255) - 0.5
        segmented_data.append(i_examples)

    min_count = min([arr.shape[0] for arr in segmented_data])

    segmented_tensor = np.zeros((10, min_count, 256))

    for i in range(10):
        segmented_tensor[i] = segmented_data[i][:min_count]

    return segmented_tensor

def load_data(filename):
    if filename == mnist_16:
        print("Loading data...")
        train_raw = genfromtxt('train_mnist_16.csv', delimiter=',')
        test_raw = genfromtxt('test_mnist_16.csv', delimiter=',')

        print("Data loaded!\nSegmenting data...")
        train_data = segment_data(train_raw)
        test_data = segment_data(test_raw)
        print("Data segmented!")

        return (train_data, test_data)
    if filename == cifar_10:
        print("Loading data...")
        (train_data, test_data) = cifar_extractor.get_cifar_10_train_test()
        print("Data loaded!")

        return (train_data, test_data)

    return
