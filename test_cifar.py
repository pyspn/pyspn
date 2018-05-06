import torch
from torch import optim, cuda
import pdb
import math
import csv
import pdb
import numpy as np
import pickle
from numpy import genfromtxt

from collections import defaultdict, deque
from struct_to_spn import *
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TorchSPN.src import network, param, nodes
from tmm import *

print("Loading data set..")
test_raw = genfromtxt('train_mnist_16.csv', delimiter=',')

def segment_data():
    segmented_data = []
    for i in range(10):
        i_examples = test_raw[test_raw[:,0] == i][:,1:] / 255 - 0.5
        segmented_data.append(i_examples)

    return segmented_data

print("Segmenting...")
segmented_data = segment_data()
print("Dataset loaded!")

def compute_prob(model, x):
    (val_dict, cond_mask_dict) = model.get_mapped_input_dict(np.array([x]))
    loss = model.ComputeProbability(
            val_dict=val_dict,
            cond_mask_dict=cond_mask_dict,
            grad=False,
            log=True)
    return loss

def predict(model, x):
    losses = []
    for img_key in model.img_keys:
        network = model.networks[img_key]
        loss = compute_prob(network, x)
        losses.append(loss)

    losses = np.array(losses)
    prediction_index = np.argmax(losses)
    predicted_digit = model.img_keys[prediction_index]
    return predicted_digit

model = pickle.load(open('mm_5', 'rb'))

num_tests = 50
error = 0
errors = defaultdict(int)
for i in range(num_tests):
    for img_key in model.img_keys:
        x = segmented_data[img_key][i]
        prediction = predict(model, x)
        if prediction != img_key:
            errors[img_key] += 1
            error += 1

print("Error " + str(error))
print("Detail " + str(errors))

