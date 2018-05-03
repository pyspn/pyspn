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
test_raw = genfromtxt('mnist_test.csv', delimiter=',')

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
    for digit in models.digits:
        network = models.networks[digit]
        loss = compute_prob(network, x)
        losses.append(loss)

    losses = np.array(losses)
    prediction_index = np.argmax(losses)
    predicted_digit = models.digit[prediction_index]

    return predicted_digit

model = pickle.load(open('tmm_[7, 8]', 'rb'))

num_tests = 100
error = 0
for i in range(num_tests):
    for digit in models.digit:
        x = segmented_data[digit][i]
        prediction = predict(model, x)
        if prediction != digit:
            error += 1

pdb.set_trace()

#num_tests = 100
#error = 0
#for test_i in range(num_tests):
#    for digit in range(10):
#        num_tests += 1
#        data = segment_data[digit][test_i]
#        y_pred = model.classify_data(data)
#        if (model.digit == digit) != y_pred:
#            error += 1

#pdb.set_trace()
