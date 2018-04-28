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

from train_mnist import *

model_name = 'spn_7'
model = pickle.load(open(model_name, 'rb'))

n_var = 784
node2var = [i for i in range(n_var)]
sample = model.network.GenSample(node2var, n_batch=100, n_var=n_var)[7]

img = np.zeros((28, 28))
for (i, pixel) in enumerate(sample):
    y_idx = i % 28
    x_idx = i / 28

    img[x_idx][y_idx] = pixel

filename = 'sample.csv'
np.savetxt(filename, img, delimiter=',') 
