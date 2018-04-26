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

def main():
    model_name = 'flatspn_6_2'
    model = pickle.load(open(model_name, 'rb'))
    leaves = model.network.concat_leaves.child_list

    img = np.zeros((28, 28))
    for (i, leaf) in enumerate(leaves):
        x_idx = i % 28
        y_idx = i / 28

        img[x_idx][y_idx] = leaf.mean

    activation_name = model_name + '_activation.csv'
    np.savetxt(activation_name, img, delimiter=",")

if __name__=='__main__':
    main()
