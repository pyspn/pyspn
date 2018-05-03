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

def main():
    print("Loading models...")

    model_name = 'tmm_16_[7, 8]'
    model = pickle.load(open(model_name, 'rb'))
    leaves_a = model.networks[7].concat_leaves.child_list
    leaves_b = model.networks[8].concat_leaves.child_list

    print("Models loaded")

    def save_activation(leaves, fname):
        sz = int(math.sqrt(len(leaves)))
        print("Saving as " + str(sz) + "x" + str(sz))
        img = np.zeros((sz, sz))
        for (i, leaf) in enumerate(leaves):
            y_idx = int(i % sz)
            x_idx = int(i / sz)
            img[x_idx][y_idx] = leaf.mean

        activation_name = fname + '_activation.csv'
        np.savetxt(activation_name, img, delimiter=",")

    save_activation(leaves_a, 'tmm_1_7')
    save_activation(leaves_b, 'tmm_1_8')

if __name__=='__main__':
    main()
