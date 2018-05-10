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

    model_name = 'mm_2_'
    model = pickle.load(open(model_name, 'rb'))

    for (i, digit) in enumerate(model.networks):
        network = model.networks[digit]
        leaves = network.concat_leaves.child_list

        structure = network.structure
        num_channels = structure.num_channels

        area = int(structure.x_size * structure.y_size)
        channel_leaves = [np.zeros((structure.x_size, structure.y_size)) for c in range(num_channels)]
        for (leaf_idx, leaf) in enumerate(leaves):
            leaf_channel = int(leaves_network_id[leaf_idx])
            input_index = network.leaves_input_indices[leaf_idx]
            x_idx = int(input_index % structure.x_size)
            y_idx = int(input_index / structure.x_size)
            channel_leaves[leaf_channel][y_idx][x_idx] = leaf.mean

        #for channel_idx in range(num_channels):
        img = np.mean( channel_leaves, axis=0 )
        activation_name = model_name + "_" + str(digit) + ".csv"
        np.savetxt(activation_name, img, delimiter=",")
        print("Activation saved as: " + activation_name)

if __name__=='__main__':
    main()
