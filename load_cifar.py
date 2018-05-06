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

    for (i, img_keys) in enumerate(model.networks):
        network = model.networks[img_keys]
        leaves = network.concat_leaves.child_list

        structure = network.structure
        num_channels = structure.num_channels

        area = int(structure.x_size * structure.y_size)
        img = np.zeros((structure.x_size, structure.y_size, num_channels))
        for (leaf_idx, leaf) in enumerate(leaves):
            leaf_channel = network.leaves_network_id[leaf_idx]
            input_index = network.leaves_input_indices[leaf_idx]
            x_idx = int(input_index % structure.x_size)
            y_idx = int(input_index / structure.x_size)
            img[y_idx][x_idx][leaf_channel] = leaf.mean

        img = (img + 0.5).clip(min=0, max=1)

        activation_name = "leaves_" + model_name + "_" + str(img_keys) + ".csv"
        np.savetxt(activation_name, img, delimiter=",")
        print("Activation saved as: " + activation_name)

if __name__=='__main__':
    main()
