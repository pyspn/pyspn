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

from train_cifar import *

def save_activations(model, model_name):
    for (i, img_keys) in enumerate(model.networks):
        network = model.networks[img_keys]
        leaves_mean = network.leaves[0].mean

        structure = network.structure
        num_channels = structure.num_channels

        area = int(structure.x_size * structure.y_size)
        img = np.zeros((area, num_channels))
        for (leaf_idx, leaf_mean) in enumerate(leaves_mean):
            leaf_channel = network.leaves_network_id[leaf_idx]
            input_index = leaf_idx % area
            img[input_index][leaf_channel] = leaf_mean

        activation_name = "leaves_" + model_name + "_" + str(img_keys) + ".csv"
        np.savetxt(activation_name, img, delimiter=",")
        print("Activation saved as: " + activation_name)

def generate_sample(model, model_name):
    for img_keys in model.networks:
        network = model.networks[img_keys]
        img_size = network.structure.x_size * network.structure.y_size
        n_var = img_size * 3
        node2var = [network.leaves_network_id[i] * img_size + network.leaves_input_indices[i] for i in range(n_var)]
        img = network.GenSample(node2var, n_batch=1, n_var=n_var)[0]

        img = (img + 0.5).clip(min=0, max=1).reshape((img_size, 3))

        sample_name = "sample_" + model_name + "_" + str(img_keys) + ".csv"
        np.savetxt(sample_name, img, delimiter=",")
        print("Sample saved as: " + sample_name)

def main():
    print("Loading models...")

    model_name = 'cifar_oneimg_2'
    model = pickle.load(open(model_name, 'rb'))

    save_activations(model, model_name)

if __name__=='__main__':
    main()
