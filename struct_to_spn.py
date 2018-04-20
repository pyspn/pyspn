import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque
import os.path
import sys
from struct_gen import *
from matrix_gen import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TorchSPN.src import network, param, nodes

debug = False
def dbg(debug_text):
    if debug:
        print(debug_text)

class MatrixSPN(torch.nn.Module):
    def __init__(self, x_size, y_size, sum_shifts, prd_subdivs, is_cuda = False):
        '''
        Initialize the lists of nodes and edges
        '''
        super(MatrixSPN, self).__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.sum_shifts = sum_shifts
        self.prd_subdivs = prd_subdivs
        self.is_cuda = is_cuda

        self.network = network.Network(is_cuda=is_cuda)
        self.parameters = param.Param()

        self.root = None
        self.leaves = []
        self.concat_leaves = None
        self.layers = []

        self.val_dict = None
        self.cond_mask_dict = None

        self.generate_network()
        self.setup_parameters()

    def feed(self, data):
        self.val_dict = {}
        self.cond_mask_dict = {}

        for (i, leaf) in enumerate(self.leaves):
            self.val_dict[leaf] = np.array([[ data[i] ]])
            self.cond_mask_dict[leaf] = np.array([[ 0 ]])

    def forward(self):
        self.val = self.network.ComputeProbability(
            val_dict=self.val_dict,
            cond_mask_dict=self.cond_mask_dict,
            grad=True,
            log=False)

        if debug:
            print("Leaf: " + str(torch.max(self.concat_leaves.val)))

            for (i, layer) in enumerate(self.layers):
                type = str(layer.__class__)
                print(type + " layer " + str(i) + " :" + str(layer.val))

        return self.val

    def initialize_weights_from_mask(self, mask):
        weights = np.random.normal(1, 0.2, mask.shape).astype('float32')
        return weights.clip(min=0.5,max=1.5)

    def generate_multinomial_leaves(self, metadata):
        num_leaves = metadata.num_nodes_by_level[-1]

        list_prob = [
            np.array([
                [0.6],
                [0.4]
            ])
        ] * num_leaves
        leaves = self.network.AddMultinomialNodes(
            n_variable=num_leaves,
            n_out=1,
            list_n_values=[2] *  num_leaves,
            list_prob=list_prob,
            parameters=self.parameters)

        self.leaves = [leaves]

        return leaves

    def generate_gaussian_leaves(self, metadata):
        num_leaves = metadata.num_nodes_by_level[-1]

        leaves = []
        for i in range(num_leaves):
            mean = np.random.normal(0, 1)
            std = np.random.normal(1, 0.2)
            if std < .6:
                std = .6
            mean = np.array([mean], dtype='float32')
            std = np.array([std], dtype='float32')

            leaf = self.network.AddGaussianNodes(
                mean,
                std,
                parameters=self.parameters)
            leaves.append(leaf)

        self.leaves = leaves
        concatenated_leaves = self.network.AddConcatLayer(leaves)
        self.concat_leaves = concatenated_leaves

        return concatenated_leaves

    def generate_network(self):
        structure = ConvSPN(
            self.x_size,
            self.y_size,
            self.sum_shifts,
            self.prd_subdivs)

        if debug:
            structure.print_stat()

        metadata = CVMetaData(structure)

        # create leaves
        leaves = self.generate_gaussian_leaves(metadata)
        # leaves = self.generate_multinomial_leaves(metadata)

        # create layers bottom-up
        prev_layer = leaves
        for level in reversed(range(metadata.depth - 1)):
            type = metadata.type_by_level[level]
            num_nodes = metadata.num_nodes_by_level[level]
            mask = metadata.masks_by_level[level]

            cur_layer = None
            if type == sum_type:
                cur_layer = self.network.AddSumNodes(num_nodes)
                weights = self.initialize_weights_from_mask(mask)

                self.network.AddSumEdges(
                    prev_layer,
                    cur_layer,
                    weights,
                    mask,
                    parameters=self.parameters)
            else:
                cur_layer = self.network.AddProductNodes(num_nodes)
                self.network.AddProductEdges(
                    prev_layer,
                    cur_layer,
                    mask)

            self.layers.append(cur_layer)
            prev_layer = cur_layer

        self.root = prev_layer

    def setup_parameters(self):
        self.parameters.register(self.network)
        self.parameters.proj()
