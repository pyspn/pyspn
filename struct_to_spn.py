import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque
import os.path
import sys
from struct_gen import *
from matrix_gen import *
import math
from random import randint

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TorchSPN.src import network, param, nodes

debug = False
def dbg(debug_text):
    if debug:
        print(debug_text)

np.random.seed(123)

class MatrixSPN(network.Network):
    def __init__(self, structure, shared_parameters, is_cuda=False):
        '''
        Initialize the lists of nodes and edges
        '''
        super(MatrixSPN, self).__init__(is_cuda=is_cuda)

        self.structure = structure
        self.is_cuda = is_cuda

        self.shared_parameters = shared_parameters

        self.root = None
        self.leaves = []
        self.concat_leaves = None
        self.layers = []

        self.leaves_input_indices = None
        self.leaves_network_id = None
        self.num_leaves_per_network = None

        self.generate_network()

    def get_mapped_input_dict(self, data):
        val_dict = {}
        cond_mask_dict = {}

        for (i, leaf) in enumerate(self.leaves):
            network_id = self.leaves_network_id[i]
            relative_input_index = self.leaves_input_indices[i]

            input_index = network_id * self.num_leaves_per_network + relative_input_index

            _val = np.array([ data[:, input_index] ])
            val_dict[leaf] = _val
            cond_mask_dict[leaf] = np.zeros(_val.shape)

        return (val_dict, cond_mask_dict)

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
        leaves = self.AddMultinomialNodes(
            n_variable=num_leaves,
            n_out=1,
            list_n_values=[2] *  num_leaves,
            list_prob=list_prob,
            parameters=self.shared_parameters)

        self.leaves = [leaves]

        return leaves

    def generate_gaussian_leaves(self, metadata):
        num_leaves = metadata.num_nodes_by_level[-1]

        leaves = []
        for i in range(num_leaves):
            mean = np.random.normal(0, 1) + randint(1, 10)
            std = np.random.uniform(0.006, 5)

            mean = np.array([mean], dtype='float32')
            std = np.array([std], dtype='float32')

            leaf = self.AddGaussianNodes(
                mean,
                std,
                parameters=self.shared_parameters)
            leaves.append(leaf)

        self.leaves = leaves
        concatenated_leaves = self.AddConcatLayer(leaves)
        self.concat_leaves = concatenated_leaves

        return concatenated_leaves

    def compute_prob(self, x):
        (val_dict, cond_mask_dict) = self.get_mapped_input_dict(np.array([x]))

        return self.ComputeProbability(
            val_dict=val_dict,
            cond_mask_dict=cond_mask_dict,
            grad=False,
            log=False)

    def ComputeTMMLoss(self, val_dict=None, cond_mask_dict={}):
        log_p_tilde = self.ComputeLogUnnormalized(val_dict)

        marginalize_dict = {}
        for k in cond_mask_dict:
            marginalize_dict[k] = 1 - cond_mask_dict[k]

        log_Z = self.ComputeLogUnnormalized(val_dict, marginalize_dict)

        J = torch.sum(- log_p_tilde + log_Z) #  negative log-likelihood

        return J

    def generate_network(self):
        if debug:
            self.structure.print_stat()

        metadata = CVMetaData(self.structure)

        self.leaves_input_indices = metadata.leaves_input_indices
        self.num_leaves_per_network = metadata.num_leaves_per_network
        self.leaves_network_id = metadata.leaves_network_id

        # create leaves
        leaves = self.generate_gaussian_leaves(metadata)

        # create layers bottom-up
        prev_layer = leaves
        for level in reversed(range(metadata.depth - 1)):
            type = metadata.type_by_level[level]
            num_nodes = metadata.num_nodes_by_level[level]
            mask = metadata.masks_by_level[level]

            cur_layer = None
            if type == sum_type:
                cur_layer = self.AddSumNodes(num_nodes)
                weights = self.initialize_weights_from_mask(mask)

                dbg("Level " + str(level) + ": " + str(weights))

                self.AddSumEdges(
                    prev_layer,
                    cur_layer,
                    weights,
                    mask,
                    parameters=self.shared_parameters)
            else:
                cur_layer = self.AddProductNodes(num_nodes)
                self.AddProductEdges(
                    prev_layer,
                    cur_layer,
                    mask)

            self.layers.append(cur_layer)
            prev_layer = cur_layer

        self.root = prev_layer
