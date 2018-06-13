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
from reordering.masks import reorder

use_sparse = True
debug = True
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
        self.masks = []

        self.leaves_input_indices = None
        self.leaves_network_id = None
        self.num_leaves_per_network = None

        self.generate_network()

    def get_mapped_input_dict(self, data):
        val_dict = {}
        cond_mask_dict = {}

        for (i, x) in enumerate(data):
            leaf = self.leaves[i]
            val_dict[leaf] = x
            cond_mask_dict[leaf] = np.zeros(x.shape)

        return (val_dict, cond_mask_dict)

    def initialize_weights_from_mask(self, mask):
        weights = np.random.uniform(10, 1000, mask.shape).astype('float32')
        
        return weights

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

        mean = np.random.uniform(-0.5, 0.5, num_leaves).astype('float32')
        std = np.random.uniform(0.006, 1, num_leaves).astype('float32')

        leaf = self.AddGaussianNodes(
            mean,
            std,
            parameters=self.shared_parameters)
        self.leaves.append(leaf)

        return leaf

    def compute_prob(self, x):
        (val_dict, cond_mask_dict) = self.get_mapped_input_dict(np.array([x]))

        return self.ComputeProbability(
            val_dict=val_dict,
            cond_mask_dict=cond_mask_dict,
            grad=False,
            log=False)

    def ComputeTMMLoss(self, val_dict=None, cond_mask_dict={}, debug=False):
        log_p_tilde = self.ComputeLogUnnormalized(val_dict)

        marginalize_dict = {}
        for k in cond_mask_dict:
            marginalize_dict[k] = 1 - cond_mask_dict[k]

        log_Z = self.ComputeLogUnnormalized(val_dict, marginalize_dict)

        J = - log_p_tilde + log_Z #  negative log-likelihood

        if debug:
            del log_Z, log_p_tilde
            return J

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

        reorder_metadata = []

        if use_sparse:
            self.AddSumNodeWeights(metadata.weights, parameters=self.shared_parameters)

        # create layers bottom-up
        prev_layer = leaves
        for level in reversed(range(metadata.depth - 1)):
            type = metadata.type_by_level[level]
            num_nodes = metadata.num_nodes_by_level[level]

            cur_layer = None
            if use_sparse:
                connections = metadata.connections_by_level[level]

                if type == sum_type:
                    cur_layer = self.AddSparseSumNodes(num_nodes)
                    weight_indices = metadata.weight_indices_by_level[level]

                    self.AddSparseSumEdges(
                        prev_layer,
                        cur_layer,
                        connections,
                        weight_indices)
                else:
                    cur_layer = self.AddSparseProductNodes(num_nodes)
                    self.AddSparseProductEdges(
                        prev_layer,
                        cur_layer,
                        connections)
            else:
                mask = metadata.masks_by_level[level]
                self.masks.append(mask)
                if type == sum_type:
                    cur_layer = self.AddSumNodes(num_nodes)
                    weights = self.initialize_weights_from_mask(mask)

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

        print("Done")
        return

def main():
    structure = MultiChannelConvSPN(8, 1, 1, 2, 2, 1)
    shared_parameters = param.Param()

    network = MatrixSPN(structure, shared_parameters, is_cuda=False)
    pass

if __name__ == '__main__':
    main()
