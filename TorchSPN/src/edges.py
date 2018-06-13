#!/usr/bin/env python3

import torch
import numpy as np
import pdb

EPSILON = 0.00001

class SparseProductEdges():
    def __init__(self, child, parent, indices):
        self.parent = parent
        self.child = child

        self.flattened_indices = None
        self.dim = None

        self.preprocess(indices)

    def preprocess(self, idx):
        max_length = max([len(a) for a in idx])
        new_idx = []
        for ii in range(len(idx)):
            id = idx[ii]
            new_idx.extend(id)

            buffer = [new_idx[-1]] * (max_length - len(id))
            new_idx.extend(buffer)

        self.flattened_indices = torch.LongTensor(new_idx)
        self.dim = (len(idx), max_length)

class SparseSumEdges():
    def __init__(self, child, parent, connections, weight_indices):
        self.parent = parent
        self.child = child

        self.flattened_indices = None
        self.connection_weight_indices = None
        self.dim = None

        self.preprocess(connections, weight_indices)

    def preprocess(self, idx, wgt):
        max_length = max([len(a) for a in idx])
        new_idx = []
        new_weight = []
        for ii in range(len(idx)):
            id = idx[ii]
            new_idx.extend(id)

            buffer = [new_idx[-1]] * (max_length - len(id))
            new_idx.extend(buffer)

            wg = wgt[ii]
            new_weight.extend(wg)

            buffer_wg = [0] * (max_length - len(wg))
            new_weight.extend(buffer_wg)

        new_weight = np.array(new_weight, dtype='float32')
        part_lg = len(idx)

        self.flattened_indices = torch.LongTensor(new_idx)
        self.connection_weight_indices = torch.from_numpy(new_weight)
        self.dim = (part_lg, max_length)

class ProductEdges():
    '''
    # the class for a set of edges
    '''
    def __init__(self, child, parent, mask):
        '''
        Initialize a set of product edges
        :param child: child layer, with size: num_in
        :param parent: parent layer, with size: num_out
        :param mask: masks out unconnected edges, with size: num_in x num_out
        '''
        self.parent = parent
        self.child = child
        self.mask = mask


class SumEdges():
    '''
    # the class for a set of edges
    '''
    def __init__(self, child, parent, weights, mask):
        '''
        Initialize a set of product edges
        :param child: child layer, with size: num_in
        :param parent: parent layer, with size: num_out
        :param weights: size: num_in x num_out
        :param mask: masks out unconnected edges, with size: num_in x num_out
        '''
        self.parent = parent
        self.child = child
        self.weights = weights
        self.mask  = mask

    def sum_weight_hook(self):
        if self.weights.size()[0] == 30:
            #pdb.set_trace()
            pass

        self.weights.data = self.weights.data.clamp(min=EPSILON)
