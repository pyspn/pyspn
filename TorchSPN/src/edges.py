#!/usr/bin/env python3

import torch
import numpy as np

EPSILON = 0.00001

class SparseProductEdges():
    def __init__(self, child, parent, mask_matrix):
        self.parent = parent
        self.child = child

        self.flattened_indices = None
        self.dim = None

        self.preprocess(mask_matrix)

    def preprocess(self, mask_matrix):
        idx = self.get_sparse_idx(mask_matrix)

        max_length = max([len(a) for a in idx])
        new_idx = []
        for ii in range(len(idx)):
            id = idx[ii]
            new_idx.extend(id)

            buffer = [new_idx[-1]] * (max_length - len(id))
            new_idx.extend(buffer)

        new_idx = torch.tensor(new_idx)
        part_lg = len(idx)

        self.flattened_indices = new_idx
        self.dim = (part_lg, max_length)

    def get_sparse_idx(self, mask_matrix):
        indices = []

        num_cols = len(mask_matrix[0])
        num_rows = len(mask_matrix)

        for c in range(num_cols):
            col_idx = []
            col_wg = []
            for r in range(num_rows):
                if mask_matrix[r][c]:
                    col_idx.append(r)
            col_idx =  torch.tensor(col_idx)
            col_wg = torch.FloatTensor(col_wg)

            indices.append(col_idx)

        return indices

class SparseSumEdges():
    def __init__(self, child, parent, mask_matrix, weight_matrix):
        self.parent = parent
        self.child = child

        self.flattened_indices = None
        self.connection_weights = None
        self.dim = None

        self.preprocess(mask_matrix, weight_matrix)

    def preprocess(self, mask_matrix, weight_matrix):
        (idx, wgt) = self.get_sparse_idx_wgt(mask_matrix, weight_matrix)

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

        new_idx = torch.tensor(new_idx)
        new_weight = torch.FloatTensor(new_weight)
        part_lg = len(idx)

        self.flattened_indices = new_idx
        self.connection_weights = new_weight
        self.dim = (part_lg, max_length)

    def get_sparse_idx_wgt(self, mask_matrix, weight_matrix):
        indices = []
        sparse_weights = []

        num_cols = len(mask_matrix[0])
        num_rows = len(mask_matrix)

        for c in range(num_cols):
            col_idx = []
            col_wg = []
            for r in range(num_rows):
                if mask_matrix[r][c]:
                    col_idx.append(r)
                    col_wg.append(weight_matrix[r][c])
            col_idx =  torch.tensor(col_idx)
            col_wg = torch.FloatTensor(col_wg)

            indices.append(col_idx)
            sparse_weights.append(col_wg)

        return (indices, sparse_weights)

    def sum_weight_hook(self):
        self.connection_weights.data = self.connection_weights.data.clamp(min=EPSILON)

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
        self.weights.data = self.weights.data.clamp(min=EPSILON)
